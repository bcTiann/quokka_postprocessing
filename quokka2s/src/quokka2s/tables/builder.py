from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="collision rates not available")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Mapping, Sequence

import logging
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from despotic.chemistry import NL99, NL99_GC

from .models import (
    AttemptRecord,
    DespoticTable,
    LineLumResult,
    LogGrid,
    SpeciesLineGrid,
    SpeciesRecord,
)
from .solver import LINE_RESULT_FIELDS, calculate_single_despotic_point


LOGGER = logging.getLogger(__name__)
DEFAULT_LINE_RESULT = LineLumResult(*([float("nan")] * len(LINE_RESULT_FIELDS)))

@dataclass(frozen=True)
class SpeciesSpec:
    name: str
    is_emitter: bool

SPECIES_SPECS = (
    SpeciesSpec("H+", False),
    SpeciesSpec("CO", True),
    SpeciesSpec("C", True),
    SpeciesSpec("C+", True),
    SpeciesSpec("HCO+", True),
    SpeciesSpec("e-", False),
)



def build_table(
    nH_grid: LogGrid,
    col_grid: LogGrid,
    T_grid: LogGrid,  
    *,
    species_specs: Sequence[SpeciesSpec],
    chem_network=NL99_GC,
    show_progress: bool = True,
    full_parallel: bool = False,
    workers: int | None = None,
) -> DespoticTable:
    """
    Run DESPOTIC across logarithmic density/column grids and cache the results.

    Parameters
    ----------
    nH_grid : LogGrid
        Hydrogen number-density grid (cm^-3); defines table rows.
    col_grid : LogGrid
        Column-density grid ; defines table columns.
    T_grid : LogGrid    
    species : Sequence[str], optional
        Species for which line luminosities/abundances are tabulated.
    chem_network : callable, optional
        Chemistry network constructor provided to DESPOTIC (defaults to NL99_GC).
    show_progress : bool, optional
        Whether to wrap the row loop with tqdm progress reporting.

    Returns
    -------
    DespoticTable
        Structured table containing per-species line grids, final gas temperatures,
        failure mask, and heating/cooling energy rates on the sampled grid.
    """

    specs = tuple(species_specs)
    nH_vals = nH_grid.sample()
    col_vals = col_grid.sample()
    T_vals = T_grid.sample()

    num_rows, num_cols, num_T = len(nH_vals), len(col_vals), len(T_vals)
    shape = (num_rows, num_cols, num_T)

    tg_table = np.full(shape, np.nan)
    failure_mask = np.zeros(shape, dtype=bool)
    abundance_map = {spec.name: np.full(shape, np.nan) for spec in specs}
    mu_grid = np.full(shape, np.nan) 

    line_buffers: dict[str, dict[str, np.ndarray]] = {
        spec.name: {
            field: np.full(shape, np.nan, dtype=float)
            for field in LINE_RESULT_FIELDS
        }
        for spec in specs
        if spec.is_emitter
    }
    energy_fields: dict[str, np.ndarray] = {}

    def _flatten_energy(term: str, value, target: dict[str, np.ndarray], idx: int) -> None:
        if isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                _flatten_energy(f"{term}.{sub_key}", sub_value, target, idx)
            return
        grid = target.setdefault(term, np.full((num_cols, num_T), np.nan, dtype=float))
        grid[idx] = float(value)

    def _solve_row(row_idx: int) -> tuple[
        int, np.ndarray, np.ndarray, dict[str, dict[str, np.ndarray]],
        dict[str, np.ndarray], dict[str, np.ndarray], list[AttemptRecord]
    ]:
        tg_row = np.full((num_cols, num_T), np.nan)
        failure_row = np.zeros((num_cols, num_T), dtype=bool)
        mu_row = np.full((num_cols, num_T), np.nan)
        line_rows: dict[str, dict[str, np.ndarray]] = {
            spec.name: {field: np.full((num_cols, num_T), np.nan) for field in LINE_RESULT_FIELDS}
            for spec in specs if spec.is_emitter
        }
        abundance_rows: dict[str, np.ndarray] = {
            spec.name: np.full((num_cols, num_T), np.nan) for spec in specs
        }
        energy_rows: dict[str, np.ndarray] = {}
        attempts_row: list[AttemptRecord] = []

        for col_idx, col_val in enumerate(col_vals):
            for t_idx, T_val in enumerate(T_vals):
                line_results, emitter_abunds, chem_abunds, mu_val, final_tg, energy_terms, failed = (
                    calculate_single_despotic_point(
                        nH_val=nH_vals[row_idx],
                        colDen_val=col_val,
                        initial_Tg_guesses=(0,),  # 不再使用，可保留占位
                        species=[spec.name for spec in specs if spec.is_emitter],
                        abundance_only=[spec.name for spec in specs if not spec.is_emitter],
                        chem_network=chem_network,
                        row_idx=row_idx,
                        col_idx=col_idx,
                        T_fixed=T_val,            # 关键
                        log_failures=True,
                        attempt_log=attempts_row,
                    )
                )
                tg_row[col_idx, t_idx] = final_tg
                failure_row[col_idx, t_idx] = failed
                mu_row[col_idx, t_idx] = mu_val 
                
                for spec in specs:
                    abundance_rows[spec.name][col_idx, t_idx] = chem_abunds.get(spec.name, float("nan"))
                    if spec.is_emitter:
                        result = line_results.get(spec.name, DEFAULT_LINE_RESULT)
                        for field in LINE_RESULT_FIELDS:
                            line_rows[spec.name][field][col_idx, t_idx] = getattr(result, field)

                for term, value in energy_terms.items():
                    if isinstance(value, Mapping):
                        for sub_key, sub_val in value.items():
                            grid = energy_rows.setdefault(f"{term}.{sub_key}", np.full((num_cols, num_T), np.nan))
                            if np.isscalar(sub_val):
                                grid[col_idx, t_idx] = float(sub_val)
                        continue
                    grid = energy_rows.setdefault(term, np.full((num_cols, num_T), np.nan))
                    grid[col_idx, t_idx] = float(value)

        return row_idx, tg_row, failure_row, line_rows, abundance_rows, energy_rows, mu_row, attempts_row

    def _solve_point(i: int, j: int, k: int):
        # 调用 solver 固定温度
        line_results, emitter_abunds, chem_abunds, mu_val, final_tg, energy_terms, failed = calculate_single_despotic_point(
            nH_val=nH_vals[i],
            colDen_val=col_vals[j],
            initial_Tg_guesses=(0,),  # 占位，不用
            species=[spec.name for spec in specs if spec.is_emitter],
            abundance_only=[spec.name for spec in specs if not spec.is_emitter],
            chem_network=chem_network,
            row_idx=i,
            col_idx=j,
            T_fixed=T_vals[k],
            log_failures=True,
            attempt_log=None,  # 可选
        )
        # 准备返回一个小包，后面写回表格
        return i, j, k, final_tg, failed, line_results, mu_val, chem_abunds, energy_terms


    if full_parallel:
        tasks = [(i, j, k) for i in range(num_rows) for j in range(num_cols) for k in range(num_T)]
        results = Parallel(n_jobs=workers, prefer="threads")(delayed(_solve_point)(i, j, k) for (i, j, k) in tasks)

        for i, j, k, final_tg, failed, line_results, mu_val, chem_abunds, energy_terms in results:
            tg_table[i, j, k] = final_tg
            failure_mask[i, j, k] = failed
            mu_grid[i, j, k] = mu_val
            for spec in specs:
                abundance_map[spec.name][i, j, k] = chem_abunds.get(spec.name, float("nan"))
                if spec.is_emitter:
                    buf = line_buffers[spec.name]
                    result = line_results.get(spec.name, DEFAULT_LINE_RESULT)
                    for field in LINE_RESULT_FIELDS:
                        buf[field][i, j, k] = getattr(result, field)
            # 能量项展开（同你现在的逻辑，避免 float(dict)）
            for term, value in energy_terms.items():
                if isinstance(value, Mapping):
                    for sub_key, sub_val in value.items():
                        grid = energy_fields.setdefault(f"{term}.{sub_key}", np.full(shape, np.nan))
                        if np.isscalar(sub_val):
                            grid[i, j, k] = float(sub_val)
                    continue
                grid = energy_fields.setdefault(term, np.full(shape, np.nan))
                grid[i, j, k] = float(value)

    else:
        
        if workers is None:
            workers = -1
        tasks = range(num_rows)
        solve_row = partial(_solve_row)

        if show_progress:
            progress = tqdm(total=num_rows, desc="DESPOTIC rows", unit="row")
            with tqdm_joblib(progress):
                results = Parallel(n_jobs=workers)(delayed(solve_row)(row_idx) for row_idx in tasks)
        else:
            results = Parallel(n_jobs=workers)(delayed(solve_row)(row_idx) for row_idx in tasks)

        attempts: list[AttemptRecord] = []
        # 解包顺序要与 _solve_row 返回一致：(..., line_rows, abundance_rows, energy_rows, mu_row, attempts_row)
        for row_idx, tg_row, failure_row, line_rows, abundance_rows, energy_rows, mu_row, attempts_row in results:
            tg_table[row_idx, :, :] = tg_row
            failure_mask[row_idx, :, :] = failure_row
            mu_grid[row_idx, :, :] = mu_row
            attempts.extend(attempts_row)

            for name, row_values in abundance_rows.items():
                abundance_map[name][row_idx, :, :] = row_values
            for name, fields in line_rows.items():
                buffer = line_buffers[name]
                for field, values in fields.items():
                    buffer[field][row_idx, :, :] = values
            for term, values in energy_rows.items():
                grid = energy_fields.setdefault(term, np.full(shape, np.nan))
                grid[row_idx, :, :] = values

        failed_cells = int(np.count_nonzero(failure_mask))
        if failed_cells:
            LOGGER.warning("DESPOTIC table: %s/%s cells failed to converge", failed_cells, num_rows * num_cols * num_T)
        else:
            LOGGER.info("DESPOTIC table converged for all %s cells", num_rows * num_cols * num_T)

        species_data: dict[str, SpeciesRecord] = {}
        for spec in specs:
            abundance_grid = abundance_map[spec.name]
            line_grid = None
            if spec.is_emitter:
                buf = line_buffers[spec.name]
                line_grid = SpeciesLineGrid(
                    freq=buf["freq"],
                    intIntensity=buf["intIntensity"],
                    intTB=buf["intTB"],
                    lumPerH=buf["lumPerH"],
                    tau=buf["tau"],
                    tauDust=buf["tauDust"],
                    abundance=abundance_grid,
                )
            species_data[spec.name] = SpeciesRecord(
                name=spec.name,
                abundance=abundance_grid,
                line=line_grid,
                is_emitter=spec.is_emitter,
            )

        return DespoticTable(
            species_data=species_data,
            tg_final=tg_table,
            nH_values=nH_vals,
            T_values=T_vals, 
            mu_values=mu_grid,
            col_density_values=col_vals,
            failure_mask=failure_mask,
            energy_terms=energy_fields or None,
            attempts=tuple(attempts),
        )



def plot_table(*_args, **_kwargs) -> None:
    """Placeholder for future plotting utilities."""
    raise NotImplementedError("plot_table is not implemented yet.")

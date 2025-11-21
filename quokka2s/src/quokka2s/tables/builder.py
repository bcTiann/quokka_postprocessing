from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="collision rates not available")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")

import csv
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
np.seterr(divide="ignore", invalid="ignore")

from despotic.chemistry import NL99, NL99_GC, GOW
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib 
from tqdm import tqdm
from functools import partial

from .models import AttemptRecord, DespoticTable, LineLumResult, LogGrid, SpeciesLineGrid
from .solver import LINE_RESULT_FIELDS, calculate_single_despotic_point

LOGGER = logging.getLogger(__name__)
DEFAULT_LINE_RESULT = LineLumResult(*([float("nan")] * len(LINE_RESULT_FIELDS)))
ABUNDANCE_ONLY_SPECIES = ("e-",)

def build_table(
    nH_grid: LogGrid,
    col_grid: LogGrid,
    tg_guesses: Sequence[float],
    *,
    species: Sequence[str],
    abundance_only: Sequence[str],
    chem_network=NL99_GC,
    show_progress: bool = True,
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
    tg_guesses : Sequence[float]
        Initial gas-temperature guesses passed to the DESPOTIC solver.
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

    species = tuple(species)
    abundance_only = tuple(abundance_only)

    nH_vals = nH_grid.sample()
    col_vals = col_grid.sample()

    num_rows = len(nH_vals)
    num_cols = len(col_vals) 

    tg_table = np.full((num_rows, num_cols), np.nan, dtype=float)
    failure_mask = np.zeros((num_rows, num_cols), dtype=bool)

    buffer_fields = tuple(LINE_RESULT_FIELDS) + ("abundance",)
    species_buffers: dict[str, dict[str, np.ndarray]] = {
        sp: {
            field: np.full((num_rows, num_cols), np.nan, dtype=float)
            for field in buffer_fields
        }
        for sp in species
    }   
    energy_fields: dict[str, np.ndarray] = {}


    def _solve_row(
        row_idx: int,
        *,
        nH_vals: np.ndarray,
        col_vals: np.ndarray,
        tg_guesses: Sequence[float],
        species: Sequence[str],
        abundance_only: Sequence[str],
        chem_network,
    ) -> tuple[int, np.ndarray, np.ndarray, dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], list[AttemptRecord]]:
        tg_row = np.full(col_vals.shape, np.nan, dtype=float)
        failure_row = np.zeros(col_vals.shape, dtype=bool)
        species_rows: dict[str, dict[str, np.ndarray]] = {
            sp: {field: np.full(col_vals.shape, np.nan, dtype=float) for field in buffer_fields}
            for sp in species
        }  
        abundance_rows: dict[str, np.ndarray] = {
            sp: np.full(col_vals.shape, np.nan, dtype=float)
            for sp in abundance_only
        }

        def _flatten_energy(term_name: str, value, target: dict[str, np.ndarray], idx: int) -> None:
            if isinstance(value, Mapping):
                for sub_key, sub_value in value.items():
                    _flatten_energy(f"{term_name}.{sub_key}", sub_value, target, idx)
                return
            grid = target.setdefault(term_name, np.full(col_vals.shape, np.nan, dtype=float))
            grid[idx] = float(value)


        energy_rows: dict[str, np.ndarray] = {}
        attempts_row: list[AttemptRecord] = []

        for col_idx, col_val in enumerate(col_vals):
            line_results, abundances, final_tg, energy_terms, failed = calculate_single_despotic_point(
                nH_val=nH_vals[row_idx],
                colDen_val=col_val,
                initial_Tg_guesses=tg_guesses,
                species=species,
                abundance_only=abundance_only,
                chem_network=chem_network,
                row_idx=row_idx,
                col_idx=col_idx,
                log_failures=True,
                attempt_log=attempts_row,
            )
            tg_row[col_idx] = final_tg
            failure_row[col_idx] = failed

            # species data
            for sp in species:
                result = line_results.get(sp, DEFAULT_LINE_RESULT)
                for field in LINE_RESULT_FIELDS:
                    species_rows[sp][field][col_idx] = getattr(result, field)
                species_rows[sp]["abundance"][col_idx] = abundances.get(sp, float("nan"))

            for extra in abundance_only:
                abundance_rows[extra][col_idx] = abundances.get(extra, float("nan"))

            # energy terms
            for term, value in energy_terms.items():
                _flatten_energy(term, value, energy_rows, col_idx)


        return row_idx, tg_row, failure_row, species_rows, abundance_rows, energy_rows, attempts_row


    attempts: list[AttemptRecord] = []
    if workers is None:
        workers = -1

    tasks = range(num_rows)
    solve_row = partial(
        _solve_row,
        nH_vals=nH_vals,
        col_vals=col_vals,      
        tg_guesses=tg_guesses,
        species=species,
        abundance_only=abundance_only,
        chem_network=chem_network,
    )

    if show_progress:
        progress = tqdm(
            total=num_rows,
            desc="DESPOTIC rows",
            unit="row",
            dynamic_ncols=True,
            colour="cyan",
            mininterval=1.0,
            smoothing=0.2,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {unit}s [{elapsed}<{remaining}]",
        )
        with tqdm_joblib(progress):
            results = Parallel(n_jobs=workers)(
                delayed(solve_row)(row_idx) for row_idx in tasks
            )
    else:
        results = Parallel(n_jobs=workers)(
            delayed(solve_row)(row_idx) for row_idx in tasks
        )

    for row_idx, tg_row, failure_row, species_rows, extra_rows, energy_rows, attempts_row in results:
        tg_table[row_idx, :] = tg_row
        failure_mask[row_idx, :] = failure_row
        attempts.extend(attempts_row)
        for sp, field_map in species_rows.items():
            buffer = species_buffers[sp]
            for field, values in field_map.items():
                buffer[field][row_idx, :] = values
                
        for extra, values in extra_rows.items():
            buffer = species_buffers.setdefault(extra, {
                field: np.full((num_rows, num_cols), np.nan, dtype=float)
                for field in buffer_fields
            })
            buffer["abundance"][row_idx, :] = values

        for term, values in energy_rows.items():
            grid = energy_fields.setdefault(
                term, np.full((num_rows, num_cols), np.nan, dtype=float)
            )
            grid[row_idx, :] = values

    total_cells = num_rows * num_cols
    failed_cells = int(np.count_nonzero(failure_mask))
    if failed_cells:
        LOGGER.warning(
            "DESPOTIC table: %s/%s cells failed to converge",
            failed_cells,
            total_cells,
        )
    else:
        LOGGER.info("DESPOTIC table converged for all %s cells", total_cells)


    species_data = {
        sp: SpeciesLineGrid(
            **{field: buf[field] for field in LINE_RESULT_FIELDS},
            abundance=buf["abundance"],
        )
        for sp, buf in species_buffers.items()
    }

    
     
    return DespoticTable(
        species_data=species_data,
        tg_final=tg_table,
        nH_values=nH_vals,
        col_density_values=col_vals,
        failure_mask=failure_mask,
        energy_terms=energy_fields or None,
        attempts=tuple(attempts),
    )



def plot_table(*_args, **_kwargs) -> None:
    """Placeholder for future plotting utilities."""
    raise NotImplementedError("plot_table is not implemented yet.")

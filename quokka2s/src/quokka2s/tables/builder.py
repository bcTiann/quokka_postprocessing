from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from despotic.chemistry import NL99, NL99_GC, GOW

from .models import AttemptRecord, DespoticTable, LineLumResult, LogGrid, SpeciesLineGrid
from .solver import LINE_RESULT_FIELDS, calculate_single_despotic_point

LOGGER = logging.getLogger(__name__)
DEFAULT_LINE_RESULT = LineLumResult(*([float("nan")] * len(LINE_RESULT_FIELDS)))

def build_table(
    nH_grid: LogGrid,
    col_grid: LogGrid,
    tg_guesses: Sequence[float],
    *,
    species: Sequence[str] = ("CO", "C+", "HCO+"),
    chem_network=NL99_GC,
    show_progress: bool = True,
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

    nH_vals = nH_grid.sample()
    col_vals = col_grid.sample()

    num_rows = len(nH_vals)
    num_cols = len(col_vals) 

    tg_table = np.full((num_rows, num_cols), np.nan, dtype=float)
    failure_mask = np.zeros((num_rows, num_cols), dtype=bool)
    energy_rate = np.full((num_rows, num_cols), np.nan, dtype=float)

    buffer_fields = tuple(LINE_RESULT_FIELDS) + ("abundance",)
    species_buffers: dict[str, dict[str, np.ndarray]] = {
        sp: {
            field: np.full((num_rows, num_cols), np.nan, dtype=float)
            for field in buffer_fields
        }
        for sp in species
    }   

    attempts: list[AttemptRecord] = []

    row_iter = range(num_rows)
    if show_progress:
        from tqdm import tqdm
        row_iter = tqdm(row_iter, desc="DESPOTIC rows")
    
    for row_idx in row_iter:
        for col_idx, col_val in enumerate(col_vals):
            line_results, abundances, final_tg, e_dot, failed = calculate_single_despotic_point(
                nH_val=nH_vals[row_idx],
                colDen_val=col_val,
                initial_Tg_guesses=tg_guesses,
                species=species,
                chem_network=chem_network,
                row_idx=row_idx,
                col_idx=col_idx,
                log_failures=True,
                attempt_log=attempts
            )
            tg_table[row_idx, col_idx] = final_tg
            failure_mask[row_idx, col_idx] = failed
            energy_rate[row_idx, col_idx] = e_dot


            for sp in species:
                buffer = species_buffers[sp]
                result = line_results.get(sp, DEFAULT_LINE_RESULT)
                for field in LINE_RESULT_FIELDS:
                    buffer[field][row_idx, col_idx] = getattr(result, field)
                buffer["abundance"][row_idx, col_idx] = abundances.get(sp, float("nan"))
    


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
        energy_rate=energy_rate,
        attempts=tuple(attempts),
    )
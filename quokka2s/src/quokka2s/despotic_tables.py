"""Utilities for building DESPOTIC lookup tables."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import warnings

import contextlib
import io


import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import RectBivariateSpline, griddata
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib


from despotic import cloud
from despotic.chemistry import NL99, NL99_GC, GOW

@dataclass(frozen=True)
class LogGrid:
    """Logarithmically spaced grid specification."""

    min_value: float
    max_value: float
    num_points: int

    def __post_init__(self) -> None:
        if self.min_value <= 0 or self.max_value <= 0:
            raise ValueError("LogGrid values must be positive for log10 sampling")
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be smaller than max_value")
        if self.num_points < 2:
            raise ValueError("num_points must be >= 2")

    # def sample(self) -> np.ndarray:
    #     return np.logspace(np.log10(self.min_value), np.log10(self.max_value), self.num_points)

    def sample(self) -> np.ndarray:
        values = np.logspace(np.log10(self.min_value), np.log10(self.max_value), self.num_points)
        return np.round(values, 6)

@dataclass(frozen=True)
class AttemptRecord:
    row_idx: int | None
    col_idx: int | None
    nH: float
    colDen: float
    tg_guess: float
    Tg_setTempEq: float
    final_Tg: float
    attempt_number: int
    attempt_type: str  # "normal_attempt" / "all_guesses_failed"
    converged: bool
    repeat_equilibrium: int
    emitter_abundance: float




@dataclass(frozen=True)
class DespoticTable:
    """Container for DESPOTIC lookup table outputs."""

    co_int_tb: np.ndarray
    tg_final: np.ndarray
    nH_values: np.ndarray
    col_density_values: np.ndarray
    attempts: Tuple[AttemptRecord, ...] = field(default_factory=tuple)

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.co_int_tb, self.tg_final


DEFAULT_EMITTER_ABUNDANCE = 8.0e-9



def calculate_single_despotic_point(
    nH_val: float,
    colDen_val: float,
    initial_Tg_guesses: Sequence[float],
    *,
    log_failures: bool = False,
    emitter_abundance: float = DEFAULT_EMITTER_ABUNDANCE,
    repeat_equilibrium: int = 0,
    chem_network=NL99, 
    row_idx: int | None = None, 
    col_idx: int | None = None,
    attempt_log: list[AttemptRecord] | None = None
) -> Tuple[float, float]:
    """Run DESPOTIC for one (nH, column density) pair.

    Returns (co_int_TB, final_Tg); (np.nan, np.nan) if all guesses fail.
    """
    last_guess: float | None = None
    attempt_number = 0
    last_final_tg = float("nan")
    last_tg_set_temp_eq = float("nan")
    for guess in initial_Tg_guesses:
        attempt_number += 1
        last_guess = guess
        try:
            cell = cloud()
            cell.nH = nH_val
            cell.colDen = colDen_val
            cell.Tg = guess

            cell.sigmaNT = 2.0e5
            cell.comp.xoH2 = 0.1
            cell.comp.xpH2 = 0.4
            cell.comp.xHe = 0.1

            cell.dust.alphaGD = 3.2e-34
            cell.dust.sigma10 = 2.0e-25
            cell.dust.sigmaPE = 1.0e-21
            cell.dust.sigmaISRF = 3.0e-22
            cell.dust.beta = 2.0
            cell.dust.Zd = 1.0

            cell.Td = 10.0
            cell.rad.TCMB = 2.73
            cell.rad.TradDust = 0.0
            cell.rad.ionRate = 2.0e-17
            cell.rad.chi = 1.0
            
            cell.comp.computeDerived(cell.nH)

            cell.addEmitter("CO", emitter_abundance)

            # cell.setTempEq()
            last_tg_set_temp_eq = float(cell.Tg)
            converge = cell.setChemEq(network=chem_network, evolveTemp="iterateDust")

            # if converge:
            #     rates = cell.dEdt(sumOnly=True)
            #     print(f"dEdtGas={rates['dEdtGas']:.3e}, dEdtDust={rates['dEdtDust']:.3e}")
            print("="*40)
            print(f"({row_idx}, {col_idx}) converge: {converge}")
            print("="*40)


            if not converge:
                print("="*40)
                print(f"({row_idx}, {col_idx}) converge failed :")
                print(f"Tg = {cell.Tg}")
                print(f"nH = {cell.nH}")
                print(f"colDen = {cell.colDen}")
                print("="*40)

                if attempt_log is not None:
                    attempt_log.append(
                        AttemptRecord(
                            row_idx=row_idx,
                            col_idx=col_idx,
                            nH=cell.nH,
                            colDen=cell.colDen,
                            tg_guess=guess,
                            Tg_setTempEq=last_tg_set_temp_eq,
                            attempt_number=attempt_number,
                            attempt_type="single_attempt",
                            converged=False,
                            final_Tg=float(cell.Tg),
                            repeat_equilibrium=repeat_equilibrium,
                            emitter_abundance=emitter_abundance,
                        )
                    )
                last_final_tg = float(cell.Tg)
                continue

            else:
                if attempt_log is not None:
                    attempt_log.append(
                            AttemptRecord(
                                row_idx=row_idx,
                                col_idx=col_idx,
                                nH=cell.nH,
                                colDen=cell.colDen,
                                tg_guess=guess,
                                Tg_setTempEq=last_tg_set_temp_eq,
                                attempt_number=attempt_number,
                                attempt_type="successful",
                                converged=True,
                                final_Tg=float(cell.Tg),
                                repeat_equilibrium=repeat_equilibrium,
                                emitter_abundance=emitter_abundance,
                            )
                        )
                last_final_tg = float(cell.Tg)
                # if repeat_equilibrium > 0:
                #     for _ in range(repeat_equilibrium):
                #         cell.setChemEq(network=NL99, evolveTemp="iterate")
                lines = cell.lineLum("CO")
                co_int_TB = lines[0]["intTB"]
                final_Tg = float(cell.Tg)
                
                return co_int_TB, final_Tg
            # if (not np.isfinite(co_int_TB)) or (co_int_TB < 0.0) or (not np.isfinite(final_Tg)) or (final_Tg < 0.0):
            #     row_str = "?" if row_idx is None else row_idx
            #     col_str = "?" if col_idx is None else col_idx
            #     index_info = f"(i={row_str}, j={col_str})"

            #     diagnostics_msg = (
            #         "DESPOTIC returned an invalid state:\n"
            #         f"  intTB = {co_int_TB:.3e} K km/s\n"
            #         f"  Tg    = {final_Tg:.3e} K\n"
            #         f"  grid  = {index_info}\n"
            #         f"  nH    = {nH_val:.3e} cm^-3\n"
            #         f"  colDen= {colDen_val:.3e} cm^-2\n"
            #         f"  Tg guess = {guess:.3e} K\n"
            #         f"  repeat_equilibrium = {repeat_equilibrium}\n"
            #         f"  emitter_abundance  = {emitter_abundance:.3e}"
            #     )

            #     warnings.warn(diagnostics_msg, RuntimeWarning)
            #     return float("nan"), float("nan")

                
        
        except Exception as exc:  # pragma: no cover - DESPOTIC exceptions vary
            if log_failures:
                warnings.warn(
                    f"DESPOTIC failed for Tg guess {guess:.2f} K (nH={nH_val}, colDen={colDen_val}): {exc}",
                    RuntimeWarning,
                )
            continue

    if attempt_log is not None:
        attempt_log.append(
            AttemptRecord(
                row_idx=row_idx,
                col_idx=col_idx,
                nH=nH_val,
                colDen=colDen_val,
                tg_guess=last_guess if last_guess is not None else float("nan"),
                Tg_setTempEq=last_tg_set_temp_eq,
                attempt_number=attempt_number,
                attempt_type="all_guesses_failed",
                converged=False,
                final_Tg=last_final_tg,
                repeat_equilibrium=repeat_equilibrium,
                emitter_abundance=emitter_abundance,
                        )

        )
    return float("nan"), float("nan")


def _compute_row(
    nH: float,
    col_den_points: np.ndarray,
    guess_list: Sequence[float],
    interpolator: Optional[RectBivariateSpline],
    *,
    chem_network=NL99,
    row_logs: list[AttemptRecord] | None = None,
    repeat_equilibrium: int = 0,
    log_failures: bool = False,
    row_idx: int,
) -> Tuple[List[float], List[float], list[AttemptRecord]]:
    
    co_row: List[float] = []
    tg_row: List[float] = []
    
    if row_logs is None:
        row_logs = []  

    for col_idx, colDen in enumerate(col_den_points):

        dynamic_guesses: Sequence[float] = guess_list
        if interpolator is not None:
            try:
                predicted_log_T = interpolator(np.log10(nH), np.log10(colDen))[0][0]
                predicted_T = float(10 ** predicted_log_T)
                if np.isfinite(predicted_T) and predicted_T > 0:
                    dynamic_guesses = [predicted_T] + list(guess_list)
            except Exception as exc:
                if log_failures:
                    warnings.warn(
                        f"Temperature interpolation failed at (nH={nH}, colDen={colDen}): {exc}",
                        RuntimeWarning,
                    )

        co_val, tg_val = calculate_single_despotic_point(
            nH_val=nH,
            colDen_val=colDen,
            initial_Tg_guesses=dynamic_guesses,
            log_failures=log_failures,
            repeat_equilibrium=repeat_equilibrium,
            chem_network=chem_network,
            row_idx=row_idx,
            col_idx=col_idx,
            attempt_log=row_logs
        )

        co_row.append(co_val)
        tg_row.append(tg_val)

    return co_row, tg_row, row_logs


def build_table(
    nH_grid: LogGrid,
    col_den_grid: LogGrid,
    tg_guesses: Sequence[float],
    *,
    chem_network=NL99,
    interpolator: Optional[RectBivariateSpline] = None,
    n_jobs: int = -1,
    repeat_equilibrium: int = 0,
    show_progress: bool = False,
    log_failures: bool = False,
) -> DespoticTable:
    """Build a DESPOTIC lookup table for a pair of logarithmic grids."""
    nH_points = nH_grid.sample()
    col_den_points = col_den_grid.sample()


    progress_bar = None
    progress_manager = contextlib.nullcontext()
    if show_progress:
        progress_bar = tqdm(total=nH_points.size, desc="DESPOTIC rows")
        progress_manager = tqdm_joblib(progress_bar)

    with progress_manager:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_row)(
                nH,
                col_den_points,
                tg_guesses,
                interpolator,
                repeat_equilibrium=repeat_equilibrium,
                chem_network=chem_network,
                log_failures=log_failures,
                row_idx=row_idx,
            )
            for row_idx, nH in enumerate(nH_points)
        )

    if progress_bar is not None:
        progress_bar.close()
    

    co_rows, tg_rows, attempt_lists = zip(*results)
    co_int_tb = np.array(co_rows)
    tg_final = np.array(tg_rows)
    from itertools import chain
    attempts = tuple(chain.from_iterable(attempt_lists))

    return DespoticTable(
        co_int_tb=co_int_tb,
        tg_final=tg_final,
        nH_values=nH_points,
        col_density_values=col_den_points,
        attempts=attempts
    )

def refine_table(
    coarse_table: DespoticTable,
    fine_nH_grid: LogGrid,
    fine_col_den_grid: LogGrid,
    tg_guesses: Sequence[float],
    *,
    interpolator: Optional[RectBivariateSpline] = None,
    n_jobs: int = -1,
    repeat_equilibrium: int = 0,
    show_progress: bool = False,
    log_failures: bool = False,
) -> DespoticTable:
    """Use a coarse table to guide computation on a finer grid."""
    if interpolator is None:
        interpolator = make_temperature_interpolator(
            coarse_table.nH_values,
            coarse_table.col_density_values,
            coarse_table.tg_final,
        )

    return build_table(
        fine_nH_grid,
        fine_col_den_grid,
        tg_guesses,
        interpolator=interpolator,
        n_jobs=n_jobs,
        repeat_equilibrium=repeat_equilibrium,
        show_progress=show_progress,
        log_failures=log_failures,
    )


def make_temperature_interpolator(
    nH_values: Sequence[float],
    col_density_values: Sequence[float],
    tg_table: np.ndarray,
    *,
    kx: int = 3,
    ky: int = 3,
) -> RectBivariateSpline:
    """Create a spline interpolator in log-space for Tg data."""
    nH_values = np.asarray(nH_values)
    col_density_values = np.asarray(col_density_values)
    tg_table = np.asarray(tg_table)

    if tg_table.shape != (nH_values.size, col_density_values.size):
        raise ValueError(
            "tg_table shape must match the lengths of nH_values and col_density_values"
        )

    log_nH = np.log10(nH_values)
    log_col = np.log10(col_density_values)
    log_tg = np.log10(tg_table)

    return RectBivariateSpline(log_nH, log_col, log_tg, kx=kx, ky=ky)



# def fill_missing_co_values(table: DespoticTable) -> DespoticTable:
#     """Return a copy of the table with non-finite CO entries filled by interpolation."""
#     co = table.co_int_tb
#     mask = ~np.isfinite(co)
#     if not mask.any():
#         return table

#     co_filled = co.copy()
#     log_nH = np.log10(table.nH_values)
#     log_col = np.log10(table.col_density_values)
#     log_col_grid, log_nH_grid = np.meshgrid(log_col, log_nH, indexing="xy")

#     points = np.column_stack((log_nH_grid[~mask], log_col_grid[~mask]))
#     values = co[~mask]
#     targets = np.column_stack((log_nH_grid[mask], log_col_grid[mask]))

#     filled = griddata(points, values, targets, method="linear")
#     if np.isnan(filled).any():
#         fallback = griddata(points, values, targets, method="nearest")
#         filled = np.where(np.isnan(filled), fallback, filled)

#     co_filled[mask] = filled
#     return DespoticTable(
#         co_int_tb=co_filled,
#         tg_final=table.tg_final,
#         nH_values=table.nH_values,
#         col_density_values=table.col_density_values,
#         failures=table.failures,
#     )


def fill_missing_values(table: DespoticTable) -> DespoticTable:
    
    log_nH = np.log10(table.nH_values)
    log_col = np.log10(table.col_density_values)
    log_col_grid, log_nH_grid = np.meshgrid(log_col, log_nH, indexing="xy")

    def _fill_grid(values: np.ndarray, *, log_space: bool) -> np.ndarray:
        grid = values.copy()
        mask = ~np.isfinite(grid) # ~ all valid grid == all invalid grid == mask
        # ~mask == all valid grid
        if log_space:
            mask |= grid <= 0
        if not mask.any():
            return grid

        if log_space:
            safe = grid.copy()
            safe[mask] = 1.0
            work = np.log10(safe)
        else:
            work = grid # work == all grid values     work[~mask] == all valid grid values
 
        points = np.column_stack((log_nH_grid[~mask], log_col_grid[~mask])) # all valid grid 2D coordinates
        targets = np.column_stack((log_nH_grid[mask], log_col_grid[mask])) # all invalid grid 2D coordinates
        filled = griddata(points, work[~mask], targets, method="linear")

        if np.isnan(filled).any():
            fallback = griddata(points, work[~mask], targets, method="nearest")
            filled = np.where(np.isnan(filled), fallback, filled)

        result = work.copy()
        result[mask] = filled
        return np.power(10.0, result) if log_space else result
    
    co_filled = _fill_grid(table.co_int_tb, log_space=False)
    tg_filled = _fill_grid(table.tg_final, log_space=True)

    return DespoticTable(
    co_int_tb=co_filled,
    tg_final=tg_filled,
    nH_values=table.nH_values,
    col_density_values=table.col_density_values,
    attempts=table.attempts,
)


def compute_average(
    components: Sequence[np.ndarray],
    *,
    method: str = "arithmetic",
) -> np.ndarray:
    """Combine column density components with the requested averaging scheme.

    Parameters
    ----------
    components
        Sequence of arrays (all same shape) to combine.
    method
        One of ``"arithmetic"``/``"mean"``, ``"geometric"``/``"geom"``, or
        ``"harmonic"``/``"inverse"``.

    Returns
    -------
    np.ndarray
        Array with the same shape as the inputs after applying the requested
        averaging operation.
    """

    if not components:
        raise ValueError("components must contain at least one array")

    stack = np.stack(components, axis=0)
    method_key = method.lower()

    if method_key in {"arithmetic", "mean", "avg"}:
        return np.mean(stack, axis=0)

    if method_key in {"geometric", "geom"}:
        if np.any(stack <= 0):
            raise ValueError("Geometric mean requires all values to be positive")
        return np.exp(np.mean(np.log(stack), axis=0))

    if method_key in {"harmonic", "inverse"}:
        if np.any(stack <= 0):
            raise ValueError("Harmonic mean requires all values to be positive")
        return stack.shape[0] / np.sum(1.0 / stack, axis=0)

    raise ValueError(
        "Unsupported method '{method}'. Choose from 'arithmetic', 'geometric', or 'harmonic'."
        .format(method=method)
    )



__all__ = [
    "LogGrid",
    "DespoticTable",
    "calculate_single_despotic_point",
    "build_table",
    "make_temperature_interpolator",
    "refine_table",
    "fill_missing_values",
    "compute_average"
]

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
    round_digits: int | None = None

    def __post_init__(self) -> None:
        if self.min_value <= 0 or self.max_value <= 0:
            raise ValueError("LogGrid values must be positive for log10 sampling")
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be smaller than max_value")
        if self.num_points < 2:
            raise ValueError("num_points must be >= 2")

    def sample(self) -> np.ndarray:
        values = np.logspace(np.log10(self.min_value), np.log10(self.max_value), self.num_points)
        if self.round_digits is not None:
            values = np.round(values, self.round_digits)
        return values

@dataclass(frozen=True)
class AttemptRecord:
    row_idx: int | None
    col_idx: int | None
    nH: float
    colDen: float
    tg_guess: float
    final_Tg: float
    attempt_number: int
    attempt_type: str  # "successful" / "single_attempt" / "co_int_below_threshold" / "exception" / "all_guesses_failed"
    converged: bool
    repeat_equilibrium: int
    co_int_TB: float
    int_intensity: float
    lum_per_h: float
    tau: float
    tau_dust: float
    tex: float
    frequency: float
    error_message: str | None = None




@dataclass(frozen=True)
class DespoticTable:
    """Container for DESPOTIC lookup table outputs."""

    co_int_tb: np.ndarray
    tg_final: np.ndarray
    int_intensity: np.ndarray
    lum_per_h: np.ndarray
    tau: np.ndarray
    tau_dust: np.ndarray
    tex: np.ndarray
    frequency: np.ndarray
    nH_values: np.ndarray
    col_density_values: np.ndarray
    attempts: Tuple[AttemptRecord, ...] = field(default_factory=tuple)

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.co_int_tb, self.tg_final


DEFAULT_EMITTER_ABUNDANCE = 8.0e-9
CO_INT_THRESHOLD = 1.0e-8



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
    attempt_log: list[AttemptRecord] | None = None,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Run DESPOTIC for one (nH, column density) pair.

    Returns
    -------
    Tuple
        (co_int_TB, final_Tg, intIntensity, lumPerH, tau, tauDust, Tex, freq)
        Values are ``nan`` if all guesses fail.
    """
    last_guess: float | None = None
    attempt_number = 0
    last_final_tg = float("nan")
    last_co_int = float("nan")
    last_int_intensity = float("nan")
    last_lum_per_h = float("nan")
    last_tau = float("nan")
    last_tau_dust = float("nan")
    last_tex = float("nan")
    last_freq = float("nan")
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

            converge = cell.setChemEq(network=chem_network, evolveTemp="iterateDust")

            if not converge:
                final_tg = float(cell.Tg)
                last_final_tg = final_tg
                last_co_int = float("nan")
                last_int_intensity = float("nan")
                last_lum_per_h = float("nan")
                last_tau = float("nan")
                last_tau_dust = float("nan")
                last_tex = float("nan")
                last_freq = float("nan")
                if attempt_log is not None:
                    attempt_log.append(
                        AttemptRecord(
                            row_idx=row_idx,
                            col_idx=col_idx,
                            nH=cell.nH,
                            colDen=cell.colDen,
                            tg_guess=guess,
                            final_Tg=final_tg,
                            attempt_number=attempt_number,
                            attempt_type="single_attempt",
                            converged=False,
                            repeat_equilibrium=repeat_equilibrium,
                            co_int_TB=float("nan"),
                            int_intensity=float("nan"),
                            lum_per_h=float("nan"),
                            tau=float("nan"),
                            tau_dust=float("nan"),
                            tex=float("nan"),
                            frequency=float("nan"),
                        )
                    )
                continue

            lines = cell.lineLum("CO")
            co_int_TB = lines[0]["intTB"]
            intensity_with_dust = lines[0]["intIntensity"]
            lumPerH = lines[0]["lumPerH"]
            tau = lines[0]["tau"]
            tau_dust = lines[0]["tauDust"]
            tex = lines[0]["Tex"]
            freq = lines[0]["freq"]


            final_Tg = float(cell.Tg)
            last_final_tg = final_Tg
            last_co_int = co_int_TB
            last_int_intensity = intensity_with_dust
            last_lum_per_h = lumPerH
            last_tau = tau
            last_tau_dust = tau_dust
            last_tex = tex
            last_freq = freq

            # if (not np.isfinite(co_int_TB)) or (co_int_TB < CO_INT_THRESHOLD):
            #     if attempt_log is not None:
            #         attempt_log.append(
            #             AttemptRecord(
            #                 row_idx=row_idx,
            #                 col_idx=col_idx,
            #                 nH=cell.nH,
            #                 colDen=cell.colDen,
            #                 tg_guess=guess,
            #                 final_Tg=final_Tg,
            #                 attempt_number=attempt_number,
            #                 attempt_type="co_int_below_threshold",
            #                 converged=False,
            #                 repeat_equilibrium=repeat_equilibrium,
            #                 co_int_TB=co_int_TB,
            #             )
            #         )
            #     continue

            if attempt_log is not None:
                attempt_log.append(
                    AttemptRecord(
                        row_idx=row_idx,
                        col_idx=col_idx,
                        nH=cell.nH,
                        colDen=cell.colDen,
                        tg_guess=guess,
                        final_Tg=final_Tg,
                        attempt_number=attempt_number,
                        attempt_type="successful",
                        converged=True,
                        repeat_equilibrium=repeat_equilibrium,
                        co_int_TB=co_int_TB,
                        int_intensity=intensity_with_dust,
                        lum_per_h=lumPerH,
                        tau=tau,
                        tau_dust=tau_dust,
                        tex=tex,
                        frequency=freq,
                    )
                )
            return (
                co_int_TB,
                final_Tg,
                intensity_with_dust,
                lumPerH,
                tau,
                tau_dust,
                tex,
                freq,
            )

        except Exception as exc:  # pragma: no cover - DESPOTIC exceptions vary
            fallback_cell = locals().get("cell")
            fallback_tg = float(getattr(fallback_cell, "Tg", float("nan"))) if fallback_cell is not None else float("nan")

            last_final_tg = fallback_tg
            last_co_int = float("nan")
            last_int_intensity = float("nan")
            last_lum_per_h = float("nan")
            last_tau = float("nan")
            last_tau_dust = float("nan")
            last_tex = float("nan")
            last_freq = float("nan")
            if attempt_log is not None:
                attempt_log.append(
                    AttemptRecord(
                        row_idx=row_idx,
                        col_idx=col_idx,
                        nH=nH_val,
                        colDen=colDen_val,
                        tg_guess=guess,
                        final_Tg=fallback_tg,
                        attempt_number=attempt_number,
                        attempt_type="exception",
                        converged=False,
                        repeat_equilibrium=repeat_equilibrium,
                        co_int_TB=float("nan"),
                        int_intensity=float("nan"),
                        lum_per_h=float("nan"),
                        tau=float("nan"),
                        tau_dust=float("nan"),
                        tex=float("nan"),
                        frequency=float("nan"),
                        error_message=str(exc),
                    )
                )
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
                final_Tg=last_final_tg,
                attempt_number=attempt_number,
                attempt_type="all_guesses_failed",
                converged=False,
                repeat_equilibrium=repeat_equilibrium,
                co_int_TB=last_co_int,
                int_intensity=last_int_intensity,
                lum_per_h=last_lum_per_h,
                tau=last_tau,
                tau_dust=last_tau_dust,
                tex=last_tex,
                frequency=last_freq,
            )
        )
    return (
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    )


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
    ) -> Tuple[
        List[float],
        List[float],
        List[float],
        List[float],
        List[float],
        List[float],
        List[float],
        List[float],
        list[AttemptRecord],
    ]:

    co_row: List[float] = []
    tg_row: List[float] = []
    intensity_row: List[float] = []
    lum_row: List[float] = []
    tau_row: List[float] = []
    tau_dust_row: List[float] = []
    tex_row: List[float] = []
    freq_row: List[float] = []

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

        (
            co_val,
            tg_val,
            intensity_val,
            lum_val,
            tau_val,
            tau_dust_val,
            tex_val,
            freq_val,
        ) = calculate_single_despotic_point(
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
        intensity_row.append(intensity_val)
        lum_row.append(lum_val)
        tau_row.append(tau_val)
        tau_dust_row.append(tau_dust_val)
        tex_row.append(tex_val)
        freq_row.append(freq_val)

    return (
        co_row,
        tg_row,
        intensity_row,
        lum_row,
        tau_row,
        tau_dust_row,
        tex_row,
        freq_row,
        row_logs,
    )


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
    

    (
        co_rows,
        tg_rows,
        intensity_rows,
        lum_rows,
        tau_rows,
        tau_dust_rows,
        tex_rows,
        freq_rows,
        attempt_lists,
    ) = zip(*results)

    co_int_tb = np.array(co_rows)
    tg_final = np.array(tg_rows)
    int_intensity = np.array(intensity_rows)
    lum_per_h = np.array(lum_rows)
    tau = np.array(tau_rows)
    tau_dust = np.array(tau_dust_rows)
    tex = np.array(tex_rows)
    freq = np.array(freq_rows)
    from itertools import chain
    attempts = tuple(chain.from_iterable(attempt_lists))

    return DespoticTable(
        co_int_tb=co_int_tb,
        tg_final=tg_final,
        int_intensity=int_intensity,
        lum_per_h=lum_per_h,
        tau=tau,
        tau_dust=tau_dust,
        tex=tex,
        frequency=freq,
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
    intensity_filled = _fill_grid(table.int_intensity, log_space=True)
    lum_filled = _fill_grid(table.lum_per_h, log_space=True)
    tau_filled = _fill_grid(table.tau, log_space=False)
    tau_dust_filled = _fill_grid(table.tau_dust, log_space=False)
    tex_filled = _fill_grid(table.tex, log_space=False)
    freq_filled = table.frequency  # frequency is fixed by transition; assume complete

    return DespoticTable(
        co_int_tb=co_filled,
        tg_final=tg_filled,
        int_intensity=intensity_filled,
        lum_per_h=lum_filled,
        tau=tau_filled,
        tau_dust=tau_dust_filled,
        tex=tex_filled,
        frequency=freq_filled,
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

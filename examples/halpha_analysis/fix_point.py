#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import warnings
import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

try:  # Prefer rich-rendered bars
    from tqdm.rich import tqdm
except Exception:  # pragma: no cover - fallback if rich support unavailable
    from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

from quokka2s.despotic_tables import (
    AttemptRecord,
    DespoticTable,
    LineLumResult,
    SpeciesLineGrid,
    calculate_single_despotic_point,
)
from despotic.chemistry import NL99, NL99_GC, GOW
from build_despotic_table import plot_table, TG_GUESSES, save_table

linear_part = np.linspace(1.0, 100.0, 30)  # 例如 1–100 取 30 个点
log_part = np.logspace(np.log10(100.0), np.log10(10000.0), 20)

TG_GUESSES = np.concatenate([linear_part, log_part]).tolist()
# 如需去重 + 排序，可选：
TG_GUESSES = sorted(set(TG_GUESSES))

NETWORK_MAP = {
    "nl99": NL99,
    "nl99_gc": NL99_GC,
    "gow": GOW,
}

LOGGER = logging.getLogger(__name__)


def configure_logging(log_path: Path) -> None:
    """Configure logging for fix_point workflow."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    logging.captureWarnings(True)
    warnings.simplefilter("always")


def load_table(npz_path: Path) -> DespoticTable:
    with np.load(npz_path, allow_pickle=True) as data:
        if "species" in data and any(key.endswith("_grid") for key in data.keys()):
            species_names = [str(name) for name in data["species"]]
            emitter_arr = data.get("emitter_abundances")
            emitter_abundances = {
                species: float(emitter_arr[idx]) if emitter_arr is not None else float("nan")
                for idx, species in enumerate(species_names)
            }
            line_grids: dict[str, SpeciesLineGrid] = {}
            for idx, species in enumerate(species_names):
                line_grids[species] = SpeciesLineGrid(
                    int_tb=data["int_tb_grid"][idx],
                    int_intensity=data["int_intensity_grid"][idx],
                    lum_per_h=data["lum_per_h_grid"][idx],
                    tau=data["tau_grid"][idx],
                    tau_dust=data["tau_dust_grid"][idx],
                    tex=data["tex_grid"][idx],
                    freq=data["freq_grid"][idx],
                )
            return DespoticTable(
                species_data=line_grids,
                tg_final=data["tg_final"],
                nH_values=data["nH"],
                col_density_values=data["col_density"],
                emitter_abundances=emitter_abundances,
                attempts=(),
            )

        # legacy format fallback
        shape = data["co_int_tb"].shape

        def get_array(key: str) -> np.ndarray:
            if key in data:
                return data[key]
            return np.full(shape, np.nan)

        legacy_grid = SpeciesLineGrid(
            int_tb=data["co_int_tb"],
            int_intensity=get_array("int_intensity"),
            lum_per_h=get_array("lum_per_h"),
            tau=get_array("tau"),
            tau_dust=get_array("tau_dust"),
            tex=get_array("tex"),
            freq=get_array("frequency"),
        )

        return DespoticTable(
            species_data={"CO": legacy_grid},
            tg_final=data["tg_final"],
            nH_values=data["nH"],
            col_density_values=data["col_density"],
            emitter_abundances={"CO": float("nan")},
            attempts=(),
        )


DEFAULT_TARGET_SPECIES = "CO"
LINE_FIELDS = (
    "int_tb",
    "int_intensity",
    "lum_per_h",
    "tau",
    "tau_dust",
    "tex",
    "freq",
)
EMPTY_LINE_RESULT = LineLumResult(
    int_tb=float("nan"),
    int_intensity=float("nan"),
    lum_per_h=float("nan"),
    tau=float("nan"),
    tau_dust=float("nan"),
    tex=float("nan"),
    freq=float("nan"),
)


def _select_species(table: DespoticTable, requested: str | None) -> tuple[str, SpeciesLineGrid]:
    if requested and requested in table.species_data:
        return requested, table.get_species_grid(requested)
    if requested and requested not in table.species_data:
        LOGGER.warning(
            "Requested species '%s' unavailable; falling back to '%s'.",
            requested,
            table.primary_species,
        )
    fallback = table.primary_species
    return fallback, table.get_species_grid(fallback)


def _ensure_species_arrays(
    buffers: dict[str, dict[str, np.ndarray]],
    species: str,
    shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    arrays = buffers.get(species)
    if arrays is None:
        arrays = {field: np.full(shape, np.nan, dtype=float) for field in LINE_FIELDS}
        buffers[species] = arrays
    return arrays



def select_indices(values: np.ndarray, span: Tuple[int, int] | None, value_range: Tuple[float, float] | None) -> np.ndarray:
    if span is not None:
        start, end = span
        return np.arange(max(start, 0), min(end, values.size))
    if value_range is not None:
        low, high = value_range
        return np.where((values >= low) & (values <= high))[0]
    return np.arange(values.size)


def recompute_low_co_cells(
    table: DespoticTable,
    *,
    threshold: float,
    chem_network,
    tg_guesses: Sequence[float],
    target_species: str | None = None,
    log_failures: bool = True,
    repeat_equilibrium: int = 0,
    n_jobs: int = 1,
    nH_values: np.ndarray,
    col_values: np.ndarray,
    row_span: Tuple[int, int] | None = None,
    col_span: Tuple[int, int] | None = None,
    nH_range: Tuple[float, float] | None = None,
    col_range: Tuple[float, float] | None = None,
    reuse_failed_tg: bool = False,
    reuse_max_insertions: int = 3,
) -> tuple[DespoticTable, str]:
    grid_shape = table.tg_final.shape
    species_name, target_grid = _select_species(table, target_species or DEFAULT_TARGET_SPECIES)

    species_buffers: dict[str, dict[str, np.ndarray]] = {
        name: {field: np.array(getattr(grid, field), copy=True) for field in LINE_FIELDS}
        for name, grid in table.species_data.items()
    }
    _ensure_species_arrays(species_buffers, species_name, grid_shape)

    target_arrays = species_buffers[species_name]
    co_grid = target_arrays["int_tb"]
    intensity_grid = target_arrays["int_intensity"]
    lum_grid = target_arrays["lum_per_h"]
    tau_grid = target_arrays["tau"]
    tau_dust_grid = target_arrays["tau_dust"]
    tex_grid = target_arrays["tex"]
    freq_grid = target_arrays["freq"]
    tg_grid = np.array(table.tg_final, copy=True)

    rows = select_indices(nH_values, row_span, nH_range)
    cols = select_indices(col_values, col_span, col_range)

    if rows.size == 0 or cols.size == 0:
        LOGGER.info("No cells match the specified region; skipping recomputation.")
        return table, species_name

    region_mask = np.zeros_like(co_grid, dtype=bool)
    region_mask[np.ix_(rows, cols)] = True

    LOGGER.info(
        "Selected region rows: %d–%d (count: %d), cols: %d–%d (count: %d)",
        rows[0],
        rows[-1],
        rows.size,
        cols[0],
        cols[-1],
        cols.size,
    )

    mask = (np.isnan(co_grid) | (co_grid < threshold)) & region_mask
    if not np.any(mask):
        LOGGER.info("No cells require recomputation (threshold=%s).", threshold)
        return table, species_name

    attempt_records: list[AttemptRecord] = []
    low_indices = np.argwhere(mask)
    LOGGER.info(
        "Recomputing %d cells with %s intensity < %.3e",
        len(low_indices),
        species_name,
        threshold,
    )
    for row_idx, col_idx in low_indices:
        nH = float(nH_values[row_idx])
        col = float(col_values[col_idx])
        current_co = float(co_grid[row_idx, col_idx])
        LOGGER.debug(
            "Task row=%d col=%d nH=%.6e cm^-3 colDen=%.6e cm^-2 %s_before=%.6e",
            row_idx,
            col_idx,
            nH,
            col,
            current_co,
            species_name,
        )

    def _evaluate_cell(row_idx: int, col_idx: int):
        row_log: list[AttemptRecord] = []
        line_map_proxy, tg_val = calculate_single_despotic_point(
            nH_val=float(nH_values[row_idx]),
            colDen_val=float(col_values[col_idx]),
            initial_Tg_guesses=tg_guesses,
            log_failures=log_failures,
            repeat_equilibrium=repeat_equilibrium,
            chem_network=chem_network,
            row_idx=int(row_idx),
            col_idx=int(col_idx),
            attempt_log=row_log,
            reuse_failed_tg=reuse_failed_tg,
            reuse_max_insertions=reuse_max_insertions,
        )
        line_results = dict(line_map_proxy)
        return (
            row_idx,
            col_idx,
            line_results,
            tg_val,
            tuple(row_log),
        )

    if n_jobs == 1:
        results = []
        for r, c in tqdm(
            [tuple(idx) for idx in low_indices],
            desc="Recomputing cells",
            unit="cell",
        ):
            results.append(_evaluate_cell(int(r), int(c)))
    else:
        from joblib import Parallel, delayed

        with tqdm_joblib(
            tqdm(
                desc="Recomputing cells",
                total=len(low_indices),
                unit="cell",
            )
        ):
            results = Parallel(n_jobs=n_jobs)(
                delayed(_evaluate_cell)(int(r), int(c)) for r, c in low_indices
            )

    for (
        row_idx,
        col_idx,
        line_map,
        tg_val,
        row_log,
    ) in results:
        tg_grid[row_idx, col_idx] = tg_val
        for species, result in line_map.items():
            arrays = _ensure_species_arrays(species_buffers, species, grid_shape)
            arrays["int_tb"][row_idx, col_idx] = result.int_tb
            arrays["int_intensity"][row_idx, col_idx] = result.int_intensity
            arrays["lum_per_h"][row_idx, col_idx] = result.lum_per_h
            arrays["tau"][row_idx, col_idx] = result.tau
            arrays["tau_dust"][row_idx, col_idx] = result.tau_dust
            arrays["tex"][row_idx, col_idx] = result.tex
            arrays["freq"][row_idx, col_idx] = result.freq
        attempt_records.extend(row_log)

    species_data = {
        species: SpeciesLineGrid(
            int_tb=arrays["int_tb"],
            int_intensity=arrays["int_intensity"],
            lum_per_h=arrays["lum_per_h"],
            tau=arrays["tau"],
            tau_dust=arrays["tau_dust"],
            tex=arrays["tex"],
            freq=arrays["freq"],
        )
        for species, arrays in species_buffers.items()
    }

    updated_table = DespoticTable(
        species_data=species_data,
        tg_final=tg_grid,
        nH_values=table.nH_values,
        col_density_values=table.col_density_values,
        emitter_abundances=table.emitter_abundances,
        attempts=table.attempts + tuple(attempt_records),
    )
    return updated_table, species_name





def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Recompute cells with CO intensity below a threshold and write an updated DESPOTIC table."
    )
    parser.add_argument("table_npz", type=Path, help="Path to an existing table_*.npz file.")
    parser.add_argument("output_npz", type=Path, help="Destination path for the recomputed table.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-8,
        help="Cells with CO intensity below this threshold will be recomputed (default: 1e-8).",
    )
    parser.add_argument(
        "--network",
        choices=NETWORK_MAP.keys(),
        default="nl99",
        help="Chemical network to use (default: nl99).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="repeat_equilibrium passed to calculate_single_despotic_point (default: 0).",
    )
    parser.add_argument(
        "--log-failures",
        action="store_true",
        help="Emit warnings when DESPOTIC raises exceptions (default: disabled).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        dest="n_jobs",
        type=int,
        default=-1,
        help="Number of parallel workers to use (default: -1 uses all cores).",
    )
    parser.add_argument(
        "--nH-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Only consider rows whose nH lies within [MIN, MAX].",
    )
    parser.add_argument(
        "--col-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Only consider columns whose colDen lies within [MIN, MAX].",
    )
    parser.add_argument(
        "--row-span",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Only consider row indices in [START, END) (0-based).",
    )
    parser.add_argument(
        "--col-span",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Only consider column indices in [START, END) (0-based).",
    )
    parser.add_argument(
        "--reuse-final",
        "--reuse-final-tg",
        dest="reuse_final_tg",
        action="store_true",
        help="Reuse the final DESPOTIC Tg as the next guess when a run fails.",
    )
    parser.add_argument(
        "--species",
        type=str,
        default=None,
        help="Emitter species to monitor and threshold (default: CO).",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_npz.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{args.output_npz.stem}.log"
    configure_logging(log_path)
    LOGGER.info("Writing logs to %s", log_path)
    LOGGER.info(
        "CLI arguments: table=%s output=%s threshold=%s network=%s repeat=%d n_jobs=%d reuse_final_tg=%s species=%s",
        args.table_npz,
        args.output_npz,
        args.threshold,
        args.network,
        args.repeat,
        args.n_jobs,
        args.reuse_final_tg,
        args.species or DEFAULT_TARGET_SPECIES,
    )
    tag = args.output_npz.stem.replace("table_", "")

    table = load_table(args.table_npz)
    reuse_failed_tg = args.reuse_final_tg
    new_table, species_name = recompute_low_co_cells(
        table,
        threshold=args.threshold,
        chem_network=NETWORK_MAP[args.network],
        tg_guesses=TG_GUESSES,
        target_species=args.species,
        log_failures=args.log_failures,
        repeat_equilibrium=args.repeat,
        n_jobs=args.n_jobs,
        nH_values=table.nH_values,
        col_values=table.col_density_values,
        row_span=tuple(args.row_span) if args.row_span else None,
        col_span=tuple(args.col_span) if args.col_span else None,
        nH_range=tuple(args.nH_range) if args.nH_range else None,
        col_range=tuple(args.col_range) if args.col_range else None,
        reuse_failed_tg=reuse_failed_tg,
    )

    species_token = species_name.replace("+", "plus").lower()

    recomputed_plot = output_dir / f"{species_token}_int_TB_{tag}.png"
    plot_table(
        table=new_table,
        data=new_table.co_int_tb,
        output_path=str(recomputed_plot),
        title=f"DESPOTIC Lookup Table ({tag})",
        cbar_label=f"{species_name} Integrated Brightness Temp (K km/s)",
    )

    tg_plot = output_dir / f"tg_final_{tag}.png"
    plot_table(
        table=new_table,
        data=new_table.tg_final,
        output_path=str(tg_plot),
        title=f"DESPOTIC Gas Temperature ({tag})",
        cbar_label="Tg (K)",
        use_log=False,
    )
    plot_table(
        table=new_table,
        data=new_table.int_intensity,
        output_path=str(output_dir / f"{species_token}_intensity_{tag}.png"),
        title=f"{species_name} Integrated Intensity ({tag})",
        cbar_label=f"{species_name} Integrated Intensity (erg cm$^{{-2}}$ s$^{{-1}}$ sr$^{{-1}}$)",
    )
    plot_table(
        table=new_table,
        data=new_table.lum_per_h,
        output_path=str(output_dir / f"{species_token}_lum_per_H_{tag}.png"),
        title=f"{species_name} Luminosity per H ({tag})",
        cbar_label=f"{species_name} Luminosity per H (erg s$^{{-1}}$ H$^{{-1}}$)",
    )
    plot_table(
        table=new_table,
        data=new_table.tau,
        output_path=str(output_dir / f"{species_token}_tau_{tag}.png"),
        title=f"{species_name} Line Optical Depth ({tag})",
        cbar_label=f"{species_name} Line Optical Depth",
        use_log=False,
    )
    plot_table(
        table=new_table,
        data=new_table.tau_dust,
        output_path=str(output_dir / f"{species_token}_tau_dust_{tag}.png"),
        title=f"Dust Optical Depth ({tag})",
        cbar_label="Dust Optical Depth",
        use_log=False,
    )
    output_base = args.output_npz.with_suffix("")
    save_table(output_base, new_table)
    output_npz_path = output_base.with_suffix(".npz")

    attempts_csv = output_npz_path.with_name(output_npz_path.stem + "_attempts.csv")
    if new_table.attempts:
        import csv

        with attempts_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "row_idx",
                    "col_idx",
                    "nH",
                    "colDen",
                    "tg_guess",
                    "final_Tg",
                    "attempt_number",
                    "attempt_type",
                    "converged",
                    "repeat_equilibrium",
                    "species",
                    "co_int_TB",
                    "int_intensity",
                    "lum_per_H",
                    "tau",
                    "tau_dust",
                    "Tex",
                    "frequency",
                    "max_residual",
                    "residual_trace",
                    "error_message",
                ]
            )
            for rec in new_table.attempts:
                residual_trace = ";".join(f"{val:.3e}" for val in rec.residual_trace)
                for species_name, line_result in rec.line_results.items():
                    writer.writerow(
                        [
                            rec.row_idx,
                            rec.col_idx,
                            rec.nH,
                            rec.colDen,
                            rec.tg_guess,
                            rec.final_Tg,
                            rec.attempt_number,
                            rec.attempt_type,
                            rec.converged,
                            rec.repeat_equilibrium,
                            species_name,
                            line_result.int_tb,
                            line_result.int_intensity,
                            line_result.lum_per_h,
                            line_result.tau,
                            line_result.tau_dust,
                            line_result.tex,
                            line_result.freq,
                            rec.max_residual,
                            residual_trace,
                            rec.error_message or "",
                        ]
                    )

    LOGGER.info("Wrote updated table to %s", output_npz_path)
    if new_table.attempts:
        LOGGER.info("Wrote recomputation attempts to %s", attempts_csv)


if __name__ == "__main__":
    main()

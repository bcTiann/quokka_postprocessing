#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from quokka2s.despotic_tables import (
    AttemptRecord,
    DespoticTable,
    calculate_single_despotic_point,
)
from despotic.chemistry import NL99, NL99_GC, GOW
from build_despotic_table import plot_table, TG_GUESSES

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


def load_table(npz_path: Path) -> DespoticTable:
    with np.load(npz_path) as data:
        shape = data["co_int_tb"].shape

        def get_array(key: str) -> np.ndarray:
            if key in data:
                return data[key]
            return np.full(shape, np.nan)

        return DespoticTable(
            co_int_tb=data["co_int_tb"],
            tg_final=data["tg_final"],
            int_intensity=get_array("int_intensity"),
            lum_per_h=get_array("lum_per_h"),
            tau=get_array("tau"),
            tau_dust=get_array("tau_dust"),
            tex=get_array("tex"),
            frequency=get_array("frequency"),
            nH_values=data["nH"],
            col_density_values=data["col_density"],
            attempts=(),
        )


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
    log_failures: bool = True,
    repeat_equilibrium: int = 0,
    n_jobs: int = 1,
    nH_values: np.ndarray,
    col_values: np.ndarray,
    row_span: Tuple[int, int] | None = None,
    col_span: Tuple[int, int] | None = None,
    nH_range: Tuple[float, float] | None = None,
    col_range: Tuple[float, float] | None = None,
) -> DespoticTable:
    co_grid = np.array(table.co_int_tb, copy=True)
    tg_grid = np.array(table.tg_final, copy=True)
    intensity_grid = np.array(table.int_intensity, copy=True)
    lum_grid = np.array(table.lum_per_h, copy=True)
    tau_grid = np.array(table.tau, copy=True)
    tau_dust_grid = np.array(table.tau_dust, copy=True)
    tex_grid = np.array(table.tex, copy=True)
    freq_grid = np.array(table.frequency, copy=True)

    rows = select_indices(nH_values, row_span, nH_range)
    cols = select_indices(col_values, col_span, col_range)

    if rows.size == 0 or cols.size == 0:
        print("No cells match the specified region; skipping recomputation.")
        return table

    region_mask = np.zeros_like(co_grid, dtype=bool)
    region_mask[np.ix_(rows, cols)] = True

    print(
        f"Selected region rows: {rows[0]}–{rows[-1]} (count: {rows.size}), "
        f"cols: {cols[0]}–{cols[-1]} (count: {cols.size})"
    )

    mask = (np.isnan(co_grid) | (co_grid < threshold)) & region_mask
    if not np.any(mask):
        print(f"No cells require recomputation (threshold={threshold}).")
        return table

    attempt_records: list = []
    low_indices = np.argwhere(mask)
    print(f"Recomputing {len(low_indices)} cells with CO intensity < {threshold:.3e}")
    for row_idx, col_idx in low_indices:
        nH = float(nH_values[row_idx])
        col = float(col_values[col_idx])
        current_co = float(co_grid[row_idx, col_idx])
        print(
            f"  (row={row_idx}, col={col_idx}) "
            f"nH={nH:.6e} cm^-3 colDen={col:.6e} cm^-2 CO_before={current_co:.6e}"
        )

    def _evaluate_cell(row_idx: int, col_idx: int):
        row_log: list[AttemptRecord] = []
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
            nH_val=float(nH_values[row_idx]),
            colDen_val=float(col_values[col_idx]),
            initial_Tg_guesses=tg_guesses,
            log_failures=log_failures,
            repeat_equilibrium=repeat_equilibrium,
            chem_network=chem_network,
            row_idx=int(row_idx),
            col_idx=int(col_idx),
            attempt_log=row_log,
        )
        return (
            row_idx,
            col_idx,
            co_val,
            tg_val,
            intensity_val,
            lum_val,
            tau_val,
            tau_dust_val,
            tex_val,
            freq_val,
            tuple(row_log),
        )

    if n_jobs == 1:
        results = [_evaluate_cell(int(r), int(c)) for r, c in low_indices]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_cell)(int(r), int(c)) for r, c in low_indices
        )

    for (
        row_idx,
        col_idx,
        co_val,
        tg_val,
        intensity_val,
        lum_val,
        tau_val,
        tau_dust_val,
        tex_val,
        freq_val,
        row_log,
    ) in results:
        co_grid[row_idx, col_idx] = co_val
        tg_grid[row_idx, col_idx] = tg_val
        intensity_grid[row_idx, col_idx] = intensity_val
        lum_grid[row_idx, col_idx] = lum_val
        tau_grid[row_idx, col_idx] = tau_val
        tau_dust_grid[row_idx, col_idx] = tau_dust_val
        tex_grid[row_idx, col_idx] = tex_val
        freq_grid[row_idx, col_idx] = freq_val
        attempt_records.extend(row_log)

    return DespoticTable(
        co_int_tb=co_grid,
        tg_final=tg_grid,
        int_intensity=intensity_grid,
        lum_per_h=lum_grid,
        tau=tau_grid,
        tau_dust=tau_dust_grid,
        tex=tex_grid,
        frequency=freq_grid,
        nH_values=table.nH_values,
        col_density_values=table.col_density_values,
        attempts=table.attempts + tuple(attempt_records),
    )





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
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel workers to use for recomputation (default: 1).",
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
    args = parser.parse_args(argv)

    output_dir = args.output_npz.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.output_npz.stem.replace("table_", "")

    table = load_table(args.table_npz)
    new_table = recompute_low_co_cells(
        table,
        threshold=args.threshold,
        chem_network=NETWORK_MAP[args.network],
        tg_guesses=TG_GUESSES,
        log_failures=args.log_failures,
        repeat_equilibrium=args.repeat,
        n_jobs=args.n_jobs,
        nH_values=table.nH_values,
        col_values=table.col_density_values,
        row_span=tuple(args.row_span) if args.row_span else None,
        col_span=tuple(args.col_span) if args.col_span else None,
        nH_range=tuple(args.nH_range) if args.nH_range else None,
        col_range=tuple(args.col_range) if args.col_range else None,
    )

    recomputed_plot = output_dir / f"co_int_TB_{tag}.png"
    plot_table(
        table=new_table,
        data=new_table.co_int_tb,
        output_path=str(recomputed_plot),
        title=f"DESPOTIC Lookup Table ({tag})",
        cbar_label="CO Integrated Brightness Temp (K km/s)",
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
        output_path=str(output_dir / f"intensity_{tag}.png"),
        title=f"CO Integrated Intensity ({tag})",
        cbar_label="Integrated Intensity (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)",
    )
    plot_table(
        table=new_table,
        data=new_table.lum_per_h,
        output_path=str(output_dir / f"lum_per_H_{tag}.png"),
        title=f"Luminosity per H ({tag})",
        cbar_label="Luminosity per H (erg s$^{-1}$ H$^{-1}$)",
    )
    plot_table(
        table=new_table,
        data=new_table.tau,
        output_path=str(output_dir / f"tau_{tag}.png"),
        title=f"Line Optical Depth ({tag})",
        cbar_label="Line Optical Depth",
        use_log=False,
    )
    plot_table(
        table=new_table,
        data=new_table.tau_dust,
        output_path=str(output_dir / f"tau_dust_{tag}.png"),
        title=f"Dust Optical Depth ({tag})",
        cbar_label="Dust Optical Depth",
        use_log=False,
    )
    np.savez_compressed(
        args.output_npz,
        co_int_tb=new_table.co_int_tb,
        tg_final=new_table.tg_final,
        int_intensity=new_table.int_intensity,
        lum_per_h=new_table.lum_per_h,
        tau=new_table.tau,
        tau_dust=new_table.tau_dust,
        tex=new_table.tex,
        frequency=new_table.frequency,
        nH=new_table.nH_values,
        col_density=new_table.col_density_values,
    )

    attempts_csv = args.output_npz.with_name(args.output_npz.stem + "_attempts.csv")
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
                    "co_int_TB",
                    "int_intensity",
                    "lum_per_H",
                    "tau",
                    "tau_dust",
                    "Tex",
                    "frequency",
                    "error_message",
                ]
            )
            for rec in new_table.attempts:
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
                        rec.co_int_TB,
                        rec.int_intensity,
                        rec.lum_per_h,
                        rec.tau,
                        rec.tau_dust,
                        rec.tex,
                        rec.frequency,
                        rec.error_message or "",
                    ]
                )

    print(f"Wrote updated table to {args.output_npz}")
    if new_table.attempts:
        print(f"Wrote recomputation attempts to {attempts_csv}")


if __name__ == "__main__":
    main()

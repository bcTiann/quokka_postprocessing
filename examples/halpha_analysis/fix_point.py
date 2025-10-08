#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from quokka2s.despotic_tables import (
    DespoticTable,
    calculate_single_despotic_point,
)
from despotic.chemistry import NL99, NL99_GC, GOW
from build_despotic_table import plot_table

TG_GUESSES = np.logspace(np.log10(10.0), np.log10(10000.0), 20).tolist()

NETWORK_MAP = {
    "nl99": NL99,
    "nl99_gc": NL99_GC,
    "gow": GOW,
}


def load_table(npz_path: Path) -> DespoticTable:
    with np.load(npz_path) as data:
        return DespoticTable(
            co_int_tb=data["co_int_tb"],
            tg_final=data["tg_final"],
            nH_values=data["nH"],
            col_density_values=data["col_density"],
            attempts=(),
        )


def recompute_low_co_cells(
    table: DespoticTable,
    *,
    threshold: float,
    chem_network,
    tg_guesses: Sequence[float],
    log_failures: bool = True,
    repeat_equilibrium: int = 0,
) -> DespoticTable:
    co_grid = np.array(table.co_int_tb, copy=True)
    tg_grid = np.array(table.tg_final, copy=True)

    mask = np.isnan(co_grid) | (co_grid < threshold)
    if not np.any(mask):
        print(f"No cells require recomputation (threshold={threshold}).")
        return table

    attempt_records: list = []
    low_indices = np.argwhere(mask)
    print(f"Recomputing {len(low_indices)} cells with CO intensity < {threshold:.3e}")
    for row_idx, col_idx in low_indices:
        nH = float(table.nH_values[row_idx])
        col = float(table.col_density_values[col_idx])
        current_co = float(co_grid[row_idx, col_idx])
        print(
            f"  (row={row_idx}, col={col_idx}) "
            f"nH={nH:.6e} cm^-3 colDen={col:.6e} cm^-2 CO_before={current_co:.6e}"
        )

    for row_idx, col_idx in low_indices:
        row_log: list = []
        co_val, tg_val = calculate_single_despotic_point(
            nH_val=np.round(float(table.nH_values[row_idx]), 2),
            colDen_val=np.round(float(table.col_density_values[col_idx]), 2),
            initial_Tg_guesses=tg_guesses,
            log_failures=log_failures,
            repeat_equilibrium=repeat_equilibrium,
            chem_network=chem_network,
            row_idx=int(row_idx),
            col_idx=int(col_idx),
            attempt_log=row_log,
        )
        co_grid[row_idx, col_idx] = co_val
        tg_grid[row_idx, col_idx] = tg_val
        attempt_records.extend(row_log)

    return DespoticTable(
        co_int_tb=co_grid,
        tg_final=tg_grid,
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
    )

    recomputed_plot = output_dir / f"co_int_TB_{tag}.png"
    plot_table(
        table=new_table,
        data=new_table.co_int_tb,
        output_path=str(recomputed_plot),
        title=f"DESPOTIC Lookup Table ({tag})",
    )

    tg_plot = output_dir / f"tg_final_{tag}.png"
    plot_table(
        table=new_table,
        data=new_table.tg_final,
        output_path=str(tg_plot),
        title=f"DESPOTIC Gas Temperature ({tag})",
    )
    np.savez_compressed(
        args.output_npz,
        co_int_tb=new_table.co_int_tb,
        tg_final=new_table.tg_final,
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
                    "Tg_setTempEq",
                    "final_Tg",
                    "attempt_number",
                    "attempt_type",
                    "converged",
                    "repeat_equilibrium",
                    "emitter_abundance",
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
                        rec.Tg_setTempEq,
                        rec.final_Tg,
                        rec.attempt_number,
                        rec.attempt_type,
                        rec.converged,
                        rec.repeat_equilibrium,
                        rec.emitter_abundance,
                    ]
                )

    print(f"Wrote updated table to {args.output_npz}")
    if new_table.attempts:
        print(f"Wrote recomputation attempts to {attempts_csv}")


if __name__ == "__main__":
    main()

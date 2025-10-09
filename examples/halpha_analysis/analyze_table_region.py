#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

@dataclass
class RegionStats:
    count: int
    nan_count: int
    min_value: float
    max_value: float
    mean_value: float

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a DESPOTIC table (npz) over a selected coordinate region."
    )
    parser.add_argument("table_npz", type=Path, help="Path to the table_*.npz file.")
    parser.add_argument(
        "--nH-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Select rows whose nH lies within [MIN, MAX].",
    )
    parser.add_argument(
        "--col-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Select columns whose colDen lies within [MIN, MAX].",
    )
    parser.add_argument(
        "--row-span",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Select rows by index span [START, END) (0-based).",
    )
    parser.add_argument(
        "--col-span",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Select columns by index span [START, END) (0-based).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0e-8,
        help="Highlight cells with CO intensity below this value (default: 1e-8).",
    )
    return parser.parse_args()

def select_indices(values: np.ndarray, span: Tuple[int, int] | None, value_range: Tuple[float, float] | None) -> np.ndarray:
    if span:
        start, end = span
        return np.arange(start, min(end, values.size))
    if value_range:
        low, high = value_range
        return np.where((values >= low) & (values <= high))[0]
    return np.arange(values.size)

def summarise(array: np.ndarray) -> RegionStats:
    finite = np.isfinite(array)
    if not finite.any():
        return RegionStats(
            count=array.size,
            nan_count=array.size,
            min_value=float("nan"),
            max_value=float("nan"),
            mean_value=float("nan"),
        )
    return RegionStats(
        count=array.size,
        nan_count=np.count_nonzero(~finite),
        min_value=float(np.nanmin(array)),
        max_value=float(np.nanmax(array)),
        mean_value=float(np.nanmean(array)),
    )

def main() -> None:
    args = parse_args()
    table_path = args.table_npz
    if not table_path.exists():
        raise SystemExit(f"File not found: {table_path}")

    with np.load(table_path) as data:
        co_int = data["co_int_tb"]
        tg_final = data["tg_final"]
        nH_values = data["nH"]
        col_values = data["col_density"]

    rows = select_indices(
        nH_values,
        span=tuple(args.row_span) if args.row_span else None,
        value_range=tuple(args.nH_range) if args.nH_range else None,
    )
    cols = select_indices(
        col_values,
        span=tuple(args.col_span) if args.col_span else None,
        value_range=tuple(args.col_range) if args.col_range else None,
    )

    co_region = co_int[np.ix_(rows, cols)]
    tg_region = tg_final[np.ix_(rows, cols)]

    co_stats = summarise(co_region)
    tg_stats = summarise(tg_region)

    print(f"Region rows: {rows[0]}–{rows[-1]} (count: {rows.size})")
    print(f"Region cols: {cols[0]}–{cols[-1]} (count: {cols.size})")
    print("CO intensity stats:")
    print(f"  count={co_stats.count}  NaN={co_stats.nan_count}")
    print(f"  min={co_stats.min_value:.3e}  max={co_stats.max_value:.3e}  mean={co_stats.mean_value:.3e}")
    print("Final Tg stats:")
    print(f"  count={tg_stats.count}  NaN={tg_stats.nan_count}")
    print(f"  min={tg_stats.min_value:.3e}  max={tg_stats.max_value:.3e}  mean={tg_stats.mean_value:.3e}")

    mask = np.isfinite(co_region) & (co_region < args.threshold)
    num_low = np.count_nonzero(mask)
    if num_low:
        print(f"Cells with CO intensity < {args.threshold:g}: {num_low}")
        for (i, j) in zip(*np.where(mask)):
            row_idx = rows[i]
            col_idx = cols[j]
            print(
                f"  (row={row_idx}, col={col_idx}) "
                f"nH={nH_values[row_idx]:.3e} cm^-3  "
                f"colDen={col_values[col_idx]:.3e} cm^-2  "
                f"CO={co_region[i, j]:.3e}  Tg={tg_region[i, j]:.3e} K"
            )
    else:
        print(f"No cells below threshold {args.threshold:g} in this region.")

if __name__ == "__main__":
    main()

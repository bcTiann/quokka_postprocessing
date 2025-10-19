#!/usr/bin/env python3
"""
Quick sanity checks for DESPOTIC attempt tables.

The script expects the two CSV files produced by the pipeline:
    * attempts_50x50_raw.csv
    * table_50x50_fixed_attempts.csv

It summarises the raw attempt grid, inspects the follow-up "fixed" attempts,
and compares the two so you can see whether the retry strategy is behaving
as expected.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# Ensure we can parse the very long residual_trace column without raising.
csv.field_size_limit(2**31 - 1)

# Ordering used when attempts share the same attempt_number but encode a
# different summary row (e.g. the trailing ``all_guesses_failed`` entry).
ATTEMPT_TYPE_ORDER = {
    "successful": 0,
    "single_attempt": 1,
    "all_guesses_failed": 2,
    "exception": 3,
}

Key = Tuple[int, int, str]


def _as_float(value: str) -> float:
    """Parse floats while tolerating blanks and NaNs."""
    if not value or value.lower() == "nan":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def _as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def load_attempt_table(path: Path) -> Dict[Key, List[dict]]:
    """Load a CSV attempt table and bucket rows by (row_idx, col_idx, species)."""
    grouped: Dict[Key, List[dict]] = defaultdict(list)
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            key = (int(raw["row_idx"]), int(raw["col_idx"]), raw["species"])
            grouped[key].append(
                {
                    "attempt_number": int(raw["attempt_number"]),
                    "attempt_type": raw["attempt_type"],
                    "converged": _as_bool(raw["converged"]),
                    "temperature_converged": _as_bool(raw["temperature_converged"]),
                    "nH": _as_float(raw["nH"]),
                    "colDen": _as_float(raw["colDen"]),
                    "tg_guess": _as_float(raw["tg_guess"]),
                    "final_Tg": _as_float(raw["final_Tg"]),
                    "max_residual": _as_float(raw.get("max_residual", "")),
                    "error_message": (raw.get("error_message") or "").strip(),
                }
            )

    for attempts in grouped.values():
        attempts.sort(
            key=lambda row: (
                row["attempt_number"],
                ATTEMPT_TYPE_ORDER.get(row["attempt_type"], 99),
            )
        )
    return grouped


def summarise_raw_attempts(grouped: Dict[Key, List[dict]]) -> None:
    total_cells = len(grouped)
    successes = 0
    first_success_attempt = Counter()
    attempts_per_cell = []
    last_failure_types = Counter()
    last_error_messages = Counter()
    temp_non_convergence = Counter()

    for attempts in grouped.values():
        attempts_per_cell.append(len(attempts))
        success_entry = next((row for row in attempts if row["converged"]), None)
        if success_entry:
            successes += 1
            first_success_attempt[success_entry["attempt_number"]] += 1
        else:
            failure = attempts[-1]
            last_failure_types[failure["attempt_type"]] += 1
            last_error_messages[failure["error_message"] or "(empty)"] += 1
            if not failure["temperature_converged"]:
                temp_non_convergence[failure["attempt_type"]] += 1

    failures = total_cells - successes
    print("=== Raw attempt grid ===")
    print(f"Cells covered: {total_cells}")
    print(f"Successful cells: {successes} ({successes / total_cells:.2%})")
    print(f"Failed cells:     {failures} ({failures / total_cells:.2%})")
    print(
        "Attempts per cell: min={:.0f} max={:.0f} mean={:.2f}".format(
            min(attempts_per_cell),
            max(attempts_per_cell),
            statistics.mean(attempts_per_cell),
        )
    )
    if first_success_attempt:
        print("First-success attempt distribution:")
        for attempt_number, count in sorted(first_success_attempt.items()):
            print(f"  attempt {attempt_number}: {count}")
    if failures:
        print("Final failure types (raw grid):")
        for attempt_type, count in last_failure_types.most_common():
            print(f"  {attempt_type}: {count}")
        print("Temperature unconverged in final attempt (raw grid):")
        for attempt_type, count in temp_non_convergence.most_common():
            print(f"  {attempt_type}: {count}")
        print("Most common error messages (raw grid failures):")
        for message, count in last_error_messages.most_common(5):
            print(f"  {count} × {message}")
    print()


def summarise_fixed_attempts(grouped: Dict[Key, List[dict]]) -> None:
    cells = len(grouped)
    attempts_per_cell = [len(attempts) for attempts in grouped.values()]
    final_types = Counter(attempts[-1]["attempt_type"] for attempts in grouped.values())
    temp_non_convergence = sum(
        not attempts[-1]["temperature_converged"] for attempts in grouped.values()
    )

    print("=== Fixed-attempt sweep (table_…_fixed_attempts.csv) ===")
    print(f"Cells retried: {cells}")
    if cells:
        print(
            "Attempts per cell: min={:.0f} max={:.0f} mean={:.2f}".format(
                min(attempts_per_cell),
                max(attempts_per_cell),
                statistics.mean(attempts_per_cell),
            )
        )
        print("Final attempt types:")
        for attempt_type, count in final_types.most_common():
            print(f"  {attempt_type}: {count}")
        print(
            f"Final attempts with temperature not converged: {temp_non_convergence}"
        )
    print()


def compare_runs(
    raw_grouped: Dict[Key, List[dict]], fixed_grouped: Dict[Key, List[dict]]
) -> None:
    overlap = set(fixed_grouped) & set(raw_grouped)
    if not overlap:
        print(
            "No overlapping cells between raw and fixed tables – nothing to compare."
        )
        return

    improved = 0
    regressed = 0
    attempt_gain = []
    final_type_transitions = Counter()
    temp_convergence_deltas = Counter()

    for key in sorted(overlap):
        raw_attempts = raw_grouped[key]
        fixed_attempts = fixed_grouped[key]
        raw_final = raw_attempts[-1]
        fixed_final = fixed_attempts[-1]

        attempt_gain.append(len(fixed_attempts) - len(raw_attempts))
        final_type_transitions[
            (raw_final["attempt_type"], fixed_final["attempt_type"])
        ] += 1

        if raw_final["converged"] and not fixed_final["converged"]:
            regressed += 1
        elif not raw_final["converged"] and fixed_final["converged"]:
            improved += 1

        if raw_final["temperature_converged"] != fixed_final["temperature_converged"]:
            temp_convergence_deltas[
                (
                    raw_final["temperature_converged"],
                    fixed_final["temperature_converged"],
                )
            ] += 1

    print("=== Raw vs fixed comparison on overlapping cells ===")
    print(f"Cells compared: {len(overlap)}")
    print(f"Cells improved (new convergence): {improved}")
    print(f"Cells regressed (lost convergence): {regressed}")
    if attempt_gain:
        print(
            "Additional attempts per cell: min={:+d} max={:+d} mean={:+.2f}".format(
                min(attempt_gain),
                max(attempt_gain),
                statistics.mean(attempt_gain),
            )
        )
    print("Final attempt type transitions (raw → fixed):")
    for (raw_type, fixed_type), count in final_type_transitions.most_common():
        print(f"  {raw_type} → {fixed_type}: {count}")
    if temp_convergence_deltas:
        print("Temperature convergence flag changes:")
        for (raw_flag, fixed_flag), count in temp_convergence_deltas.items():
            print(f"  {raw_flag} → {fixed_flag}: {count}")
    print()


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarise DESPOTIC attempt grids for sanity checking."
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("examples/halpha_analysis/output_tables_Oct18"),
        help="Directory containing the CSV files to inspect.",
    )
    parser.add_argument(
        "--raw-file",
        type=str,
        default="attempts_50x50_raw.csv",
        help="Filename for the raw attempts table.",
    )
    parser.add_argument(
        "--fixed-file",
        type=str,
        default="table_50x50_fixed_attempts.csv",
        help="Filename for the post-processing fixed attempts table.",
    )
    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    raw_path = args.directory / args.raw_file
    fixed_path = args.directory / args.fixed_file

    if not raw_path.exists():
        raise SystemExit(f"Raw table not found: {raw_path}")
    if not fixed_path.exists():
        raise SystemExit(f"Fixed-attempt table not found: {fixed_path}")

    raw_grouped = load_attempt_table(raw_path)
    fixed_grouped = load_attempt_table(fixed_path)

    summarise_raw_attempts(raw_grouped)
    summarise_fixed_attempts(fixed_grouped)
    compare_runs(raw_grouped, fixed_grouped)


if __name__ == "__main__":
    main()

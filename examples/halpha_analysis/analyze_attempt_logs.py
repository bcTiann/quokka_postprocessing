from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ATTEMPT_FILE_PATTERN = "attempts_*.csv"
DEFAULT_OUTPUT_DIR = Path("output_tables")
CO_INT_THRESHOLD = 1.0e-6
LOW_CO_PRINT_LIMIT = 10


@dataclass(frozen=True)
class PointSummary:
    """Summary statistics for a single grid cell."""

    attempts: int
    ever_converged: bool
    first_attempt_success: bool
    final_converged: bool
    final_attempt_type: str
    last_attempt: Dict[str, str] | None


def _to_bool(value: str | bool | None) -> bool:
    """Return True for common truthy strings (case-insensitive)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return value.strip().lower() in {"true", "1", "yes", "y"}


def load_attempts(csv_path: Path) -> Dict[Tuple[int, int], PointSummary]:
    """Group attempt logs by (row_idx, col_idx) and summarise convergence."""
    grouped: Dict[Tuple[int, int], List[dict]] = defaultdict(list)

    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                row_idx = int(row["row_idx"])
                col_idx = int(row["col_idx"])
            except (KeyError, TypeError, ValueError):
                # Skip malformed rows rather than aborting the entire file.
                continue
            grouped[(row_idx, col_idx)].append(row)

    summaries: Dict[Tuple[int, int], PointSummary] = {}
    for key, attempts in grouped.items():
        # Ensure chronological order.
        attempts.sort(key=lambda record: int(record.get("attempt_number", 0)))

        ever_converged = any(
            _to_bool(record.get("converged")) or record.get("attempt_type") == "successful"
            for record in attempts
        )
        first_attempt_success = (
            attempts
            and (
                _to_bool(attempts[0].get("converged"))
                or attempts[0].get("attempt_type") == "successful"
            )
        )
        final_converged = (
            attempts
            and (
                _to_bool(attempts[-1].get("converged"))
                or attempts[-1].get("attempt_type") == "successful"
            )
        )
        summaries[key] = PointSummary(
            attempts=len(attempts),
            ever_converged=ever_converged,
            first_attempt_success=bool(first_attempt_success),
            final_converged=bool(final_converged),
            final_attempt_type=attempts[-1].get("attempt_type", ""),
            last_attempt=dict(attempts[-1]) if attempts else None,
        )

    return summaries


def summarise_points(points: Dict[Tuple[int, int], PointSummary]) -> str:
    """Generate a human-readable summary for a collection of point statistics."""
    total_points = len(points)
    if total_points == 0:
        return "  No attempt records found."

    total_attempts = sum(summary.attempts for summary in points.values())
    ever_converged = [key for key, summary in points.items() if summary.ever_converged]
    final_converged = [key for key, summary in points.items() if summary.final_converged]
    recovered = [
        key
        for key, summary in points.items()
        if summary.ever_converged and not summary.first_attempt_success
    ]
    never_converged = [key for key, summary in points.items() if not summary.ever_converged]

    lines = [
        f"  Grid cells analysed      : {total_points}",
        f"  Total attempts logged    : {total_attempts}",
        f"  Average attempts / cell  : {total_attempts / total_points:.2f}",
        f"  Ever converged           : {len(ever_converged)}",
        f"    • one-shot successes   : {len(ever_converged) - len(recovered)}",
        f"    • recovered after retry: {len(recovered)}",
        f"  Final state converged    : {len(final_converged)}",
        f"  Never converged          : {len(never_converged)}",
    ]

    if never_converged:
        sample = ", ".join(f"({i}, {j})" for i, j in never_converged[:5])
        lines.append(f"  Example non-converged cells: {sample}")
        lines.append("  Detailed non-converged attempts:")
        for (row_idx, col_idx) in never_converged[:10]:
            detail = points[(row_idx, col_idx)]
            last = detail.last_attempt or {}
            info = {
                "attempts": detail.attempts,
                "last_type": detail.final_attempt_type,
                "tg_guess": last.get("tg_guess"),
                "final_Tg": last.get("final_Tg"),
                "nH": last.get("nH"),
                "colDen": last.get("colDen"),
            }
            lines.append(
                "    ({row}, {col}) attempts={attempts} last={last_type} "
                "tg_guess={tg_guess} final_Tg={final_Tg} nH={nH} colDen={colDen}".format(
                    row=row_idx,
                    col=col_idx,
                    **info,
                )
            )

    return "\n".join(lines)


def find_attempt_files(directory: Path) -> List[Path]:
    """Return all attempt CSV files under the given directory."""
    return sorted(directory.glob(ATTEMPT_FILE_PATTERN))


def load_table_npz(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load table arrays corresponding to an attempt log CSV.

    Expects the matching file to be named like ``table_*.npz`` beside the CSV.
    """
    table_stem = csv_path.stem.replace("attempts_", "table_")
    table_path = csv_path.with_name(f"{table_stem}.npz")
    if not table_path.exists():
        return None

    with np.load(table_path) as data:
        try:
            co_int = data["co_int_tb"]
            tg_final = data["tg_final"]
            nH_values = data["nH"]
            col_values = data["col_density"]
        except KeyError:
            return None

    return co_int, tg_final, nH_values, col_values


def summarise_low_co_points(
    co_int: np.ndarray,
    tg_final: np.ndarray,
    nH_values: np.ndarray,
    col_values: np.ndarray,
    *,
    threshold: float = CO_INT_THRESHOLD,
    limit: int = LOW_CO_PRINT_LIMIT,
) -> str:
    """Generate a summary of cells whose CO intensity falls below ``threshold``."""
    mask = np.isfinite(co_int) & (co_int < threshold)
    if not np.any(mask):
        return f"  No cells with CO intensity < {threshold:g}."

    indices = np.argwhere(mask)
    lines = [
        f"  Cells with CO intensity < {threshold:g}: {indices.shape[0]} total"
    ]

    for row_idx, col_idx in indices[:limit]:
        lines.append(
            "    (row={row}, col={col}) nH={nH:.3e} cm^-3 colDen={col_den:.3e} cm^-2 "
            "CO={co:.3e} Tg={tg:.3e} K".format(
                row=row_idx,
                col=col_idx,
                nH=nH_values[row_idx],
                col_den=col_values[col_idx],
                co=co_int[row_idx, col_idx],
                tg=tg_final[row_idx, col_idx],
            )
        )

    remaining = indices.shape[0] - min(limit, indices.shape[0])
    if remaining > 0:
        lines.append(f"    … and {remaining} more.")

    return "\n".join(lines)


def main(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    csv_files = find_attempt_files(output_dir)
    if not csv_files:
        print(f"No attempt CSV files matching '{ATTEMPT_FILE_PATTERN}' found in {output_dir}.")
        return

    for csv_path in csv_files:
        print(csv_path.name)
        summaries = load_attempts(csv_path)
        print(summarise_points(summaries))

        table_arrays = load_table_npz(csv_path)
        if table_arrays is not None:
            co_int, tg_final, nH_values, col_values = table_arrays
            print(
                summarise_low_co_points(
                    co_int,
                    tg_final,
                    nH_values,
                    col_values,
                )
            )
        else:
            print("  Matching table_*.npz not found or invalid; skipping CO threshold check.")
        print()


if __name__ == "__main__":
    main()

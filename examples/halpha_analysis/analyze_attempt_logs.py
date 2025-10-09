from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ATTEMPT_FILE_PATTERN = "*.csv"
DEFAULT_OUTPUT_DIR = Path("output_tables_NL99_large")
CO_INT_THRESHOLD = 1.0e-7
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


def load_attempts(csv_path: Path, rows: List[dict] | None = None) -> Dict[Tuple[int, int], PointSummary]:
    """Group attempt logs by (row_idx, col_idx) and summarise convergence."""
    grouped: Dict[Tuple[int, int], List[dict]] = defaultdict(list)

    source_rows: List[dict]
    if rows is None:
        with csv_path.open(newline="") as fh:
            source_rows = list(csv.DictReader(fh))
    else:
        source_rows = rows

    for row in source_rows:
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
    rows: List[dict],
    *,
    threshold: float = CO_INT_THRESHOLD,
    limit: int = LOW_CO_PRINT_LIMIT,
) -> str:
    """Generate a summary of cells whose CO intensity falls below ``threshold``."""
    filtered = []
    for row in rows:
        value = row.get("co_int_TB")
        try:
            if value is not None and value != "" and float(value) < threshold:
                filtered.append(row)
        except (TypeError, ValueError):
            continue

    if not filtered:
        return f"  No cells with CO intensity < {threshold:g}."

    lines = [f"  Cells with CO intensity < {threshold:g}: {len(filtered)} total"]
    for row in filtered[:limit]:
        try:
            nH = float(row["nH"])
            col_den = float(row["colDen"])
            co_val = float(row["co_int_TB"])
            tg_val = float(row.get("final_Tg", "nan"))
        except (KeyError, TypeError, ValueError):
            continue

        lines.append(
            "    (row={row}, col={col}) nH={nH:.3e} cm^-3 colDen={col_den:.3e} cm^-2 "
            "CO={co:.3e} Tg={tg:.3e} K".format(
                row=row.get("row_idx"),
                col=row.get("col_idx"),
                nH=nH,
                col_den=col_den,
                co=co_val,
                tg=tg_val,
            )
        )

    remaining = len(filtered) - min(limit, len(filtered))
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
        with csv_path.open(newline="") as fh:
            raw_rows = list(csv.DictReader(fh))

        summaries = load_attempts(csv_path, rows=raw_rows)
        print(summarise_points(summaries))

        if raw_rows and "co_int_TB" in raw_rows[0]:
            print(summarise_low_co_points(rows=raw_rows, threshold=CO_INT_THRESHOLD))
        else:
            table_arrays = load_table_npz(csv_path)
            if table_arrays is not None:
                co_int, tg_final, nH_values, col_values = table_arrays
                valid_mask = np.isfinite(co_int)
                fallback_rows: List[dict] = [
                    {
                        "row_idx": str(i),
                        "col_idx": str(j),
                        "nH": str(nH_values[i]),
                        "colDen": str(col_values[j]),
                        "co_int_TB": str(co_int[i, j]),
                        "final_Tg": str(tg_final[i, j]),
                    }
                    for i, j in np.argwhere(valid_mask)
                ]
                print(summarise_low_co_points(rows=fallback_rows, threshold=CO_INT_THRESHOLD))
            else:
                print("  Matching table_*.npz not found or invalid; skipping CO threshold check.")
        print()


if __name__ == "__main__":
    main()

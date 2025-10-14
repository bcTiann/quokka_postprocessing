#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
from pathlib import Path

RAW_ATTEMPTS = Path("output_tables_NL99_large_TempEqOff/attempts_100x100_raw.csv")
FIX_ATTEMPTS = Path("output_tables_NL99_large_TempEqOff/table_100x100_fixed_attempts.csv")


def load_attempts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_candidates = {
        "row_idx",
        "col_idx",
        "tg_guess",
        "final_Tg",
        "attempt_number",
        "nH",
        "colDen",
        "co_int_TB",
    }
    for col in numeric_candidates & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["attempt_number"] = pd.to_numeric(df.get("attempt_number", pd.Series(dtype=float)), errors="coerce")
    return df


def first_success_after_failure(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (row, col), group in df.groupby(["row_idx", "col_idx"]):
        g = group.sort_values("attempt_number")
        successes = g[g["attempt_type"] == "successful"]
        if successes.empty:
            continue
        first_success = successes.iloc[0]
        prior = g[g["attempt_number"] < first_success["attempt_number"]]
        if prior.empty:
            continue
        first_failure = prior.iloc[0]
        records.append(
            {
                "row_idx": row,
                "col_idx": col,
                "failure_guess": first_failure["tg_guess"],
                "success_guess": first_success["tg_guess"],
                "final_Tg": first_success["final_Tg"],
                "failure_gap": abs(first_failure["tg_guess"] - first_success["final_Tg"]),
                "success_gap": abs(first_success["tg_guess"] - first_success["final_Tg"]),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    raw = load_attempts(RAW_ATTEMPTS)
    fixed = load_attempts(FIX_ATTEMPTS)

    raw_summary = first_success_after_failure(raw)
    print("---- Raw table: guesses that failed before succeeding ----")
    if raw_summary.empty:
        print("No cells had failure followed by success.")
    else:
        print(f"Cells considered: {len(raw_summary)}")
        print(f"Mean |failure guess - final Tg| : {raw_summary['failure_gap'].mean():.3f}")
        print(f"Median |failure guess - final Tg| : {raw_summary['failure_gap'].median():.3f}")
        print(f"Mean |success guess - final Tg| : {raw_summary['success_gap'].mean():.3f}")
        print(f"Median |success guess - final Tg| : {raw_summary['success_gap'].median():.3f}")
        print("Worst 5 failure gaps:")
        print(raw_summary.nlargest(5, "failure_gap")[["row_idx", "col_idx", "failure_guess", "final_Tg", "failure_gap"]])

    # Cells that succeeded on the first attempt
    print("\n---- Raw table: cells that converged on the first attempt ----")
    first_try_records = []
    for (row, col), group in raw.groupby(["row_idx", "col_idx"]):
        g = group.sort_values("attempt_number")
        if not g.empty and g.iloc[0]["attempt_type"] == "successful":
            first = g.iloc[0]
            first_try_records.append(
                {
                    "row_idx": row,
                    "col_idx": col,
                    "tg_guess": first["tg_guess"],
                    "final_Tg": first["final_Tg"],
                    "gap": abs(first["tg_guess"] - first["final_Tg"]),
                }
            )
    if not first_try_records:
        print("No cells converged on the first attempt.")
    else:
        first_try_df = pd.DataFrame(first_try_records)
        print(f"Cells converged on first try: {len(first_try_df)}")
        print(f"Mean |guess - final Tg| : {first_try_df['gap'].mean():.3f}")
        print(f"Median |guess - final Tg| : {first_try_df['gap'].median():.3f}")
        print("Sample first-try successes:")
        print(first_try_df.nsmallest(5, "gap")[["row_idx", "col_idx", "tg_guess", "final_Tg", "gap"]])

    raw_failed_cells = {
        (row, col)
        for (row, col), group in raw.groupby(["row_idx", "col_idx"])
        if not (group["attempt_type"] == "successful").any()
    }

    fixed_success_records = []
    for (row, col), group in fixed.groupby(["row_idx", "col_idx"]):
        if (row, col) not in raw_failed_cells:
            continue
        g = group.sort_values("attempt_number")
        successes = g[g["attempt_type"] == "successful"]
        if successes.empty:
            continue
        first_success = successes.iloc[0]
        raw_subset = raw[(raw["row_idx"] == row) & (raw["col_idx"] == col)]
        if raw_subset.empty:
            raw_first_attempt = float("nan")
        else:
            raw_first_attempt = raw_subset.sort_values("attempt_number").iloc[0]["tg_guess"]

        fixed_success_records.append(
            {
                "row_idx": row,
                "col_idx": col,
                "fixed_guess": first_success["tg_guess"],
                "fixed_final_Tg": first_success["final_Tg"],
                "fixed_gap": abs(first_success["tg_guess"] - first_success["final_Tg"]),
                "raw_first_guess": raw_first_attempt,
            }
        )

    print("\n---- Fixed table: formerly failed cells now succeeding ----")
    if not fixed_success_records:
        print("No previously failed cells were recovered.")
    else:
        fixed_df = pd.DataFrame(fixed_success_records)
        print(f"Recovered cells: {len(fixed_df)}")
        print(f"Mean |fixed guess - final Tg| : {fixed_df['fixed_gap'].mean():.3f}")
        print(f"Median |fixed guess - final Tg| : {fixed_df['fixed_gap'].median():.3f}")
        print("Sample recoveries:")
        print(
            fixed_df.nsmallest(5, "fixed_gap")[
                ["row_idx", "col_idx", "fixed_guess", "fixed_final_Tg", "fixed_gap", "raw_first_guess"]
            ]
        )


if __name__ == "__main__":
    main()

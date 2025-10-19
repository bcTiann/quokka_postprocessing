#!/usr/bin/env python3
"""
Diagnose a single DESPOTIC grid cell using the same settings as the table builder.

This helper lets you pick a failed (row, col, species) combination from either
`attempts_50x50_raw.csv` or `table_50x50_fixed_attempts.csv`, reconstruct the
matching DESPOTIC configuration, and rerun the temperature / chemistry solver
interactively to see whether it converges with custom guesses.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from despotic import cloud
from despotic.chemistry import GOW, NL99, NL99_GC

from analyze_attempt_tables import ATTEMPT_TYPE_ORDER, load_attempt_table  # type: ignore


DEFAULT_DIR = Path("examples/halpha_analysis/output_tables_Oct18")
RAW_FILENAME = "attempts_50x50_raw.csv"
FIXED_FILENAME = "table_50x50_fixed_attempts.csv"

EMITTER_ABUNDANCES = {"CO": 8.0e-9, "C+": 1.1e-4}
NETWORKS = {"nl99": NL99, "nl99_gc": NL99_GC, "gow": GOW}
BASE_TG_GUESSES = (5.12, 100.414, 1000.321)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", type=int, required=True, help="Row index from the CSV.")
    parser.add_argument("--col", type=int, required=True, help="Column index from the CSV.")
    parser.add_argument(
        "--species",
        choices=sorted(EMITTER_ABUNDANCES),
        default="CO",
        help="Emitter to diagnose (default: CO).",
    )
    parser.add_argument(
        "--source",
        choices=("raw", "fixed"),
        default="fixed",
        help="Select which CSV to read (raw attempts or fixed attempts).",
    )
    parser.add_argument(
        "--attempt-number",
        type=int,
        help="Restrict to this attempt number (defaults to last attempt for the cell).",
    )
    parser.add_argument(
        "--attempt-type",
        type=str,
        help="Optionally match a specific attempt_type (e.g. single_attempt, all_guesses_failed).",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_DIR,
        help="Directory containing the CSV files.",
    )
    parser.add_argument(
        "--raw-file",
        type=str,
        default=RAW_FILENAME,
        help="Filename for the raw attempts table.",
    )
    parser.add_argument(
        "--fixed-file",
        type=str,
        default=FIXED_FILENAME,
        help="Filename for the fixed attempts table.",
    )
    parser.add_argument(
        "--network",
        choices=NETWORKS.keys(),
        default="nl99",
        help="Chemical network to use for reruns.",
    )
    parser.add_argument(
        "--tg-override",
        type=float,
        help="Explicit starting Tg guess (placed at front of the guess queue).",
    )
    parser.add_argument(
        "--reuse-final",
        action="store_true",
        help="Reuse the final Tg from failed attempts as the next guess (like the table builder).",
    )
    parser.add_argument(
        "--reuse-max",
        type=int,
        default=5,
        help="Maximum times to insert the failed Tg back into the guess queue when --reuse-final is set.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="maxTempIter argument for setChemEq (default mirrors table builder).",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=1e20,
        help="maxTime argument for setChemEq.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance passed to setChemEq.",
    )
    parser.add_argument(
        "--no-lines",
        action="store_true",
        help="Skip computing line luminosities even if chemistry converges.",
    )
    return parser.parse_args(argv)


def _load_attempt(
    *,
    directory: Path,
    raw_file: str,
    fixed_file: str,
    source: str,
    key: Tuple[int, int, str],
) -> List[dict]:
    path = directory / (raw_file if source == "raw" else fixed_file)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    grouped = load_attempt_table(path)
    if key not in grouped:
        keys = sorted(grouped)
        raise KeyError(
            f"Cell {key} not present in {path.name}. "
            f"Available keys include {keys[:5]}..."
        )
    return grouped[key]


def _select_attempt(
    attempts: List[dict],
    *,
    attempt_number: int | None,
    attempt_type: str | None,
) -> dict:
    filtered = attempts
    if attempt_number is not None:
        filtered = [row for row in filtered if row["attempt_number"] == attempt_number]
        if not filtered:
            raise ValueError(
                f"No rows matching attempt_number={attempt_number}. "
                f"Available numbers: {sorted({row['attempt_number'] for row in attempts})}"
            )
    if attempt_type is not None:
        filtered = [row for row in filtered if row["attempt_type"] == attempt_type]
        if not filtered:
            raise ValueError(
                f"No rows matching attempt_type={attempt_type}. "
                f"Available types: {sorted({row['attempt_type'] for row in attempts})}"
            )
    return filtered[-1]


def _merge_guesses(primary: Iterable[float], secondary: Iterable[float]) -> List[float]:
    guesses: List[float] = []
    for value in list(primary) + list(secondary):
        if value is None:
            continue
        if not np.isfinite(value) or value <= 0:
            continue
        value = float(value)
        if any(math.isclose(value, existing, rel_tol=5e-2, abs_tol=1e-2) for existing in guesses):
            continue
        guesses.append(value)
    return guesses


def _configure_cell(nH: float, col_den: float, tg_guess: float) -> cloud.cloud:
    cell = cloud()
    cell.noWarn = True
    cell.nH = nH
    cell.colDen = col_den
    cell.Tg = tg_guess

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

    for species, abundance in EMITTER_ABUNDANCES.items():
        cell.addEmitter(species, abundance)
    return cell


def _run_single_attempt(
    *,
    nH: float,
    col_den: float,
    guess: float,
    chem_network,
    tol: float,
    max_time: float,
    max_iter: int,
) -> Tuple[bool, float, str]:
    cell = _configure_cell(nH, col_den, guess)
    cell.setTempEq()
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        converged = cell.setChemEq(
            network=chem_network,
            tol=tol,
            maxTime=max_time,
            maxTempIter=max_iter,
            evolveTemp="iterateDust",
            verbose=True,
        )
    output = stdout_buffer.getvalue()
    return converged, float(cell.Tg), output


def _summarise_attempt(
    *,
    attempt_idx: int,
    guess: float,
    final_tg: float,
    converged: bool,
    duration: float,
    stdout_capture: str,
) -> None:
    status = "SUCCESS" if converged else "FAIL"
    print(
        f"[Attempt {attempt_idx:02d}] guess={guess:.4g} K -> "
        f"Tg={final_tg:.4g} K | {status} | Δt={duration:.2f}s"
    )
    if stdout_capture.strip():
        print("  setChemEq output:")
        for line in stdout_capture.strip().splitlines():
            print(f"    {line}")


def run_cell_diagnosis(args: argparse.Namespace) -> None:
    key = (args.row, args.col, args.species)
    attempts = _load_attempt(
        directory=args.directory,
        raw_file=args.raw_file,
        fixed_file=args.fixed_file,
        source=args.source,
        key=key,
    )

    # Attempts are already sorted in load_attempt_table by (attempt_number, type)
    target = _select_attempt(
        attempts,
        attempt_number=args.attempt_number,
        attempt_type=args.attempt_type,
    )

    print("=== Selected attempt ===")
    print(f"location   : row={args.row}, col={args.col}, species={args.species}")
    print(f"source file: {args.source}")
    print(
        "attempt    : number={attempt_number}, type={attempt_type}, "
        "converged={converged}, temp_converged={temperature_converged}".format(**target)
    )
    print(
        f"nH={target['nH']:.6g} cm^-3 | colDen={target['colDen']:.6g} cm^-2 | "
        f"tg_guess={target['tg_guess']:.6g} K | final_Tg={target['final_Tg']:.6g} K"
    )
    if target.get("error_message"):
        print(f"error message: {target['error_message']}")
    print()

    primary_guesses = []
    if args.tg_override is not None:
        primary_guesses.append(args.tg_override)
    else:
        primary_guesses.append(target["tg_guess"])
    secondary_guesses = [target["final_Tg"], *BASE_TG_GUESSES]
    guesses = _merge_guesses(primary_guesses, secondary_guesses)
    if not guesses:
        raise RuntimeError("No usable Tg guesses assembled.")

    print(f"Initial guess queue: {', '.join(f'{g:.4g}' for g in guesses)}")
    if args.reuse_final:
        print(f"Reuse of failed Tg enabled (max insertions: {args.reuse_max}).")
    print()

    chem_network = NETWORKS[args.network]
    attempt_log: List[Tuple[float, bool, float]] = []
    reuse_insertions = 0
    attempt_idx = 0

    while guesses:
        guess = guesses.pop(0)
        attempt_idx += 1
        start = time.perf_counter()
        converged, final_tg, stdout_capture = _run_single_attempt(
            nH=target["nH"],
            col_den=target["colDen"],
            guess=guess,
            chem_network=chem_network,
            tol=args.tol,
            max_time=args.max_time,
            max_iter=args.max_iterations,
        )
        duration = time.perf_counter() - start
        _summarise_attempt(
            attempt_idx=attempt_idx,
            guess=guess,
            final_tg=final_tg,
            converged=converged,
            duration=duration,
            stdout_capture=stdout_capture,
        )
        attempt_log.append((guess, converged, final_tg))
        if converged:
            break
        if args.reuse_final and reuse_insertions < args.reuse_max:
            if np.isfinite(final_tg) and final_tg > 0:
                if not any(math.isclose(final_tg, existing, rel_tol=5e-2, abs_tol=1e-2) for _, _, existing in attempt_log):
                    guesses.insert(0, final_tg)
                    reuse_insertions += 1
        if not guesses:
            print("Guess queue exhausted without convergence.\n")

    final_converged = attempt_log and attempt_log[-1][1]
    final_tg = attempt_log[-1][2] if attempt_log else float("nan")
    print("=== Summary ===")
    print(f"Total attempts: {len(attempt_log)}")
    print(f"Converged     : {final_converged}")
    print(f"Final Tg      : {final_tg:.6g} K")

    if final_converged and not args.no_lines:
        cell = _configure_cell(target["nH"], target["colDen"], final_tg)
        cell.setTempEq()
        print("\nLine luminosities:")
        for species in sorted(EMITTER_ABUNDANCES):
            try:
                transitions = cell.lineLum(species)
            except Exception as exc:  # pragma: no cover - DESPOTIC variations
                print(f"  {species}: failed to compute lineLum ({exc})")
                continue
            if not transitions:
                print(f"  {species}: no transitions returned.")
                continue
            first = transitions[0]
            int_tb = first.get("intTB", float("nan"))
            tex = first.get("Tex", float("nan"))
            tau = first.get("tau", float("nan"))
            print(
                f"  {species}: intTB={int_tb:.6g}, Tex={tex:.6g}, tau={tau:.6g}"
            )


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    run_cell_diagnosis(args)


if __name__ == "__main__":
    main(sys.argv[1:])

import argparse
import csv
import json
import logging
import sys
import time
import warnings
from dataclasses import asdict, fields
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from despotic.chemistry import NL99, NL99_GC, GOW
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata

from quokka2s.despotic_tables import (
    DespoticTable,
    LogGrid,
    SpeciesLineGrid,
    build_table,
    fill_missing_values,
    make_temperature_interpolator,
    refine_table,
)

# OUTPUT_DIR = Path("output_tables_GC")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# # Old Range
# N_H_RANGE = (1e-4, 1e4)
# COL_DEN_RANGE = (1e18, 1e24)

# # Success Range
# N_H_RANGE = (1e1, 1e5)
# COL_DEN_RANGE = (1e20, 1e23)

N_H_RANGE = (1e-4, 1e5)
COL_DEN_RANGE = (1e18, 1e24)



TG_GUESSES = [5.12, 100.414, 1000.321]
PLOT_DPI = 600
SHOW_PLOTS = False

NETWORK_MAP = {
    "nl99": NL99,
    "nl99_gc": NL99_GC,
    "gow": GOW,
}

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_SPECIES = "CO"
LINE_FIELD_NAMES = tuple(field.name for field in fields(SpeciesLineGrid))


def configure_logging(log_path: Path) -> None:
    """Configure root logging to stream to stdout and the provided file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate logs when re-running
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
    try:
        from tqdm import TqdmExperimentalWarning  # type: ignore
    except ImportError:  # pragma: no cover - tqdm < 4.62
        pass
    else:
        warnings.simplefilter("ignore", TqdmExperimentalWarning)




def parse_cli_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DESPOTIC tables with selectable resolution/network/output."
    )
    parser.add_argument("resolution", type=int, nargs="+",
                        help="grid resolution: 10 or 10 20")
    parser.add_argument("network", choices=NETWORK_MAP.keys(),
                        help="chemical network : NL99 / NL99_GC / GOW")
    parser.add_argument("output_dir", type=Path,
                        help="output dir (auto create)")
    parser.add_argument("--repeat", type=int, default=0,
                        help="repeat_equilibrium (defualt=0)")
    parser.add_argument("--fill", action="store_true",
                        help="run fill_missing_values after table created")
    parser.add_argument("--round", dest="round_digits", type=int, default=None,
                        help="round LogGrid values to this many decimal places (omit for full precision)")
    parser.add_argument(
        "-j",
        "--jobs",
        dest="n_jobs",
        type=int,
        default=-1,
        help="Number of parallel workers (default: -1 uses all cores).",
    )
    parser.add_argument(
        "--reuse-final",
        "--reuse-final-tg",
        dest="reuse_final_tg",
        action="store_true",
        help="If set, reuse the final Tg from failed DESPOTIC runs as the next temperature guess.",
    )
    return parser.parse_args(argv[1:])


# def plot_table(data: np.ndarray, output_path: str, title: str, show: bool = SHOW_PLOTS) -> None:
def plot_table(
    *,
    table: DespoticTable,
    data: np.ndarray,
    output_path: str,
    title: str,
    cbar_label: str,
    show: bool = SHOW_PLOTS,
    use_log: bool = True,
) -> None:
    """a lookup table heatmap."""

    invalid = ~np.isfinite(data)
    if use_log:
        invalid |= data <= 0
    masked = np.ma.masked_array(data, mask=invalid)

    def _log_edges(values: np.ndarray) -> np.ndarray:
        if values.size < 2:
            raise ValueError("Need at least two grid points to compute edges.")
        log_values = np.log10(values)
        deltas = np.diff(log_values)
        edges = np.empty(values.size + 1, dtype=float)
        edges[1:-1] = log_values[:-1] + deltas / 2.0
        edges[0] = log_values[0] - deltas[0] / 2.0
        edges[-1] = log_values[-1] + deltas[-1] / 2.0
        return np.power(10.0, edges)

    col_edges = _log_edges(table.col_density_values)
    nH_edges = _log_edges(table.nH_values)

    fig, ax = plt.subplots(figsize=(8, 6))
    norm = None
    if use_log:
        valid_values = masked.compressed()
        if valid_values.size:
            norm = LogNorm(vmin=valid_values.min(), vmax=valid_values.max())
    mesh = ax.pcolormesh(col_edges, nH_edges, masked, shading="auto", cmap="viridis", norm=norm)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Column Density (cm$^{-2}$)")
    ax.set_ylabel("n$_\\mathrm{H}$ (cm$^{-3}$)")
    ax.set_title(title)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(cbar_label)

    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def save_table(prefix: str, table: DespoticTable) -> None:
    """Persist the lookup table plus axis metadata to an .npz archive."""
    species_names = np.array(table.species, dtype=object)
    data: dict[str, np.ndarray] = {
        "species": species_names,
        "tg_final": table.tg_final,
        "nH": table.nH_values,
        "col_density": table.col_density_values,
        "emitter_abundances": np.asarray(
            [table.emitter_abundances[sp] for sp in species_names],
            dtype=float,
        ),
    }

    for field in LINE_FIELD_NAMES:
        stacked = np.stack(
            [getattr(table.species_data[sp], field) for sp in species_names],
            axis=0,
        )
        data[f"{field}_grid"] = stacked

    np.savez_compressed(prefix, **data)



def build_table_at_resolution(points: int,
                              seed_table: DespoticTable | None,
                              repeat_equilibrium: int = 0,
                              chem_network=NL99,
                              round_digits: int | None = None,
                              n_jobs: int = -1,
                              reuse_failed_tg: bool = False,) -> DespoticTable:
    suffix = " (seeded by previous refinement)" if seed_table else ""
    LOGGER.debug("Preparing DESPOTIC table at resolution %dx%d%s", points, points, suffix)

    nH_grid = LogGrid(*N_H_RANGE, points, round_digits=round_digits)
    col_grid = LogGrid(*COL_DEN_RANGE, points, round_digits=round_digits)

    if seed_table is None:
        return build_table(
            nH_grid,
            col_grid,
            TG_GUESSES,
            chem_network=chem_network,
            show_progress=True,
            n_jobs=n_jobs,
            repeat_equilibrium=repeat_equilibrium,
            log_failures=True,
            reuse_failed_tg=reuse_failed_tg,
            reuse_max_insertions=3
        )

    interpolator = make_temperature_interpolator(
        seed_table.nH_values,
        seed_table.col_density_values,
        seed_table.tg_final,
    )
    return refine_table(
        seed_table,
        nH_grid,
        col_grid,
        TG_GUESSES,
        chem_network=chem_network,
        interpolator=interpolator,
        show_progress=True,
        repeat_equilibrium=repeat_equilibrium,
        n_jobs=n_jobs,
        reuse_failed_tg=reuse_failed_tg,
    )


def refine_same_resolution(table: DespoticTable, repeat_equilibrium: int = 0,
                           round_digits: int | None = None,
                           n_jobs: int = -1,
                           reuse_failed_tg: bool = False) -> DespoticTable:
    _, grid = table.require_species(DEFAULT_OUTPUT_SPECIES)
    points = grid.int_tb.shape[0]
    LOGGER.info("Refining temperature guesses on existing %dx%d grid", points, points)

    nH_grid = LogGrid(table.nH_values[0], table.nH_values[-1], points, round_digits=round_digits)
    col_grid = LogGrid(table.col_density_values[0], table.col_density_values[-1], points, round_digits=round_digits)

    interpolator = make_temperature_interpolator(
        table.nH_values,
        table.col_density_values,
        table.tg_final,
        kx=3,
        ky=3
    )

    return refine_table(
        table,
        nH_grid,
        col_grid,
        TG_GUESSES,
        interpolator=interpolator,
        show_progress=True,
        repeat_equilibrium=repeat_equilibrium,
        n_jobs=n_jobs,
        reuse_failed_tg=reuse_failed_tg,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_cli_args(sys.argv if argv is None else argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "build.log"
    configure_logging(log_path)
    LOGGER.info("Writing logs to %s", log_path)
    LOGGER.info(
        "CLI arguments: resolution=%s, network=%s, repeat=%d, fill=%s, round=%s, jobs=%d, reuse_final=%s",
        args.resolution,
        args.network,
        args.repeat,
        args.fill,
        args.round_digits,
        args.n_jobs,
        args.reuse_final_tg,
    )

    resolution_steps = tuple(args.resolution)
    chem_network = NETWORK_MAP[args.network]
    repeat_equilibrium = args.repeat
    fill_requested = args.fill
    round_digits = args.round_digits
    n_jobs = args.n_jobs
    reuse_failed_tg = args.reuse_final_tg
    previous_refined: DespoticTable | None = None



    for points in resolution_steps:

        tag = f"{points}x{points}"
##########################################################################################
        # Build Table
        LOGGER.info("Building DESPOTIC table at resolution %s", tag)
        raw_table = build_table_at_resolution(
            points,
            previous_refined,
            repeat_equilibrium=repeat_equilibrium,
            chem_network=chem_network,
            round_digits=round_digits,   
            n_jobs=n_jobs,
            reuse_failed_tg=reuse_failed_tg,
        )

        # Save Table
        raw_prefix = output_dir / f"table_{tag}_raw"
        save_table(raw_prefix, raw_table)
        LOGGER.info("Raw %s table saved to %s", tag, raw_prefix.with_suffix(".npz"))
        
        # Plot Table
        for species_name, species_grid in raw_table.species_data.items():
            species_token = species_name.replace("+", "plus").lower()
            plot_table(
                table=raw_table,
                data=species_grid.int_tb,
                output_path=str(output_dir / f"{species_token}_int_TB_{tag}_raw.png"),
                title=f"{species_name} Integrated Brightness ({tag} raw)",
                cbar_label=f"{species_name} Integrated Brightness Temp (K km/s)",
            )
            plot_table(
                table=raw_table,
                data=species_grid.int_intensity,
                output_path=str(output_dir / f"{species_token}_intensity_{tag}_raw.png"),
                title=f"{species_name} Integrated Intensity ({tag} raw)",
                cbar_label=f"{species_name} Integrated Intensity (erg cm$^{{-2}}$ s$^{{-1}}$ sr$^{{-1}}$)",
            )
            plot_table(
                table=raw_table,
                data=species_grid.lum_per_h,
                output_path=str(output_dir / f"{species_token}_lum_per_H_{tag}_raw.png"),
                title=f"{species_name} Luminosity per H ({tag} raw)",
                cbar_label=f"{species_name} Luminosity per H (erg s$^{{-1}}$ H$^{{-1}}$)",
            )
            plot_table(
                table=raw_table,
                data=species_grid.tau,
                output_path=str(output_dir / f"{species_token}_tau_{tag}_raw.png"),
                title=f"{species_name} Line Optical Depth ({tag} raw)",
                cbar_label=f"{species_name} Line Optical Depth",
                use_log=False,
            )
            plot_table(
                table=raw_table,
                data=species_grid.tau_dust,
                output_path=str(output_dir / f"{species_token}_tau_dust_{tag}_raw.png"),
                title=f"Dust Optical Depth ({tag} raw)",
                cbar_label="Dust Optical Depth",
                use_log=False,
            )
        plot_table(
            table=raw_table,
            data=raw_table.tg_final,
            output_path=str(output_dir / f"tg_final_{tag}_raw.png"),
            title=f"DESPOTIC Gas Temperature ({tag} raw)",
            cbar_label="Tg (K)"
        )

        if raw_table.attempts:
            attempts_path = output_dir / f"attempts_{tag}_raw.csv"
            with attempts_path.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([
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
                    "temperature_converged",
                    "max_residual",
                    "residual_trace",
                    "error_message",
                ])
                for record in raw_table.attempts:
                    residual_trace = ";".join(f"{val:.3e}" for val in record.residual_trace)
                    for species_name, line_result in record.line_results.items():
                        writer.writerow([
                            record.row_idx,
                            record.col_idx,
                            record.nH,
                            record.colDen,
                            record.tg_guess,
                            record.final_Tg,
                            record.attempt_number,
                            record.attempt_type,
                            record.converged,
                            record.repeat_equilibrium,
                            species_name,
                            line_result.int_tb,
                            line_result.int_intensity,
                            line_result.lum_per_h,
                            line_result.tau,
                            line_result.tau_dust,
                            line_result.tex,
                            line_result.freq,
                            record.temperature_converged,
                            record.max_residual,
                            residual_trace,
                            record.error_message or "",
                        ])
            LOGGER.info("%d attempts logged to %s", len(raw_table.attempts), attempts_path)
        else:
            LOGGER.info("No DESPOTIC attempts recorded for this table.")
        
        next_seed = raw_table
##########################################################################################


        # # Build Table
        # refined_table = refine_same_resolution(raw_table, repeat_equilibrium=REPEAT)

        # # Save Table
        # refined_prefix = OUTPUT_DIR / f"table_{tag}_fine"
        # save_table(refined_prefix, refined_table)
        # print(f"Refined {tag} table saved to 'table_{tag}_refined.npz'")

        # # Plot Table
        # fine_plot_path = OUTPUT_DIR / f"co_int_TB_{tag}_fine.png"
        # plot_table(
        #     str(fine_plot_path),
        #     f"DESPOTIC Lookup Table ({tag} refined)",
        # )

        # plot_table(
        #     refined_table.tg_final,
        #     str(OUTPUT_DIR / f"tg_final_{tag}_refined.png"),
        #     f"DESPOTIC Gas Temperature ({tag} refined)",
        # )

     

        # next_seed = refined_table

##########################################################################################


        if fill_requested:
            # Fill table
            filled = fill_missing_values(raw_table)
            filled_prefix = output_dir / f"table_{tag}_filled"
            save_table(filled_prefix, filled)

            # Plot Table
            for filled_species_name, filled_grid in filled.species_data.items():
                filled_species_token = filled_species_name.replace("+", "plus").lower()
                plot_table(
                    table=filled,
                    data=filled_grid.int_tb,
                    output_path=str(output_dir / f"{filled_species_token}_int_TB_{tag}_filled.png"),
                    title=f"{filled_species_name} Integrated Brightness ({tag} filled)",
                    cbar_label=f"{filled_species_name} Integrated Brightness Temp (K km/s)",
                )

                plot_table(
                    table=filled,
                    data=filled_grid.int_intensity,
                    output_path=str(output_dir / f"{filled_species_token}_intensity_{tag}_filled.png"),
                    title=f"{filled_species_name} Integrated Intensity ({tag} filled)",
                    cbar_label=f"{filled_species_name} Integrated Intensity (erg cm${{-2}}$ s${{-1}}$ sr${{-1}}$)",
                )
                plot_table(
                    table=filled,
                    data=filled_grid.lum_per_h,
                    output_path=str(output_dir / f"{filled_species_token}_lum_per_H_{tag}_filled.png"),
                    title=f"{filled_species_name} Luminosity per H ({tag} filled)",
                    cbar_label=f"{filled_species_name} Luminosity per H (erg s${{-1}}$ H${{-1}}$)",
                )
                plot_table(
                    table=filled,
                    data=filled_grid.tau,
                    output_path=str(output_dir / f"{filled_species_token}_tau_{tag}_filled.png"),
                    title=f"{filled_species_name} Line Optical Depth ({tag} filled)",
                    cbar_label=f"{filled_species_name} Line Optical Depth",
                    use_log=False,
                )
                plot_table(
                    table=filled,
                    data=filled_grid.tau_dust,
                    output_path=str(output_dir / f"{filled_species_token}_tau_dust_{tag}_filled.png"),
                    title=f"Dust Optical Depth ({tag} filled)",
                    cbar_label="Dust Optical Depth",
                    use_log=False,
                )

            plot_table(
                table=filled,
                data=filled.tg_final,
                output_path=str(output_dir / f"tg_final_{tag}_filled.png"),
                title=f"DESPOTIC Gas Temperature ({tag} filled)",
                cbar_label="Tg (K)",
                use_log=False,
            )


            next_seed = filled

        previous_refined = next_seed

    LOGGER.info("Multi-resolution DESPOTIC tables generated successfully.")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start
    LOGGER.info("Total runtime: %.2f s", elapsed)

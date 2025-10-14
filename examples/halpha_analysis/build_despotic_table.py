import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
from pathlib import Path
from typing import Sequence
import time
from despotic.chemistry import NL99, NL99_GC, GOW
import argparse
from quokka2s.despotic_tables import (
    DespoticTable,
    LogGrid,
    build_table,
    make_temperature_interpolator,
    refine_table,
    fill_missing_values,
)

# OUTPUT_DIR = Path("output_tables_GC")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# # Old Range
# N_H_RANGE = (1e-4, 1e4)
# COL_DEN_RANGE = (1e18, 1e24)

# # Success Range
# N_H_RANGE = (1e1, 1e5)
# COL_DEN_RANGE = (1e20, 1e23)

N_H_RANGE = (1e1, 1e5)
COL_DEN_RANGE = (1e18, 1e24)



TG_GUESSES = [5.12, 10.234, 25.245, 50.42, 100.414, 1000.321]
PLOT_DPI = 600
SHOW_PLOTS = False

NETWORK_MAP = {
    "nl99": NL99,
    "nl99_gc": NL99_GC,
    "gow": GOW,
}
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
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers (default: -1 uses all available cores).",
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
    use_log: bool = False,
) -> None:
    """a lookup table heatmap."""

    masked = np.ma.masked_where(np.isnan(data), data)

    X, Y = np.meshgrid(table.col_density_values, table.nH_values)

    fig, ax = plt.subplots(figsize=(8, 6))
    norm = LogNorm() if use_log else None
    mesh = ax.pcolormesh(X, Y, masked, shading="auto", cmap="viridis", norm=norm)
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
    np.savez_compressed(
        prefix,
        co_int_tb=table.co_int_tb,
        tg_final=table.tg_final,
        int_intensity=table.int_intensity,
        lum_per_h=table.lum_per_h,
        tau=table.tau,
        tau_dust=table.tau_dust,
        tex=table.tex,
        frequency=table.frequency,
        nH=table.nH_values,
        col_density=table.col_density_values,
    )



def build_table_at_resolution(points: int, 
                              seed_table: DespoticTable | None, 
                              repeat_equilibrium: int = 0,
                              chem_network=NL99,
                              round_digits: int | None = None,
                              n_jobs: int = -1,) -> DespoticTable:
    suffix = " (seeded by previous refinement)" if seed_table else ""
    print(f"Building DESPOTIC table at resolution {points}x{points}{suffix}")

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
            log_failures=True
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
    )


def refine_same_resolution(table: DespoticTable, repeat_equilibrium: int = 0,
                           round_digits: int | None = None,
                           n_jobs: int = -1) -> DespoticTable:
    points = table.co_int_tb.shape[0]
    print(f"Refining temperature guesses on existing {points}x{points} grid")

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
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_cli_args(sys.argv if argv is None else argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution_steps = tuple(args.resolution)
    chem_network = NETWORK_MAP[args.network]
    repeat_equilibrium = args.repeat
    fill_requested = args.fill
    round_digits = args.round_digits
    n_jobs = args.n_jobs
    previous_refined: DespoticTable | None = None



    for points in resolution_steps:

        tag = f"{points}x{points}"
##########################################################################################
        # Build Table
        print(f"Building DESPOTIC table at resolution {tag}")
        raw_table = build_table_at_resolution(
            points,
            previous_refined,
            repeat_equilibrium=repeat_equilibrium,
            chem_network=chem_network,
            round_digits=round_digits,   
            n_jobs=n_jobs,
        )

        # Save Table
        raw_prefix = output_dir / f"table_{tag}_raw"
        save_table(raw_prefix, raw_table)
        print(f"Raw {tag} table saved to {raw_prefix.with_suffix('.npz')}")
        
        # Plot Table
        plot_table(
            table=raw_table,
            data=raw_table.co_int_tb,
            output_path=str(output_dir / f"co_int_TB_{tag}_raw.png"),
            title=f"DESPOTIC Lookup Table ({tag} raw)",
            cbar_label="CO Integrated Brightness Temp (K km/s)",
        )
        plot_table(
            table=raw_table,
            data=raw_table.tg_final,
            output_path=str(output_dir / f"tg_final_{tag}_raw.png"),
            title=f"DESPOTIC Gas Temperature ({tag} raw)",
            cbar_label="Tg (K)",
            use_log=False,
        )
        plot_table(
            table=raw_table,
            data=raw_table.int_intensity,
            output_path=str(output_dir / f"intensity_{tag}_raw.png"),
            title=f"CO Integrated Intensity ({tag} raw)",
            cbar_label="Integrated Intensity (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)",
        )
        plot_table(
            table=raw_table,
            data=raw_table.lum_per_h,
            output_path=str(output_dir / f"lum_per_H_{tag}_raw.png"),
            title=f"Luminosity per H ({tag} raw)",
            cbar_label="Luminosity per H (erg s$^{-1}$ H$^{-1}$)",
        )
        plot_table(
            table=raw_table,
            data=raw_table.tau,
            output_path=str(output_dir / f"tau_{tag}_raw.png"),
            title=f"Line Optical Depth ({tag} raw)",
            cbar_label="Line Optical Depth",
            use_log=False,
        )
        plot_table(
            table=raw_table,
            data=raw_table.tau_dust,
            output_path=str(output_dir / f"tau_dust_{tag}_raw.png"),
            title=f"Dust Optical Depth ({tag} raw)",
            cbar_label="Dust Optical Depth",
            use_log=False,
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
                    "co_int_TB",
                    "int_intensity",
                    "lum_per_H",
                    "tau",
                    "tau_dust",
                    "Tex",
                    "frequency",
                    "error_message",
                ])
                for record in raw_table.attempts:
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
                    record.co_int_TB,
                    record.int_intensity,
                    record.lum_per_h,
                    record.tau,
                    record.tau_dust,
                    record.tex,
                    record.frequency,
                    record.error_message or "",
                ])
            print(f"{len(raw_table.attempts)} attempts logged to {attempts_path}")
        else:
            print("No DESPOTIC attempts recorded for this table.")
        
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
        #     refined_table.co_int_tb,
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
            filled_plot_path = output_dir / f"co_int_TB_{tag}_filled.png"

            plot_table(
                table=filled,
                data=filled.co_int_tb,
                output_path=str(filled_plot_path),
                title=f"DESPOTIC Lookup Table ({tag} filled)",
                cbar_label="CO Integrated Brightness Temp (K km/s)",
            )

            plot_table(
                table=filled,
                data=filled.tg_final,
                output_path=str(output_dir / f"tg_final_{tag}_filled.png"),
                title=f"DESPOTIC Gas Temperature ({tag} filled)",
                cbar_label="Tg (K)",
                use_log=False,
            )

            plot_table(
                table=filled,
                data=filled.int_intensity,
                output_path=str(output_dir / f"intensity_{tag}_filled.png"),
                title=f"CO Integrated Intensity ({tag} filled)",
                cbar_label="Integrated Intensity (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)",
            )
            plot_table(
                table=filled,
                data=filled.lum_per_h,
                output_path=str(output_dir / f"lum_per_H_{tag}_filled.png"),
                title=f"Luminosity per H ({tag} filled)",
                cbar_label="Luminosity per H (erg s$^{-1}$ H$^{-1}$)",
            )
            plot_table(
                table=filled,
                data=filled.tau,
                output_path=str(output_dir / f"tau_{tag}_filled.png"),
                title=f"Line Optical Depth ({tag} filled)",
                cbar_label="Line Optical Depth",
                use_log=False,
            )
            plot_table(
                table=filled,
                data=filled.tau_dust,
                output_path=str(output_dir / f"tau_dust_{tag}_filled.png"),
                title=f"Dust Optical Depth ({tag} filled)",
                cbar_label="Dust Optical Depth",
                use_log=False,
            )

            next_seed = filled

        previous_refined = next_seed

    print("Multi-resolution DESPOTIC tables generated successfully.")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start
    print(f"Total runtime: {elapsed:.2f} s")

import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
from pathlib import Path
import time
from quokka2s.despotic_tables import (
    DespoticTable,
    LogGrid,
    build_table,
    make_temperature_interpolator,
    refine_table,
    fill_missing_values,
)

OUTPUT_DIR = Path("output_tables_filters")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# N_H_RANGE = (1e-4, 1e4)
# COL_DEN_RANGE = (1e18, 1e24)
N_H_RANGE = (1e1, 1e5)
COL_DEN_RANGE = (1e20, 1e23)
RESOLUTION_STEPS = (5, 10)
FILL = False
TG_GUESSES = np.logspace(np.log10(10.0), np.log10(10000.0), 20).tolist()
PLOT_DPI = 600
SHOW_PLOTS = False
REPEAT = 0

def plot_table(data: np.ndarray, output_path: str, title: str, show: bool = SHOW_PLOTS) -> None:
    """Render and optionally display a lookup table heatmap."""
    masked = np.ma.masked_where(np.isnan(data), data)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        masked,
        origin="lower",
        cmap="viridis",
        norm=LogNorm(),
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("CO Integrated Brightness Temp (K km/s)")
    ax.set_xlabel("Column Density Index")
    ax.set_ylabel("nH Density Index")
    ax.set_title(title)

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
        nH=table.nH_values,
        col_density=table.col_density_values,
    )



def build_table_at_resolution(points: int, seed_table: DespoticTable | None, repeat_equilibrium: int = 0) -> DespoticTable:
    suffix = " (seeded by previous refinement)" if seed_table else ""
    print(f"Building DESPOTIC table at resolution {points}x{points}{suffix}")

    nH_grid = LogGrid(*N_H_RANGE, points)
    col_grid = LogGrid(*COL_DEN_RANGE, points)

    if seed_table is None:
        return build_table(
            nH_grid,
            col_grid,
            TG_GUESSES,
            show_progress=True,
            repeat_equilibrium=repeat_equilibrium
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
        interpolator=interpolator,
        show_progress=True,
        repeat_equilibrium=repeat_equilibrium
    )


def refine_same_resolution(table: DespoticTable, repeat_equilibrium: int = 0) -> DespoticTable:
    points = table.co_int_tb.shape[0]
    print(f"Refining temperature guesses on existing {points}x{points} grid")

    nH_grid = LogGrid(table.nH_values[0], table.nH_values[-1], points)
    col_grid = LogGrid(table.col_density_values[0], table.col_density_values[-1], points)

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
        repeat_equilibrium=repeat_equilibrium
    )


def main() -> None:
    previous_refined: DespoticTable | None = None


    for points in RESOLUTION_STEPS:

        tag = f"{points}x{points}"
##########################################################################################
        # Build Table
        raw_table = build_table_at_resolution(points, previous_refined, repeat_equilibrium=REPEAT)

        # Save Table
        raw_prefix = OUTPUT_DIR / f"table_{tag}_raw"
        save_table(raw_prefix, raw_table)
        print(f"Raw {tag} table saved to {raw_prefix.with_suffix('.npz')}")
        
        # Plot Table
        raw_plot_path = OUTPUT_DIR / f"co_int_TB_{tag}_raw.png"
        plot_table(
            raw_table.co_int_tb,
            str(raw_plot_path),
            f"DESPOTIC Lookup Table ({tag} raw)",
        )
        plot_table(
            raw_table.tg_final,
            str(OUTPUT_DIR / f"tg_final_{tag}_raw.png"),
            f"DESPOTIC Gas Temperature ({tag} raw)",
        )

        if raw_table.failures:
            failure_path = OUTPUT_DIR / f"failures_{tag}_raw.csv"
            with failure_path.open("w", newline="") as fh:
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
                    "repeat_equilibrium",
                    "emitter_abundance",
                ])
                for record in raw_table.failures:
                    writer.writerow([
                    record.row_idx,
                    record.col_idx,
                    record.nH,
                    record.colDen,
                    record.tg_guess,
                    record.final_Tg,
                    record.attempt_number,
                    record.attempt_type,
                    record.repeat_equilibrium,
                    record.emitter_abundance,
                ])
            print(f"{len(raw_table.failures)} failures logged to {failure_path}")
        else:
            print("No DESPOTIC failures recorded for this table.")
        
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


        if FILL:

            # Fill table
            filled_table = fill_missing_values(raw_table)
            

            # Save Table
            filled_prefix = OUTPUT_DIR / f"table_{tag}_refined_filled"
            save_table(filled_prefix, filled_table)
            print(f"Filled {tag} table saved to {filled_prefix.with_suffix('.npz')}")
            
            # Plot Table
            filled_plot_path = OUTPUT_DIR / f"co_int_TB_{tag}_filled.png"
            plot_table(
                filled_table.co_int_tb,
                str(filled_plot_path),
                f"DESPOTIC Lookup refine Table ({tag} filled)",
            )

            plot_table(
                filled_table.tg_final,
                str(OUTPUT_DIR / f"tg_final_{tag}_filled.png"),
                f"DESPOTIC Gas Temperature ({tag} filled)",
            )


            next_seed = filled_table

        previous_refined = next_seed

    print("Multi-resolution DESPOTIC tables generated successfully.")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start
    print(f"Total runtime: {elapsed:.2f} s")

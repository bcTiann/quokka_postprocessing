#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from quokka2s.tables import load_table
from quokka2s.tables.plotting import plot_table_overview
from quokka2s.pipeline.prep import config as cfg


def main():
    table = load_table(cfg.DESPOTIC_TABLE_PATH)
    samples = np.load("log_samples.npy")

    tokens = [
        "tg_final",
        "mu",
        "species:CO:abundance",
        "species:CO:lumPerH",
        "species:CO:intTB",

        "species:C+:abundance",
        "species:C+:lumPerH",
        "species:C+:intTB",

        "species:C:abundance",
        "species:C:lumPerH",
        "species:C:intTB",

        "species:HCO+:abundance",
        "species:HCO+:lumPerH",
        "species:HCO+:intTB",

        "species:e-:abundance",
        "species:H:abundance",
        "species:H2:abundance",
        "species:H+:abundance",

    ]
    T_targets = [1.0, 5e1, 5e2, 5e3, 5e4, 5e5, 5e6]
    T_indices = [int(np.argmin(np.abs(table.T_values - T))) for T in T_targets]
    table_path = Path(cfg.DESPOTIC_TABLE_PATH)
    base_dir = Path("plots") / table_path.parent.name
    base_dir.mkdir(parents=True, exist_ok=True)

    for t_idx, T_val in zip(T_indices, T_targets):
        fig = plot_table_overview(
            table,
            t_index=t_idx,
            fields=tokens,
            ncols=3,
            figsize=(20, 20),
            separate=False,  # 拼成一张大图
            samples=samples,
        )
        fname = f"T_{T_val:.0f}K.png"
        fig.savefig(base_dir / fname, dpi=400)
        plt.close(fig)


if __name__ == "__main__":
    main()

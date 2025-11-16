from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="collision rates not available",
    category=UserWarning,
    module=r"DESPOTIC.*emitterData",
)
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in log",
    category=RuntimeWarning,
    module=r"DESPOTIC.*NL99_GC",
)

import math
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .models import DespoticTable, SpeciesLineGrid


def _get_field_data(table: DespoticTable, token: str) -> tuple[np.ndarray, str]:
    """Retrieve data array and label for a given field token."""
    if token == "tg_final":
        return table.tg_final, "T_g (K)"
    if token == "failure_mask":
        return table.failure_mask.astype(float), "Failure Mask"
    
    if token.startswith("energy:"):
        key = token.split(":", 1)[1]
        if not table.energy_terms or key not in table.energy_terms:
            raise ValueError(f"Energy term '{key}' not found in the table.")
        return table.energy_terms[key], f"Energy Term: {key}"
    
    if token.startswith("species:"):
        _, spec, field = token.split(":")
        _, grid = table.require_species(spec)
        if not hasattr(grid, field):
            raise ValueError(f"Species field '{field}' not found")
        return getattr(grid, field), f"{spec}:{field}"
    
    raise ValueError(f"Unknown field token: {token}")

DEFAULT_FIELDS: tuple[str, ...] = (
    "tg_final",
    "energy:dEdtGas",
    "energy:GammaPE",
    "species:CO:lumPerH",
    "species:C+:lumPerH",
)

def _plot_panel(ax, data, title, table, cmap, show_colorbar, fig):
    im = ax.imshow(
        data.T,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        # extent=[
        #     math.log10(table.nH_values[0]),
        #     math.log10(table.nH_values[-1]),
        #     math.log10(table.col_density_values[0]),
        #     math.log10(table.col_density_values[-1]),
        # ],
    )

    ax.set_title(title)
    ax.set_xlabel("Column Density")
    ax.set_ylabel("Hydrogen Number Density (cm$^{-3}$)")
    ax.set_xticks(range(len(table.col_density_values)))
    ax.set_xticklabels([f"{v:1e}" for v in table.col_density_values], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(table.nH_values)))
    ax.set_yticklabels([f"{v:1e}" for v in table.nH_values], fontsize=8)


    if table.failure_mask is not None:
        mask = np.ma.masked_where(~table.failure_mask, table.failure_mask)
        ax.imshow(mask, origin="lower", aspect="auto", cmap="Greys", alpha=0.3)

    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    

def plot_table_overview(
    table: DespoticTable,
    *,
    fields: Sequence[str] | None = None,
    ncols: int = 3,
    figsize: tuple[float, float] = (15, 10),
    cmap: str = "viridis",
    show_colorbar: bool = True,
    separate: bool = False,
) -> plt.Figure | list[plt.Figure]:
    """Plot an overview of selected fields from a DespoticTable.

    Parameters:
        table: DespoticTable
            The DESPOTIC table to plot.
        fields: Sequence[str] | None
            List of field tokens to plot. If None, defaults are used.
        ncols: int
            Number of columns in the subplot grid.
        figsize: tuple[float, float]
            Size of the figure.
        cmap: str
            Colormap to use for the plots.
        show_colorbar: bool
            Whether to display colorbars for each subplot.

    Returns:
        plt.Figure
            The matplotlib Figure object containing the plots.
    """

    tokens = list(fields) if fields is not None else list(DEFAULT_FIELDS)
    if not tokens:
        raise ValueError("At least one field token must be specified for plotting.")
    if separate:
        figs = []
        for token in tokens:
            data, title = _get_field_data(table, token)
            fig, ax = plt.subplots(figsize=figsize)
            _plot_panel(ax, data, title, table, cmap, show_colorbar, fig)
            figs.append(fig)
        return figs

    n_panels = len(tokens)
    ncols = max(1, ncols)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)

    log_nH = [f"{v:1e}" for v in table.nH_values]
    log_colDen = [f"{v:1e}" for v in table.col_density_values]

    for idx, token in enumerate(tokens):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        data, title = _get_field_data(table, token)
        _plot_panel(ax, data, title, table, cmap, show_colorbar, fig)


    # Hide any unused subplots
    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis("off")

    fig.tight_layout()
    return fig



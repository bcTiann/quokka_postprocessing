from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .models import DespoticTable

def plot_failure_overlay(*_args, **_kwargs) -> None:
    """Placeholder for failure-visualization helper."""
    raise NotImplementedError("plot_failure_overlay is not implemented yet.")

def summarize_failures(*_args, **_kwargs) -> None:
    """Placeholder for failure summary helper."""
    raise NotImplementedError("summarize_failures is not implemented yet.")



def plot_sampling_histogram(
        table: DespoticTable,
        samples: np.ndarray,
        *,
        ax: plt.Axes | None = None,
        cmap: str = "viridis",
        log_space: bool = True,
) -> plt.Axes:
    """Plot a histogram showing the sampling density of given samples in the table.

    Parameters:
        table: DespoticTable
            The DESPOTIC table to reference for sampling density.
        samples: np.ndarray
            An array of shape (N, 2) containing N samples of (nH, column density).
        ax: plt.Axes | None
            The axes to plot on. If None, a new figure and axes will be created.
        cmap: str
            The colormap to use for the histogram.
        log_space: bool
            if True, samples are already in log10 space. If False, samples are in linear space and will be applied to log10.
    Returns:
        plt.Axes
            The axes containing the histogram plot.
    """

    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("samples must be a 2D array with shape (N, 2)")
    

    nH_vals = np.log10(table.nH_values) if log_space else table.nH_values
    col_vals = np.log10(table.col_density_values) if log_space else table.col_density_values

    data = samples.copy()
    if not log_space:
        data = np.log10(data)
    
    bins = (np.append(nH_vals, nH_vals[-1] + np.diff(nH_vals).mean()),
                np.append(col_vals, col_vals[-1] + np.diff(col_vals).mean()))

    hist, _, _ = np.histogram2d(
        data[:, 0],
        data[:, 1],
        bins=bins,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    mesh = ax.pcolormesh(
        nH_vals,
        col_vals,
        hist.T,
        cmap=cmap,
        shading="auto",
    )
    ax.set_xlabel("log10 nH [cm^-3]")
    ax.set_ylabel("log10 N_H [cm^-2]")
    fig = ax.get_figure()
    fig.colorbar(mesh, ax=ax, label="Counts")

    if table.failure_mask is not None:
        print("Overlaying failure regions...")
        mask = np.ma.masked_where(~table.failure_mask, table.failure_mask)
        overlay = np.where(table.failure_mask, 1.0, np.nan)
        ax.imshow(
            overlay.T,
            origin="lower",
            aspect="auto",
            cmap="Reds",
            alpha=0.8,
            vmin=0.0,
            vmax=1.0,
            extent=[
                nH_vals[0],
                nH_vals[-1],
                col_vals[0],
                col_vals[-1],
            ],
        )
    return ax



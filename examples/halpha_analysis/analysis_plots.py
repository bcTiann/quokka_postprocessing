import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from build_despotic_table import plot_table


def report_raw_table_coverage(
    *,
    table,
    species_grid,
    raw_log_interp,
    nH_cgs,
    colDen_cgs,
    valid_mask,
    output_path: str | None = None,
    log_color: bool = True,
) -> np.ndarray:
    """Report how simulation sampling maps onto the raw (unfilled) table.

    Returns the log10 luminosity values evaluated with the raw interpolator so
    that callers can build additional masks (e.g., strict coverage masks).
    """
    if raw_log_interp is None:
        return np.full_like(nH_cgs, np.nan, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_nH_all = np.log10(nH_cgs)
        log_col_all = np.log10(colDen_cgs)

    raw_log_values = np.asarray(raw_log_interp(log_nH_all, log_col_all))
    sampled_points = raw_log_values.size
    raw_nan_mask_any = ~np.isfinite(raw_log_values)
    raw_nan_count_any = np.count_nonzero(raw_nan_mask_any)

    raw_nan_mask = valid_mask & raw_nan_mask_any
    raw_nan_count = np.count_nonzero(raw_nan_mask)

    print(
        f"[DESPOTIC] Simulation voxels mapped to raw table: {sampled_points}; "
        f"original NaN hits (any range): {raw_nan_count_any} "
        f"({raw_nan_count_any / sampled_points:.3%})."
    )

    sampled_in_bounds = np.count_nonzero(valid_mask)
    if sampled_in_bounds > 0:
        coverage_pct = 100.0 * (sampled_in_bounds - raw_nan_count) / sampled_in_bounds
        print(
            f"[DESPOTIC] Within table bounds: {sampled_in_bounds}; "
            f"original NaN hits: {raw_nan_count} "
            f"({raw_nan_count / sampled_in_bounds:.3%})."
        )
        print(
            f"[DESPOTIC] Fraction retaining original table values (no fill): "
            f"{coverage_pct:.2f}%."
        )

    if output_path:
        table_valid = np.isfinite(species_grid.lum_per_h)

        def _log_edges(values: np.ndarray) -> np.ndarray:
            log_values = np.log10(values)
            deltas = np.diff(log_values)
            edges = np.empty(values.size + 1, dtype=float)
            edges[1:-1] = log_values[:-1] + deltas / 2.0
            edges[0] = log_values[0] - deltas[0] / 2.0
            edges[-1] = log_values[-1] + deltas[-1] / 2.0
            return edges

        nH_edges = _log_edges(table.nH_values)
        col_edges = _log_edges(table.col_density_values)
        finite_samples = np.isfinite(log_nH_all) & np.isfinite(log_col_all)
        counts, _, _ = np.histogram2d(
            log_nH_all[finite_samples].ravel(),
            log_col_all[finite_samples].ravel(),
            bins=[nH_edges, col_edges],
        )
        counts = counts.astype(float)
        masked = np.ma.masked_where(~table_valid, counts)

        plot_table(
            table=table,
            data=masked,
            output_path=output_path,
            title="Simulation Sampling vs Raw Table Coverage",
            cbar_label="Voxel count per table cell",
            use_log=log_color,
            overlay_mask=~table_valid,
            overlay_alpha=0.35,
        )
        print(f"figure saved to '{output_path}'")

    return raw_log_values


def _build_masked_array(data: np.ndarray, mask: np.ndarray | None) -> np.ma.MaskedArray:
    array = np.asarray(data, dtype=float)
    if mask is None:
        masked = np.ma.masked_invalid(array)
    else:
        masked = np.ma.masked_where(~mask, array)
    return masked


def plot_masked_image(
    *,
    data: np.ndarray,
    mask: np.ndarray | None,
    extent,
    title: str,
    cbar_label: str,
    output_path: str,
    xlabel: str = "",
    ylabel: str = "",
    cmap: str = "viridis",
    log_norm: bool = True,
) -> None:
    """Render a masked 2D field with optional logarithmic color scaling."""
    masked = _build_masked_array(data, mask)

    norm = None
    if log_norm:
        valid_values = masked.compressed()
        valid_values = valid_values[np.isfinite(valid_values)]
        positive = valid_values[valid_values > 0]
        if positive.size:
            norm = LogNorm(vmin=positive.min(), vmax=positive.max())

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        masked.T,
        extent=extent,
        origin="lower",
        cmap=cmap,
        norm=norm,
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=800, bbox_inches="tight")
    print(f"figure saved to '{output_path}'")
    plt.close(fig)

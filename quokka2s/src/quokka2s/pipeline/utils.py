from ..utils.axes import axis_label
from typing import Iterable
import numpy as np
from yt.units.yt_array import YTArray
from matplotlib.colors import LogNorm

_PLANE_AXES = {
    "x": ("y", "z"),
    "y": ("x", "z"),
    "z": ("x", "y"),
}


def plane_axes(axis: str) -> tuple[str, str]:
    """Return the two axes that span the plotting plane for the given projection axis."""
    canonical = axis_label(axis)
    return _PLANE_AXES[canonical]


def make_axis_labels(axis: str, units: str) -> tuple[str, str]:
    """Produce human-readable axis labels with units."""
    horiz, vert = plane_axes(axis)
    return f"{horiz.upper()} ({units})", f"{vert.upper()} ({units})"



def shared_lognorm(*arrays: Iterable[YTArray]) -> LogNorm | None:
    """
    Build a LogNorm spanning the finite positive values of all provided arrays.
    Returns None if no positive values exist.
    """

    merged = np.concatenate(arrays)
    vmin = merged.min()
    vmax = merged.max()

    return LogNorm(vmin=vmin, vmax=vmax)

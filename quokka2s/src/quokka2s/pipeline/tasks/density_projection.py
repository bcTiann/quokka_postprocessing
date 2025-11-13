"""Density projection task."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LogNorm

from ...plotting import create_plot
from ...utils.axes import axis_index
from ..base import AnalysisTask, PipelinePlotContext
from ..utils import make_axis_labels


class DensityProjectionTask(AnalysisTask):
    """Simple density projection along the chosen axis."""

    def __init__(
        self,
        config,
        axis: str | None = None,
        figure_units: str | None = None,
        title: str = "Density Projection",
        filename: str = "density.png",
    ) -> None:
        super().__init__(config)
        self.axis = axis
        self.axis_idx = axis_index(self.axis)
        self.figure_units = figure_units or config.figure_units
        self.xlabel, self.ylabel = make_axis_labels(self.axis, self.figure_units)
        self.title = title
        self.filename = filename
        self._rho_3d = None
        self._extent = None
        self._norm = LogNorm()

    def prepare(self, context: PipelinePlotContext) -> None:
        provider = context.provider
        self._rho_3d, self._extent = provider.get_slab_z(("gas", "density"))

    def compute(self, context: PipelinePlotContext):
        density_projection = np.sum(self._rho_3d, axis=self.axis_idx)
        context.results["density_projection"] = density_projection
        return {"map": density_projection, "extent": self._extent[self.axis]}

    def plot(self, context: PipelinePlotContext, results):
        output = self.config.output_dir / self.filename
        create_plot(
            data_2d=results["map"].T.to_ndarray(),
            title=self.title,
            cbar_label=f"Density ({results['map'].units})",
            filename=str(output),
            extent=results["extent"],
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            norm=self._norm,
        )

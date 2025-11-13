from __future__ import annotations

import numpy as np
from yt.units import cm

from ...analysis import calculate_attenuation, calculate_cumulative_column_density
from ...plotting import create_plot
from ...utils.axes import axis_index
from ..base import AnalysisTask, PipelinePlotContext
from ..utils import make_axis_labels


class HalphaWithDustTask(AnalysisTask):
    """Integrate H-alpha luminosity including dust attenuation."""

    def __init__(self, config, axis: str | None = None, figure_units: str | None = None):
        super().__init__(config)
        self.axis = axis
        self.axis_idx = axis_index(self.axis)
        self.figure_units = figure_units or config.figure_units
        self.xlabel, self.ylabel = make_axis_labels(self.axis, self.figure_units)
        self._lum_3d = None
        self._rho_3d = None
        self._dx_3d = None
        self._extent = None

    def prepare(self, context: PipelinePlotContext) -> None:
        provider = context.provider
        self._lum_3d, self._extent = provider.get_slab_z(("gas", "Halpha_luminosity"))
        self._rho_3d, _ = provider.get_slab_z(("gas", "density"))
        self._dx_3d, _ = provider.get_slab_z(("boxlib", "dx"))

    def compute(self, context: PipelinePlotContext):
        X_H = 1.0   # Hydrogen mass fraction
        A_LAMBDA_OVER_NH =  4e-22 * cm**2 # Attenuation 

        N_H = calculate_cumulative_column_density(self._rho_3d, self._dx_3d, axis=self.axis_idx, X_H=X_H, sign="+")
        attenuation, _ = calculate_attenuation(N_H, A_LAMBDA_OVER_NH)
        attenuated_map = np.sum(self._lum_3d * attenuation * self._dx_3d, axis=self.axis_idx)

        context.results["halpha_with_dust"] = attenuated_map


        return {"map": attenuated_map, "extent": self._extent[self.axis]}

    def plot(self, context: PipelinePlotContext, results):
        pass 
        # output = self.config.output_dir / "halpha_with_dust_shared.png"
        # create_plot(
        #     data_2d=results["map"].T.to_ndarray(),
        #     title="H-alpha (Dust)",
        #     cbar_label=f"Surface Brightness ({results['map'].units})",
        #     filename=str(output),
        #     extent=results["extent"],
        #     xlabel=self.xlabel,
        #     ylabel=self.ylabel,
        #     norm=self.config.extra_options.get("halpha_norm"),
        # )

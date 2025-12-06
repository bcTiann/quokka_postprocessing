from __future__ import annotations

import numpy as np
from yt.units import cm
from matplotlib.colors import LogNorm

from quokka2s.tables import load_table
from quokka2s.tables.lookup import TableLookup
from quokka2s.pipeline.prep import config as cfg

from ...analysis import calculate_attenuation, calculate_cumulative_column_density
from ...plotting import create_plot, plot_multiview_grid
from ...utils.axes import axis_index
from ..base import AnalysisTask, PipelinePlotContext
from ..utils import make_axis_labels, shared_lognorm


class EmitterTask(AnalysisTask):
    """Integrate Emitter luminosity without dust attenuation."""

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
        self._CO_lum_3d = None
        self._Cplus_lum_3d = None
        self._T = None

    def prepare(self, context: PipelinePlotContext) -> None:
        provider = context.provider
        self._CO_lum_3d, self._extent = provider.get_slab_z(("gas", "CO_luminosity"))
        self._Cplus_lum_3d, self._extent = provider.get_slab_z(("gas", "C+_luminosity"))
        self._rho_3d, _ = provider.get_slab_z(("gas", "density"))
        self._dx_3d, _ = provider.get_slab_z(("boxlib", "dx"))
        # self._dy_3d, self._bottom_extent = provider.get_slab_z(("boxlib", "dy"))
        # self._dz_3d, _ = provider.get_slab_z(("boxlib", "dz"))
        self._conv_mask, _ = provider.get_slab_z(("gas", "temperature_converged"))  # 新增
        self._T, _ = provider.get_slab_z(("gas", "temperature"))


    def compute(self, context: PipelinePlotContext):

        ############### No Dust ################
        CO_surface_brightness = np.sum(self._CO_lum_3d * self._dx_3d, axis=0)
        Cplus_surface_brightness = np.sum(self._Cplus_lum_3d * self._dx_3d, axis=0)
        context.results["CO"] = CO_surface_brightness
        context.results["Cplus"] = Cplus_surface_brightness

        # 温度上下限掩膜：等于 T_min 或 T_max 视为裁剪
        T_vals = self._T.in_cgs().to_ndarray()
        lookup = TableLookup(load_table(cfg.DESPOTIC_TABLE_PATH))
        T_min = lookup.table.T_values.min()
        T_max = lookup.table.T_values.max()
        clip_low = (T_vals <= T_min * 1.0001)
        clip_high = (T_vals >= T_max * 0.9999)
        clip_mask = clip_low | clip_high

        # 收敛统计
        conv = np.asarray(self._conv_mask)
        frac_conv = np.count_nonzero(conv) / conv.size if conv.size else np.nan

        context.results["converged_fraction"] = frac_conv

        return {
            "CO": CO_surface_brightness,
            "Cplus": Cplus_surface_brightness,
            "extent": self._extent[self.axis],
            "converged_fraction": frac_conv,
            "temperature": self._T,
            "density": self._rho_3d,
            "conv_mask": conv,
            "clip_mask": clip_mask,
        }


    def plot(self, context: PipelinePlotContext, results):
        frac = results.get("converged_fraction")
        print(f"Temperature field converged fraction: {frac:.3f}")
        extent = [float(v.to(self.figure_units).value) for v in results["extent"]]

        shared_norm = shared_lognorm(
            results["CO"],
            results["Cplus"],
        )
        conv_mask = results["conv_mask"]
        clip_mask = results["clip_mask"]
        
        # 在温度图上叠加 clip_mask
        idx = 64
        temp2d = results["temperature"].in_cgs().to_ndarray()[idx, :, :]
        mask2d = clip_mask[idx, :, :]
        temp_masked = np.ma.masked_where(mask2d, temp2d)

        unit_label = str(results["CO"].in_cgs().units)
        
        plots_info = [
            {
                "title": "CO Emission",
                "label": f"Surface Brightness ({unit_label})",
                "norm": LogNorm(),
                "data_top": results["CO"].in_cgs().to_ndarray().T,
            },
            {
                "title": "C+ Emssion",
                "label": f"Surface Brightness ({unit_label})",
                "norm": LogNorm(),
                "data_top": results["Cplus"].in_cgs().to_ndarray().T,
            },
            {
                "title": "Temperature Slice",
                "label": "Temperature (K)",
                "norm": LogNorm(),
                "data_top": results["temperature"].in_cgs().to_ndarray()[idx,: , :].T,
            },
            {
                "title": "Temperature Slice (clipped shaded)",
                "label": "Temperature (K)",
                "norm": LogNorm(),
                "data_top": temp_masked.T,
                "overlay_mask": mask2d.T,  # 你可以在 plot_multiview_grid 里支持 overlay
            },
            {
                "title": "density Slice",
                "label": rf"$\rho$ ({self._rho_3d.units})",
                "norm": LogNorm(),
                "data_top": results["density"].in_cgs().to_ndarray()[idx,: , :].T,
            },
        ]

        plot_multiview_grid(
            plots_info=plots_info,
            extent_top=extent,
            filename=str(self.config.output_dir / "emitter.png"),
            top_ylabel=self.ylabel,
            top_xlabel=self.xlabel,
            include_bottom=False,
            units=self.figure_units,
        )

        
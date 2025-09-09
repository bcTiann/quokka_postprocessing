import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter 
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter, LogFormatterMathtext, MaxNLocator
import yt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def create_plot(data_2d: np.ndarray,
                title: str,
                cbar_label: str,
                filename: str,
                extent: list,
                xlabel: str = None,
                ylabel: str = None,
                norm: Optional[LogNorm] = None,
                camp: str = 'viridis',
                use_scientific_formatter: bool = True
                ):
    """
    Generates and saves a 2D plot from a NumPy array.
    Parameters:
    - data_2d: 2D NumPy array to plot.  
    - title: Title of the plot.
    - cbar_label: Label for the color bar.
    - filename: Name of the file to save the plot.
    - extent: List defining the physical extent of the plot [Horizontal_min, Horizental_max, Vertical_min, Vertical_max].
    - xlabel: Label for the horizontal axis.
    - ylabel: Label for the vertical axis.
    - norm: Normalization for color scaling (e.g., LogNorm).
    - camp: Colormap to use for the plot.
    - use_scientific_formatter: Whether to use scientific notation for color bar ticks.
    """
    print(f"Plotting'{filename}'... Data Range: min={np.min(data_2d):.2e}, max={np.max(data_2d):.2e}")
    fig, ax = plt.subplots(figsize=(6, 10))

    im = ax.imshow(data_2d, 
                   origin='lower',
                   extent=extent,
                   aspect='auto',
                   cmap=camp,
                   norm=norm)

    formatter = None
    if use_scientific_formatter:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0,0))
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close(fig)
    print("="*40)
    print(f"Plot saved as {filename}")
    print("="*40)
    return fig, ax


def plot_multiview_grid(plots_info: List[Dict],
                        extent_top: List[float],
                        extent_bottom: List[float],
                        filename: str,
                        top_ylabel: str = "z",
                        bottom_ylabel: str = "x",
                        units: str = 'pc'):
    """
    Generates and saves multi-plots.
    Parameters: 
    - plots_info: A list of dictionaries, where each dictionary contains the
                  info for one subplot: {'data', 'title', 'cbar_label', 'norm', 'xlabel'}.
    - extent_top: The physical extent for the top row of plots [Horizontal_min, Horizental_max, Vertical_min, Vertical_max].
    - extent_bottom: The physical extent for the bottom row of plots [Horizontal_min, Horizental_max, Vertical_min, Vertical_max].
    - filename: Name of the file to save the entire figure.
    - top_ylabel: The y-axis label for the top row of plots.
    - bottom_ylabel: The y-axis label for the bottom row of plots.
    - units: The unit string to append to the axis labels.
    """
    N_COLS = len(plots_info)
    fig = plt.figure(figsize=(2* N_COLS + 1.5, 8), layout="constrained")

    top_aspect = (extent_top[3] - extent_top[2]) / (extent_top[1] - extent_top[0])
    bottom_aspect = (extent_bottom[3] - extent_bottom[2]) / (extent_bottom[1] - extent_bottom[0])
    
    gs = gridspec.GridSpec(3, N_COLS, 
                           figure=fig,
                           height_ratios=[top_aspect, bottom_aspect * 0.8, 0.1], 
                           hspace=0.05, wspace=0.3)

    top_axes = [fig.add_subplot(gs[0, i]) for i in range(N_COLS)]
    bottom_axes = [fig.add_subplot(gs[1, i]) for i in range(N_COLS)]

    for i, info in enumerate(plots_info):
        ax_top = top_axes[i]
        ax_bottom = bottom_axes[i]
        norm = info.get('norm')

        im = ax_top.imshow(info['data_top'], origin='lower', extent=extent_top,
                           aspect='auto', cmap='viridis', norm=info['norm'])
        ax_bottom.imshow(info['data_bottom'], origin='lower', extent=extent_bottom, 
                         aspect='auto', cmap=info.get('cmap', 'viridis'), norm=info['norm'])

        ax_cbar = fig.add_subplot(gs[2, i])

        formatter = None
  
        if isinstance(norm, LogNorm):

            formatter = LogFormatterMathtext()
        else:

            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True) # Use scientific notation
            formatter.set_powerlimits((0,1))
            formatter.set_useOffset(False) 
            

        fig.colorbar(im, cax=ax_cbar, orientation='horizontal', format=formatter)
        ax_cbar.set_title(info['label'], fontsize=8, y=0.9) 



    for i, (ax_t, ax_b) in enumerate(zip(top_axes, bottom_axes)):
        ax_t.tick_params(axis='x', labelbottom=False)
        if i == 0:
            ax_t.set_ylabel(top_ylabel + " (" + units + ") ", fontsize=8)
            ax_b.set_ylabel(bottom_ylabel + " ()" + units + ") ", fontsize=8)
        else:
            ax_t.tick_params(axis='y', labelleft=False)
            ax_b.tick_params(axis='y', labelleft=False)

    # plt.tight_layout()
    plt.savefig(filename, dpi=800, bbox_inches='tight', pad_inches=0.5)
    plt.show()



def create_horizontal_subplots(plots_info: List[Dict],
                               shared_extent: List,
                               shared_xlabel: str,
                               shared_ylabel: str,
                               filename: str):
    """
    Generates and saves a single figure with multiple horizontally arranged subplots.

    Parameters:
    - plots_info: A list of dictionaries, where each dictionary contains the
                  info for one subplot: {'data', 'title', 'cbar_label', 'norm'}.
    - shared_extent: The physical extent, shared by all subplots.
    - shared_xlabel: The x-axis label, shared by all subplots.
    - shared_ylabel: The y-axis label, shared by all subplots.
    - filename: Name of the file to save the entire figure.
    """
    num_plots = len(plots_info)
    # Create a subplot figure with 1 row and num_plots columns
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 8))
    if num_plots == 1:
        axes = [axes]  # Ensure axes is always a list for consistency

    for i, ax in enumerate(axes):
        
        info = plots_info[i]
        data_2d = info['data']
        norm = info.get('norm', None)

        print(f"Plotting subplot '{info['title']}'... Data Range: min={np.min(data_2d):.2e}, max={np.max(data_2d):.2e}")
        
        im = ax.imshow(data_2d,
                        origin='lower',
                        extent=shared_extent,
                        aspect='auto',
                        cmap='viridis',
                        norm=norm)
        
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0,0))

        fig.colorbar(im, ax=ax, label=info['cbar_label'], format=formatter, pad=0.02)
        ax.set_title(info['title'])
        ax.set_xlabel(shared_xlabel)
            
    axes[0].set_ylabel(shared_ylabel)
    plt.subplots_adjust(wspace=0.4)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close(fig)
    print("="*40)
    print(f"Subplots saved as {filename}")
    print("="*40)
    return fig, axes

class YTDataProvider:
    def __init__(self, ds):
        self.ds = ds
        self._axis_map = {'x': 0, 'y': 1, 'z': 2, 0:'x', 1:'y', 2:'z'}

    def get_slice(self,
                  field: Tuple[str, str],
                  axis: Union[str, int],
                  coord: Optional[float] = None,
                  resolution: Tuple[int, int] = (800, 800)) -> np.ndarray:
        """
        Get a slice of 2D Numpy array for the specified field and axis.
        """
        axis_str = self._axis_map[axis] if isinstance(axis, int) else axis
        axis_index = self._axis_map[axis_str]

        if coord is None:
            coord = self.ds.domain_center[axis_index]

        full_domain_width = self.ds.domain_width
        plane_axes = [i for i in range(3) if i != axis_index]
        
        slice_width = full_domain_width[plane_axes[0]]
        slice_height = full_domain_width[plane_axes[1]]

        slc = self.ds.slice(axis_str, coord=coord)

        frb = slc.to_frb(width=slice_width, height=slice_height, resolution=resolution)
        
        data_with_units = frb[field]
        numpy_data = np.array(data_with_units)
        unit_string = str(data_with_units.units)



        print("="*40)
        print(f"Slice: field = {field}, axis = {axis_str}, units = {unit_string}")
        print("="*40)
        
        return numpy_data, unit_string
    
    def get_projection(self,
                       field: Tuple[str, str],
                       axis: Union[str, int],
                       weight_field: Optional[Tuple[str, str]] = None,
                       resolution: Tuple[int, int] = (800, 800)) -> np.ndarray:
        """
        Get a projection of 2D Numpy array for the specified field and axis.
        """
        axis_str = self._axis_map[axis] if isinstance(axis, int) else axis
        axis_index = self._axis_map[axis_str]
        
        full_domain_width = self.ds.domain_width
        plane_axes = [i for i in range(3) if i != axis_index]
        
        proj_width = full_domain_width[plane_axes[0]]
        proj_height = full_domain_width[plane_axes[1]]
        
        prj = self.ds.proj(field, axis_str, weight_field=weight_field)

        frb = prj.to_frb(width=proj_width, height=proj_height, resolution=resolution)
        
        data_with_units = frb[field]
        numpy_data = np.array(data_with_units)
        unit_string = str(data_with_units.units)
        print("="*40)
        print(f"Projection: field = {field}, axis = {axis_str}, weight_field = {weight_field}, units = {unit_string}")
        print("="*40)
        return numpy_data, unit_string
    
    def get_plot_extent(self, axis: Union[str, int], units: str = 'pc') -> List[float]:
        """
        Get the physical extent of the plot for the specified axis.
        """
        axis_str = self._axis_map[axis] if isinstance(axis, int) else axis
        axis_index = self._axis_map[axis_str]
        axes = [i for i in range(3) if i != axis_index]

        horizon_min, horizen_max = self.ds.domain_left_edge.in_units(units)[axes[0]].value, self.ds.domain_right_edge.in_units(units)[axes[0]].value
        vertical_min, vertical_max = self.ds.domain_left_edge.in_units(units)[axes[1]].value, self.ds.domain_right_edge.in_units(units)[axes[1]].value
        
        return [horizon_min, horizen_max, vertical_min, vertical_max]
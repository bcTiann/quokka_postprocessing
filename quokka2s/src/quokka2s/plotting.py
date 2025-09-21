import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter 
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter, LogFormatterMathtext, MaxNLocator
import yt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import collections



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
    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(data_2d, 
                   origin='lower',
                   extent=extent,
                   aspect='equal',
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
                        units: str = 'pc',
                        particles_top: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                        particles_bottom: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                        particle_style: Optional[Dict] = None
                        ):
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

    - particles_top: Optional tuple (px, py) of particle coordinates for the top row.
    - particles_bottom: Optional tuple (px, py) of particle coordinates for the bottom row.
    - particle_style: Optional dictionary to style the particles (e.g., {'s': 1, 'c': 'red'}).
    """
    N_COLS = len(plots_info)
    fig = plt.figure(figsize=(2* N_COLS + 1.5, 8), layout="constrained")

    top_aspect = (extent_top[3] - extent_top[2]) / (extent_top[1] - extent_top[0])
    bottom_aspect = (extent_bottom[3] - extent_bottom[2]) / (extent_bottom[1] - extent_bottom[0])
    
    gs = gridspec.GridSpec(3, N_COLS, 
                           figure=fig,
                           height_ratios=[top_aspect, bottom_aspect, 0.1], 
                           hspace=0.05, wspace=0.5)

    top_axes = [fig.add_subplot(gs[0, i]) for i in range(N_COLS)]
    bottom_axes = [fig.add_subplot(gs[1, i]) for i in range(N_COLS)]
    
    # --- Define a default style for the particles ---
    if particle_style is None:
        particle_style = {'s': 1, 'c': 'red', 'alpha': 0.7, 'marker': '.'}

    for i, info in enumerate(plots_info):
        ax_top = top_axes[i]
        ax_bottom = bottom_axes[i]
        norm = info.get('norm')
        cmap = info.get('cmap')


        im = ax_top.imshow(info['data_top'], origin='lower', extent=extent_top,
                           aspect='equal', cmap=cmap, norm=info['norm'])
        
        ax_bottom.imshow(info['data_bottom'], origin='lower', extent=extent_bottom, 
                         aspect='equal', cmap=cmap, norm=info['norm'])

        if particles_top is not None:
            ax_top.scatter(particles_top[0], particles_top[1], **particle_style)
        
        if particles_bottom is not None:
            ax_bottom.scatter(particles_bottom[0], particles_bottom[1], **particle_style)


        quiver_style = {
            'color': 'black',
            'scale': 20000,          # 尝试减小这个值让箭头变长 (例如从 1000 减到 500)
            'scale_units': 'xy',   # 使用坐标轴单位来衡量箭头长度
            'width': 0.005,        # 增加这个值让箭头变粗
            'headwidth': 3,        # 箭头宽度
            'headlength': 5       # 箭头长度
        }
        
        if 'vector_field_top' in info and info['vector_field_top'] is not None:
            X, Y, U, V = info['vector_field_top']
            ax_top.quiver(X, Y, U, V, **quiver_style)
            
        if 'vector_field_bottom' in info and info['vector_field_bottom'] is not None:
            X, Y, U, V = info['vector_field_bottom']
            ax_bottom.quiver(X, Y, U, V, **quiver_style)

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



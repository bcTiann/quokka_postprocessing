# main_analysis.py

import yt
import numpy as np
import os
from yt.units import kpc
# --- Import our custom modules ---
import config as cfg
import quokka2s as q2s
import physics_models as phys
from matplotlib import pyplot as plt

PROJECTION_AXIS = 'x' 

ds = yt.load(cfg.YT_DATASET_PATH)


phys.add_all_fields(ds) # Add derived fields
provider = q2s.YTDataProvider(ds)


slice_axis = 'x'


axis_map = {'x': 0, 'y': 1, 'z': 2}
proj_axis_idx = axis_map[cfg.PROJECTION_AXIS]


print(f"domain_width = {ds.domain_width}")


slab, extent_slab = provider.get_slab_z(
    field=('gas', 'density')
    )

cube, extent_cube = provider.get_cubic_box(
    field=('gas', 'density')
    )

cube_shape = cube.shape






projection_cube = cube.sum(axis=0)
projection_slab = slab.sum(axis=0)


############################# cube #########################################
q2s.create_plot(
    data_2d=projection_cube.T.to_ndarray(), # Use the calculated ratio map
    title="Dust Transmission (XY Projection)",
    cbar_label="Fraction of Light Transmitted",
    filename=os.path.join(cfg.OUTPUT_DIR, "halpha_dust_ratio.png"),
    extent=extent_cube['x'],
    xlabel=f"{'XYZ'[proj_axis_idx-2]} ({cfg.FIGURE_UNITS})",
    ylabel=f"{'XYZ'[proj_axis_idx-1]} ({cfg.FIGURE_UNITS})",
    norm=None,  # Use a LINEAR scale for the ratio map, not LogNorm!
    camp='viridis_r' # Use a reversed colormap so dense areas are dark
)



q2s.create_plot(
    data_2d=cube[6, :, :].T.to_ndarray(), # Use the calculated ratio map
    title="Dust Transmission (With Dust / Without Dust)",
    cbar_label="Fraction of Light Transmitted",
    filename=os.path.join(cfg.OUTPUT_DIR, "halpha_dust_ratio.png"),
    extent=extent_cube['x'],
    xlabel=f"{'XYZ'[proj_axis_idx-2]} ({cfg.FIGURE_UNITS})",
    ylabel=f"{'XYZ'[proj_axis_idx-1]} ({cfg.FIGURE_UNITS})",
    norm=None,  # Use a LINEAR scale for the ratio map, not LogNorm!
    camp='viridis_r' # Use a reversed colormap so dense areas are dark
)

################################ slab ######################################
q2s.create_plot(
    data_2d=projection_slab.T.to_ndarray(), # Use the calculated ratio map
    title="Dust Transmission (XY Projection)",
    cbar_label="Fraction of Light Transmitted",
    filename=os.path.join(cfg.OUTPUT_DIR, "halpha_dust_ratio.png"),
    extent=extent_slab['x'],
    xlabel=f"{'XYZ'[proj_axis_idx-2]} ({cfg.FIGURE_UNITS})",
    ylabel=f"{'XYZ'[proj_axis_idx-1]} ({cfg.FIGURE_UNITS})",
    norm=None,  # Use a LINEAR scale for the ratio map, not LogNorm!
    camp='viridis_r' # Use a reversed colormap so dense areas are dark
)



q2s.create_plot(
    data_2d=slab[6, :, :].T.to_ndarray(), # Use the calculated ratio map
    title="Dust Transmission (With Dust / Without Dust)",
    cbar_label="Fraction of Light Transmitted",
    filename=os.path.join(cfg.OUTPUT_DIR, "halpha_dust_ratio.png"),
    extent=extent_slab['x'],
    xlabel=f"{'XYZ'[proj_axis_idx-2]} ({cfg.FIGURE_UNITS})",
    ylabel=f"{'XYZ'[proj_axis_idx-1]} ({cfg.FIGURE_UNITS})",
    norm=None,  # Use a LINEAR scale for the ratio map, not LogNorm!
    camp='viridis_r' # Use a reversed colormap so dense areas are dark
)

######################################################################


# projection_pixel = cube_pixel.sum(axis=0)


# q2s.create_plot(
#     data_2d=projection_pixel.T.to_ndarray(), # Use the calculated ratio map
#     title="Dust Transmission (XY Projection)",
#     cbar_label="Fraction of Light Transmitted",
#     filename=os.path.join(cfg.OUTPUT_DIR, "halpha_dust_ratio.png"),
#     extent=extent['x'],
#     xlabel=f"{'XYZ'[proj_axis_idx-2]} ({cfg.FIGURE_UNITS})",
#     ylabel=f"{'XYZ'[proj_axis_idx-1]} ({cfg.FIGURE_UNITS})",
#     norm=None,  # Use a LINEAR scale for the ratio map, not LogNorm!
#     camp='viridis_r' # Use a reversed colormap so dense areas are dark
# )


print(cube.shape) 
print(cube.units)
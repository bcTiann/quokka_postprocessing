# from plot_interface import YTDataProvider, create_plot, create_horizontal_subplots, plot_multiview_grid
import yt
from yt.units import mp, kb, mh 
from matplotlib.colors import LogNorm
import numpy as np
from quokka2s import *

# 1. define field: temperature
def _temperature(field, data):
    mu = 0.6
    # gas_internal_energy_density = data[('boxlib', 'gasInternalEnergy')]
    gas_internal_energy_density = data[('gas', 'internal_energy_density')]
    # gas_density = data[('boxlib', 'gasDensity')]
    gas_density = data[('gas', 'density')]
    temp = 2/3 * gas_internal_energy_density * mu * mh / gas_density / kb
    return temp

# Load your simulation data
ds = yt.load("plt01000") # Replace with your dataset file



# Instantiate your data provider
provider = YTDataProvider(ds)


# Add derived fields to the dataset
ds.add_field(name=('gas', 'temperature'),
             function=_temperature,
             sampling_type="cell",  
             units="K")             


# 3. initialize data provider
provider = YTDataProvider(ds)
# specify the desired unit for the plot extent
units = 'pc'
extent_x = provider.get_plot_extent(axis='x', units=units)
extent_z = provider.get_plot_extent(axis='z', units=units)

print("\nFetching particle data...")
# Calculate the depth for particle selection (e.g., xx % of the domain width)
depth_x_pc = ds.domain_width[provider._axis_map['x']].to(units).value / 20
depth_z_pc = ds.domain_width[provider._axis_map['z']].to(units).value / 20

# Get particles for the top row (x-slice)
px_top, py_top = provider.get_particle_positions(axis='x', depth=depth_x_pc, units=units)

# Get particles for the bottom row (z-slice)
px_bottom, py_bottom = provider.get_particle_positions(axis='z', depth=depth_z_pc, units=units)
print("...Particle data fetched.\n")


print("\nFetching vector field data...")
vec_field_top = provider.get_velocity_field(axis='x', downsample_factor=30, units=units)
vec_field_bottom = provider.get_velocity_field(axis='z', downsample_factor=30, units=units)
print("...Vector field data fetched.\n")

# Plot 1. Density along x-axis
density_x = provider.get_slice(field=('gas', 'density'), axis='x')
density_z = provider.get_slice(field=('gas', 'density'), axis='z')

# Plot 2. column density along x-axis
column_density_x = provider.get_projection(field=('gas', 'density'), axis='x')
column_density_z = provider.get_projection(field=('gas', 'density'), axis='z')

# Plot 3. Density-weighted V_z along x-axis
weighted_vz_x = provider.get_projection(field=('gas', 'four_velocity_z'), 
                                                             weight_field=('gas', 'density'), 
                                                             axis='x')
weighted_vz_z = provider.get_projection(field=('gas', 'four_velocity_z'), 
                                                             weight_field=('gas', 'density'), 
                                                             axis='z')

# Plot 4. Density-weighted Temperature slice along x-axis
weighted_T_x = provider.get_projection(field=('gas', 'temperature'), 
                                     weight_field=('gas', 'density'),
                                     axis='x')
weighted_T_z = provider.get_projection(field=('gas', 'temperature'), 
                                     weight_field=('gas', 'density'),
                                     axis='z')  


plots_info = [
    {
        'data_top': density_x.to_ndarray(),
        'data_bottom': density_z.to_ndarray(),
        'label': f'Density ({density_x.units})',
        'norm': LogNorm(),
        'cmap': 'viridis',
        # 'vector_field_top': vec_field_top,
        # 'vector_field_bottom': vec_field_bottom,
    },
    {
        'data_top': column_density_x.to_ndarray(),
        'data_bottom': column_density_z.to_ndarray(),
        'label': f'Column Density ({column_density_x.units})',
        'norm': LogNorm(),
        'cmap': 'viridis',
    },
    {
        'data_top': weighted_vz_x.to_ndarray(),
        'data_bottom': weighted_vz_z.to_ndarray(),
        'label': f'Density Weighted $v_z$ ({weighted_vz_x.units})',
        'norm': None,
        'cmap': 'seismic',
    },
    {
        'data_top': weighted_T_x.to_ndarray(),
        'data_bottom': weighted_T_z.to_ndarray(),
        'label': f'Density Weighted Tempeature ({weighted_T_x.units})',
        'norm': LogNorm(),
        'cmap': 'viridis',
    }
]



plot_multiview_grid(
    plots_info=plots_info,
    extent_top=extent_x,
    extent_bottom=extent_z,
    filename="multiview_figure_particles.png",
    units=units,
    particles_top=(px_top.to_ndarray(), py_top.to_ndarray()),
    particles_bottom=(px_bottom.to_ndarray(), py_bottom.to_ndarray())
)


# create_horizontal_subplots(
#     plots_info=plots_info,              
#     filename="combined_plots_x.png",
#     shared_extent=extent_x,
#     shared_xlabel=f"Y ({units})",
#     shared_ylabel=f"Z ({units})"
#     )

# create_plot(data_2d=density_x,
#             title="Density Slice along x-axis",
#             cbar_label=r'Density (density_units)',
#             filename="density_x.png",
#             extent=extent_x,
#             xlabel=f"Y ({units})",
#             ylabel=f"Z ({units})",
#             norm=LogNorm())

# create_plot(data_2d=column_density_x,
#             title="Column Density along x-axis",
#             cbar_label=r'Column Density (column_dens_units)',
#             filename="column_density_x.png",
#             extent=extent_x,
#             xlabel=f"Y ({units})",
#             ylabel=f"Z ({units})",
#             norm=LogNorm()) 

# create_plot(data_2d=weighted_vz_x,
#             title="Density-weighted vz along x-axis",
#             cbar_label=r'Velocity (weighted_vz_x_units)',
#             filename="weighted_vz_x.png",
#             extent=extent_x,
#             xlabel=f"Y ({units})",
#             ylabel=f"Z ({units})",
#             # norm=LogNorm()
#             ) 

# create_plot(data_2d=weighted_T_x,
#             title="Density-weighted Temperature along x-axis",
#             cbar_label=r'Temperature (weighted_T_x_units)',
#             filename="weighted_T_x.png",
#             extent=extent_x,
#             xlabel=f"Y ({units})",
#             ylabel=f"Z ({units})",
#             norm=LogNorm())

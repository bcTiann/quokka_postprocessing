from plot_interface import YTDataProvider, create_plot, create_horizontal_subplots, plot_multiview_grid
import yt
from yt.units import mp, kb, mh 
from matplotlib.colors import LogNorm

# 1. define field: temperature
def _temperature(field, data):
    mu = 0.6
    # gas_internal_energy_density = data[('boxlib', 'gasInternalEnergy')]
    gas_internal_energy_density = data[('gas', 'internal_energy_density')]
    # gas_density = data[('boxlib', 'gasDensity')]
    gas_density = data[('gas', 'density')]
    temp = 2/3 * gas_internal_energy_density * mu * mh / gas_density / kb
    return temp

# 2. Load the dataset
ds = yt.load("plt00500")  
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

# Plot 1. Density along x-axis
density_x, density_x_units = provider.get_slice(field=('gas', 'density'), axis='x')
density_z, density_z_units = provider.get_slice(field=('gas', 'density'), axis='z')

# Plot 2. column density along x-axis
column_density_x, column_density_x_units = provider.get_projection(field=('gas', 'density'), axis='x')
column_density_z, column_density_z_units = provider.get_projection(field=('gas', 'density'), axis='z')

# Plot 3. Density-weighted V_z along x-axis
weighted_vz_x, weighted_vz_x_units = provider.get_projection(field=('gas', 'four_velocity_z'), 
                                                             weight_field=('gas', 'density'), 
                                                             axis='x')
weighted_vz_z, weighted_vz_z_units = provider.get_projection(field=('gas', 'four_velocity_z'), 
                                                             weight_field=('gas', 'density'), 
                                                             axis='z')

# Plot 4. Density-weighted Temperature slice along x-axis
weighted_T_x, weighted_T_x_units = provider.get_projection(field=('gas', 'temperature'), 
                                     weight_field=('gas', 'density'),
                                     axis='x')
weighted_T_z, weighted_T_z_units = provider.get_projection(field=('gas', 'temperature'), 
                                     weight_field=('gas', 'density'),
                                     axis='z')  


plots_info = [
    {
        'data_top': density_x,
        'data_bottom': density_z,
        'label': f'density ({density_x_units})',
        'norm': LogNorm(),
    },
    {
        'data_top': column_density_x,
        'data_bottom': column_density_z,
        'label': f'column density ({column_density_x_units})',
        'norm': LogNorm(),
    },
    {
        'data_top': weighted_vz_x,
        'data_bottom': weighted_vz_z,
        'label': f'weighted vz ({weighted_vz_x_units})',
        'norm': None,
    },
    {
        'data_top': weighted_T_x,
        'data_bottom': weighted_T_z,
        'label': f'weighted_T_x ({weighted_T_x_units})',
        'norm': LogNorm(),
    }
]

plot_multiview_grid(
    plots_info=plots_info,
    extent_top=extent_x,
    extent_bottom=extent_z,
    filename="elegant_multiview_figure.png"
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

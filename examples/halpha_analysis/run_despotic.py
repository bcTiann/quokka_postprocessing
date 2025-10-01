import yt
import numpy as np
import os
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg
# --- Import our custom modules ---
import config as cfg
import quokka2s as q2s
import physics_models as phys
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from matplotlib.ticker import LogLocator, LogFormatter
from quokka2s.despotic_tables import (
    DespoticTable,
    compute_average,
)


def load_table(path: str) -> DespoticTable:
    data = np.load(path)
    return DespoticTable(
        co_int_tb=data["co_int_tb"],
        tg_final=data["tg_final"],
        nH_values=data["nH"],
        col_density_values=data["col_density"],
    )
axis_map = {'x': 0, 'y': 1, 'z': 2}
proj_axis_idx = axis_map[cfg.PROJECTION_AXIS]
factor = 1
mid_z = 256//factor//2
mid_x = 128//factor//2
ds = yt.load(cfg.YT_DATASET_PATH)
phys.add_all_fields(ds)
provider = q2s.YTDataProvider(ds)



###################### grid size ##############################
dx_3d, dx_3d_extent = provider.get_cubic_box(('boxlib', 'dx'))
dx_projection = dx_3d.sum(axis=0)

dy_3d, dy_3d_extent = provider.get_cubic_box(('boxlib', 'dy'))
dy_projection = dy_3d.sum(axis=0)

dz_3d, dz_3d_extent = provider.get_cubic_box(('boxlib', 'dz'))
dz_projection = dz_3d.sum(axis=0)
################################################################

###################### number density ##############################


print("####################################")
print(f"dx_3d.mean : {dx_3d.mean()}")
print(f"dx_3d.var : {dx_3d.var()}")

print("...3D data retrieval complete.")
print("####################################")
print("####################################")
print(f"dy_3d.mean : {dy_3d.mean()}")
print(f"dy_3d.var : {dy_3d.var()}")

print("...3D data retrieval complete.")
print("####################################")
print("####################################")
print(f"dz_3d.mean : {dz_3d.mean()}")
print(f"dz_3d.var : {dz_3d.var()}")

print("...3D data retrieval complete.")
print("####################################")


X_H = cfg.X_H
m_H = mh.in_cgs()
density_3d, density_3d_extent = provider.get_cubic_box(
    field=('gas', 'density')
)
density_3d = provider.downsample_3d_array(density_3d, factor=factor)

n_H_3d = (density_3d * X_H) / m_H




print("\n--- Downsampling by factor of 2 ---")


print(f"Downsampled data type: {type(n_H_3d)}")
print(f"Downsampled data units: {n_H_3d.units}")
print(f"Downsampled data shape: {n_H_3d.shape}")

dx_3d, dx_3d_extent = provider.get_cubic_box(
    field=('boxlib', 'dx')
)
dy_3d, dy_3d_extent = provider.get_cubic_box(
    field=('boxlib', 'dy')
)
dz_3d, dz_3d_extent = provider.get_cubic_box(
    field=('boxlib', 'dz')
)


dx_3d = provider.downsample_3d_array(dx_3d, factor=factor)
dy_3d = provider.downsample_3d_array(dy_3d, factor=factor)
dz_3d = provider.downsample_3d_array(dz_3d, factor=factor)

temp_3d, temp_3d_extent = provider.get_cubic_box(
    field=('gas', 'temperature')
)
temp_3d = provider.downsample_3d_array(temp_3d, factor=factor)
print(f"temp_3d[:, :, 8] = {temp_3d.in_cgs()[:, :, mid_z]}")

Nx_p = q2s.along_sight_cumulation(n_H_3d, axis="x", sign="+") * dx_3d
Ny_p = q2s.along_sight_cumulation(n_H_3d, axis="y", sign="+") * dy_3d
Nz_p = q2s.along_sight_cumulation(n_H_3d, axis="z", sign="+") * dz_3d

Nx_n = q2s.along_sight_cumulation(n_H_3d, axis="x", sign="-") * dx_3d
Ny_n = q2s.along_sight_cumulation(n_H_3d, axis="y", sign="-") * dy_3d
Nz_n = q2s.along_sight_cumulation(n_H_3d, axis="z", sign="-") * dz_3d


print(f"Nx_p.shape = {Nx_p.shape}")
print(f"Nx_p.units = {Nx_p.units}")


average_N_3d = compute_average(
    [Nx_p, Ny_p, Nz_p, Nx_n, Ny_n, Nz_n],
    method="mean",
)


print(f"average_N_3d.shape = {average_N_3d.shape}")
print(f"average_N_3d.units = {average_N_3d.units}")


# fig, ax = plt.subplots(figsize=(8, 6))
# image = ax.imshow(average_N_3d[mid_x, :, :].T,
#                   extent=dx_3d_extent['x'],
#                   origin='lower',
#                   cmap='viridis',
#                 #   norm=LogNorm())
# )

# cbar = fig.colorbar(image, ax=ax)
# cbar.set_label(f"K km/s")
# plt.savefig('average_N_3d_Slice', dpi=600, bbox_inches='tight')
# print("figure saved as 'average_N_3d_Slice'")
# plt.show()

# # =======================================================
# from distribution import analyze_and_plot_distribution
# analyze_and_plot_distribution(n_H_3d, average_N_3d, output_prefix="my_simulation")
# # =======================================================


X_H = cfg.X_H
m_H = mh.in_cgs()
density_3d, density_3d_extent = provider.get_cubic_box(
    field=('gas', 'density')
)
density_3d = provider.downsample_3d_array(density_3d, factor=factor)

n_H_3d = (density_3d * X_H) / m_H


fig, ax = plt.subplots(figsize=(8, 6))
image = ax.imshow(n_H_3d[mid_x, :, :].T,
                  extent=dx_3d_extent['x'],
                  origin='lower',
                  cmap='viridis',
                  norm=LogNorm()
)

cbar = fig.colorbar(image, ax=ax)
cbar.set_label(f"K km/s")
plt.savefig('n_H_3d_Slice', dpi=600, bbox_inches='tight')
print("figure saved as 'n_H_3d_Slice'")
plt.show()

######################### Load table #######################
table = load_table("fine_table.npz")
log_nH_grid = np.log10(table.nH_values)
log_col_grid = np.log10(table.col_density_values)
co_interpolator = RectBivariateSpline(log_nH_grid, log_col_grid, table.co_int_tb)
################################################################

nH_cgs = n_H_3d.in_cgs().to_ndarray()
colDen_cgs = average_N_3d.in_cgs().to_ndarray()

log_nH_vals = np.log10(nH_cgs)
log_col_vals = np.log10(colDen_cgs)


log_nH_vals = np.clip(log_nH_vals, log_nH_grid[0], log_nH_grid[-1])
log_col_vals = np.clip(log_col_vals, log_col_grid[0], log_col_grid[-1])

co_int_3d = co_interpolator.ev(log_nH_vals.ravel(), log_col_vals.ravel())
co_int_3d = co_int_3d.reshape(nH_cgs.shape)

co_slice = co_int_3d[mid_x, :, : ]

fig, ax = plt.subplots(figsize=(8, 6))
image = ax.imshow(co_slice.T,
                  extent=dx_3d_extent['x'],
                  origin='lower',
                  cmap='viridis',
                  norm=LogNorm())

cbar = fig.colorbar(image, ax=ax)
cbar.set_label(f"K km/s")
plt.savefig('CO_int_Slice', dpi=600, bbox_inches='tight')
print("figure saved as 'CO_int_Slice_Table'")
plt.show()

# fig, ax = plt.subplots(figsize=(8, 6))
# image = ax.imshow(temp_3d.in_cgs().to_ndarray()[:, :, mid_z],
#                   extent=dx_3d_extent['x'],
#                   origin='lower',
#                   cmap='viridis',
#                   norm=LogNorm())

# cbar = fig.colorbar(image, ax=ax)
# cbar.set_label(f"Temp ({temp_3d.units})")
# plt.savefig('Initial_temperature', dpi=600, bbox_inches='tight')
# print("figure saved as 'Initial_temperature'")
# plt.show()



co_map_K_kms, Tg_map = q2s.run_despotic_on_map(
                                    nH_map=n_H_3d.in_cgs().to_ndarray()[:, :, mid_z], 
                                    colDen_map=average_N_3d.in_cgs().to_ndarray()[:, :, mid_z],
                                    Tg_map=temp_3d.in_cgs().to_ndarray()[:, :, mid_z]
)

co_map_masked = np.ma.masked_where(np.isnan(co_map_K_kms), co_map_K_kms)


# fig, ax = plt.subplots(figsize=(8, 6))
# image = ax.imshow(Tg_map,
#                   extent=dx_3d_extent['x'],
#                   origin='lower',
#                   cmap='viridis',
#                   norm=LogNorm())
# cbar = fig.colorbar(image, ax=ax)
# cbar.set_label(f"Temp ({temp_3d.units})")
# plt.savefig('Final_temperature', dpi=600, bbox_inches='tight')
# print("figure saved as 'Final_temperature'")
# plt.show()


params = cfg.ANALYSES["co_despotic"]

q2s.create_plot(
    data_2d=co_map_masked.T, # .T to transpose for correct orientation
    title=params['title'],
    cbar_label=params['cbar_label'],
    filename='CO_map_Despotic_oneByeone',
    extent=dx_3d_extent['z'],
    xlabel='y',
    ylabel='z',
    norm=params['norm'],
    camp='viridis' 
)

# q2s.create_plot(
#     data_2d=Tg_map.T,
#     title="Tg map",
#     cbar_label="Tg (K)",
#     filename="Tg_map",
#     extent=dx_3d_extent['z'],
#     xlabel="y",
#     ylabel="z",
#     norm=params['norm'],
#     camp='viridis'
# )

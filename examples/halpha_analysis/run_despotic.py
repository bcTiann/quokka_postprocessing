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
from quokka2s.despotic_tables import DespoticTable, SpeciesLineGrid, compute_average


def load_table(path: str) -> DespoticTable:
    data = np.load(path, allow_pickle=True)
    if "species" in data and any(key.endswith("_grid") for key in data.keys()):
        species_names = [str(name) for name in data["species"]]
        emitter_arr = data.get("emitter_abundances")
        emitter_abundances = {
            species: float(emitter_arr[idx]) if emitter_arr is not None else float("nan")
            for idx, species in enumerate(species_names)
        }
        line_grids: dict[str, SpeciesLineGrid] = {}
        for idx, species in enumerate(species_names):
            line_grids[species] = SpeciesLineGrid(
                int_tb=data["int_tb_grid"][idx],
                int_intensity=data["int_intensity_grid"][idx],
                lum_per_h=data["lum_per_h_grid"][idx],
                tau=data["tau_grid"][idx],
                tau_dust=data["tau_dust_grid"][idx],
                tex=data["tex_grid"][idx],
                freq=data["freq_grid"][idx],
            )
        return DespoticTable(
            species_data=line_grids,
            tg_final=data["tg_final"],
            nH_values=data["nH"],
            col_density_values=data["col_density"],
            emitter_abundances=emitter_abundances,
        )

    shape = data["co_int_tb"].shape

    def get_array(key: str) -> np.ndarray:
        if key in data:
            return data[key]
        return np.full(shape, np.nan)

    legacy_grid = SpeciesLineGrid(
        int_tb=data["co_int_tb"],
        int_intensity=get_array("int_intensity"),
        lum_per_h=get_array("lum_per_h"),
        tau=get_array("tau"),
        tau_dust=get_array("tau_dust"),
        tex=get_array("tex"),
        freq=get_array("frequency"),
    )

    return DespoticTable(
        species_data={"CO": legacy_grid},
        tg_final=data["tg_final"],
        nH_values=data["nH"],
        col_density_values=data["col_density"],
        emitter_abundances={"CO": float("nan")},
    )

axis_map = {'x': 0, 'y': 1, 'z': 2}
proj_axis_idx = axis_map[cfg.PROJECTION_AXIS]

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

dv_3d = dx_3d * dy_3d * dz_3d
################################################################

factor = 1
nx, ny, nz = dy_3d.shape
mid_z = nz//factor//2
mid_x = nx//factor//2

# ================= DEBUGGING BLOCK =================
print("\n--- DEBUGGING CELL VOLUME ---")
# 打印一个网格的平均体积，使用它自己的原始单位
print(f"Mean cell volume in original units: {dv_3d.mean()}")

# 打印它在CGS单位制 (cm^3) 下的数值
print(f"Mean cell volume in CGS (cm^3): {dv_3d.in_cgs().mean().v}") 
print("-----------------------------\n")
# ===================================================

###################### number density ##############################



print("############ dx #################")
print(f"dx_3d.mean : {dx_3d.mean()}")
print(f"dx_3d.var : {dx_3d.var()}")
print(f"dx units: {dx_3d.units}")
print("####################################")

print("############# dy #####################")
print(f"dy_3d.mean : {dy_3d.mean()}")
print(f"dy_3d.var : {dy_3d.var()}")
print(f"dy units: {dy_3d.units}")
print("####################################")

print("############## dz ####################")
print(f"dz_3d.mean : {dz_3d.mean()}")
print(f"dz_3d.var : {dz_3d.var()}")
print(f"dz units: {dz_3d.units}")
print("####################################")

print("############## dv ####################")
print(f"dv_3d.mean : {dv_3d.mean()}")
print(f"dv_3d.var : {dv_3d.var()}")
print(f"dv units: {dv_3d.units}")
print("####################################")

print("...3D data retrieval complete.")
print("####################################")




X_H = cfg.X_H
m_H = mh.in_cgs()
density_3d, density_3d_extent = provider.get_cubic_box(
    field=('gas', 'density')
)

density_3d = provider.downsample_3d_array(density_3d, factor=factor)
##################################
n_H_3d = (density_3d * X_H) / m_H
# units cm**(-3) its a volumn density! 
##################################
print("####################################")
print(f"n_H_3d.mean : {n_H_3d.mean()}")
print(f"n_H_3d.var : {n_H_3d.var()}")
print(f"n_H_3d.units = {n_H_3d.units}")

print("####################################")

# print("\n--- Downsampling by factor of 2 ---")


# print(f"Downsampled data type: {type(n_H_3d)}")
# print(f"Downsampled data units: {n_H_3d.units}")
# print(f"Downsampled data shape: {n_H_3d.shape}")

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
    method="harmonic",
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


nH_slice = n_H_3d.in_cgs().to_ndarray()[mid_x, :, :]        # cm^-3
colDen_slice = average_N_3d.in_cgs().to_ndarray()[mid_x, :, :]  # cm^-2

# n_H (cm^-3)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    nH_slice.T,
    extent=density_3d_extent['x'],
    origin='lower',
    cmap='viridis',
    norm=LogNorm()
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(f"{n_H_3d.in_cgs().units}")
ax.set_xlabel("y (cm)")
ax.set_ylabel("z (cm)")
plt.savefig("plots/nH_slice_cgs.png", dpi=600, bbox_inches="tight")
plt.close(fig)

# 柱密度 N_H (cm^-2)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    colDen_slice.T,
    extent=density_3d_extent['x'],
    origin='lower',
    cmap='viridis',
    norm=LogNorm()
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(f"{average_N_3d.in_cgs().units}")
ax.set_xlabel("y (cm)")
ax.set_ylabel("z (cm)")
plt.savefig("plots/colDen_slice_cgs.png", dpi=600, bbox_inches="tight")




######################### Load table #######################
table = load_table("output_tables_2species/table_100x100_fixed.npz")
target_species = "CO" 
species_grid = table.get_species_grid(target_species)

log_nH_grid = np.log10(table.nH_values)
log_col_grid = np.log10(table.col_density_values)
log_lumPerH = np.log10(species_grid.lum_per_h)
log_lumPerH_interpolator = RectBivariateSpline(log_nH_grid, log_col_grid, log_lumPerH)
################################################################

nH_cgs = n_H_3d.in_cgs().to_ndarray()
colDen_cgs = average_N_3d.in_cgs().to_ndarray()


valid_nH = (nH_cgs >= table.nH_values.min()) & (nH_cgs <= table.nH_values.max())
valid_col = (colDen_cgs >= species_grid.col_density_values.min()) & (colDen_cgs <= species_grid.col_density_values.max())
valid_mask = valid_nH & valid_col

clipped_nH = np.clip(nH_cgs, table.nH_values.min(), table.nH_values.max())
clipped_col = np.clip(colDen_cgs, species_grid.col_density_values.min(), species_grid.col_density_values.max())
log_nH_vals = np.log10(clipped_nH)
log_col_vals = np.log10(clipped_col)



# log_nH_vals = np.log10(nH_cgs)
# log_col_vals = np.log10(colDen_cgs)


# ############
# # Restric each log(nH) in the tables avaliable values
# log_nH_vals = np.clip(log_nH_vals, log_nH_grid[0], log_nH_grid[-1])

# # Restric each log(colDen) in the tables avaliable values
# log_col_vals = np.clip(log_col_vals, log_col_grid[0], log_col_grid[-1])
# # so that the interplation won't use outside table value which is not stable
# #############


# ravel --> flatten the 3D array to 1D
log_lumPerH_3d = log_lumPerH_interpolator.ev(log_nH_vals.ravel(), log_col_vals.ravel())
log_lumPerH_3d = log_lumPerH_3d.reshape(nH_cgs.shape)

lumPerH_3d = np.power(10.0, log_lumPerH_3d) # erg s^-1 H^-1
lumPerH_3d[~valid_mask] = np.nan


# Calculate lum from lumPerH_3d * n_H_3d * dv_3d
lum = lumPerH_3d * dv_3d.in_cgs().to_ndarray() * n_H_3d.in_cgs().to_ndarray() # (erg s^-1 H^-1) * cm^3 * (H * cm^-3) 
# Set invalid grid to np.nan
lum[~valid_mask] = np.nan


# convert to YTArray
lum_per_H_yt = yt.YTArray(lumPerH_3d, "erg/s")
lum = yt.YTArray(lum, "erg/s")

##################### Slice Plot ##################### 
# lum_slice = lum[mid_x, :, : ]
lum_slice = np.ma.masked_where(~valid_mask[mid_x, :, :], lum[mid_x, :, :])

fig, ax = plt.subplots(figsize=(8, 6))
cmap = plt.cm.get_cmap("viridis").copy()
cmap.set_bad("lightgray", alpha=0.7)
image = ax.imshow(lum_slice.T,
                  extent=dx_3d_extent['x'],
                  origin='lower',
                  cmap=cmap,
                  norm=LogNorm())

cbar = fig.colorbar(image, ax=ax)
cbar.set_label(f"Luminosity ({lum.units})")
plt.savefig('plots/luminosity_CO_xSlice.png', dpi=600, bbox_inches='tight')
print("figure saved as 'CO_luminosity_xSlice'")
plt.show()

################################################# 


##################### Projection Plot ##################### 
lum_x_projection = lum.sum(axis=0)
lum_z_projection = lum.sum(axis=2)


fig, ax = plt.subplots(figsize=(8, 6))
image = ax.imshow(lum_x_projection.T,
                  extent=dx_3d_extent['x'],
                  origin='lower',
                  cmap='viridis',
                  norm=LogNorm())

cbar = fig.colorbar(image, ax=ax)
cbar.set_label(f"Luminosity ({lum.units})")
plt.savefig('plots/luminosity_CO_xProjection.png', dpi=600, bbox_inches='tight')
print("figure saved as 'CO_luminosity_xProjection'")
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
image = ax.imshow(lum_z_projection.T,
                  extent=dx_3d_extent['x'],
                  origin='lower',
                  cmap='viridis',
                  norm=LogNorm())

cbar = fig.colorbar(image, ax=ax)
cbar.set_label(f"luminosity")
plt.savefig('plots/luminosity_CO_zProjection.png', dpi=600, bbox_inches='tight')
print("figure saved as 'CO_luminosity_zProjection'")
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



# co_map_K_kms, Tg_map = q2s.run_despotic_on_map(
#                                     nH_map=n_H_3d.in_cgs().to_ndarray()[:, :, mid_z], 
#                                     colDen_map=average_N_3d.in_cgs().to_ndarray()[:, :, mid_z],
#                                     Tg_map=temp_3d.in_cgs().to_ndarray()[:, :, mid_z]
# )

# co_map_masked = np.ma.masked_where(np.isnan(co_map_K_kms), co_map_K_kms)


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


# params = cfg.ANALYSES["co_despotic"]

# q2s.create_plot(
#     data_2d=co_map_masked.T, # .T to transpose for correct orientation
#     title=params['title'],
#     cbar_label=params['cbar_label'],
#     filename='CO_map_Despotic_oneByeone',
#     extent=dx_3d_extent['z'],
#     xlabel='y',
#     ylabel='z',
#     norm=params['norm'],
#     camp='viridis' 
# )

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

import yt
import numpy as np
import os
from pathlib import Path
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg
# --- Import our custom modules ---
import config as cfg
import quokka2s as q2s
import physics_models as phys
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter
from quokka2s.despotic_tables import DespoticTable, SpeciesLineGrid, compute_average
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import analysis_plots as plots
from build_despotic_table import plot_table

USE_INTERPOLATED_TABLE = True  # False: original table; True: interplated_table
USE_BOUNDARY_CLAMP = True  # True: clip the simulation of nH, colDen to its boundary at avaibale table
def _unit_to_latex(unit) -> str:
    """Return a LaTeX-ready string for a yt Unit-like object."""
    latex = getattr(unit, "latex_representation", None)
    if latex is None:
        return str(unit)
    if callable(latex):
        return latex()
    return str(latex)

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
dx_3d, dx_3d_extent = provider.get_slab_z(('boxlib', 'dx'))
dx_projection = dx_3d.sum(axis=0)

dy_3d, dy_3d_extent = provider.get_slab_z(('boxlib', 'dy'))
dy_projection = dy_3d.sum(axis=0)

dz_3d, dz_3d_extent = provider.get_slab_z(('boxlib', 'dz'))
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
density_3d, density_3d_extent = provider.get_slab_z(
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

dx_3d, dx_3d_extent = provider.get_slab_z(
    field=('boxlib', 'dx')
)
dy_3d, dy_3d_extent = provider.get_slab_z(
    field=('boxlib', 'dy')
)
dz_3d, dz_3d_extent = provider.get_slab_z(
    field=('boxlib', 'dz')
)


dx_3d = provider.downsample_3d_array(dx_3d, factor=factor)
dy_3d = provider.downsample_3d_array(dy_3d, factor=factor)
dz_3d = provider.downsample_3d_array(dz_3d, factor=factor)

temp_3d, temp_3d_extent = provider.get_slab_z(
    field=('gas', 'temperature')
)
temp_3d = provider.downsample_3d_array(temp_3d, factor=factor)
print(f"temp_3d[:, :, 8] = {temp_3d.in_cgs()[:, :, mid_z]}")

Nx_p = q2s.along_sight_cumulation(n_H_3d * dx_3d, axis="x", sign="+") 
Ny_p = q2s.along_sight_cumulation(n_H_3d * dy_3d, axis="y", sign="+") 
Nz_p = q2s.along_sight_cumulation(n_H_3d * dz_3d, axis="z", sign="+")

Nx_n = q2s.along_sight_cumulation(n_H_3d * dx_3d, axis="x", sign="-")
Ny_n = q2s.along_sight_cumulation(n_H_3d * dy_3d, axis="y", sign="-")
Nz_n = q2s.along_sight_cumulation(n_H_3d * dz_3d, axis="z", sign="-")


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
density_3d, density_3d_extent = provider.get_slab_z(
    field=('gas', 'density')
)
density_3d = provider.downsample_3d_array(density_3d, factor=factor)

n_H_3d = (density_3d * X_H) / m_H


nH_slice = n_H_3d.in_cgs().to_ndarray()[mid_x, :, :]        # cm^-3
colDen_slice = average_N_3d.in_cgs().to_ndarray()[mid_x, :, :]  # cm^-2
colDen_projection_x = average_N_3d.in_cgs().sum(axis=0)
# n_H (cm^-3)
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("Number Density H Slice x cgs")
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
plt.savefig("plots/nH_number_density_slice_cgs.png", dpi=800, bbox_inches="tight")


# 柱密度 N_H (cm^-2)
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("colDen slice x cgs")
im = ax.imshow(
    colDen_projection_x.T,
    extent=density_3d_extent['x'],
    origin='lower',
    cmap='viridis',
    norm=LogNorm()
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(f"{average_N_3d.in_cgs().units}")
ax.set_xlabel("y (cm)")
ax.set_ylabel("z (cm)")
plt.savefig("plots/colDen_slice_x_cgs.png", dpi=800, bbox_inches="tight")




######################### Load table #######################
table = load_table("output_tables_Oct21_50/table_50x50_fixed.npz")
species_list = ["CO", "C+"]

################################################################
# nH, colDens from simulation
nH_cgs = n_H_3d.in_cgs().to_ndarray()
colDen_cgs = average_N_3d.in_cgs().to_ndarray()

print("=" * 50)
print(f"nH_cgs min: {nH_cgs.min():.3e}, max: {nH_cgs.max():.3e}")
print(f"colDen_cgs min: {colDen_cgs.min():.3e}, max: {colDen_cgs.max():.3e}")
print("=" * 50)

valid_nH = (nH_cgs >= table.nH_values.min()) & (nH_cgs <= table.nH_values.max())
valid_col = (colDen_cgs >= table.col_density_values.min()) & (colDen_cgs <= table.col_density_values.max())
valid_mask = valid_nH & valid_col

for species in species_list:
    print(f"=== Processing species {species} ===")
    species_grid = table.get_species_grid(species)
    species_dir = Path("plots") / species
    species_dir.mkdir(parents=True, exist_ok=True)

    if not USE_INTERPOLATED_TABLE:
        log_nH_grid = np.log10(table.nH_values)
        log_col_grid = np.log10(table.col_density_values)
        log_nH_mesh, log_col_mesh = np.meshgrid(log_nH_grid, log_col_grid, indexing="ij")

        lumPerH_grid = species_grid.lum_per_h
        table_valid_mask = np.isfinite(lumPerH_grid)
        log_lumPerH = np.zeros_like(lumPerH_grid, dtype=float)
        log_lumPerH[table_valid_mask] = np.log10(lumPerH_grid[table_valid_mask])

        interp_points = np.column_stack((log_nH_mesh[table_valid_mask], log_col_mesh[table_valid_mask]))
        interp_values = log_lumPerH[table_valid_mask]
        log_lumPerH_interp = LinearNDInterpolator(interp_points, interp_values, fill_value=np.nan)
        raw_log_interp = log_lumPerH_interp
    else:
        lumPerH_grid = species_grid.lum_per_h
        table_mask = np.isfinite(lumPerH_grid)
        log_nH_grid = np.log10(table.nH_values)
        log_col_grid = np.log10(table.col_density_values)
        log_nH_mesh, log_col_mesh = np.meshgrid(log_nH_grid, log_col_grid, indexing="ij")
        log_lumPerH_grid = np.full_like(lumPerH_grid, np.nan, dtype=float)
        log_lumPerH_grid[table_mask] = np.log10(lumPerH_grid[table_mask])

        interp_points = np.column_stack((log_nH_mesh[table_mask], log_col_mesh[table_mask]))
        interp_values = log_lumPerH_grid[table_mask]
        linear_interp = LinearNDInterpolator(interp_points, interp_values, fill_value=np.nan)
        nearest_interp = NearestNDInterpolator(interp_points, interp_values)

        log_lumPerH_filled = linear_interp(log_nH_mesh, log_col_mesh)
        nan_after_linear = ~np.isfinite(log_lumPerH_filled)
        if np.any(nan_after_linear):
            log_lumPerH_filled[nan_after_linear] = nearest_interp(
                log_nH_mesh[nan_after_linear],
                log_col_mesh[nan_after_linear],
            )
        lumPerH_filled = np.power(10.0, log_lumPerH_filled)
        interpolated_mask = ~table_mask

        interp_points_full = np.column_stack([log_nH_mesh.ravel(), log_col_mesh.ravel()])
        interp_values_full = log_lumPerH_filled.ravel()
        log_lumPerH_interp = LinearNDInterpolator(
            interp_points_full,
            interp_values_full,
            fill_value=np.nan,
        )

        plot_table(
            table=table,
            data=lumPerH_filled,
            output_path=str(species_dir / f"table_{species}_lum_per_h_filled.png"),
            title=f"{species} lum_per_h table (filled)",
            cbar_label="lum_per_h (erg/s per H)",
            use_log=True,
            overlay_mask=interpolated_mask,
            overlay_alpha=0.3,
        )
        raw_log_interp = linear_interp

    if USE_BOUNDARY_CLAMP:
        clipped_nH = np.clip(nH_cgs, table.nH_values.min(), table.nH_values.max())
        clipped_col = np.clip(colDen_cgs, table.col_density_values.min(), table.col_density_values.max())
        log_nH_vals = np.log10(clipped_nH)
        log_col_vals = np.log10(clipped_col)
    else:
        log_nH_vals = np.log10(nH_cgs)
        log_col_vals = np.log10(colDen_cgs)

    log_lumPerH_3d = log_lumPerH_interp(
        log_nH_vals.ravel(),
        log_col_vals.ravel(),
    ).reshape(nH_cgs.shape)
    table_interp_mask = np.isfinite(log_lumPerH_3d)
    lumPerH_3d = np.power(10.0, log_lumPerH_3d)

    raw_log_values = plots.report_raw_table_coverage(
        table=table,
        species_grid=species_grid,
        raw_log_interp=raw_log_interp,
        nH_cgs=nH_cgs,
        colDen_cgs=colDen_cgs,
        valid_mask=valid_mask,
        output_path=str(species_dir / f"{species}_simulation_vs_raw_table.png"),
        log_color=USE_INTERPOLATED_TABLE,
    )
    strict_mask = valid_mask & table_interp_mask & np.isfinite(raw_log_values)

    if USE_BOUNDARY_CLAMP:
        combined_mask = table_interp_mask
    else:
        combined_mask = valid_mask & table_interp_mask

    lumPerH_3d[~combined_mask] = np.nan
    lum = (
        lumPerH_3d
        * dv_3d.in_cgs().to_ndarray()
        * n_H_3d.in_cgs().to_ndarray()
    )
    lum_density = (
        lumPerH_3d * n_H_3d.in_cgs().to_ndarray()
    )


    lum[~combined_mask] = np.nan
    lum_density[~combined_mask] = np.nan

    lum_loose = np.nan_to_num(lum, copy=True, nan=0.0)
    lum_strict = lum.copy()
    lum_strict[~strict_mask] = np.nan

    lum_density_loose = np.nan_to_num(lum_density, copy=True, nan=0.0)
    lum_density_strict = lum_density.copy()
    lum_density_strict[~strict_mask] = np.nan

    lum = yt.YTArray(lum, "erg/s")
    lum_loose = yt.YTArray(lum_loose, "erg/s")
    lum_strict = yt.YTArray(lum_strict, "erg/s")

    lum_density = yt.YTArray(lum_density, "erg/(s*cm**3)")
    lum_density_loose = yt.YTArray(lum_density_loose, "erg/(s*cm**3)")
    lum_density_strict = yt.YTArray(lum_density_strict, "erg/(s*cm**3)")


    lum_nd = lum.to_ndarray()
    lum_strict_nd = lum_strict.to_ndarray()
    lum_loose_nd = lum_loose.to_ndarray()

    lum_density_nd = lum_density.to_ndarray()
    lum_density_strict_nd = lum_density_strict.to_ndarray()
    lum_density_loose_nd = lum_density_loose.to_ndarray()

    dx_3d_cgs = dx_3d.in_cgs()
    dy_3d_cgs = dy_3d.in_cgs()
    dz_3d_cgs = dz_3d.in_cgs()

    dx_3d_nd = dx_3d_cgs.to_ndarray()
    dy_3d_nd = dy_3d_cgs.to_ndarray()
    dz_3d_nd = dz_3d_cgs.to_ndarray()

    lum_density_unit_latex = _unit_to_latex(lum_density.units)
    lum_density_strict_unit_latex = _unit_to_latex(lum_density_strict.units)
    lum_density_loose_unit_latex = _unit_to_latex(lum_density_loose.units)
    sb_unit_x_latex = _unit_to_latex((lum_density * dx_3d_cgs).units)
    sb_unit_z_latex = _unit_to_latex((lum_density * dz_3d_cgs).units)
    sb_unit_x_strict_latex = _unit_to_latex((lum_density_strict * dx_3d_cgs).units)
    sb_unit_z_strict_latex = _unit_to_latex((lum_density_strict * dz_3d_cgs).units)
    sb_unit_x_loose_latex = _unit_to_latex((lum_density_loose * dx_3d_cgs).units)
    sb_unit_z_loose_latex = _unit_to_latex((lum_density_loose * dz_3d_cgs).units)


    slice_extent = dx_3d_extent['x']
    plots.plot_masked_image(
        data=lum_density_nd[mid_x, :, :],
        mask=combined_mask[mid_x, :, :],
        extent=slice_extent,
        title=f"{species} Luminosity Density x Slice",
        cbar_label=rf"Luminosity Density ($\mathrm{{{lum_density_unit_latex}}}$)",
        output_path=str(species_dir / f"{species}_lum_Density_xSlice.png"),
        xlabel="y (cm)",
        ylabel="z (cm)",
    )

    plots.plot_masked_image(
        data=lum_density_strict_nd[mid_x, :, :],
        mask=strict_mask[mid_x, :, :],
        extent=slice_extent,
        title=f"{species} Luminosity Density x Slice (masked)",
        cbar_label=rf"Luminosity Density ($\mathrm{{{lum_density_strict_unit_latex}}}$)",
        output_path=str(species_dir / f"{species}_lum_Density_xSlice_masked.png"),
        xlabel="y (cm)",
        ylabel="z (cm)",
    )

    loose_slice = lum_density_loose_nd[mid_x, :, :]
    plots.plot_masked_image(
        data=loose_slice,
        mask=loose_slice > 0.0,
        extent=slice_extent,
        title=f"{species} Luminosity Density x Slice (loose)",
        cbar_label=rf"Luminosity Density ($\mathrm{{{lum_density_loose_unit_latex}}}$)",
        output_path=str(species_dir / f"{species}_lum_Density_xSlice_loose.png"),
        xlabel="y (cm)",
        ylabel="z (cm)",
    )

    surface_brightness_x = np.nansum(lum_density_nd * dx_3d_nd, axis=0)
    surface_brightness_x_valid = np.all(combined_mask, axis=0)
    plots.plot_masked_image(
        data=surface_brightness_x,
        mask=surface_brightness_x_valid,
        extent=dx_3d_extent['x'],
        title=f"{species} Luminosity Density x Projection",
        cbar_label=rf"Surface Brightness ($\mathrm{{{sb_unit_x_latex}}}$)",
        output_path=str(species_dir / f"{species}_lum_Density_xProjection.png"),
        xlabel="y (cm)",
        ylabel="z (cm)",
    )

    surface_brightness_z = np.nansum(lum_density_nd * dz_3d_nd, axis=2)
    surface_brightness_z_valid = np.all(combined_mask, axis=2)
    plots.plot_masked_image(
        data=surface_brightness_z,
        mask=surface_brightness_z_valid,
        extent=dx_3d_extent['z'],
        title=f"{species} Luminosity Density z Projection",
        cbar_label=rf"Surface Brightness ($\mathrm{{{sb_unit_z_latex}}}$)",
        output_path=str(species_dir / f"{species}_lum_Density_zProjection.png"),
        xlabel="x (cm)",
        ylabel="y (cm)",
    )

    strict_surface_brightness_x = np.nansum(lum_density_strict_nd * dx_3d_nd, axis=0)
    strict_surface_brightness_x_valid = np.all(strict_mask, axis=0)
    plots.plot_masked_image(
        data=strict_surface_brightness_x,
        mask=strict_surface_brightness_x_valid,
        extent=dx_3d_extent['x'],
        title=f"{species} Luminosity Density x Projection (masked)",
        cbar_label=rf"Surface Brightness ($\mathrm{{{sb_unit_x_strict_latex}}}$)",
        output_path=str(species_dir / f"{species}_lum_Density_xProjection_masked.png"),
        xlabel="y (cm)",
        ylabel="z (cm)",
    )

    strict_surface_brightness_z = np.nansum(lum_density_strict_nd * dz_3d_nd, axis=2)
    strict_surface_brightness_z_valid = np.all(strict_mask, axis=2)
    plots.plot_masked_image(
        data=strict_surface_brightness_z,
        mask=strict_surface_brightness_z_valid,
        extent=dx_3d_extent['z'],
        title=f"{species} Luminosity Density z Projection (masked)",
        cbar_label=rf"Surface Brightness ($\mathrm{{{sb_unit_z_strict_latex}}}$)",
        output_path=str(species_dir / f"{species}_lum_Density_zProjection_masked.png"),
        xlabel="x (cm)",
        ylabel="y (cm)",
    )

    loose_surface_brightness_x = np.nansum(lum_density_loose_nd * dx_3d_nd, axis=0)
    loose_surface_brightness_z = np.nansum(lum_density_loose_nd * dz_3d_nd, axis=2)
    plots.plot_masked_image(
        data=loose_surface_brightness_x,
        mask=loose_surface_brightness_x > 0.0,
        extent=dx_3d_extent['x'],
        title=f"{species} Luminosity Density x Projection (loose)",
        cbar_label=rf"Surface Brightness ($\mathrm{{{sb_unit_x_loose_latex}}}$)",
        output_path=str(species_dir / f"{species}_lum_Density_xProjection_loose.png"),
        xlabel="y (cm)",
        ylabel="z (cm)",
    )

    plots.plot_masked_image(
        data=loose_surface_brightness_z,
        mask=loose_surface_brightness_z > 0.0,
        extent=dx_3d_extent['z'],
        title=f"{species} Luminosity Density z Projection (loose)",
        cbar_label=rf"Surface Brightness ($\mathrm{{{sb_unit_z_loose_latex}}}$)",
        output_path=str(species_dir / f"{species}_lum_Density_zProjection_loose.png"),
        xlabel="x (cm)",
        ylabel="y (cm)",
    )

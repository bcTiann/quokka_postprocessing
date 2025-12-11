import numpy as np
import matplotlib.pyplot as plt
from quokka2s.tables import load_table, plot_sampling_histogram
import yt
from yt.units import mh
from quokka2s.pipeline.prep import config as cfg
import quokka2s as q2s
from quokka2s.pipeline.prep import physics_fields as phys
from quokka2s.despotic_tables import compute_average
from pathlib import Path



table = load_table(cfg.DESPOTIC_TABLE_PATH)


ds = yt.load(cfg.YT_DATASET_PATH)
phys.add_all_fields(ds)
provider = q2s.YTDataProvider(ds)
dx_3d, dx_3d_extent = provider.get_slab_z(('boxlib', 'dx'))
dx_projection = dx_3d.sum(axis=0)

dy_3d, dy_3d_extent = provider.get_slab_z(('boxlib', 'dy'))
dy_projection = dy_3d.sum(axis=0)

dz_3d, dz_3d_extent = provider.get_slab_z(('boxlib', 'dz'))
dz_projection = dz_3d.sum(axis=0)

dv_3d = dx_3d * dy_3d * dz_3d

factor = 1
nx, ny, nz = dy_3d.shape
mid_z = nz//factor//2
mid_x = nx//factor//2



X_H = cfg.X_H
m_H = mh.in_cgs()
density_3d, density_3d_extent = provider.get_slab_z(
    field=('gas', 'density')
)

density_3d = provider.downsample_3d_array(density_3d, factor=factor)
##################################
n_H_3d = (density_3d * X_H) / m_H


dx_3d, dx_3d_extent = provider.get_slab_z(
    field=('boxlib', 'dx')
)
dy_3d, dy_3d_extent = provider.get_slab_z(
    field=('boxlib', 'dy')
)
dz_3d, dz_3d_extent = provider.get_slab_z(
    field=('boxlib', 'dz')
)
temp_3d, temp_3d_extent = provider.get_slab_z(
    field=('gas', 'temperature')
)


Nx_p = q2s.along_sight_cumulation(n_H_3d * dx_3d, axis="x", sign="+") 
Ny_p = q2s.along_sight_cumulation(n_H_3d * dy_3d, axis="y", sign="+") 
Nz_p = q2s.along_sight_cumulation(n_H_3d * dz_3d, axis="z", sign="+")

Nx_n = q2s.along_sight_cumulation(n_H_3d * dx_3d, axis="x", sign="-")
Ny_n = q2s.along_sight_cumulation(n_H_3d * dy_3d, axis="y", sign="-")
Nz_n = q2s.along_sight_cumulation(n_H_3d * dz_3d, axis="z", sign="-")


average_N_3d = compute_average(
    [Nx_p, Ny_p, Nz_p, Nx_n, Ny_n, Nz_n],
    method="harmonic",
)


# 取温度并裁剪到表的范围
T_arr = temp_3d.in_cgs().to_ndarray()
T_min, T_max = table.T_values.min(), table.T_values.max()
T_safe = np.clip(T_arr, T_min, T_max)


X_H = cfg.X_H
m_H = mh.in_cgs()
density_3d, density_3d_extent = provider.get_slab_z(
    field=('gas', 'density')
)


n_H_3d = (density_3d * X_H) / m_H


n_H_array = n_H_3d.in_cgs().to_ndarray()
col_den_array = average_N_3d.in_cgs().to_ndarray()

nan_nH = ~np.isfinite(n_H_array)
nan_col = ~np.isfinite(col_den_array)
nan_T   = ~np.isfinite(T_safe)
print("NaN counts -> nH:", nan_nH.sum(), "col:", nan_col.sum(), "T:", nan_T.sum())

finite_mask = (
    np.isfinite(n_H_array)
    & np.isfinite(col_den_array)
    & np.isfinite(T_safe)
    & (n_H_array > 0.0)
    & (col_den_array > 0.0)
)
log_samples = np.column_stack(
    (
        np.log10(n_H_array[finite_mask]),
        np.log10(col_den_array[finite_mask]),
        np.log10(T_safe[finite_mask]),
    )
)
np.save("plots/log_samples.npy", log_samples)

# 选几个温度切片（确保在表范围内）
T_min, T_max = table.T_values.min(), table.T_values.max()
T_targets_raw = [1e2, 1e3, 1e4, 5e4, 1e5, 2e5, 5e5, 8e5, 1e6, 6e6]
T_targets = [T for T in T_targets_raw if T_min <= T <= T_max]
T_indices = [int(np.argmin(np.abs(table.T_values - Tt))) for Tt in T_targets]


# 计算每个样本对应的最近 T 切片索引
sample_T_log = log_samples[:, 2]
table_T_log = np.log10(table.T_values)
slice_idx_for_sample = np.argmin(np.abs(sample_T_log[:, None] - table_T_log[None, :]), axis=1)

base_dir = Path("plots") / "hist_slices"
base_dir.mkdir(parents=True, exist_ok=True)

for Tt, t_idx in zip(T_targets, T_indices):
    mask = slice_idx_for_sample == t_idx
    samples_2d = log_samples[mask][:, :2]
    if samples_2d.size == 0:
        print(f"No samples near T={Tt:.1g} K; skip")
        continue

    failure_2d = table.failure_mask[:, :, t_idx]
    ax = plot_sampling_histogram(
        table,
        samples_2d,
        log_space=True,
        failure_mask=failure_2d
    )
    ax.set_title(f"Sampling vs failures @ T≈{Tt:.1g} K")
    plt.savefig(base_dir / f"hist_T{int(Tt)}.png", dpi=400)
    plt.close()

# ---------- 统计部分（不画图） ----------
nH_vals = table.nH_values
col_vals = table.col_density_values
T_vals = table.T_values

nH_min, nH_max = nH_vals.min(), nH_vals.max()
col_min, col_max = col_vals.min(), col_vals.max()
T_min, T_max = T_vals.min(), T_vals.max()

# 样本（已裁剪温度）
nH_s = n_H_array[finite_mask]
col_s = col_den_array[finite_mask]
T_s = np.clip(T_safe[finite_mask], T_min, T_max)
total = len(nH_s)

# 判断是否落在表格范围
in_bounds = (
    (nH_s >= nH_min) & (nH_s <= nH_max) &
    (col_s >= col_min) & (col_s <= col_max) &
    (T_s >= T_min) & (T_s <= T_max)
)
out_of_bounds = (~in_bounds).sum()

# 最近邻索引（对数空间）
def to_index_nearest(arr, grid):
    idx = np.searchsorted(grid, arr, side="left")
    idx = np.clip(idx, 1, len(grid) - 1)
    left = grid[idx - 1]
    right = grid[idx]
    use_left = np.abs(np.log10(arr) - np.log10(left)) <= np.abs(np.log10(arr) - np.log10(right))
    return np.where(use_left, idx - 1, idx)

nH_idx = to_index_nearest(nH_s[in_bounds], nH_vals)
col_idx = to_index_nearest(col_s[in_bounds], col_vals)
T_idx  = to_index_nearest(T_s[in_bounds],  T_vals)


# 命中 failure_mask
fail_mask = table.failure_mask

# 把发射线 lumPerH 非有限或 <=0 也并入失败
emitters = ["CO", "C+", "HCO+"]  # 按需调整
bad_em_mask = np.zeros_like(fail_mask, dtype=bool)
for em in emitters:
    rec = table.species_data.get(em)
    lum = getattr(rec, "lumPerH", None)
    if lum is None:
        continue
    bad_em_mask |= (~np.isfinite(lum)) | (lum <= 0.0)

combined_fail_mask = fail_mask | bad_em_mask

# 后面的命中统计用 combined_fail_mask
fail_hits = combined_fail_mask[nH_idx, col_idx, T_idx]
n_fail = fail_hits.sum()


# 命中 NaN（以 mu_values 为例，也可换成 tg_final 或某个物种）
mu = table.mu_values
nan_hits = ~np.isfinite(mu[nH_idx, col_idx, T_idx])
n_nan = nan_hits.sum()

# 逐物种 abundance<=0 或非有限
species_list = list(table.species_data.keys())
species_bad_counts = {}
for sp in species_list:
    sp_rec = table.species_data[sp]
    abund = getattr(sp_rec, "abundance", None)
    if abund is None:
        continue
    vals = abund[nH_idx, col_idx, T_idx]
    bad_mask = (~np.isfinite(vals)) | (vals <= 0.0)
    species_bad_counts[sp] = bad_mask.sum()

print("各物种 abundance<=0 或非有限值的样本数：")
for sp, cnt in species_bad_counts.items():
    print(f"  {sp}: {cnt}")

print(f"总样本数: {total}")
print(f"越界样本: {out_of_bounds}")
print(f"在表内的样本: {in_bounds.sum()}")
print(f"命中 failure_mask 的样本: {n_fail}")
print(f"命中 NaN (mu_values) 的样本: {n_nan}")

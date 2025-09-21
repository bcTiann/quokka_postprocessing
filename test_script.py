import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import yt 
from plot_interface import YTDataProvider, create_plot, create_horizontal_subplots, plot_multiview_grid

# --- 1. 加载数据并初始化 Provider ---
ds = yt.load("plt01000")
provider = YTDataProvider(ds)

# --- 2. 定义切片和粒子参数 ---
slice_axis = 'x'
field = ('gas', 'density')
particle_units = 'pc'
depth_pc = float(ds.domain_width[provider._axis_map[slice_axis]].in_units(particle_units).value)

# --- 3. 从 Provider 获取所有需要的数据 ---

# 获取2D网格数据 (密度)
density_data, density_units = provider.get_slice(field=field, axis=slice_axis)

# 获取绘图范围
plot_extent = provider.get_plot_extent(axis=slice_axis, units=particle_units)

# 获取粒子位置数据
# 注意：我们将深度设置为总宽度的 1/10，与你的原始 annotate_particles 示例匹配
px, py = provider.get_particle_positions(axis=slice_axis, depth=depth_pc / 10, units=particle_units)


# --- 4. 使用 Matplotlib 绘图 ---

fig, ax = plt.subplots(figsize=(8, 10))

# 绘制背景密度图
# 使用 LogNorm 来匹配 yt 的对数颜色条
# origin='lower' 确保 y 轴方向正确
im = ax.imshow(density_data,
               origin='lower',
               extent=plot_extent,
               norm=LogNorm())

# 绘制粒子
# 使用 scatter 函数，这相当于 yt 的 annotate_particles
ax.scatter(px, py, s=5., c='red', marker='o')


# --- 5. 设置图像格式 ---
ax.set_aspect('auto') # 或者 'equal'，取决于你想要的效果

# --- Corrected Lines ---
# Determine the plot axes from the slice axis
slice_axis_index = provider._axis_map[slice_axis]
plot_axes_indices = [i for i in range(3) if i != slice_axis_index]

# Set the labels correctly
xlabel = f"{provider._axis_map[plot_axes_indices[0]]} ({particle_units})"
ylabel = f"{provider._axis_map[plot_axes_indices[1]]} ({particle_units})"

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)

ax.set_title(f'Slice of {field[1]} along {slice_axis}-axis')

# 添加颜色条
cbar = fig.colorbar(im)
cbar.set_label(f'Density ({density_units})')


# --- 6. 显示并保存图像 ---
plt.tight_layout()
plt.savefig("matplotlib_density_slice_with_particles.png", dpi=300)
plt.show()
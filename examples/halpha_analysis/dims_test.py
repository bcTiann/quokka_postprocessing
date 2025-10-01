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



ds = yt.load(cfg.YT_DATASET_PATH)

# 获取整个模拟域的物理宽度和像素数量
domain_width = ds.domain_width
domain_dimensions = ds.domain_dimensions

print("--------------------------------------------------")
print("模拟域的总物理宽度 (ds.domain_width):")
print(f"X: {domain_width[0]}")
print(f"Y: {domain_width[1]}")
print(f"Z: {domain_width[2]}")
print("\n")

print("模拟域的总像素数量 (ds.domain_dimensions):")
print(f"X: {domain_dimensions[0]}")
print(f"Y: {domain_dimensions[1]}")
print(f"Z: {domain_dimensions[2]}")
print("\n")

# 计算每个像素在物理空间中的尺寸 (即 dx, dy, dz)
pixel_widths = domain_width / domain_dimensions

print("单个像素的物理尺寸 (pixel_widths):")
print(f"dx (X轴): {pixel_widths[0]}")
print(f"dy (Y轴): {pixel_widths[1]}")
print(f"dz (Z轴): {pixel_widths[2]}")
print("--------------------------------------------------")

dx = ds.index.get_smallest_dx()
# dy = ds.index.get_smallest_dy()
# dz = ds.index.get_smallest_dz()

print(f"dx = {dx}")
# print(f"dy = {dy}")
# print(f"dz = {dz}")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
from pathlib import Path
from quokka2s.despotic_tables import (
    DespoticTable,
    LogGrid,
    build_table,
    make_temperature_interpolator,
    refine_table,
    fill_missing_values,
)


table = np.load("output_tables_2REPATE/table_10x10_raw_8.npz")
co = table["co_int_tb"]
tg = table["tg_final"]

print("nonfinite co:", np.count_nonzero(~np.isfinite(co)))
print("nonfinite tg:", np.count_nonzero(~np.isfinite(tg)))
print("nonpositive tg:", np.count_nonzero(tg <= 0))
print("zeros in co:", np.count_nonzero(co == 0))

print("+"*40)
print("tg_final")
arr = table["tg_final"]
idx = np.unravel_index(arr.argmin(), arr.shape)
print(idx, arr[idx])

print(table["tg_final"])

print("+"*40)
print("co_int_tb")

arr = table["co_int_tb"]
idx = np.unravel_index(arr.argmin(), arr.shape)
print(idx, arr[idx])

print(table["co_int_tb"])

row, col = 2, 8
co_val = table["co_int_tb"][row, col]
tg_val = table["tg_final"][row, col]

print(f"co_int_tb[{row}, {col}] = {co_val}")
print(f"tg_final[{row}, {col}] = {tg_val}")

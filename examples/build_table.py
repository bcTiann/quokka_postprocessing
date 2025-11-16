from __future__ import annotations

import warnings
warnings.filterwarnings(
    "ignore",
    message="collision rates not available",
    category=UserWarning,
    module=r"DESPOTIC.*emitterData",
)
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in log",
    category=RuntimeWarning,
    module=r"DESPOTIC.*NL99_GC",
)



from pathlib import Path

from quokka2s.tables import LogGrid, build_table, save_table



N_H_RANGE = (1e-5, 1e5)
COL_DEN_RANGE = (1e15, 1e24)
points = 10
# 网格设置，可按需要修改
nH_grid = LogGrid(*N_H_RANGE, num_points=points)
col_grid = LogGrid(*COL_DEN_RANGE, num_points=points)
tg_guesses = [10.0, 30.0, 100.0, 300.0]

# 构建 DESPOTIC 查找表
table = build_table(nH_grid, col_grid, tg_guesses, show_progress=True)

# 保存到 npz
output_dir = Path("output_tables_new")
output_dir.mkdir(parents=True, exist_ok=True)
save_table(table, output_dir / "despotic_table.npz")



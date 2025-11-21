from __future__ import annotations
from pathlib import Path
from quokka2s.tables import LogGrid, build_table, save_table

N_H_RANGE = (1e-5, 1e5)
COL_DEN_RANGE = (1e15, 1e24)
points = 25
nH_grid = LogGrid(*N_H_RANGE, num_points=points)
col_grid = LogGrid(*COL_DEN_RANGE, num_points=points)
tg_guesses = [10.0, ]
# SPECIES = ('CO', 'C+', "C", 'HCO+', 'O')
SPECIES = ('CO', 'C+', 'HCO+', 'C')
# ABUNDANCES = ('H+', 'H2', 'H3+', 'He+', 'OHx', 'CHx', 'CO', 'C', 
#               'C+', 'HCO+', 'O', 'M+', 'H', 'He', 'M', 'e-')
ABUNDANCES = ('e-', 'H-')

table = build_table(nH_grid, col_grid, tg_guesses, species=SPECIES, abundance_only=ABUNDANCES, show_progress=True, workers=-1)
output_dir = Path("output_tables_testSmall")
output_dir.mkdir(parents=True, exist_ok=True)
save_table(table, output_dir / "despotic_table.npz")



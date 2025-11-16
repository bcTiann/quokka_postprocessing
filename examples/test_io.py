from pathlib import Path
from quokka2s.tables import LogGrid, build_table, save_table, load_table
import numpy as np


nH = LogGrid(1e2, 1e3, 2)
col = LogGrid(1e20, 1e21, 2)

table = build_table(nH, col, tg_guesses=[10, 30], show_progress=True)


tmp = Path("tmp_table.npz")
save_table(table, tmp)
round_trip = load_table(tmp)

assert table.species == round_trip.species
assert np.allclose(table.tg_final, round_trip.tg_final, equal_nan=True)
assert np.array_equal(table.failure_mask, round_trip.failure_mask)
assert len(table.attempts) == len(round_trip.attempts)


# 新增：验证 energy_terms round-trip
assert table.energy_terms.keys() == round_trip.energy_terms.keys()
for name, grid in table.energy_terms.items():
    assert np.allclose(grid, round_trip.energy_terms[name], equal_nan=True)

tmp.unlink()

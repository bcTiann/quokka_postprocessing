import numpy as np
import sys
sys.path.append('quokka2s/src')
from quokka2s.tables import load_table
from quokka2s.pipeline.prep import config as cfg

table = load_table(cfg.DESPOTIC_TABLE_PATH)
arr = table.require_species('C').abundance

print('shape', arr.shape)
print('nan', np.isnan(arr).sum(), 'inf', np.isinf(arr).sum())
print('<=0', (arr <= 0).sum(), '<0', (arr < 0).sum(), '==0', (arr == 0).sum())
finite = arr[np.isfinite(arr)]
print('finite min/max', finite.min(), finite.max())

rows, cols = np.where(~np.isfinite(arr) | (arr <= 0))
print('first few problematic entries:')
for r, c in list(zip(rows, cols))[:]:
    print(f'  idx=({r},{c}) value={arr[r,c]} nH={table.nH_values[r]} colDen={table.col_density_values[c]}')

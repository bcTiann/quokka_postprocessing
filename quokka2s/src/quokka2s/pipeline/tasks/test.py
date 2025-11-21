import yt
from quokka2s.pipeline.prep.physics_fields import add_all_fields
from quokka2s.pipeline.prep import config as cfg

ds = yt.load(cfg.YT_DATASET_PATH)
add_all_fields(ds)

cg = ds.covering_grid(level=ds.max_level,
                      left_edge=ds.domain_left_edge,
                      dims=ds.domain_dimensions * (2**ds.max_level))

n_H = cg[('gas', 'number_density_H')].to('cm**-3')
col = cg[('gas', 'column_density_H')].to('cm**-2')

print(f"nH: {n_H.min():.3e}, {n_H.max():.3e}, {n_H.mean():.3e}")
print(f"colDen: {col.min():.3e}, {col.max():.3e}, {col.mean():.3e}")

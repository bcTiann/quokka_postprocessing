from quokka2s.tables import load_table
from quokka2s.tables.lookup import TableLookup
from quokka2s.config import config as cfg
from yt.units import K, mh, kb
import numpy as np

table = load_table(cfg.DESPOTIC_TABLE_PATH)
lookup = TableLookup(table)
T_min, T_max = table.T_values.min(), table.T_values.max()

gamma = 5/3

def iterate_T_mu(nH, colDen, rho, E_int_density, T_init=1e3, max_iter=10, tol=1e-3):
    T = T_init

    for _ in range(max_iter):
        T = np.clip(T, T_min, T_max)
        mu = lookup.mu(nH, colDen, T)
        T_new = (gamma - 1.0) * (E_int_density / rho) * mu * (mh / kb)
        if np.abs(T_new - T) / T < tol:
            return T_new, mu
        T = T_new
    return T, mu

def query_point(nH, colDen, rho, E_int_density):
    T_final, mu_final = iterate_T_mu(nH, colDen, rho, E_int_density)
    co_abund = lookup.abundance("CO", nH, colDen, T_final)
    co_lum   = lookup.line_field("CO", "lumPerH", nH, colDen, T_final)
    return {"T": T_final, "mu": mu_final, "CO_abund": co_abund, "CO_lumPerH": co_lum}
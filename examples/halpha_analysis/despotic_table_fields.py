from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from yt.units import mh

import config as cfg
import quokka2s as q2s
from quokka2s.despotic_tables import compute_average
from quokka2s.tables import DespoticTable, TableLookup



def compute_inputs(provider) -> dict[str, np.ndarray]:
    """Return nH and column Density arrays suitable for DESPOTIC table lookups

    Args:
        provider (_type_): _description_

    Returns:
        dict[str, np.ndarray]: _description_
    """

    density_3d, extent = provider.get_slab_z(("gas", "density"))
    dx_3d, _ = provider.get_slab_z(("boxlib", "dx"))
    dy_3d, _ = provider.get_slab_z(("boxlib", "dy"))
    dz_3d, _ = provider.get_slab_z(("boxlib", "dz"))

    m_H = mh.in_cgs()
    n_H_3d = (density_3d * cfg.X_H) / m_H

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

    return {
        "n_H": n_H_3d.to_cgs(),
        "column_density": average_N_3d.to_cgs(),
    }



def evaluate_table_fields(
        table: DespoticTable,
        n_H_cgs: np.ndarray,
        colDen_cgs: np.ndarray,
        *,
        species: Sequence[str],
) -> dict[str, object]:
    """Sample tempearture/abundances/electron density from the DESPOTIC table.

    Args:
        table (DespoticTable): _description_
        n_H_cgs (np.ndarray): _description_
        colDen_cgs (np.ndarray): _description_
        species (Sequence[str]): _description_

    Returns:
        dict[str, object]: _description_
    """
    lookup = TableLookup(table)
    tempearture = lookup.temperature(n_H_cgs, colDen_cgs)
    abundances = {
        sp: lookup.abundance(sp, n_H_cgs, colDen_cgs)
        for sp in species
    }

    return {
        "lookup": lookup,
        "temperature": tempearture,
        "abundances": abundances,
    }



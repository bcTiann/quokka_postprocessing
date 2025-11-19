# physics_models.py

import yt
import numpy as np
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg
import quokka2s as q2s
from quokka2s.despotic_tables import compute_average
import config as cfg
from quokka2s.tables.lookup import TableLookup
from examples.halpha_analysis.run_despotic import load_table


# --- Fundamental Physical Constants ---
m_H = mh.in_cgs()
lambda_Halpha = 656.3e-7 * cm
h = planck_constant
speed_of_light_value_in_ms = 299792458 
c = speed_of_light_value_in_ms * m / s
TABLE_LOOKUP_CACHE: TableLookup | None = None
TABLE_LOOKUP_SPECIES: tuple[str, ...] = ()

def ensure_table_lookup(path: str | None) -> TableLookup:
    global TABLE_LOOKUP_CACHE
    if TABLE_LOOKUP_CACHE is None:
        table = load_table(path or cfg.DESPOTIC_TABLE_PATH)
        TABLE_LOOKUP_CACHE = TableLookup(table)
    return TABLE_LOOKUP_CACHE


def _number_density_H(field, data):
    density_3d = data[('gas', 'density')].in_cgs()
    n_H_3d = (density_3d * cfg.X_H) / m_H
    return n_H_3d.to('cm**-3')


def _column_density_H(field, data):
    density_3d = data[('gas', 'density')].in_cgs()
    dx_3d = data[("boxlib", "dx")].in_cgs()
    dy_3d = data[("boxlib", "dy")].in_cgs()
    dz_3d = data[("boxlib", "dz")].in_cgs()

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
    return average_N_3d.to('cm**-2')

# --- YT Derived Fields ---


def _temperature(field, data):
    n_H = data[('gas', 'number_density_H')]
    colDen_H = data[('gas', 'column_density_H')]
    # for each point (nH, colDen) we get lookup our despotic table, get tempearture for that (nH, colDen)
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)

    temps = lookup.temperature(nH_cgs=n_H.value, colDen_cgs=colDen_H.value)

    return temps * K


# def _number_density_electron(field, data):
#     n_H = data[('gas', 'number_density_H')]
#     colDen_H = data[('gas', 'column_density_H')]
#     lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
#     number_density_electron = lookup.number_densities('e-', nH_cgs=n_H.value, colDen_cgs=colDen_H.value)

#     return number_density_electron * cm**-3

def _make_number_density_field(species: str):
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
    token = species
    yt_safe_name = species.replace('+', '_plus').replace('-','_minus')
    def _field(field, data):
        n_H = data[('gas','number_density_H')].to('cm**-3').value
        colDen = data[('gas','column_density_H')].to('cm**-2').value
        densities = lookup.number_densities([token], n_H, colDen)
        return densities[token] * cm**-3
    _field.__name__ = f"_number_density_{yt_safe_name}"
    return yt_safe_name, _field


def _Halpha_luminosity(field, data):
    """
    Calculate H-alpha Luminosity Density
    Units: erg / s / cm**3
    
    Draine (2011) Eq. 14.6
    """
    E_Halpha = (h * c) / lambda_Halpha # Energy of a single H-alpha photon
    density_3d = data[('gas', 'density')].in_cgs()
    temp = data[('gas', 'temperature')].in_cgs()
   
    n_e = data[('gas', 'e-')]
    n_ion = data[('gas', 'H+')]
    n_H = (density_3d * cfg.X_H) / m_H

    Z = 1.0
    T4 = temp / (1e4 * yt.units.K)

    exponent = -0.8163 - 0.0208 * np.log(T4 / Z**2)

    alpha_B = (2.54e-13 * Z**2 * (T4 / Z**2)**exponent) * cm**3 / s

    luminosity_density = 0.45 * E_Halpha * alpha_B * n_e * n_ion
    print(f"lum density units:{luminosity_density.units}")
    luminosity_density = luminosity_density.in_cgs()
    print(f"lum density units in cgs:{luminosity_density.units}")
    return luminosity_density


def add_all_fields(ds):
    """Adds all derived fields to the yt dataset."""
    ds.add_field(name=('gas', 'number_density_H'), function=_number_density_H, sampling_type="cell", units="cm**-3", force_override=True)
    ds.add_field(name=('gas', 'column_density_H'), function=_column_density_H, sampling_type="cell", units="cm**-2", force_override=True)
    ds.add_field(name=('gas', 'temperature'), function=_temperature, sampling_type="cell", units="K", force_override=True)
    
    SPECIES = ['H+', 'H2', 'H3+', 'He+', 'OHx', 'CHx', 'CO', 'C',
           'C+', 'HCO+', 'O', 'M+', 'H', 'He', 'M', 'e-']
    for sp in SPECIES:
        _, func = _make_number_density_field(species=sp)
        ds.add_field(
            name=('gas', f'{sp}'),
            function=func,
            sampling_type="cell", 
            units="cm**-3", 
            force_override=True
        )
    ds.add_field(name=('gas', 'Halpha_luminosity'), function=_Halpha_luminosity, sampling_type="cell", units="erg/s/cm**3", force_override=True)
    print("Added derived fields: 'temp_neutral', 'temperature', 'ionized_mask', 'Halpha_luminosity'.")


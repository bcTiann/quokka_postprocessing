# physics_models.py

import yt
import numpy as np
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg
from quokka2s import *

# --- Fundamental Physical Constants ---
m_H = mh.in_cgs()
lambda_Halpha = 656.3e-7 * cm
h = planck_constant
speed_of_light_value_in_ms = 299792458 
c = speed_of_light_value_in_ms * m / s
E_Halpha = (h * c) / lambda_Halpha # Energy of a single H-alpha photon

# --- YT Derived Fields ---

def _temperature(field, data):
    mu = 0.6

    gas_internal_energy_density = data[('gas', 'internal_energy_density')].in_cgs()
    print(f"gas_internal_Energy_density units: {gas_internal_energy_density.units}")

    gas_density = data[('gas', 'density')].in_cgs()
    print(f"gas_density units: {gas_density.units} ")
   

    temp = 2/3 * gas_internal_energy_density * mu * mh / gas_density / kb
    print(f"temperature units: {temp.units}")
    return temp


def _Halpha_emission(field, data):
    """
    Calculate H-alpha Luminosity Density
    Units: erg / s / cm**3
    
    Draine (2011) Eq. 14.6
    """
    X_H = 0.76
    m_H = mh

    gas_density = data[('gas', 'density')].in_cgs()
    print(f"gas density units: {gas_density.units}")
    temperature = data[('gas', 'temperature')].in_cgs()

    n_H = (gas_density * X_H) / m_H

    Z = 1.0
    T4 = temperature / (1e4 * yt.units.K)
    T4_safe = np.maximum(T4, 1e-6)

    exponent = -0.8163 - 0.0208 * np.log(T4_safe / Z**2)

    alpha_B = (2.54e-13 * Z**2 * (T4_safe / Z**2)**exponent) * cm**3 / s
    luminosity_density = 0.45 * E_Halpha * alpha_B * n_H**2
    print(f"lum density units:{luminosity_density.units}")
    luminosity_density = luminosity_density.in_cgs()
    print(f"lum density units in cgs:{luminosity_density.units}")
    return luminosity_density


def add_all_fields(ds):
    """Adds all derived fields to the yt dataset."""
    ds.add_field(name=('gas', 'temperature'), function=_temperature, sampling_type="cell", units="K", force_override=True)
    ds.add_field(name=('gas', 'Halpha_luminosity'), function=_Halpha_emission, sampling_type="cell", units="erg/s/cm**3", force_override=True)
    print("Added derived fields: 'temperature' and 'Halpha_luminosity'.")


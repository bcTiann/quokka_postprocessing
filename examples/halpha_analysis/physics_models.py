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


# --- YT Derived Fields ---

def _temp_neutral(field, data):
    """
    (内部辅助字段)
    计算假设 mu=1.0 时的“中性温度”(T_neutral 或 temp_init)
    """
    mu = 1.0
    eint = data[('gas', 'internal_energy_density')].in_cgs()
    rho = data[('gas', 'density')].in_cgs()
    
    # number density (假设 mu=1.0)
    n = rho / (mu * mp)
    
    temp_init = (2.0 / 3.0) * eint / (n * kb)
    return temp_init.to('K')

def _temperature(field, data):
    temp_init = data[('gas', 'temp_neutral')]
    high_temp_mask = temp_init > (1.0e4 * K)

    corrected_temp = temp_init.copy()
    corrected_temp[high_temp_mask] = temp_init[high_temp_mask] * 0.5

    return corrected_temp.to('K')

def _ionized_mask(field, data):
    temp_init = data[('gas', 'temp_neutral')]
    mask = (temp_init > (1.0e4 * K)).astype('float64')
    return mask



def _Halpha_luminosity(field, data):
    """
    Calculate H-alpha Luminosity Density
    Units: erg / s / cm**3
    
    Draine (2011) Eq. 14.6
    """
    E_Halpha = (h * c) / lambda_Halpha # Energy of a single H-alpha photon
    rho = data[('gas', 'density')].in_cgs()
    temp = data[('gas', 'temperature')].in_cgs()
    # dx = data[('boxlib', 'dx')].in_cgs()
    # dy = data[('boxlib', 'dy')].in_cgs()
    # dz = data[('boxlib', 'dz')].in_cgs()
    # V_cell = dx * dy * dz
    ionized_mask = data[('gas', 'ionized_mask')]


    n_H = rho / mh
    n_e = n_H * ionized_mask
    n_ion = n_e # hold only for fully ionized H gas

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
    ds.add_field(name=('gas', 'temp_neutral'), function=_temp_neutral, sampling_type="cell", units="K", force_override=True)
    ds.add_field(name=('gas', 'temperature'), function=_temperature, sampling_type="cell", units="K", force_override=True)
    ds.add_field(name=('gas', 'ionized_mask'), function=_ionized_mask, sampling_type="cell", units='', force_override=True)
    ds.add_field(name=('gas', 'Halpha_luminosity'), function=_Halpha_luminosity, sampling_type="cell", units="erg/s/cm**3", force_override=True)
    print("Added derived fields: 'temp_neutral', 'temperature', 'ionized_mask', 'Halpha_luminosity'.")


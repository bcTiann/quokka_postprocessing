from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg

def get_attenuation_factor(
        A_lambda_over_NH=8e-22, # (mag * cm^2 / N_H)
        column_density=None #N_H
        ):
    
    # --- 1. 计算每个位置对应的总消光 A_lambda ---
    A_lambda_3d = column_density * A_lambda_over_NH

    print(f"Max A_lambda: {A_lambda_3d.max():.2f} mag")

    # --- 2. 计算衰减因子 10^(-A_lambda / 2.5) ---
    attenuation_factor_3d = 10.0**(-A_lambda_3d / 2.5)
    print(f"Attenuation factor range: min={attenuation_factor_3d.min():.2e}, max={attenuation_factor_3d.max():.2e}")

    return attenuation_factor_3d



def along_sight_cumulation(data: np.ndarray,
                axis: Union[str, int], 
                ):
    
    return np.flip(np.cumsum(np.flip(data, axis=axis), axis=axis), axis=axis)

# --- Core Physics Calculations ---

def calculate_cumulative_column_density(density_3d: np.ndarray, 
                             dx_3d: np.ndarray, 
                             axis: int, 
                             X_H: float):
    """Calculates the cumulative hydrogen column density along a given axis."""
    m_H = mh.in_cgs()
    n_H_3d = (density_3d * X_H) / m_H
    N_H_cell_3d = n_H_3d * dx_3d

    N_H_cumulative = along_sight_cumulation(N_H_cell_3d, axis=axis)
    return N_H_cumulative


def calculate_attenuation(column_density_3d: np.ndarray, A_lambda_over_NH: float):
    """Calculates the dust attenuation factor from column density."""
    A_lambda_3d = column_density_3d * A_lambda_over_NH
    attenuation_factor = 10.0**(-A_lambda_3d / 2.5)
    return attenuation_factor, A_lambda_3d

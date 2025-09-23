from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg
from despotic.chemistry import NL99
from despotic import cloud
from tqdm import tqdm # 使用标准的tqdm库
import time



def run_despotic_on_map(nH_map: np.ndarray, 
                        Tg_map: np.ndarray, 
                        colDen_map: np.ndarray) -> np.ndarray:
    """
    Iterates over a 2D slice and runs DESPOTIC on each pixel.

    Args:
        nH_map (np.ndarray): 2D map of hydrogen number density [cm^-3].
        Tg_map (np.ndarray): 2D map of gas temperature [K].
        colDen_map (np.ndarray): 2D map of total column density [cm^-2].

    Returns:
        np.ndarray: 2D map of the calculated CO (J=1-0) integrated brightness temperature [K km/s].
    """
    shape = nH_map.shape
    co_line_map = np.zeros(shape)

    print(f"\n--- Running DESPOTIC on a {shape[0]}x{shape[1]} map ---")
    start_time = time.time()

    
    for i in tqdm(range(shape[0]), desc="DESPOTIC Processing Rows"):
        for j in range(shape[1]):
            # --- 1.  DESPOTIC Inputs for each Cell:  ---
            
            cell = cloud()
            cell.nH = nH_map[i, j]
            cell.colDen = colDen_map[i, j]
            cell.Tg = Tg_map[i, j]
            
            
            # --- same constant for all cells ---
            cell.sigmaNT = 2.0e5
            cell.comp.xoH2 = 0.1
            cell.comp.xpH2 = 0.4
            cell.comp.xHe = 0.1

            cell.dust.alphaGD   = 3.2e-34    # Dust-gas coupling coefficient
            cell.dust.sigma10   = 2.0e-25    # Cross section for 10K thermal radiation
            cell.dust.sigmaPE   = 1.0e-21    # Cross section for photoelectric heating
            cell.dust.sigmaISRF = 3.0e-22    # Cross section to the ISRF
            cell.dust.beta      = 2.0        # Dust spectral index
            cell.dust.Zd        = 1.0        # Abundance relative to Milky Way
            cell.Td             = 10.0       # Dust temperature
            cell.rad.TCMB       = 2.73       # CMB temperature
            cell.rad.TradDust   = 0.0        # IR radiation field seen by the dust
            cell.rad.ionRate    = 2.0e-17    # Primary ionization rate
            cell.rad.chi        = 1.0        # ISRF normalized to Solar neighborhood


            cell.addEmitter("CO", 1.0e-4) # 添加CO作为发射物种

            # --- 2. Calculate  ---
            try:
                cell.setChemEq(network=NL99, evolveTemp='iterate')
                lines = cell.lineLum('CO')
                co_int_TB = lines[0]['intTB'] # J=1-0 线的积分亮度温度 [K km/s]
                co_line_map[i, j] = co_int_TB
            except Exception as e:
                # 如果计算失败，记录一个特殊值 (比如NaN)，并继续
                co_line_map[i, j] = np.nan 
                # print(f"Warning: Pixel ({i}, {j}) failed. Error: {e}") # 可以取消注释来调试
                continue

    end_time = time.time()
    total_time = end_time - start_time
    num_pixels = shape[0] * shape[1]
    time_per_pixel = total_time / num_pixels if num_pixels > 0 else 0

    print(f"\n--- DESPOTIC run complete ---")
    print(f"Processed {num_pixels} pixels in {total_time:.2f} seconds.")
    print(f"Average time per pixel: {time_per_pixel*1000:.2f} ms.")

    return co_line_map

def get_attenuation_factor(
        number_column_density , #N_H
        A_lambda_over_NH=8e-22, # (mag * cm^2) make it globe var

        ):
    
    # --- 1. 计算每个位置对应的总消光 A_lambda ---
    A_lambda_3d = number_column_density * A_lambda_over_NH

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

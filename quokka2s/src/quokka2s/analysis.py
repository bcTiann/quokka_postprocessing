from typing import Optional, Union
import time

import numpy as np
from despotic import cloud
from despotic.chemistry import NL99
from tqdm import tqdm
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg

from .despotic_tables import calculate_single_despotic_point
from .utils.axes import axis_index


def run_despotic_on_map(
    nH_map: np.ndarray,
    colDen_map: np.ndarray,
    Tg_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Iterates over a 2D slice and runs DESPOTIC on each pixel.

    Args:
        nH_map (np.ndarray): 2D map of Volume density of H nuclei [cm^-3].
        Tg_map (np.ndarray): 2D map of gas temperature [K].
        colDen_map (np.ndarray): 2D map of Column density of H nuclei [cm^-2].

    Returns:
        np.ndarray: 2D map of the calculated CO (J=1-0) integrated brightness temperature [K km/s].
    """
    shape = nH_map.shape
    print(f"nH_map.shape = {nH_map.shape}")

    co_line_map = np.zeros(shape)
    Tg_map_final = np.zeros(shape)

    print(f"\n--- Running DESPOTIC on a {shape[0]}x{shape[1]} map ---")
    start_time = time.time()

    Tg_working = []
    tg_guesses = [
        1958.3639773669447,
        1876.3811984430195,
        1904.6735038533895,
        2269.7464280911563,
        5529.099690585343,
        4467.07705368385,
        1921.7595997264211,
        1903.3063609939322,
        1907.7950789311126,
        1923.5653681769838,
        3128.178080797415,
        1921.9298726833636,
        1896.4011480919055,
        1909.3440156155311,
        13996.874562092222,
        3022.9801308742763,
        1907.308357562927,
        1909.4572682127873,
        2229.9653914481332,
        1947.0623389936507,
        2388.5525460121485,
        1901.115961026267,
        2393.6290610541246,
        2693.9753954376056,
        1907.308357562927,
        1907.308357562927,
        1906.8138296348143,
        2959.997148072065,
        208072.04383595262,
        503398.0480434042,
        6199404.503674414,
        1907.3083575629323,
        1951.2451612775776,
        1907.2675792788657,
        1907.3096438825598,
        2098.333042303612,
        37739.033264610196,
        25301.5132811833,
        1907.1986667866572,
        1906.0004035267134,
        1940.9177912829907,
        1907.2974533513202,
        1907.978989716906,
        1907.308433427053,
        1926.1281732989792,
        2043.475598201943,
        2061.8065832019997,
        1906.219433232716,
        34590.689516202736,
        24173.3696307906,
        1123275.1871095176,
        1998.2232263030933,
        1891.6056493165058,
        1908.05451292517,
        1909.1151153089127,
        1907.308357562927,
        889293.6619643234,
        67831.41869508532,
        69094.70519253935,
        2621.014000572547,
        2080.154582484991,
        1890.6529949503965,
        2618.834115138999,
        1907.3084022789865,
    ]
    working = 0
    for i in tqdm(range(shape[0]), desc="DESPOTIC Processing Rows"):
        for j in range(shape[1]):
            success = False
            for guess in tg_guesses:
                try:
                    cell = cloud()
                    cell.nH = nH_map[i, j]
                    cell.colDen = colDen_map[i, j]
                    cell.Tg = guess

                    cell.sigmaNT = 2.0e5
                    cell.comp.xoH2 = 0.1
                    cell.comp.xpH2 = 0.4
                    cell.comp.xHe = 0.1
                    cell.comp.mu = 0.6

                    cell.dust.alphaGD = 3.2e-34
                    cell.dust.sigma10 = 2.0e-25
                    cell.dust.sigmaPE = 1.0e-21
                    cell.dust.sigmaISRF = 3.0e-22
                    cell.dust.beta = 2.0
                    cell.dust.Zd = 1.0
                    cell.Td = 10.0
                    cell.rad.TCMB = 2.73
                    cell.rad.TradDust = 0.0
                    cell.rad.ionRate = 2.0e-17
                    cell.rad.chi = 1.0

                    cell.addEmitter("CO", 8.0e-9)
                    co_abundance = cell.emitters["CO"].abundance
                    print("++++++++++++++++++++++++\n")
                    print(f"initial CO abundance = {co_abundance}")
                    print(f"initial Tg = {cell.Tg}")
                    print("haven't pass")
                    cell.setChemEq(network=NL99, evolveTemp="iterate")
                    print("pass!!")
                    lines = cell.lineLum("CO")
                    co_int_TB = lines[0]["intTB"]

                    co_line_map[i, j] = co_int_TB
                    Tg_map_final[i, j] = cell.Tg

                    Tg_working.append(guess)

                    co_abundance = cell.emitters["CO"].abundance

                    print(f"after ChemEq CO abundance = {co_abundance}")
                    print(f"final Tg = {cell.Tg}")
                    print("++++++++++++++++++++++++\n")

                    success = True
                    print(f"guess T = {guess} successed at ({i}, {j})")
                    working += 1
                    break

                except Exception:
                    print(f"guess T = {guess} failed at ({i}, {j})")
                    continue

            if not success:
                print(f"Cell ({i}, {j}) failed for all guesses.")

    end_time = time.time()
    total_time = end_time - start_time
    num_pixels = shape[0] * shape[1]
    time_per_pixel = total_time / num_pixels if num_pixels > 0 else 0

    print(f"\n--- DESPOTIC run complete ---")
    print(f"Processed {num_pixels} pixels in {total_time:.2f} seconds.")
    print(f"Average time per pixel: {time_per_pixel*1000:.2f} ms.")
    print(f"Tg_working = {Tg_working}")
    print(f"working times = {working}")
    return co_line_map, Tg_map_final


def get_attenuation_factor(
    number_column_density,
    A_lambda_over_NH=8e-22,
):
    A_lambda_3d = number_column_density * A_lambda_over_NH

    print(f"Max A_lambda: {A_lambda_3d.max():.2f} mag")

    attenuation_factor_3d = 10.0 ** (-A_lambda_3d / 2.5)
    print(
        "Attenuation factor range: min="
        f"{attenuation_factor_3d.min():.2e}, max={attenuation_factor_3d.max():.2e}"
    )

    return attenuation_factor_3d



def along_sight_cumulation(
    data: np.ndarray,
    axis: Union[str, int],
    sign: str,
):
    """Cumulative sum along a requested axis and direction."""
    axis = axis_index(axis)

    if sign == "+":
        return np.flip(np.cumsum(np.flip(data, axis=axis), axis=axis), axis=axis)

    if sign == "-":
        return np.cumsum(data, axis=axis)

    raise ValueError("Direction must be '+' or '-'.")


def calculate_cumulative_column_density(
    density_3d: np.ndarray,
    dx_3d: np.ndarray,
    axis: Union[str, int],
    X_H: float,
    sign: str,
):
    """Calculates the cumulative hydrogen column density along a given axis."""
    m_H = mh.in_cgs()
    n_H_3d = (density_3d * X_H) / m_H
    N_H_cell_3d = n_H_3d * dx_3d

    N_H_cumulative = along_sight_cumulation(N_H_cell_3d, axis=axis, sign=sign)
    return N_H_cumulative


def calculate_attenuation(
    column_density_3d: np.ndarray,
    A_lambda_over_NH: float,
):
    """Calculates the dust attenuation factor from column density."""
    A_lambda_3d = column_density_3d * A_lambda_over_NH
    attenuation_factor = 10.0 ** (-A_lambda_3d / 2.5)
    return attenuation_factor, A_lambda_3d

# config.py

from matplotlib.colors import LogNorm
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg 

# --- Input/Output ---
YT_DATASET_PATH = "~/quokka_postprocessing/plt01000"
OUTPUT_DIR = "plots/"

# --- Physics Parameters / Model Assumptions ---
X_H = 0.76  # Mass fraction of Hydrogen
A_LAMBDA_OVER_NH = 4e-22 * cm**2  # Dust extinction cross-section (mag * cm^2 / N_H)

# --- Simulation Control ---
PROJECTION_AXIS = 'x'     # Axis for projection ('x', 'y', or 'z')

# --- Plotting Parameters ---
FIGURE_UNITS = 'pc'
CMAP = 'viridis'

# --- Analysis Tasks ---
# Define which analyses to run and their parameters
ANALYSES = {
    "halpha_no_dust": {
        "title": "H-alpha Emission (No Dust)",
        "filename": "halpha_no_dust.png",
        "cbar_label": "Surface Brightness (erg/s/cm$^2$)",
        "norm": LogNorm(),
        "enabled": True
    },
    "halpha_with_dust": {
        "title": "H-alpha Emission (With Dust Absorption)",
        "filename": "halpha_with_dust.png",
        "cbar_label": "Surface Brightness ({surface_brightness_with_dust.units})",
        "norm": LogNorm(),
        "enabled": True
    },
    "co_despotic": {
        "title": "CO (J=1-0) Emission (DESPOTIC)",
        "filename": "co_despotic_map.png",
        "cbar_label": "Integrated Brightness Temperature (K km/s)",
        "norm": None,  
        "enabled": True
    }
}



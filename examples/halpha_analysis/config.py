# config.py

from matplotlib.colors import LogNorm
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg 

# --- Input/Output ---
# YT_DATASET_PATH = "~/quokka_postprocessing/plt01000"
YT_DATASET_PATH = "~/quokka_postprocessing/examples/halpha_analysis/plt263168"
OUTPUT_DIR = "plots/"

# --- Physics Parameters / Model Assumptions ---
X_H = 0.74  # Mass fraction of Hydrogen
A_LAMBDA_OVER_NH = 4e-22 * cm**2  # Dust extinction cross-section (mag * cm^2 / N_H)

# --- Simulation Control ---
PROJECTION_AXIS = 'x'     # Axis for projection ('x', 'y', or 'z')

# --- Plotting Parameters ---
FIGURE_UNITS = 'pc'
CMAP = 'viridis'

# --- Analysis Tasks ---
# Define which analyses to run and their parameters
ANALYSES = {
    "density": {
        "title": "density projection along x",
        "filename": "density.png",
        "cbar_label": "density ($g/cm^{3}$)",
        "norm": LogNorm(),
        "enabled": True
    },
    "halpha_no_dust": {
        "title": "H-alpha (No Dust)",
        "filename": "halpha_no_dust.png",
        "cbar_label": "Surface Brightness (erg/s/cm$^2$)",
        "norm": LogNorm(),
        "enabled": True
    },
    "halpha_with_dust": {
        "title": "H-alpha (Dust)",
        "filename": "halpha_with_dust.png",
        "cbar_label": "Surface Brightness (erg/s/cm$^2$)",
        "norm": LogNorm(),
        "enabled": True
    },
    "co_despotic": {
        "title": "CO (J=1-0) (DESPOTIC)",
        "filename": "co_despotic_map_(2).png",
        "cbar_label": "Integrated Brightness Temperature (K km/s)",
        "norm": None,  
        "enabled": False
    }
}



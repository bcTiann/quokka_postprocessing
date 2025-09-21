# Quokka Post-processing Pipeline (quokka2s)

[![PyPI version](https://badge.fury.io/py/quokka2s.svg)](https://badge.fury.io/py/quokka2s)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python post-processing toolkit designed for the [QUOKKA](https://github.com/quokka-astro/quokka) radiation-magnetohydrodynamics (R-MHD) code. [cite_start]The primary goal of this project is to bridge the gap between theoretical simulation data and synthetic observational data, enabling direct comparisons between simulation results and real astronomical observations[cite: 1, 13].

This repository contains two main components:
1.  **`quokka2s`**: An installable Python library that provides core utilities for data handling, physics calculations, and visualization.
2.  **`/examples`**: A directory containing example scripts that use the `quokka2s` library to perform specific scientific analyses.

---

## Core Features

* [cite_start]**Convenient Data Interface**: Built on the powerful `yt` library, the `YTDataProvider` class makes it easy to extract slices, projections, and 3D grid data from QUOKKA simulation outputs[cite: 15, 82].
* **Physics Analysis Module**: Offers a suite of functions for common astrophysical calculations, such as cumulative column density along a line of sight and dust attenuation factors.
* **Publication-Quality Visualizations**: Includes a variety of plotting functions to rapidly generate high-quality figures, including single panels, multi-plot grids, and complex views with particle overlays and vector fields.
* **Modular Design**: A clean and organized code structure that separates data handling, analysis, and plotting concerns, making the library easy to understand, maintain, and extend.

## Installation

You can install the core `quokka2s` library directly from PyPI using `pip`:

```bash
pip install quokka2s
```

For the latest development version, or to run the example scripts, we recommend cloning this repository and installing in "editable" mode:

```bash
# 1. Clone the repository
git clone [https://github.com/bcTiann/quokka_postprocessing.git](https://github.com/bcTiann/quokka_postprocessing.git)

# 2. Navigate to the project directory
cd quokka_postprocessing

# 3. Install the quokka2s library in editable mode
pip install -e ./quokka2s
```
*The `-e` flag ensures that any changes you make to the library's source code are immediately reflected in your environment, which is ideal for development and testing.*

## Quick Start: Running the H-alpha Emission Analysis Example

This repository includes a complete example that calculates and visualizes H-alpha emission from simulation data, both with and without the effects of dust absorption.

#### **Prerequisites**

1.  Ensure you have installed the `quokka2s` library using one of the methods above.
2.  You will need access to a QUOKKA simulation output dataset (e.g., a `plt01000` directory).

#### **Running the Example**

1.  **Configure Paths**:
    Open the `examples/halpha_analysis/config.py` file and modify the `YT_DATASET_PATH` variable to point to your local QUOKKA data directory.

    ```python
    # in: examples/halpha_analysis/config.py
    YT_DATASET_PATH = "/path/to/your/plt01000"
    ```

2.  **Execute the Analysis Script**:
    Navigate to the example directory and run the main script.

    ```bash
    cd examples/halpha_analysis/
    python main_analysis.py
    ```

The script will automatically perform the following steps:
* [cite_start]Load the QUOKKA dataset and add custom derived fields (temperature, H-alpha luminosity)[cite: 83].
* [cite_start]Calculate the H-alpha surface brightness map without dust absorption[cite: 86].
* [cite_start]Calculate the H-alpha surface brightness map with dust absorption included[cite: 89].
* Generate a ratio map to visually demonstrate the effect of dust attenuation.

All generated figures will be saved to the `examples/halpha_analysis/plots/` directory by default.

## `quokka2s` Library API Usage

Here is a basic example of how to import and use the core components of the `quokka2s` library in your own script.

```python
import yt
import quokka2s as q2s
from matplotlib.colors import LogNorm

# 1. Load data and initialize the data provider
ds = yt.load("/path/to/your/simulation_data")
provider = q2s.YTDataProvider(ds)

# 2. Fetch Data (from the data_handling module)
# Get a projection of gas density along the z-axis
density_projection = provider.get_projection(field=('gas', 'density'), axis='z')

# 3. Perform Physics Analysis (from the analysis module)
# Assuming you have a 3D column density array N_H_3d
# attenuation_factor = q2s.calculate_attenuation(N_H_3d, ...)

# 4. Visualize the Result (from the plotting module)
plot_extent = provider.get_plot_extent(axis='z', units='kpc')
q2s.create_plot(
    data_2d=density_projection,
    title="Gas Density Projection",
    cbar_label="Column Density (g/cm^2)",
    filename="density_projection.png",
    extent=plot_extent,
    norm=LogNorm()
)
```

## Contributing

Contributions are welcome! If you find a bug or have a suggestion for a new feature, please feel free to open an Issue or submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).
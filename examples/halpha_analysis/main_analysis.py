"""Entry point for the H-alpha analysis pipeline."""
from __future__ import annotations

from pathlib import Path

import yt
import numpy as np
import os
import time
from matplotlib.colors import LogNorm

# --- Import our custom modules ---
import config as cfg
import quokka2s as q2s
import physics_models as phys
from quokka2s.pipeline.base import Pipeline, PipelineConfig
from quokka2s.pipeline.tasks import (
    DensityProjectionTask,
    HalphaComparisonTask,
    HalphaNoDustTask,
    HalphaWithDustTask,
)

from quokka2s.tables import load_table
from despotic_table_fields import compute_inputs, evaluate_table_fields



def _compute_shared_lognorm(*arrays):
    """Return a LogNorm spanning the positive values of all provided arrays."""
    valid_arrays = [arr for arr in arrays if arr is not None]
    if not valid_arrays:
        return None
    ref_units = valid_arrays[0].units
    positives = []
    for arr in valid_arrays:
        # Convert to common units, flatten, and keep positive finite values
        arr_vals = np.asarray(arr.to_value(ref_units)).ravel()
        finite_vals = arr_vals[np.isfinite(arr_vals)]
        finite_positive = finite_vals[finite_vals > 0]
        if finite_positive.size:
            positives.append(finite_positive)
    if not positives:
        return None
    combined = np.concatenate(positives)
    vmin = combined.min()
    vmax = combined.max()
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmin <= 0:
        vmin = combined[combined > 0].min()
    if vmin == vmax:
        vmax = vmin * (1 + 1e-9)
    return LogNorm(vmin=vmin, vmax=vmax)





def build_pipeline() -> Pipeline:
    """Configure and assemble the pipeline with the desired tasks."""
    pipeline_config = PipelineConfig(
        dataset_path=cfg.YT_DATASET_PATH,
        output_dir=Path(cfg.OUTPUT_DIR),
        figure_units="kpc",
        projection_axis="x",
        field_setup=phys.add_all_fields,
    )

    pipeline = Pipeline(pipeline_config)
    pipeline.register_task(DensityProjectionTask(pipeline_config, axis="x"))
    pipeline.register_task(HalphaNoDustTask(pipeline_config, axis="x"))
    pipeline.register_task(HalphaWithDustTask(pipeline_config, axis="x"))
    pipeline.register_task(HalphaComparisonTask(pipeline_config, axis="x"))
    return pipeline



def main() -> None:
    pipeline = build_pipeline()
    pipeline.run()



if __name__ == '__main__':
    start = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start
    print(f"\nTotal analysis time: {elapsed/60:.2f} minutes")

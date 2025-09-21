# main_analysis.py

import yt
import numpy as np
import os

# --- Import our custom modules ---
import config as cfg
import quokka2s as q2s
import physics_models as phys

def main():
    """Main function to run the H-alpha emission analysis."""
    
    # --- 1. Setup and Data Loading ---
    ds = yt.load(cfg.YT_DATASET_PATH)
    phys.add_all_fields(ds) # Add derived fields
    provider = q2s.YTDataProvider(ds)

    # Create output directory if it doesn't exist
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Map axis string to integer index for numpy
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    proj_axis_idx = axis_map[cfg.PROJECTION_AXIS]

    # --- 2. Get Base 3D Data from Simulation ---
    # These are needed for both analyses, so we get them once.
    print("Retrieving 3D data from simulation...")
    lum_3d = provider.get_grid_data(('gas', 'Halpha_luminosity'))
    rho_3d = provider.get_grid_data(('gas', 'density'))
    dx_3d = provider.get_grid_data(('boxlib', 'dx'))
    print("...3D data retrieval complete.")

    # --- 3. Run "No Dust" Analysis ---
    if cfg.ANALYSES["halpha_no_dust"]["enabled"]:
        print("\n--- Starting H-alpha analysis WITHOUT dust ---")
        params = cfg.ANALYSES["halpha_no_dust"]
        
        # Perform the integration (projection)
        surface_brightness_no_dust = np.sum(lum_3d * dx_3d, axis=proj_axis_idx)

        # Plot the result
        plot_extent = provider.get_plot_extent(axis=cfg.PROJECTION_AXIS, units=cfg.FIGURE_UNITS)
        q2s.create_plot(
            data_2d=surface_brightness_no_dust.T.to_ndarray(),
            title=params['title'],
            cbar_label=params['cbar_label'],
            filename=os.path.join(cfg.OUTPUT_DIR, params['filename']),
            extent=plot_extent,
            xlabel=f"{'XYZ'[proj_axis_idx-2]} ({cfg.FIGURE_UNITS})", # Clever way to get other axes
            ylabel=f"{'XYZ'[proj_axis_idx-1]} ({cfg.FIGURE_UNITS})",
            norm=params['norm']
        )

    # --- 4. Run "With Dust" Analysis ---
    if cfg.ANALYSES["halpha_with_dust"]["enabled"]:
        print("\n--- Starting H-alpha analysis WITH dust ---")
        params = cfg.ANALYSES["halpha_with_dust"]
        
        # Physics Calculation Steps
        N_H_3d = q2s.calculate_cumulative_column_density(rho_3d, dx_3d, axis=proj_axis_idx, X_H=cfg.X_H)
        
        # ======================= DEBUG STARTS HERE =======================
        print(f"Column density of H (N_H_3d):")
        print(f"  - Min: {N_H_3d.min():.2e} {N_H_3d.units}")
        print(f"  - Max: {N_H_3d.max():.2e} {N_H_3d.units}")
        print(f"  - Mean: {N_H_3d.mean():.2e} {N_H_3d.units}")
        # =================================================================


        attenuation_3d, A_lambda_3d= q2s.calculate_attenuation(N_H_3d, cfg.A_LAMBDA_OVER_NH)
        
        # ======================= DEBUG CONTINUES =======================
        print(f" (A_lambda_3d) (dimensionless):")
        print(f"  - Min: {A_lambda_3d.min():.2e}")
        print(f"  - Max: {A_lambda_3d.max():.2e}")
        print(f"  - Mean: {A_lambda_3d.mean():.2e}")

        print(f" (attenuation_3d) range (0 to 1):")
        print(f"  - Min: {attenuation_3d.min():.5f}") # 使用 .5f 看更精确的小数
        print(f"  - Max: {attenuation_3d.max():.5f}")
        print(f"  - Mean: {attenuation_3d.mean():.5f}")
        # ======================== DEBUG ENDS HERE ========================
        
        # Apply attenuation and perform integration
        L_attenuated_3d = lum_3d * attenuation_3d
        surface_brightness_with_dust = np.sum(L_attenuated_3d * dx_3d, axis=proj_axis_idx)
        
        # Plot the result
        plot_extent = provider.get_plot_extent(axis=cfg.PROJECTION_AXIS, units=cfg.FIGURE_UNITS)
        q2s.create_plot(
            data_2d=surface_brightness_with_dust.T.to_ndarray(),
            title=params['title'],
            cbar_label=params['cbar_label'],
            filename=os.path.join(cfg.OUTPUT_DIR, params['filename']),
            extent=plot_extent,
            xlabel=f"{'XYZ'[proj_axis_idx-2]} ({cfg.FIGURE_UNITS})",
            ylabel=f"{'XYZ'[proj_axis_idx-1]} ({cfg.FIGURE_UNITS})",
            norm=params['norm']
        )

        # --- 5. (New!) Create and Plot a Ratio Map to Visualize Attenuation ---
    if cfg.ANALYSES["halpha_no_dust"]["enabled"] and cfg.ANALYSES["halpha_with_dust"]["enabled"]:
        print("\n--- Creating a ratio map to visualize dust attenuation ---")

       
        # Add a small number to the denominator to prevent errors in empty regions
        epsilon = ds.quan(1e-30, surface_brightness_no_dust.units)
        # Add the two quantities with matching units
        ratio_map = surface_brightness_with_dust / (surface_brightness_no_dust + epsilon)


        # # In very optically thick regions, the ratio might be noisy if both numerator 
        # # and denominator are close to zero. We can clip the values to be between 0 and 1.
        # ratio_map = np.clip(ratio_map, 0, 1)

        # Get the plot extent again
        plot_extent = provider.get_plot_extent(axis=cfg.PROJECTION_AXIS, units=cfg.FIGURE_UNITS)

        # Plot the ratio map
        q2s.create_plot(
            data_2d=ratio_map.T,  # Use the calculated ratio map
            title="Dust Transmission (With Dust / Without Dust)",
            cbar_label="Fraction of Light Transmitted",
            filename=os.path.join(cfg.OUTPUT_DIR, "halpha_dust_ratio.png"),
            extent=plot_extent,
            xlabel=f"{'XYZ'[proj_axis_idx-2]} ({cfg.FIGURE_UNITS})",
            ylabel=f"{'XYZ'[proj_axis_idx-1]} ({cfg.FIGURE_UNITS})",
            norm=None,  # Use a LINEAR scale for the ratio map, not LogNorm!
            camp='viridis_r' # Use a reversed colormap so dense areas are dark
        )



if __name__ == '__main__':
    main()
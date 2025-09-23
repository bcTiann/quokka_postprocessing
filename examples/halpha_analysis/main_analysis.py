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

    # --- Initialize all possible result variables to None ---
    surface_brightness_no_dust = None
    surface_brightness_with_dust = None
    ratio_map = None
    co_map_K_kms = None
    
    # Map axis string to integer index for numpy
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    proj_axis_idx = axis_map[cfg.PROJECTION_AXIS]

    # --- 2. Get Base 3D Data from Simulation ---
    # These are needed for both analyses, so we get them once.
    print("Retrieving 3D data from simulation...")
    lum_3d = provider.get_grid_data(('gas', 'Halpha_luminosity'))
    rho_3d = provider.get_grid_data(('gas', 'density'))

    dx_3d = provider.get_grid_data(('boxlib', 'dx'))
    print(f"dx_3d.mean : {dx_3d.mean}")
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

    if cfg.ANALYSES["co_despotic"]["enabled"]:

        print("\n--- Starting CO Line analysis with DESPOTIC ---")
        params = cfg.ANALYSES["co_despotic"]


        print("Preparing 2D input maps for DESPOTIC...")
        
        # DESPOTIC , analysis on x center slice
        slice_axis = 'z'
        slice_coord = ds.domain_center[0]
        resolution = (15, 15)
        # 获取氢核数密度和气体温度的 2D 切片
        # Physics Calculation Steps
        # N_H_3d = q2s.calculate_cumulative_column_density(rho_3d, dx_3d, axis=proj_axis_idx, X_H=cfg.X_H)

        
        # # ======================= DEBUG STARTS HERE =======================
        # print(f"Column density of H (N_H_3d):")
        # print(f"N_H_3d shape:{ N_H_3d.shape }")
        # print(f"  - Min: {N_H_3d.min():.2e} {N_H_3d.units}")
        # print(f"  - Max: {N_H_3d.max():.2e} {N_H_3d.units}")
        # print(f"  - Mean: {N_H_3d.mean():.2e} {N_H_3d.units}")
        # # =================================================================

        nH_map = provider.get_slice(('gas', 'number_density'), 
                            axis=slice_axis, 
                            coord=slice_coord, 
                            resolution=resolution)
        
        Tg_map = provider.get_slice(('gas', 'temperature'), 
                            axis=slice_axis, 
                            coord=slice_coord,
                            resolution=resolution)
        
        # 
        # number_density_z = provider.get_slice(field=('gas', 'number_density'), 
        #                     axis=slice_coord, 
        #                     resolution=resolution)
        colDen_map = provider.get_projection(('gas', 'density'), 
                            axis=slice_axis,
                            resolution=resolution)
        
        print(f"colDen_map units: {colDen_map.units}")

        print("...input maps are ready.")
        print(f"Shape of nH_map: {nH_map.shape}")
        print(f"Shape of Tg_map: {Tg_map.shape}")
        print(f"Shape of colDen_map: {colDen_map.shape}")
        assert nH_map.shape == Tg_map.shape == colDen_map.shape, "Input maps must have the same shape!"

        # --- 5b. 调用我们的新函数运行计算 ---

        co_map_K_kms = q2s.run_despotic_on_map(
            nH_map.to_ndarray(), 
            Tg_map.to_ndarray(), 
            colDen_map.to_ndarray()
        )

        # --- 5c. 可视化结果 ---
        # 屏蔽计算失败的点 (我们之前设为了 NaN)
        co_map_masked = np.ma.masked_where(np.isnan(co_map_K_kms), co_map_K_kms)

        plot_extent = provider.get_plot_extent(axis=slice_axis, units=cfg.FIGURE_UNITS)
        q2s.create_plot(
            data_2d=co_map_masked.T, # .T to transpose for correct orientation
            title=params['title'],
            cbar_label=params['cbar_label'],
            filename=os.path.join(cfg.OUTPUT_DIR, params['filename']),
            extent=plot_extent,
            xlabel=f"X ({cfg.FIGURE_UNITS})",
            ylabel=f"Y ({cfg.FIGURE_UNITS})",
            norm=params['norm'],
            camp='viridis' 
        )


    # --- 6. Generate Combined Multi-plot Figure ---
    print("\n--- Generating combined multi-plot figure ---")
    
    # Create the list of dictionaries, one for each subplot
    plots_info = []
    
    if surface_brightness_no_dust is not None:
        params = cfg.ANALYSES["halpha_no_dust"]
        plots_info.append({
            'data': surface_brightness_no_dust.T.to_ndarray(),
            'title': params['title'],
            'cbar_label': params['cbar_label'],
            'norm': params['norm'],
        })

    if surface_brightness_with_dust is not None:
        params = cfg.ANALYSES["halpha_with_dust"]
        plots_info.append({
            'data': surface_brightness_with_dust.T.to_ndarray(),
            'title': params['title'],
            'cbar_label': "Surface Brightness (erg/s/cm$^2$)", # Using a fixed label for consistency
            'norm': params['norm'],
        })

    if ratio_map is not None:
        plots_info.append({
            'data': ratio_map.T,
            'title': "Dust Transmission Ratio",
            'cbar_label': "Fraction of Light Transmitted",
            'norm': None, # Linear scale for ratio map
        })

    # Get the shared plot extent and labels
    plot_extent = provider.get_plot_extent(axis=cfg.PROJECTION_AXIS, units=cfg.FIGURE_UNITS)
    xlabel = f"{'XYZ'[proj_axis_idx-2]} ({cfg.FIGURE_UNITS})"
    ylabel = f"{'XYZ'[proj_axis_idx-1]} ({cfg.FIGURE_UNITS})"

    # Make the single call to the plotting function
    q2s.create_horizontal_subplots(
        plots_info=plots_info,
        shared_extent=plot_extent,
        shared_xlabel=xlabel,
        shared_ylabel=ylabel,
        filename=os.path.join(cfg.OUTPUT_DIR, "halpha_analysis_combined.png")
    )


if __name__ == '__main__':
    main()
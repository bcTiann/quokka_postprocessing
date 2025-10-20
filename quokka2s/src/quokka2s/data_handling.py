# data_handling.py
import yt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from unyt import unyt_array

from .utils.axes import axis_index, axis_label

class YTDataProvider:
    def __init__(self, ds):
        self.ds = ds

    def get_slice(self,
                  field: Tuple[str, str],
                  axis: Union[str, int],
                  coord: Optional[float] = None,
                  resolution: Tuple[int, int] = (800, 800)) -> np.ndarray:
        """
        Get a slice of 2D YTNdarray for the specified field and axis.
        """
        axis_str = axis_label(axis)
        axis_idx = axis_index(axis)

        if coord is None:
            coord = self.ds.domain_center[axis_idx]

        full_domain_width = self.ds.domain_width
        plane_axes = [i for i in range(3) if i != axis_idx]
        
        slice_width = full_domain_width[plane_axes[0]]
        slice_height = full_domain_width[plane_axes[1]]

        slc = self.ds.slice(axis_str, coord=coord)

        frb = slc.to_frb(width=slice_width, height=slice_height, resolution=resolution)
        
        data_with_units = frb[field]
        # numpy_data = np.array(data_with_units)
        # unit_string = str(data_with_units.units)

        print("="*40)
        print(f"Slice: field = {field}, axis = {axis_str}, units = {data_with_units.units}")
        print("="*40)
        
        return data_with_units
    

    def get_grid_data(self,
                    field: Tuple[str, str],
                    level: Optional[int] = None,
                    dims: Tuple[int, int, int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 3D array of the entire dataset in [cgs units].
        Returns:
            - 3D YT Narray of the field data (with units)
            - 
        """
        
        if level is None:
            level = self.ds.max_level
        if dims is None:
            dims = self.ds.domain_dimensions * (2**level)
        

        grid = self.ds.covering_grid(level=level, left_edge=self.ds.domain_left_edge, dims=dims)
        data_with_units = grid[field].in_cgs()

        print(f"Retrieved 3D grid data for field '{field}' with shape {data_with_units.shape}")
        print(f"units: {data_with_units.units}")

        return data_with_units
    


    def downsample_3d_array(self,
                        data_cube: unyt_array,
                        factor: int
                        ) -> unyt_array:
        """
        Downsamples a 3D unyt_array by an integer factor by averaging blocks,
        preserving the units.

        Parameters:
        - data_cube: The input 3D unyt_array (e.g., shape (128, 128, 128)).
        - factor: The integer factor to downsample by (e.g., 2, 4, 8).
                The dimensions of the data_cube must be divisible by the factor.

        Returns:
        - The downsampled 3D unyt_array with the same units as the input.
        """

        orig_shape = np.array(data_cube.shape)

        if not np.all(orig_shape % factor == 0):
            raise ValueError(f"The shape of the data cube {orig_shape} is notdivisible by the facot {factor}.")
        
        new_shape = (orig_shape // factor).astype(int)

        reshaped_cube = data_cube.reshape(new_shape[0], factor,
                                        new_shape[1], factor,
                                        new_shape[2], factor)
        downsampled = reshaped_cube.mean(axis=(1, 3, 5))

        return downsampled



    def get_cubic_box(self,
                      field: Tuple[str, str],
                      box_width: Optional[unyt_array] = None,
                      center: Optional[unyt_array] = None,
                      level: Optional[int] = None
                      ):
        """
        Extracts a data box.
        """

        if center is None:
            center = self.ds.domain_center
            print(f"Center not provided. Using domain_center: {center}")

        if level is None:
            level = self.ds.max_level


        if box_width is None:
            min_side_length = self.ds.domain_width.min()
            box_width = self.ds.arr([min_side_length] * 3)
            print(f"Box width not provided. Defaulting to the largest possible width: {box_width}")
        
        half_width = box_width / 2.0
        left_edge = center - half_width
        right_edge = center + half_width
        print(f"Defining a physical box from {left_edge} to {right_edge}")

        pixel_widths = self.ds.domain_width / self.ds.domain_dimensions

        dims = np.round(box_width / pixel_widths).astype(int) * 2**level
        print(f"Calculated corresponding pixel dims: {dims}")

        box_region = self.ds.region(center, left_edge, right_edge)

        grid = self.ds.covering_grid(
            level=level,
            left_edge=left_edge,
            dims=dims,
            data_source=box_region
        )
        
        data_box = grid[field].in_cgs()
        print(f"Retrieved data box for field '{field}', with shape {data_box.shape}")

        extents = {
            'x': [left_edge[1], right_edge[1], left_edge[2], right_edge[2]],
            'y': [left_edge[0], right_edge[0], left_edge[2], right_edge[2]],
            'z': [left_edge[0], right_edge[0], left_edge[1], right_edge[1]]
        }

        return data_box, extents


    def get_slab_z(self,
                      field: Tuple[str, str],
                      slab_width: Optional[unyt_array] = None,
                      center: Optional[unyt_array] = None,
                      level: Optional[int] = None
                      ):
        """
        Extracts a data slab oriented along the Z-axis.
        The slab covers the full extent of the X and Y dimensions.
        The width of the slab in the Z direction is specified by the user.
        """

        if center is None:
            center = self.ds.domain_center
            print(f"Center not provided. Using domain_center: {center}")

        if level is None:
            level = self.ds.max_level


        if slab_width is None:
            slab_width = self.ds.domain_width[2]
            print(f"Slab width not provided. Defaulting to Full of Z-domain width: {slab_width}")
        
        left_edge_xy = self.ds.domain_left_edge[0:2]
        right_edge_xy = self.ds.domain_right_edge[0:2]

        half_width_z = slab_width / 2.0
        left_edge_z = center[2] - half_width_z
        right_edge_z = center[2] + half_width_z


        left_edge = self.ds.arr([left_edge_xy[0], left_edge_xy[1], left_edge_z])
        right_edge = self.ds.arr([right_edge_xy[0], right_edge_xy[1], right_edge_z])
        print(f"Defining a physical slab from {left_edge} to {right_edge}")
        
        pixel_widths = self.ds.domain_width / self.ds.domain_dimensions
        dims_xy = self.ds.domain_dimensions[0:2]

        num_z_pixels = np.round(slab_width / pixel_widths[2]).astype(int)

        dims = np.array([dims_xy[0], dims_xy[1], num_z_pixels]) * 2**level
        print(f"Calculated corresponding pixel dimensions: {dims}")

        box_region = self.ds.box(left_edge, right_edge)

        grid = self.ds.covering_grid(
            level=level,
            left_edge=left_edge,
            dims=dims,
            
            data_source=box_region
        )

        data_slab = grid[field].in_cgs()
        print(f"Retrieved data slab for field '{field}' with shape {data_slab.shape}")

        extents = {
            'x': [left_edge[1], right_edge[1], left_edge[2], right_edge[2]],
            'y': [left_edge[0], right_edge[0], left_edge[2], right_edge[2]],
            'z': [left_edge[0], right_edge[0], left_edge[1], right_edge[1]]
        }

        return data_slab, extents


    # def get_cubic_box_pixel(self,
    #                   field: Tuple[str, str],
    #                   box_size: Optional[int] = None,
    #                   box_center: Optional[Tuple[int, int, int]] = None):
    #     print("Step 1: Loading the entire grid into memory (this may take a while)...")
    #     full_grid_data = self.get_grid_data(field=field)
    #     nx, ny, nz = full_grid_data.shape

    #     if box_center is None:
    #         center_x_idx, center_y_idx, center_z_idx = nx // 2, ny // 2, nz // 2
    #         center = self.ds.domain_center
    #         print(f" center = {center}")
    #         print(f"box_center not provided, using geometric center ({center_x_idx}, {center_y_idx}, {center_z_idx}).")
    #     else: 
    #         center_x_idx, center_y_idx, center_z_idx = box_center
    #         print(f"Using user-provided box_center ({center_x_idx}, {center_y_idx}, {center_z_idx}).")

    #     print("++++++++++++++++++++++++DEBUG+++++++++++++++++++++")
    #     print(f"nx = {nx}")
    #     if box_size is None:
    #         box_size = np.min(full_grid_data.shape)
    #     print("++++++++++++++++++++++++DEBUG+++++++++++++++++++++")
    #     print(f"box_size = {box_size}")
    #     half_size = box_size // 2

    #     start_x = max(0, center_x_idx - half_size)
    #     end_x   = min(nx, center_x_idx + half_size)

    #     start_y = max(0, center_y_idx - half_size)
    #     end_y   = min(ny, center_y_idx + half_size)

    #     start_z = max(0, center_z_idx - half_size)
    #     end_z   = min(nz, center_z_idx + half_size)

    #     print(f"Step 3: Calculated slice indices: X({start_x}:{end_x}), Y({start_y}:{end_y}), Z({start_z}:{end_z})")
    #     center_cube_data = full_grid_data[start_x:end_x, start_y:end_y, start_z:end_z]
    #     print(f"--- Slicing complete. Final cube shape: {center_cube_data.shape} ---")
    #     extent = 0
    #     return center_cube_data, extent




    def get_projection(self,
                       field: Tuple[str, str],
                       axis: Union[str, int],
                       weight_field: Optional[Tuple[str, str]] = None,
                       resolution: Tuple[int, int] = (800, 800)) -> np.ndarray:
        """
        Get a projection of 2D Numpy array for the specified field and axis.
        """
        axis_str = axis_label(axis)
        axis_idx = axis_index(axis)
        
        full_domain_width = self.ds.domain_width
        plane_axes = [i for i in range(3) if i != axis_idx]
        
        proj_width = full_domain_width[plane_axes[0]]
        proj_height = full_domain_width[plane_axes[1]]
        
        prj = self.ds.proj(field, axis_str, weight_field=weight_field)

        frb = prj.to_frb(width=proj_width, height=proj_height, resolution=resolution)
        
        data_with_units = frb[field]
        # numpy_data = np.array(data_with_units)
        # unit_string = str(data_with_units.units)
        print("="*40)
        print(f"Slice: field = {field}, axis = {axis_str}, units = {data_with_units.units}")
        print("="*40)
        
        return data_with_units
    
    def get_plot_extent(self, axis: Union[str, int], units: str = 'pc') -> List[float]:
        """
        Get the physical extent of the plot for the specified axis.
        """
        axis_idx = axis_index(axis)
        axes = [i for i in range(3) if i != axis_idx]

        horizon_min, horizen_max = self.ds.domain_left_edge.in_units(units)[axes[0]].value, self.ds.domain_right_edge.in_units(units)[axes[0]].value
        vertical_min, vertical_max = self.ds.domain_left_edge.in_units(units)[axes[1]].value, self.ds.domain_right_edge.in_units(units)[axes[1]].value
        
        return [horizon_min, horizen_max, vertical_min, vertical_max]


    def get_particle_positions(self,
                               axis: Union[str, int],
                               depth: float,
                               coord: Optional[float] = None,
                               ptype: str = 'all',
                               units: str = 'pc'
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get particle positions within a slab for plotting on a slice.

        Args:
            axis: The axis PERPENDICULAR to the slice plane (e.g., 'x' for a y-z plot).
            depth: The thickness of the slab along the given axis.
            coord: The center of the slab along the given axis. Defaults to domain center.
            ptype: The particle type to select (e.g., 'all', 'io').
            units: The units for the returned positions.

        Returns:
            A tuple of two NumPy arrays: (particle_x_coords, particle_y_coords) 
            for the plotting plane.
        """
        axis_idx = axis_index(axis)

        if coord is None:
            coord = self.ds.domain_center[axis_idx].in_units(units).value


        # 1. Define the slab boundaries
        min_coord = self.ds.quan(coord - depth / 2.0, units)
        max_coord = self.ds.quan(coord + depth / 2.0, units)

        # 2. Create the geometric box (the "slab")
        # Start with the full domain edges
        left_edge = self.ds.domain_left_edge.copy()
        right_edge = self.ds.domain_right_edge.copy()

        # Modify the edges along the slice axis to define the slab's thickness
        left_edge[axis_idx] = min_coord
        right_edge[axis_idx] = max_coord
        
        # Create the data object representing only the data within this box
        slab_particles = self.ds.box(left_edge, right_edge)
        
        # -----------------------------------------------

        # Determine the axes for the plot
        plot_axes = [axis_label(i) for i in range(3) if i != axis_idx]

        # Get the particle positions for the plotting axes from the new slab object
        p_x = slab_particles[ptype, f'particle_position_{plot_axes[0]}'].in_units(units)
        p_y = slab_particles[ptype, f'particle_position_{plot_axes[1]}'].in_units(units)

        print("="*40)
        print(f"Particles: Found {len(p_x)} particles of type '{ptype}' in slab.")
        print("="*40)
        
        return p_x, p_y
    

    def get_velocity_field(self,
                           axis: Union[str, int],
                           resolution: Tuple[int, int] = (800, 800),
                           units: str = 'pc',
                           vel_units: str = 'km/s',
                           downsample_factor: int = 25
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets the 2D velocity field on a slice plane for quiver plots.

        Args:
            axis: The axis PERPENDICULAR to the slice plane.
            resolution: The resolution of the underlying grid.
            units: The spatial units for the arrow positions.
            vel_units: The units for the velocity vectors.
            downsample_factor: The factor by which to downsample the vector field to avoid overcrowding.

        Returns:
            A tuple of four 2D NumPy arrays: (X, Y, U, V)
        """
        axis_idx = axis_index(axis)

        # Determine the axes for the plot and corresponding velocity components
        plot_axes_indices = [i for i in range(3) if i != axis_idx]
        plot_axes_str = [axis_label(i) for i in plot_axes_indices]
        
        vel_u_field = ('gas', f'velocity_{plot_axes_str[0]}')
        vel_v_field = ('gas', f'velocity_{plot_axes_str[1]}')

        # get_slice  unyt_array
        u_data = self.get_slice(field=vel_u_field, axis=axis, resolution=resolution)
        v_data = self.get_slice(field=vel_v_field, axis=axis, resolution=resolution)


        extent = self.get_plot_extent(axis=axis, units=units)
        x_coords = np.linspace(extent[0], extent[1], resolution[0]) * yt.units.Unit(units)
        y_coords = np.linspace(extent[2], extent[3], resolution[1]) * yt.units.Unit(units)
        X, Y = np.meshgrid(x_coords, y_coords)

        # downsample
        skip = downsample_factor
        X_down = X[::skip, ::skip]
        Y_down = Y[::skip, ::skip]
        

        U_down = u_data[::skip, ::skip].in_units(vel_units)
        V_down = v_data[::skip, ::skip].in_units(vel_units)
        
        # unyt_array
        return X_down, Y_down, U_down, V_down

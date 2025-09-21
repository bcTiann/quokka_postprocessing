# data_handling.py
import yt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

class YTDataProvider:
    def __init__(self, ds):
        self.ds = ds
        self._axis_map = {'x': 0, 'y': 1, 'z': 2, 0:'x', 1:'y', 2:'z'}

    def get_slice(self,
                  field: Tuple[str, str],
                  axis: Union[str, int],
                  coord: Optional[float] = None,
                  resolution: Tuple[int, int] = (800, 800)) -> np.ndarray:
        """
        Get a slice of 2D YTNdarray for the specified field and axis.
        """
        axis_str = self._axis_map[axis] if isinstance(axis, int) else axis
        axis_index = self._axis_map[axis_str]

        if coord is None:
            coord = self.ds.domain_center[axis_index]

        full_domain_width = self.ds.domain_width
        plane_axes = [i for i in range(3) if i != axis_index]
        
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
                    level: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 3D array of the entire dataset in [cgs units].
        Returns:
            - 3D NumPy array of the field data.
            - string of units
        """

        if level is None:
            level = self.ds.max_level

        dims = self.ds.domain_dimensions * (2**level)
        
        grid = self.ds.covering_grid(level=level, left_edge=self.ds.domain_left_edge, dims=dims)
        data_with_units = grid[field].in_cgs()
        print(f"Retrieved 3D grid data for field '{field}' with shape {data_with_units.shape}")
        print(f"units: {data_with_units.units}")
        # numpy_data = data_with_units.to_ndarray()
        # unit_string = str(data_with_units.units)
        # print(f"Retrieved 3D grid data for field '{field}' with shape {numpy_data.shape}")
        # print(f"units: {unit_string}")
        return data_with_units

    


    def get_projection(self,
                       field: Tuple[str, str],
                       axis: Union[str, int],
                       weight_field: Optional[Tuple[str, str]] = None,
                       resolution: Tuple[int, int] = (800, 800)) -> np.ndarray:
        """
        Get a projection of 2D Numpy array for the specified field and axis.
        """
        axis_str = self._axis_map[axis] if isinstance(axis, int) else axis
        axis_index = self._axis_map[axis_str]
        
        full_domain_width = self.ds.domain_width
        plane_axes = [i for i in range(3) if i != axis_index]
        
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
        axis_str = self._axis_map[axis] if isinstance(axis, int) else axis
        axis_index = self._axis_map[axis_str]
        axes = [i for i in range(3) if i != axis_index]

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
        axis_str = self._axis_map[axis] if isinstance(axis, int) else axis
        axis_index = self._axis_map[axis_str]

        if coord is None:
            coord = self.ds.domain_center[axis_index].in_units(units).value


        # 1. Define the slab boundaries
        min_coord = self.ds.quan(coord - depth / 2.0, units)
        max_coord = self.ds.quan(coord + depth / 2.0, units)

        # 2. Create the geometric box (the "slab")
        # Start with the full domain edges
        left_edge = self.ds.domain_left_edge.copy()
        right_edge = self.ds.domain_right_edge.copy()

        # Modify the edges along the slice axis to define the slab's thickness
        left_edge[axis_index] = min_coord
        right_edge[axis_index] = max_coord
        
        # Create the data object representing only the data within this box
        slab_particles = self.ds.box(left_edge, right_edge)
        
        # -----------------------------------------------

        # Determine the axes for the plot
        plot_axes = [self._axis_map[i] for i in range(3) if i != axis_index]

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
        axis_str = self._axis_map[axis] if isinstance(axis, int) else axis
        axis_index = self._axis_map[axis_str]

        # Determine the axes for the plot and corresponding velocity components
        plot_axes_indices = [i for i in range(3) if i != axis_index]
        plot_axes_str = [self._axis_map[i] for i in plot_axes_indices]
        
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



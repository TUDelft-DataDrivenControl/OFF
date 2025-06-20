# Copyright (C) <2024>, M Becker (TUDelft), M Lejeune (UCLouvain)

# List of the contributors to the development of OFF: see LICENSE file.
# Description and complete License: see LICENSE file.
	
# This program (OFF) is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program (see COPYING file).  If not, see <https://www.gnu.org/licenses/>.

"""Wake solver classes, used to call the underlying parametric models."""
import numpy as np
from abc import ABC, abstractmethod
import off.windfarm as wfm
import off.wake_model as wm
import off.utils as ot
import copy
from os import path
import logging
import matplotlib.pyplot as plt

lg = logging.getLogger(__name__)


class WakeSolver(ABC):
    
    settings_sol: dict
    settings_vis: dict
    _flag_plot_wakes: bool
    _flag_plot_tile: bool
    _flow_field_x: np.ndarray
    _flow_field_y: np.ndarray
    _flow_field_z: np.ndarray
    _flow_field_u: np.ndarray

    def __init__(self, settings_sol: dict, settings_vis: dict):
        """
        Object to connect OFF to the wake model.
        The common interface is the get_wind_speeds function.

        Parameters
        ----------
        settings_sol: dict
            Wake solver settings
        """
        self.settings_sol = settings_sol
        self.settings_vis = settings_vis
        self._flag_plot_wakes = False
        self._flag_plot_tile  = False
        self._flag_plot_OP_mountain = False
        
        lg.info('Wake solver settings:')
        lg.info(settings_sol)

    @abstractmethod
    def get_measurements(self, i_t: int, wind_farm: wfm.WindFarm) -> tuple:
        """
        Get the wind speed at the location of all OPs and the rotor plane of turbine with index i_t

        Parameters
        ----------
        i_t : int
            index of the turbine to find the wind speeds for
        wind_farm
            influencing wind farm

        Returns
        -------
        tuple(np.ndarray, np.ndarray, nd.array)
            [u,v] wind speeds at the rotor plane (entry 1) and OPs (entry 2)
            m further measurements, depending on the used wake model
        """

    def raise_flag_plot_wakes(self):
        """
        Raises a flag to plot the wakes duing the next call of the wake model
        """
        self._flag_plot_wakes = True

    def _lower_flag_plot_wakes(self):
        """
        Lowers the flag to plot the wakes after they have been plotted
        """
        self._flag_plot_wakes = False

    def raise_flag_plot_tile(self, x:np.ndarray, y:np.ndarray, z:np.ndarray):
        """
        Raises a flag to plot the TWF tile during the next call of the wake model

        Parameters
        ----------
        x : np.ndarray
            x coordinates of the points to visualize
        y : np.ndarray
            y coordinates of the points to visualize
        z : np.ndarray
            z coordinates of the points to visualize
        """
        self._flag_plot_tile = True
        self._flow_field_x = x
        self._flow_field_y = y
        self._flow_field_z = z

    def _lower_flag_plot_tile(self):
        """
        Lowers the flag to plot the tile after it has been plotted
        """
        self._flag_plot_tile = False

    def raise_flag_plot_OP_mountain(self, x:np.ndarray, y:np.ndarray, z:np.ndarray):
        """
        Raises a flag to plot a mountain range like visualization of the wind speed along the OP location during the next call of the wake model

        Parameters
        ----------
        x : np.ndarray
            x coordinates of the points to visualize
        y : np.ndarray
            y coordinates of the points to visualize
        z : np.ndarray
            z coordinate of the OP
        """
        self._flag_plot_OP_mountain = True
        self._OP_mountain_x = x
        self._OP_mountain_y = y
        self._OP_mountain_z = z
        self._mountain_u = np.zeros(x.shape)

    def _lower_flag_plot_OP_mountains(self):
        """
        Lowers the flag to plot the mountain range after it has been plotted
        """
        self._flag_plot_OP_mountain = False

    def get_OP_mountain_u(self):
        """
        Returns the |u| component of the flow field
        """
        return self._mountain_u

    def get_tile_u(self):
        """
        Returns the |u| component of the flow field
        """
        return self._flow_field_u
    
    def vis_turbine_eff_wind_speed_field(self, wind_farm: wfm.WindFarm, sim_dir, t):
        """
        Moves a dummy turbine around to extract the effective wind speed at a given grid

        Parameters
        ----------
        wind_farm: wfm.WindFarm
            Object containing all real wind turbines

        """
        # Create meshgrid according to settings
        grid_x, grid_y = np.meshgrid(
            np.linspace(
                self.settings_vis["grid"]["boundaries"][0][0],
                self.settings_vis["grid"]["boundaries"][0][1],
                self.settings_vis["grid"]["resolution"][0]),
            np.linspace(
                self.settings_vis["grid"]["boundaries"][1][0],
                self.settings_vis["grid"]["boundaries"][1][1],
                self.settings_vis["grid"]["resolution"][1]))

        if self.settings_vis["grid"]["unit"][0] == 'D':
            grid_x = grid_x * self.settings_vis["grid"]["diameter"][0]
            grid_y = grid_y * self.settings_vis["grid"]["diameter"][0]

        grid_u_eff = np.zeros(shape=grid_x.shape)

        # Create measurement turbine
        #   Copy first turbine
        measurement_turbine = copy.deepcopy(wind_farm.turbines[0])  # TODO: Probably creates an unwanted reference here, not copy
        #   Move turbine to grid point
        measurement_turbine.base_location = np.array([grid_x[0, 0], grid_y[0, 0], 0])

        #   Set the yaw angle of the Turbine to 0
        # TODO: The wind direction at the grid point should be a product of the surrounding particles or of an external
        #  source, the current solution assumes a uniform wind direction:
        wind_dir_at_grid_point = measurement_turbine.ambient_states.get_turbine_wind_dir()
        measurement_turbine.set_yaw(wind_dir_at_grid_point, 0.0)

        # Add measurement turbine to the wind farm
        ind_m_turbine = wind_farm.add_turbine(measurement_turbine)

        # For loop moving the turbine through all grid points and storing the effective wind speed
        # TODO: This would be a prime spot for a parallelized  approach
        for (row_index, col_index), x in np.ndenumerate(grid_x):
            wind_farm.turbines[ind_m_turbine].base_location = np.array([x, grid_y[row_index, col_index], 0])
            u_rp, measurements = self._get_wind_speeds_rp(ind_m_turbine, wind_farm)
            grid_u_eff[row_index, col_index] = ot.ot_uv2abs(u_rp[0], u_rp[1])

        # Remove the measurement turbine again from the wind farm
        wind_farm.rmv_turbine(ind_m_turbine)

        # Plot the flow field if desired
        if self.settings_vis["debug"]["turbine_effective_wind_speed_plot"]:
            fig, axs = plt.subplots()
            axs.axis('equal')

            if self.settings_vis["grid"]["unit"][0] == 'D':
                plt.contourf(grid_x/self.settings_vis["grid"]["diameter"][0],
                             grid_y/self.settings_vis["grid"]["diameter"][0], grid_u_eff, 20)
                plt.xlabel('Distance (D)')
                plt.ylabel('Distance (D)')
            else:
                plt.contourf(grid_x, grid_y, grid_u_eff, 20)
                plt.xlabel('Distance (m)')
                plt.ylabel('Distance (m)')

            plt.colorbar()
            plt.title('Turbine effective wind speed (m/s)')

            # Add observation points
            if self.settings_vis["debug"]["turbine_effective_wind_speed_plot_ops"]:
                coord = wind_farm.get_op_world_coordinates()
                if self.settings_vis["grid"]["unit"][0] == 'D':
                    plt.scatter(coord[:, 0]/self.settings_vis["grid"]["diameter"][0],
                                coord[:, 1]/self.settings_vis["grid"]["diameter"][0], color='white', s=5)
                else:
                    plt.scatter(coord[:, 0], coord[:, 1], color='white', s=10)

            # Enforce axis limits
            axs.set(xlim=(self.settings_vis["grid"]["boundaries"][0][0], self.settings_vis["grid"]["boundaries"][0][1]),
                    ylim=(self.settings_vis["grid"]["boundaries"][1][0], self.settings_vis["grid"]["boundaries"][1][1]))

            if self.settings_vis["debug"]["turbine_effective_wind_speed_plot"]:
                plt.savefig(sim_dir + "/turbine_effective_wind_speed_at_" + str(int(t)).zfill(6) + "s.png")

            if self.settings_vis["debug"]["turbine_effective_wind_speed_plot"]:
                plt.show()

        if self.settings_vis["debug"]["turbine_effective_wind_speed_store_data"]:
            np.savetxt(sim_dir + "/turbine_effective_wind_speed_at_" + str(int(t)).zfill(6) + "s.csv",
                       grid_u_eff, delimiter=',')
            if not path.exists(sim_dir + "/turbine_effective_wind_speed_x_grid.csv"):
                np.savetxt(sim_dir + "/turbine_effective_wind_speed_x_grid.csv",
                           grid_x, delimiter=',')
                np.savetxt(sim_dir + "/turbine_effective_wind_speed_y_grid.csv",
                           grid_y, delimiter=',')
                
    def _get_wind_speeds_location(self, loc: np.ndarray, wind_farm: wfm.WindFarm) -> np.ndarray:
        """
        Calculates the effective wind speed at a location
        Parameters
        ----------
        loc : np.ndarray
            x,y,z coordinates with which respect to the effective wind speed should be derived
        wind_farm
            influencing wind farm

        Returns
        -------
        nd.array
            abs wind speed at location
        """
        raise NotImplementedError("_get_wind_speeds_location() is not implemented in the base class, please implement it in the derived class.")
    
    def vis_OP_mountains(self, wind_farm: wfm.WindFarm, sim_dir, t):
        """
        Goes through all OPs and plots the wind speed along the OP location
        Creates a plot like this ASCI art:
        |    |    |   |   |   |
         `-.  `\.  \  \   |   |
            )    )  )  )   )  |
         ,-'  ,/´  /  /   |   |
        |    |    |   |   |   |

        Parameters
        ----------
        wind_farm: wfm.WindFarm
            Object containing all real wind turbines
        sim_dir: str
            Directory to save the plots
        t: float
            Time of the simulation
        """

        # create an empty list to store the x,y,u values
        data_x = np.array([])
        data_y = np.array([])
        if self.settings_vis["flow_field_plots"]["mountains_3d"]:
            data_z = np.array([])
        data_u = np.array([])
        data_v = np.array([])

        # Get the scale of the grid
        scale_grid = 1
        if self.settings_vis["grid"]["unit"][0] == 'D':
            # Scale the coordinates to the grid size
            scale_grid = self.settings_vis["grid"]["diameter"][0]

        # If the grid starts at 0, FLORIS throws errors, thus we need a small offset to the floor
        floor_offset = 0
        if self.settings_vis["grid"]["boundaries"][2][0] == 0:
            floor_offset = 1

        # Go through all Turbines
        for  i_t, tur in enumerate(wind_farm.turbines):
            # Get all OP coordinates of the turbine
            op_coord = tur.observation_points.get_world_coord()

            # Go through the list of all OPs
            for i in range(
                self.settings_vis["flow_field_plots"]["mountains_offset"],
                len(op_coord),
                self.settings_vis["flow_field_plots"]["mountains_stride"]):

                # Get the x,y coordinates of the OP
                x = op_coord[i, 0]
                y = op_coord[i, 1]
                    
                # Check if OP is outside the grid ± one diameter
                if (x < self.settings_vis["grid"]["boundaries"][0][0] * scale_grid - self.settings_vis["grid"]["diameter"][0] or
                    x > self.settings_vis["grid"]["boundaries"][0][1] * scale_grid + self.settings_vis["grid"]["diameter"][0] or
                    y < self.settings_vis["grid"]["boundaries"][1][0] * scale_grid - self.settings_vis["grid"]["diameter"][0] or
                    y > self.settings_vis["grid"]["boundaries"][1][1] * scale_grid + self.settings_vis["grid"]["diameter"][0]):
                    continue

                # Get wind direction at the OP
                wind_dir_at_op = ot.ot_deg2rad(tur.ambient_states.get_wind_dir_ind(i))

                # Generate a line of points orthogonal to the wind direction or generate a 3D meshgrid from the floor to the upper boundary of the domain
                if self.settings_vis["flow_field_plots"]["mountains_3d"]:
                    # Generate 3D meshgrid
                    mesh_width = 4 * self.settings_vis["grid"]["diameter"][0] # TODO width hardcoded

                    

                    mesh_x, mesh_z = np.meshgrid(
                        np.linspace(- mesh_width / 2,
                                    mesh_width / 2,
                                    num=101),
                        np.linspace(self.settings_vis["grid"]["boundaries"][2][0] * scale_grid + floor_offset,
                                    self.settings_vis["grid"]["boundaries"][2][1] * scale_grid,
                                    num=101))
                    
                    mesh_y = mesh_x * np.cos(wind_dir_at_op) + y
                    mesh_x = mesh_x * np.sin(wind_dir_at_op) + x

                    mesh_x = mesh_x.flatten()
                    mesh_y = mesh_y.flatten()
                    mesh_z = mesh_z.flatten()

                    # Pass meshgrid to raise_flag_plot_OP_mountain
                    self.raise_flag_plot_OP_mountain(mesh_x, mesh_y, mesh_z)
                    # Get wind speed at the meshgrid points
                    self._get_wind_speeds_location(op_coord[i,:], wind_farm)
                    uv_mesh = ot.ot_abs2uv(self._mountain_u_abs[0], tur.ambient_states.get_wind_dir_ind(i))

                    if data_x.size == 0:
                        data_x = mesh_x[np.newaxis, :]
                        data_y = mesh_y[np.newaxis, :]
                        data_z = mesh_z[np.newaxis, :]
                        data_u = uv_mesh[0, :]
                        data_v = uv_mesh[1, :]
                    else:
                        # Concatenate the data
                        data_x = np.vstack((data_x, mesh_x))
                        data_y = np.vstack((data_y, mesh_y))
                        data_z = np.vstack((data_z, mesh_z))
                        data_u = np.vstack((data_u, uv_mesh[0, :]))
                        data_v = np.vstack((data_v, uv_mesh[1, :]))
                    
                                    
                else:
                    line_length = 4 * self.settings_vis["grid"]["diameter"][0] # TODO Line width hardcoded
                    line_x = np.linspace(x - line_length * np.sin(wind_dir_at_op) / 2,
                                        x + line_length * np.sin(wind_dir_at_op) / 2,
                                        num=101)
                    line_y = np.linspace(y + line_length * np.cos(wind_dir_at_op) / 2,
                                        y - line_length * np.cos(wind_dir_at_op) / 2,
                                        num=101)
                
                    # Get wind speed at the line points
                    #u_line = np.zeros(line_x.shape)                                    TODO: This is not used, remove it?
                    self.raise_flag_plot_OP_mountain(line_x, line_y, op_coord[i, 2])
                    self._get_wind_speeds_location(op_coord[i,:], wind_farm)
                    uv_line = ot.ot_abs2uv(self._mountain_u_abs[0], tur.ambient_states.get_wind_dir_ind(i))
                
                    # Attach the data to the list
                    if data_x.size == 0:
                        data_x = line_x[np.newaxis, :]
                        data_y = line_y[np.newaxis, :]
                        data_u = uv_line[0, :]
                        data_v = uv_line[1, :]
                    else:
                        # Concatenate the data
                        data_x = np.vstack((data_x, line_x))
                        data_y = np.vstack((data_y, line_y))
                        data_u = np.vstack((data_u, uv_line[0, :]))
                        data_v = np.vstack((data_v, uv_line[1, :]))

        # Store the data in a csv file
        np.savetxt(sim_dir + "/mountain_plot_x_" + str(int(t)).zfill(6) + "s.csv",
                       data_x, delimiter=',')
        np.savetxt(sim_dir + "/mountain_plot_y_" + str(int(t)).zfill(6) + "s.csv",
                       data_y, delimiter=',')
        np.savetxt(sim_dir + "/mountain_plot_u_" + str(int(t)).zfill(6) + "s.csv",
                       data_u, delimiter=',')
        np.savetxt(sim_dir + "/mountain_plot_v_" + str(int(t)).zfill(6) + "s.csv",
                       data_v, delimiter=',')
        
        
        # Don't plot if the 3d data has been collected
        if self.settings_vis["flow_field_plots"]["mountains_3d"]:
            np.savetxt(sim_dir + "/mountain_plot_z_" + str(int(t)).zfill(6) + "s.csv",
                       data_z, delimiter=',')
            return 
        
        # Plot data as line plot
        fig, ax = plt.subplots()

        max_u = np.max(data_u)
        max_v = np.max(data_v)
        amplification_factor = 10.0 #m/(m/s)
        #ax.plot(data_x.flatten(), data_y.flatten(), 'o', markersize=2, color='lightgrey')
        for i in range(0,data_x.shape[0]):
            ax.fill(np.hstack((data_x[i, :] + (max_u - data_u[i, :]) * amplification_factor, data_x[i, ::-1])),
                    np.hstack((data_y[i, :] + (max_v - data_v[i, :]) * amplification_factor, data_y[i, ::-1])), 
                    color='#0c2340', alpha=0.5, edgecolor='none')#'#0c2340')
            
        ax.plot(data_x[:,data_x.shape[1]//2], data_y[:,data_x.shape[1]//2], 'o', markersize=2, color='#ec6842')

        # Plot yawed turbines
        for i_t, tur in enumerate(wind_farm.turbines):
            # Get the turbine position
            x = tur.base_location[0]
            y = tur.base_location[1]
            # Get the yaw angle
            yaw = ot.ot_deg2rad(tur.get_yaw_orientation())
            
            # Plot the turbine as a line
            ax.plot([x - 0.5 * tur.diameter * np.sin(yaw), x + 0.5 * tur.diameter * np.sin(yaw)],
                    [y + 0.5 * tur.diameter * np.cos(yaw), y - 0.5 * tur.diameter * np.cos(yaw)],
                    color='black', linewidth=1.5)
            
            ax.plot([x, x + 0.2 * tur.diameter * np.cos(yaw)],
                    [y, y + 0.2 * tur.diameter * np.sin(yaw)],
                    color='black', linewidth=1.5)
        
        #ax.tricontourf(data_x.flatten(), 
        #              data_y.flatten(), 
        #              np.sqrt(data_u.flatten()**2 + data_v.flatten()**2), 
        #              levels=np.linspace(0, 9, 10))
        
        ax.set_aspect('equal')
        ax.set_title('Wind speed along OPs')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_xlim(self.settings_vis["grid"]["boundaries"][0][0] * scale_grid, self.settings_vis["grid"]["boundaries"][0][1] * scale_grid)
        ax.set_ylim(self.settings_vis["grid"]["boundaries"][1][0] * scale_grid, self.settings_vis["grid"]["boundaries"][1][1] * scale_grid)
        plt.savefig(sim_dir + "/mountain_plot_at_" + str(int(t)).zfill(6) + "s.png")
        #plt.show()

        #print("Plot done!")


        """
        fig, ax = plt.subplots()
        for i in range(data_x.shape[0]):
            ax.plot(data_x[i, :] + (8-data_u), data_y[i, :] + (8-data_v), color='blue', alpha=0.5)
        
        ax.set_aspect('equal')
        ax.set_title('Wind speed along OPs')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_xlim(self.settings_vis["grid"]["boundaries"][0][0] * scale_grid, self.settings_vis["grid"]["boundaries"][0][1] * scale_grid)
        ax.set_ylim(self.settings_vis["grid"]["boundaries"][1][0] * scale_grid, self.settings_vis["grid"]["boundaries"][1][1] * scale_grid)
        plt.savefig(sim_dir + "/mountain_plot_at_" + str(int(t)).zfill(6) + "s.png")
        plt.show()

        print("Plot done!")
        """


class TWFSolver(WakeSolver):
    floris_wake: wm.WakeModel

    def __init__(self, settings_wke: dict, settings_sol: dict, settings_vis: dict):
        """
        FLORIDyn temporary wind farm wake solver, based on [1].

        Parameters
        ----------
        settings_wke: dict
            Wake settings, including all parameters the wake needs to run
        settings_sol: dict
            Wake solver settings
        settings_vis: dict
            Visualization settings
        """
        super(TWFSolver, self).__init__(settings_sol, settings_vis)
        lg.info('FLORIDyn TWF solver created.')

        if settings_sol["wake_model"].startswith("FLORIS"):
            self.floris_wake = wm.Floris4Wake(settings_wke, np.array([]), np.array([]), np.array([]))
        elif settings_sol["wake_model"] == "PythonGaussianWake":
            self.floris_wake = wm.PythonGaussianWake(settings_wke, np.array([]), np.array([]), np.array([]))
        else:
            raise ImportError('Wake model unknown!')

    def get_measurements(self, i_t: int, wind_farm: wfm.WindFarm) -> tuple:
        """
        Get the wind speed at the location of all OPs and the rotor plane of turbine with index i_t

        Parameters
        ----------
        i_t : int
            index of the turbine to find the wind speeds for
        wind_farm
            influencing wind farm

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            [u,v] wind speeds at the rotor plane (entry 1) and OPs (entry 2)
        """
        u_rp, m = self._get_wind_speeds_rp(i_t, wind_farm)
        u_op = self._get_wind_speeds_op(i_t, wind_farm)
        return u_rp, u_op, m

    def _get_wind_speeds_rp(self, i_t: int, wind_farm: wfm.WindFarm) -> tuple:
        """
        Calculates the effective wind speed at the rotor plane of turbine i_t
        Parameters
        ----------
        i_t : int
            index of the turbine to find the wind speeds for
        wind_farm
            influencing wind farm

        Returns
        -------
        nd.array
            [u,v] wind speeds at the rotor plane
        """
        # Load current data
        wind_farm_layout = wind_farm.get_layout()

        # Create an index range over all nT turbines and only select the ones saved in the dependencies
        inf_turbines = np.arange(wind_farm.nT)[wind_farm.dependencies[i_t, :]]
        # index i_t is not correct anymore as only a subset of turbines are considered
        i_t_tmp = np.sum(wind_farm.dependencies[i_t, 0:i_t])

        twf_layout = np.zeros((inf_turbines.shape[0], 4))  # Allocation of x, y, z coordinates of the turbines
        twf_t_states = []
        twf_a_states = []

        # Get reference point of main wind turbine
        rotor_center_i_t = wind_farm.turbines[i_t].get_rotor_pos()

        # Go through dependencies
        for idx in np.arange(inf_turbines.shape[0]):
            if idx == i_t_tmp:
                # Turbine itself
                twf_layout[idx, :] = wind_farm_layout[i_t_tmp, :]
                twf_t_states.append(wind_farm.turbines[i_t_tmp].turbine_states.create_interpolated_state(0, 1, 0, 1))
                twf_a_states.append(wind_farm.turbines[i_t_tmp].ambient_states.create_interpolated_state(0, 1, 0, 1))
                continue

            lg.debug('Ambient states: Two OP interpolation')
            # Interpolation of turbine states
            #   Step 1 retrieve closest up and downstream OPs
            op_locations = wind_farm.turbines[inf_turbines[idx]].observation_points.get_world_coord()
            ind_op = ot.ot_get_closest_2_points_3d_sorted(rotor_center_i_t, op_locations)

            #   Step 2 calculate interpolation weights
            point_a = op_locations[ind_op[0], 0:2].transpose()
            point_b = op_locations[ind_op[1], 0:2].transpose()
            point_c = rotor_center_i_t[0:2].transpose()

            weight_d = ((point_b - point_a) @ (point_c - point_a)) / ((point_b - point_a) @ (point_b - point_a))

            # Logging Interpolation OPs
            lg.info('2 OP interpolation: T%s influence on T%s, OP1 (index: %s, loc: %s), OP2 (index: %s, loc: %s)' %
                    (inf_turbines[idx], i_t, ind_op[0], point_a, ind_op[1], point_b))
            lg.info('TWF - OP interpolation weight (should be between 0 and 1): %s' % weight_d)
            weight_d = np.fmin(np.fmax(weight_d, 0), 1)
            lg.info('TWF - Used OP interpolation weight:  %s' % weight_d)

            r0 = 1 - weight_d
            r1 = weight_d

            #   Interpolate states
            #       1. OP location
            tmp_op = op_locations[ind_op[0], 0:3] * r0 + op_locations[ind_op[1], 0:3] * r1
            #       2. Ambient
            twf_a_states.append(wind_farm.turbines[idx].ambient_states.create_interpolated_state(ind_op[0],
                                                                                                     ind_op[1], r0, r1))
            #       3. Turbine state
            twf_t_states.append(wind_farm.turbines[idx].turbine_states.create_interpolated_state(ind_op[0],
                                                                                                     ind_op[1], r0, r1))
            #   Reconstruct turbine location
            tmp_phi = twf_a_states[-1].get_turbine_wind_dir()
            tmp_phi = ot.ot_deg2rad(tmp_phi)
            #       1. Get vector from OP to related turbine
            vec_op2t = wind_farm.turbines[inf_turbines[idx]].observation_points.get_vec_op_to_turbine(ind_op[0]) * r0 \
                + wind_farm.turbines[inf_turbines[idx]].observation_points.get_vec_op_to_turbine(ind_op[1]) * r1
            #       2. Set turbine location
            twf_layout[idx, 0:3] = tmp_op + np.array([[np.cos(tmp_phi), -np.sin(tmp_phi), 0],
                                                    [np.sin(tmp_phi), np.cos(tmp_phi),  0],
                                                    [0, 0, 1]]) @ vec_op2t
            #       3. Set diameter
            twf_layout[idx, 3] = wind_farm_layout[inf_turbines[idx], 3]

        lg.info('TWF layout for turbine %s:' % i_t)
        lg.info(twf_layout)

        # Set wind farm in the wake model
        self.floris_wake.set_wind_farm(twf_layout, twf_t_states, twf_a_states)

        # Visualizations
        # Plot of effective wind farm layout
        if self._flag_plot_wakes:
            self.floris_wake.vis_flow_field()
            self._lower_flag_plot_wakes()

        # Plot of the effective wind speed from TWF "tile"
        if self._flag_plot_tile:
            self._flow_field_u = self.floris_wake.vis_tile(
                self._flow_field_x, self._flow_field_y, self._flow_field_z)
            self._lower_flag_plot_tile()

        # Plot of the effective wind speed along the cross stream direction of an OP
        if self._flag_plot_OP_mountain:
            self._mountain_u_abs = self.floris_wake.vis_tile(
                self._OP_mountain_x, self._OP_mountain_y, self._OP_mountain_z)
            self._lower_flag_plot_OP_mountains()

        # Get the measurements
        ueff, m = self.floris_wake.get_measurements_i_t(i_t_tmp)
        lg.info('Effective wind speed of turbine %s : %s m/s' % (i_t, ueff))
        [u_eff, v_eff] = ot.ot_abs2uv(ueff, twf_a_states[i_t_tmp].get_turbine_wind_dir())
        
        return np.array([u_eff, v_eff]), m

    def _get_wind_speeds_location(self, loc: np.ndarray, wind_farm: wfm.WindFarm) -> np.ndarray:
        """
        Calculates the effective wind speed at a location
        Parameters
        ----------
        loc : np.ndarray
            x,y,z coordinates with which respect to the effective wind speed should be derived
        wind_farm
            influencing wind farm

        Returns
        -------
        nd.array
            abs wind speed at location
        """
        # Load current data
        wind_farm_layout = wind_farm.get_layout()

        # Create an index range over all nT turbines
        inf_turbines = np.arange(wind_farm.nT)

        twf_layout = np.zeros((inf_turbines.shape[0], 4))  # Allocation of x, y, z coordinates of the turbines
        twf_t_states = []
        twf_a_states = []

        # Go through dependencies
        for idx in np.arange(inf_turbines.shape[0]):

            lg.debug('Ambient states: Two OP interpolation')
            # Interpolation of turbine states
            #   Step 1 retrieve closest up and downstream OPs
            op_locations = wind_farm.turbines[inf_turbines[idx]].observation_points.get_world_coord()
            ind_op = ot.ot_get_closest_2_points_3d_sorted(loc, op_locations)

            #   Step 2 calculate interpolation weights
            point_a = op_locations[ind_op[0], 0:2].transpose()
            point_b = op_locations[ind_op[1], 0:2].transpose()
            point_c = loc[0:2].transpose()

            weight_d = ((point_b - point_a) @ (point_c - point_a)) / ((point_b - point_a) @ (point_b - point_a))
            # Limit weight_d to [0,1]
            weight_d = np.fmin(np.fmax(weight_d, 0), 1)

            r0 = 1 - weight_d
            r1 = weight_d

            #   Interpolate states
            #       1. OP location
            tmp_op = op_locations[ind_op[0], 0:3] * r0 + op_locations[ind_op[1], 0:3] * r1
            #       2. Ambient
            twf_a_states.append(wind_farm.turbines[idx].ambient_states.create_interpolated_state(ind_op[0],
                                                                                                     ind_op[1], r0, r1))
            #       3. Turbine state
            twf_t_states.append(wind_farm.turbines[idx].turbine_states.create_interpolated_state(ind_op[0],
                                                                                                     ind_op[1], r0, r1))
            #   Reconstruct turbine location
            tmp_phi = twf_a_states[-1].get_turbine_wind_dir()
            tmp_phi = ot.ot_deg2rad(tmp_phi)
            #       1. Get vector from OP to related turbine
            vec_op2t = wind_farm.turbines[inf_turbines[idx]].observation_points.get_vec_op_to_turbine(ind_op[0]) * r0 \
                + wind_farm.turbines[inf_turbines[idx]].observation_points.get_vec_op_to_turbine(ind_op[1]) * r1
            #       2. Set turbine location
            twf_layout[idx, 0:3] = tmp_op + np.array([[np.cos(tmp_phi), -np.sin(tmp_phi), 0],
                                                    [np.sin(tmp_phi), np.cos(tmp_phi),  0],
                                                    [0, 0, 1]]) @ vec_op2t
            #       3. Set diameter
            twf_layout[idx, 3] = wind_farm_layout[inf_turbines[idx], 3]

        lg.info('TWF layout for the location is:')
        lg.info(twf_layout)

        # Set wind farm in the wake model
        self.floris_wake.set_wind_farm(twf_layout, twf_t_states, twf_a_states)

        # Run FLORIS and get velocities for points
        vel_at_loc = self.floris_wake.get_point_vel(
                loc[0], loc[1], loc[2])

        # Visualizations
        # Plot of effective wind farm layout
        if self._flag_plot_wakes:
            self.floris_wake.vis_flow_field()
            self._lower_flag_plot_wakes()

        # Plot of the effective wind speed from TWF "tile"
        if self._flag_plot_tile:
            self._flow_field_u = self.floris_wake.vis_tile(
                self._flow_field_x, self._flow_field_y, self._flow_field_z)
            self._lower_flag_plot_tile()

        # Plot of the effective wind speed along the cross stream direction of an OP
        if self._flag_plot_OP_mountain:
            self._mountain_u_abs = self.floris_wake.vis_tile(
                self._OP_mountain_x, self._OP_mountain_y, self._OP_mountain_z)
            self._lower_flag_plot_OP_mountains()

        return vel_at_loc

    def _get_wind_speeds_op(self, i_t: int, wind_farm: wfm.WindFarm) -> np.ndarray:
        """
        Calculates the free wind speeds for the OPs of turbine i_t
        Parameters
        ----------
        i_t : int
            index of the turbine to find the wind speeds for
        wind_farm
            influencing wind farm

        Returns
        -------
        nd.array
            [u,v] wind speeds at the OP locations
        """
        # TODO combine the influence of different wakes
        return wind_farm.turbines[i_t].ambient_states.get_wind_speed()
    

# [1] FLORIDyn - A dynamic and flexible framework for real - time wind farm control,
# Becker et al., 2022

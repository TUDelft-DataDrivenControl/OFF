# Copyright (C) <2023>, M Becker (TUDelft), M Lejeune (UCLouvain)

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

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import off.utils as ot
import logging
from floris.tools import FlorisInterface
from floris.tools.visualization import visualize_cut_plane
import matplotlib.pyplot as plt

lg = logging.getLogger(__name__)


class WakeModel(ABC):
    settings = dict()
    wind_farm_layout = np.array([])
    turbine_states = np.array([])
    ambient_states = np.array([])
    rp_s = np.array([])
    rp_w = float

    def __init__(self, settings: dict, wind_farm_layout: np.ndarray, turbine_states, ambient_states):
        """
        Class to calculate the impact of a wind farm onto a specific turbine

        Parameters
        ----------
        settings: dict
            wake settings, such as parameters
        wind_farm_layout : np.ndarray
            [n_t x 4] array with x,y,z,D entries for each turbine rotor (D = diameter)
        turbine_states : array of TurbineStates objects
            array with n_t TurbineStates objects with each one state
            Obejcts have access to methods such as get_current_Ct()
        ambient_states : array of AmbientStates objects
            array with n_t AmbientStates objects with each one state
            Obejcts have access to methods such as get_turbine_wind_dir()
        """

        self.settings = settings
        self.wind_farm_layout = wind_farm_layout
        self.turbine_states = turbine_states
        self.ambient_states = ambient_states
        self.rp_s, self.rp_w = ot.ot_isocell(settings['nRP'])

    @abstractmethod
    def get_measurements_i_t(self, i_t: int) -> tuple:
        """
        Returns the measurements of the wake model including the effective wind speed at the turbine i_t

        Parameters
        ----------
        i_t : int
            Index of the turbine of interest
        wind_farm : off.windfarm
            Wind farm object to retrieve the wakes from

        Returns
        -------
        tuple:
            float: u_eff at turbine i_t
            pandas.dataframe: m other measurements (Power gen, added turbulence, etc.)
        """
        pass

    def set_wind_farm(self, wind_farm_layout: np.ndarray, turbine_states, ambient_states):
        """
        Changes the states of the stored wind farm

        Parameters
        ----------
        wind_farm_layout: np.ndarray
            n_t x 4 array with [x,y,z,D] - world coordinates of the rotor center & diameter
        turbine_states: np.ndarray
            n_t x 2 array with [axial ind., yaw]
        ambient_states: np.ndarray
            1 x 2 : [u_abs, phi] - absolute background wind speed and direction
        """
        self.wind_farm_layout = wind_farm_layout
        self.turbine_states = turbine_states
        self.ambient_states = ambient_states


class DummyWake(WakeModel):
    """
    Dummy wake with funky shape for testing
    """

    def __init__(self, settings: dict, wind_farm_layout: np.ndarray, turbine_states, ambient_states):
        """
        Wake with funky shape for testing

        Parameters
        ----------
        settings : dict
            .dw down wind wave number
            .cw cross wind wave number
            .sig dw down wind weight
            .sig r  radial weight
        wind_farm_layout : np.ndarray
            n_t x 4 array with [x,y,z,D] - world coordinates of the rotor center & diameter
        turbine_states : np.ndarray
            n_t x 2 array with [axial ind., yaw]
        ambient_states : np.ndarray
            1 x 2 : [u_abs, phi] - absolute background wind speed and direction
        """
        super(DummyWake, self).__init__(settings, wind_farm_layout, turbine_states, ambient_states)

    def get_measurements_i_t(self, i_t: int) -> tuple:
        """
        Returns the measurements of the wake model including the effective wind speed at the turbine i_t

        Parameters
        ----------
        i_t : int
            Index of the turbine of interest

        Returns
        -------
        tuple:
            float: u_eff at turbine i_t
            pandas.dataframe: m other measurements (Power gen, added turbulence, etc.)
        """
        # Adapt basic rotor points to location and orientation of turbine i_t
        n_rps = self.rp_s.shape[0]
        rps = np.zeros((n_rps, 3))
        phi = ot.ot_deg2rad(
            ot.ot_get_orientation(self.ambient_states[i_t].get_turbine_wind_dir(),
                                  self.turbine_states[i_t].get_current_yaw()))
        phi_u = ot.ot_deg2rad(self.ambient_states[i_t].get_turbine_wind_dir())

        rps[:, 0] = self.wind_farm_layout[i_t, 0] - np.sin(phi) * self.rp_s[:, 0] * self.wind_farm_layout[i_t, 3]
        rps[:, 1] = self.wind_farm_layout[i_t, 1] + np.cos(phi) * self.rp_s[:, 0] * self.wind_farm_layout[i_t, 3]
        rps[:, 2] = self.wind_farm_layout[i_t, 2] + self.rp_s[:, 1] * self.wind_farm_layout[i_t, 3]

        #  Iterate over all turbines besides the relevant one and calculate reduction at the rotor points and
        #  subsequently across the plane
        red = np.ones((self.wind_farm_layout.shape[0], 1))

        for idx, tur in enumerate(self.wind_farm_layout):
            if idx == i_t:
                # Wind shear
                red[idx] = np.sum((rps[:, 2] / self.wind_farm_layout[i_t, 3]) ** 0.2) / n_rps
                continue

            # calculate wake influence at rotor points
            #   calculate down, crosswind distance and difference in height
            dist_wc = np.transpose(np.array([self.wind_farm_layout[idx, 0] - rps[:, 0],
                                             np.transpose(self.wind_farm_layout[idx, 1] - rps[:, 1])]))
            dist_dw = np.cos(phi_u) * dist_wc[:, 0] + np.sin(phi_u) * dist_wc[:, 1]

            if np.average(dist_dw) > -.1:
                lg.debug(f'Turbine {idx} has no influence on turbine {i_t}')
                lg.debug(f'Location Turbine {idx}: {self.wind_farm_layout[idx, 0:3]}')
                lg.debug(f'Location Turbine {i_t}: {self.wind_farm_layout[i_t, 0:3]}')
                lg.debug(f'Wind direction: {ot.ot_get_orientation(self.ambient_states[i_t].get_turbine_wind_dir())} '
                         f'deg')
                continue

            dist_cw = -np.sin(phi_u) * dist_wc[:, 0] + np.cos(phi_u) * dist_wc[:, 1]
            dist_h = self.wind_farm_layout[idx, 2] - rps[:, 2]
            dist_r = np.sqrt(dist_cw ** 2 + dist_h ** 2)

            #   calculate resulting reduction
            r = (0.7 + 0.3 * np.cos(dist_dw * np.pi / self.settings['dw'])) * \
                (0.7 + 0.3 * np.sin(dist_r * np.pi / self.settings['cw'])) * \
                (1 - np.exp(-.5 * (dist_dw / self.settings['sig dw']) ** 2) *
                 np.exp(-.5 * (dist_r / self.settings['sig r']) ** 2))

            # average
            red[idx] = np.sum(r) / n_rps

        # Multiply with background wind speed and return
        m = pd.DataFrame([[i_t, self.ambient_states[i_t].get_turbine_wind_speed_abs() * np.prod(red), np.prod(red)]],
                         columns=['t_idx', 'u_abs_eff', 'red'])
        return self.ambient_states[i_t].get_turbine_wind_speed_abs() * np.prod(red), m


class FlorisGaussianWake(WakeModel):
    """
    Interface to the FLORIS wake models
    """

    def __init__(self, settings: dict, wind_farm_layout: np.ndarray, turbine_states, ambient_states):
        """
        Interface to the FLORIS wake models

        Parameters
        ----------
        settings : dict
            .["yaml_path"] path to settings, e.g. gch.yaml
                example files can be found at https://github.com/NREL/floris/tree/main/examples/inputs
        wind_farm_layout : np.ndarray
            n_t x 4 array with [x,y,z,D] - world coordinates of the rotor center & diameter
        turbine_states : array of TurbineStates objects
            array with n_t TurbineStates objects with each one state
            Obejcts have access to methods such as get_current_Ct()
        ambient_states : array of AmbientStates objects
            array with n_t AmbientStates objects with each one state
            Obejcts have access to methods such as get_turbine_wind_dir()
        turbine_states : np.ndarray
            n_t x 2 array with [axial ind., yaw]
        ambient_states : np.ndarray
            1 x 2 : [u_abs, phi] - absolute background wind speed and direction
        """
        super(FlorisGaussianWake, self).__init__(settings, wind_farm_layout, turbine_states, ambient_states)
        lg.info(f'Interface for the ' + ' model initialized')  # TODO: Add which model has been initialized
        self.fi = FlorisInterface(self.settings['sim_dir'] + self.settings['gch_yaml_path'])
        lg.info(f'FLORIS object created.')

    def set_wind_farm(self, wind_farm_layout: np.ndarray, turbine_states, ambient_states):
        """
        Changes the states of the stored wind farm

        Parameters
        ----------
        wind_farm_layout: np.ndarray
            n_t x 4 array with [x,y,z,D] - world coordinates of the rotor center & diameter
        turbine_states: np.ndarray
            n_t x 2 array with [axial ind., yaw]
        ambient_states: np.ndarray
            1 x 2 : [u_abs, phi] - absolute background wind speed and direction
        """
        self.wind_farm_layout = wind_farm_layout
        self.turbine_states = turbine_states
        self.ambient_states = ambient_states

        self.fi.reinitialize(
            layout_x=wind_farm_layout[:, 0],
            layout_y=wind_farm_layout[:, 1],
            wind_directions=[ambient_states[0].get_turbine_wind_dir()],  # TODO Assign wind vel from main turbine
            wind_speeds=[ambient_states[0].get_turbine_wind_speed_abs()],    # TODO Assign wind dir from main turbine
        )

    def get_measurements_i_t(self, i_t: int) -> tuple:
        """
        Returns the measurements of the wake model including the effective wind speed at the turbine i_t

        Parameters
        ----------
        i_t : int
            Index of the turbine of interest

        Returns
        -------
        tuple:
            float: u_eff
                effective wind speed at turbine i_t
            pandas.dataframe: measurements
                all measurements (Power gen, added turbulence, etc.)
        """
        n_t = len(self.turbine_states)
        yaw_ang = np.zeros([1, 1, n_t])

        for ii_t in np.arange(n_t):
            yaw_ang[0, 0, ii_t] = self.turbine_states[ii_t].get_current_yaw()

        self.fi.calculate_wake(yaw_angles=yaw_ang)
        
        avg_vel = self.fi.turbine_average_velocities
        Cts = self.fi.get_turbine_Cts()
        # AIs = self.fi.get_turbine_ais()               # TODO: Fix since FLORIS v3.4 has a bug causing this line to crash
        TIs = self.fi.get_turbine_TIs()

        measurements = pd.DataFrame(
            [[
                i_t,
                avg_vel[:, :, i_t].flatten()[0],
                Cts[:, :, i_t].flatten()[0],
                # AIs[:, :, i_t].flatten()[0],          # TODO: FLORIS v3.4 fix
                TIs[:, :, i_t].flatten()[0],
            ]],
            # columns=['t_idx', 'u_abs_eff', 'Ct', 'AI', 'TI']  # TODO: FLORIS v3.4 fix
            columns=['t_idx', 'u_abs_eff', 'Ct', 'TI']
        )

        return avg_vel[:, :, i_t].flatten()[0], measurements

    def vis_flow_field(self):
        """
        Creates a plot of the wind farm applied to the given turbine using the FLORIS interface
        """

        n_t = len(self.turbine_states)
        yaw_ang = np.zeros([1, 1, n_t])

        for ii_t in np.arange(n_t):
            yaw_ang[0, 0, ii_t] = self.turbine_states[ii_t].get_current_yaw()

        # Don't know if the calculate_wake is needed, but probably for yaw angles
        self.fi.calculate_wake(yaw_angles=yaw_ang)
        horizontal_plane = self.fi.calculate_horizontal_plane(height=self.wind_farm_layout[0, 2])

        fig, ax_horo_plane = plt.subplots()
        visualize_cut_plane(horizontal_plane, ax=ax_horo_plane, title="Horizontal")
        plt.show()



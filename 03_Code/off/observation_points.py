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

import logging
lg = logging.getLogger(__name__)

import numpy as np
from abc import ABC, abstractmethod
from off.states import States
from off.utils import ot_abs_wind_speed


class ObservationPoints(States, ABC):
    """
    ObservationPoints is the abstract base class for a list of wake tracers / particles
    The class inherits get, set & iterate methods from the abstract States class, init is overwritten
    """

    def __init__(self, number_of_time_steps: int, number_of_states: int, state_names: list):
        """
        ObservationPoints is the abstract base class for a list of wake tracers / particles
        The class inherits get, set & iterate methods from the abstract States class, init is overwritten

        Parameters
        ----------
        number_of_time_steps : int
            number of time steps the states should go back / chain length
        number_of_states : int
            number of states per time step
        state_names : list
            name and unit of the states
        """
        super(ObservationPoints, self).__init__(number_of_time_steps, number_of_states, state_names)
        self.op_propagation_speed = np.zeros((number_of_time_steps, 2))

    @abstractmethod
    def get_world_coord(self) -> np.ndarray:
        """
        Returns the x, y, z coordinates of all OPs

        Returns
        -------
        np.ndarray
            m x 3 matrix where the columns are the x,y,z coordinates
        """       
        pass

    @abstractmethod
    def get_vec_op_to_turbine(self, index: int) -> np.ndarray:
        """
        Returns a x, y, z vector pointing from the OP to the turbine location in the OP coordinate system.
        OP coordinate system is usually the wake coordinate system based on the OP data.

        Returns
        -------
        np.ndarray
            1 x 3 matrix where the columns are the x,y,z coordinates
        """
        pass

    @abstractmethod
    def init_all_states(self, wind_speed_u: float, wind_speed_v: float, rotor_pos: np.ndarray, time_step: float):
        """
        Creates a downstream chain of OPs
        Overwrites the base method of the States class

        Parameters
        ----------
        wind_speed_u : float
            Wind speed in x direction in m/s
        wind_speed_v : float
            Wind speed in y direction in m/s
        rotor_pos : np.ndarray
            1 x 3 vector with x,y,z location of the rotor in the world coordinate system
        time_step : float
            simulation time step in s
        """
        pass

    @abstractmethod
    def propagate_ops(self, time_step: float):
        """
        Propagates the OPs based on the u and v velocity component

        Parameters
        ----------
        uv_op : np.ndarray
            m x 2 matrix with wind speeds of all OPs in x and y direction in m/s
        time_step : float
            Time step of the simulation in s
        """
        pass

    def set_op_propagation_speed(self, op_propagation_speed: np.ndarray):
        """
        Sets the propagation speed of the OPs, meant as a temporay storage, which ensures that different lengths of OP
        chains can be used in the simulation

        Parameters
        ----------
        op_propagation_speed: np.ndarray
            m x 2 matrix with [u,v] velocity component for all OPs
        """
        self.op_propagation_speed = op_propagation_speed


class FLORIDynOPs4(ObservationPoints):

    def __init__(self, number_of_time_steps: int):
        """
        FLORIDynOPs have four states, three in the world coordinate system (x,y,z) and one in the wake coordinate system
        (downstream).

        Parameters
        ----------
        number_of_time_steps : int
            equivalent to OP chain length
        """
        super(FLORIDynOPs4, self).__init__(number_of_time_steps, 4, ['x0 (m)', 'y0 (m)', 'z0 (m)', 'x1 (m)'])

    def get_world_coord(self) -> np.ndarray:
        """
        Returns the world coordinates of the OPs

        Returns
        -------
        np.ndarray
            [x, y, z] coordinates in world coordinate system
        """
        return self.states[:, 0:3]

    def get_vec_op_to_turbine(self, index: int) -> np.ndarray:
        """
        Returns a x, y, z vector pointing from the OP to the turbine location in the OP coordinate system.
        OP coordinate system is usually the wake coordinate system based on the OP data.

        Returns
        -------
        np.ndarray
            1 x 3 matrix where the columns are the x,y,z coordinates
        """
        return np.array([-self.states[index, 3], 0.0, 0.0])

    def init_all_states(self, wind_speed_u: float, wind_speed_v: float, rotor_pos: np.ndarray, time_step: float):
        """        
        Creates a downstream chain of OPs
        Overwrites the base method of the States class

        Parameters
        ----------
        wind_speed_u : float
            wind speed in x direction in m/s
        wind_speed_v : float
            wind speed in y direction in m/s
        rotor_pos : np.ndarray
            1 x 3 vector with x,y,z location of the rotor in the world coordinate system
        time_step : float
            simulation time step in s
        """

        self.states[:, 0] = np.arange(self.n_time_steps) * wind_speed_u * time_step + rotor_pos[0]
        self.states[:, 1] = np.arange(self.n_time_steps) * wind_speed_v * time_step + rotor_pos[1]
        self.states[:, 2] = rotor_pos[2]
        self.states[:, 3] = np.arange(self.n_time_steps) * ot_abs_wind_speed(wind_speed_u, wind_speed_v) * time_step

    def propagate_ops(self, time_step: float):
        """
        Propagates the OPs based on the u and v velocity component

        Parameters
        ----------
        uv_op : np.ndarray
            m x 2 matrix with wind speeds of all OPs in x and y direction in m/s
        time_step : float
            Time step of the simulation in s
        """
        self.states[1:, 0] = self.states[:-1, 0] + self.op_propagation_speed[:-1, 0] * time_step
        self.states[1:, 1] = self.states[:-1, 1] + self.op_propagation_speed[:-1, 1] * time_step
        self.states[1:, 3] = self.states[:-1, 3] + np.sqrt(self.op_propagation_speed[:-1, 0]**2 +
                                                           self.op_propagation_speed[:-1, 1]**2) * time_step


class FLORIDynOPs6(ObservationPoints):

    def __init__(self, number_of_time_steps: int):
        """
        FLORIDyn OPs with six states, three in the world coordinate system (x,y,z) and one in the wake coordinate system
        (x,y,z). This method requires more memory but less calculations at runtime.

        Parameters
        ----------
        number_of_time_steps : int
            equivalent to OP chain length
        """
        super(FLORIDynOPs6, self).__init__(number_of_time_steps, 6,
                                           ['x0 (m)', 'y0 (m)', 'z0 (m)', 'x1 (m)', 'y1 (m)', 'z1 (m)'])

    def get_world_coord(self) -> np.ndarray:
        """
        Returns the world coordinates of the OPs

        Returns
        -------
        np.ndarray
            [x, y, z] coordinates in world coordinate system
        """
        return self.op_list[:, 0:3]

    def init_all_states(self, wind_speed_u: float, wind_speed_v: float, rotor_pos: np.ndarray, time_step: float):
        """
        Creates a downstream chain of OPs
        Overwrites the base method of the States class

        Parameters
        ----------
        wind_speed_u : float
            wind speed in x direction in m/s
        wind_speed_v : float
            wind speed in y direction in m/s
        rotor_pos : np.ndarray
            1 x 3 vector with x,y,z location of the rotor in the world coordinate system
        time_step : float
            simulation time step in s
        """

        self.states[:, 0] = np.arange(self.n_time_steps) * wind_speed_u + rotor_pos[0]
        self.states[:, 1] = np.arange(self.n_time_steps) * wind_speed_v + rotor_pos[1]
        self.states[:, 2] = rotor_pos[2]
        self.states[:, 3] = np.arange(self.n_time_steps) * ot_abs_wind_speed(wind_speed_u, wind_speed_v)

    def propagate_ops(self, time_step: float):
        """
        Propagates the OPs based on the u and v velocity component

        Parameters
        ----------
        uv_op : np.ndarray
            m x 2 matrix with wind speeds of all OPs in x and y direction in m/s
        time_step : float
            Time step of the simulation in s
        """
        self.states[1:, 0] = self.states[:-1, 0] + self.op_propagation_speed[:-1, 0] * time_step
        self.states[1:, 1] = self.states[:-1, 1] + self.op_propagation_speed[:-1, 1] * time_step
        self.states[1:, 3] = self.states[:-1, 3] + np.sqrt(self.op_propagation_speed[:-1, 0] ** 2 +
                                                           self.op_propagation_speed[:-1, 1] ** 2) * time_step

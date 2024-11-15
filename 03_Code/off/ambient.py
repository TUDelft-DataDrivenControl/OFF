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
from typing import List
lg = logging.getLogger(__name__)

import numpy as np
from abc import ABC, abstractmethod
from off.states import States
from off.utils import ot_deg2rad



class AmbientStates(States, ABC):

    def __init__(self, number_of_time_steps: int, number_of_states: int, state_names: list):
        """
        Abstract base class for the ambient states, such as wind speed and direction.

        Parameters
        ----------
        number_of_time_steps : int
            number of time steps the states should go back / chain length
        number_of_states : int
            number of states per time step
        state_names : list
            name and unit of the states
        """
        super(AmbientStates, self).__init__(number_of_time_steps, number_of_states, state_names)
        lg.info('Ambient states chain created with %s time steps and %s states' %
                (number_of_time_steps, number_of_states))
        lg.info(state_names)

    @abstractmethod
    def get_wind_speed_abs(self) -> np.ndarray:
        """
        Returns the absolute wind speed

        Returns
        -------
        np.ndarray
             m x 1 vector of the absolute wind speed in m/s
        """
        pass

    @abstractmethod
    def get_wind_speed(self) -> np.ndarray:
        """
        Returns u,v component of all wind speeds

        Returns
        -------
        np.ndarray
            m x 2 matrix of [u,v] wind speeds in m/s
        """
        pass

    @abstractmethod
    def get_wind_speed_u(self) -> np.ndarray:
        """
        Returns the u component of the wind speed (x direction)

        Returns
        -------
        np.ndarray
             m x 1 vector of the u wind speed in m/s
        """
        pass

    @abstractmethod
    def get_wind_speed_v(self) -> np.ndarray:
        """
        Returns the v component of the wind speed (y direction)

        Returns
        -------
        np.ndarray
             m x 1 vector of the v wind speed in m/s
        """
        pass

    @abstractmethod
    def get_wind_dir(self) -> np.ndarray:
        """
        Returns all wind directions

        Returns
        -------
        np.ndarray
            m x 1 vector of wind direction states in deg
        """
        pass

    @abstractmethod
    def get_wind_dir_ind(self, ind: int):
        """
        Returns wind directions at an index

        Parameters
        ----------
        ind: int
            Index

        Returns
        -------
        np.ndarray
            m x 1 vector of wind direction states in deg
        """
        pass

    @abstractmethod
    def get_turbine_wind_speed_abs(self) -> float:
        """
        Returns the absolute wind speed at the turbine location (first entry)

        Returns
        -------
        float
            absolute wind speed
        """
        pass

    @abstractmethod
    def get_turbine_wind_speed(self) -> np.ndarray:
        """
        Returns u,v component wind speed at the turbine location

        Returns
        -------
        np.ndarray
            1 x 2 vector of [u,v] wind speeds in m/s
        """
        pass

    @abstractmethod
    def get_turbine_wind_speed_u(self) -> float:
        """
        Returns the u component of the wind speed (x direction) at the turbine location

        Returns
        -------
        float
            u wind speed
        """
        pass

    @abstractmethod
    def get_turbine_wind_speed_v(self) -> float:
        """
        Returns the v component of the wind speed (y direction) at the turbine location

        Returns
        -------
        float
            v wind speed
        """
        pass

    @abstractmethod
    def get_turbine_wind_dir(self) -> float:
        """
        Returns wind direction at the turbine location

        Returns
        -------
        float
            wind direction (deg)
        """
        pass

    @abstractmethod
    def create_interpolated_state(self, index1: int, index2: int, w1, w2):
        """
        Creates an AmbientStates object of its own kind with only one state entry, based on two weighted states.
        The returned object then still has access to functions such as get_turbine_wind_dir()

        Parameters
        ----------
        index1 : int
            Index of the first state
        index2 : int
            Index of the second state
        w1 : float
            Weight for first index (has to be w1 = 1 - w2, and [0,1])
        w2 : float
            Weight for second index (has to be w2 = 1 - w1, and [0,1])

        Returns
        -------
        AmbientStates
            ambient state object with single entry
        """
        pass



class FLORIDynAmbient(AmbientStates):
    def __init__(self, number_of_time_steps: int):
        """
        Ambient flow field based on the FLORIDyn formulation.
        The states are tied to the OP locations.
        The states are wind speed, wind direction and ambient turbulence intensity.

        Parameters
        ----------
        number_of_time_steps : int
            number of time steps the states should go back / chain length
        """
        super(FLORIDynAmbient, self).__init__(number_of_time_steps, 3, ['Abs. wind speed (m/s)', 'Wind direction (deg)',
                                                                        'Ambient turbulence intensity (%)'])

    def get_wind_speed_at(self, location: np.ndarray, op_coord: np.ndarray) -> np.ndarray:
        """
        Returns the absolute wind speed at a requested location

        Parameters
        ----------
        location : np.ndarray
            m x 3 matrix where the columns are [x,y,z] locations in m
        op_coord : np.ndarray
            n x 3 matrix of the OP world coordinate states in m

        Returns
        -------
        np.ndarray
            m x 1 vector with absolute wind speeds in m/s
        """

        # TODO
        pass

    def get_wind_direction_at(self, location: np.ndarray, op_coord: np.ndarray):
        """
        Returns the wind direction at a requested location

        Parameters
        ----------
        location : np.ndarray
            m x 3 matrix where the columns are [x,y,z] locations in m
        op_coord : np.ndarray
            n x 3 matrix of the OP world coordinate states in m

        Returns
        -------
        np.ndarray
            m x 1 vector with absolute wind speeds in deg
        """

        # TODO
        pass

    def get_turbine_wind_speed_abs(self) -> float:
        """
        Returns the absolute wind speed at the turbine location (first entry)

        Returns
        -------
        float
            absolute wind speed
        """
        if self.n_time_steps > 1:
            return self.states[0, 0]
        else:
            return self.states[0]

    def get_turbine_wind_speed(self) -> np.ndarray:
        """
        Returns u,v component of u & v wind speed at the turbine location

        Returns
        -------
        np.ndarray
            1 x 2 matrix of [u,v] wind speeds in m/s
        """
        return np.array([self.get_turbine_wind_speed_u(), self.get_turbine_wind_speed_v()])

    def get_turbine_wind_speed_u(self) -> float:
        """
        Returns the u component of the wind speed (x direction) at the turbine location

        Returns
        -------
        float
            u wind speed
        """
        if self.n_time_steps > 1:
            return self.states[0, 0] * np.cos(ot_deg2rad(self.states[0, 1]))
        else:
            return self.states[0] * np.cos(ot_deg2rad(self.states[1]))

    def get_turbine_wind_speed_v(self) -> float:
        """
        Returns the v component of the wind speed (y direction) at the turbine location

        Returns
        -------
        float
            v wind speed
        """
        if self.n_time_steps > 1:
            return self.states[0, 0] * np.sin(ot_deg2rad(self.states[0, 1]))
        else:
            return self.states[0] * np.sin(ot_deg2rad(self.states[1]))

    def get_wind_speed_abs(self) -> np.ndarray:
        """
        Returns the absolute wind speed

        Returns
        -------
        np.ndarray
             m x 1 vector of the absolute wind speed in m/s
        """
        return self.states[:, 0]

    def get_wind_speed(self) -> np.ndarray:
        """
        Returns u,v component of all wind speeds

        Returns
        -------
        np.ndarray
            m x 2 matrix of [u,v] wind speeds in m/s
        """
        return np.transpose(np.array([self.get_wind_speed_u(), self.get_wind_speed_v()]))

    def get_wind_speed_u(self) -> np.ndarray:
        """
        Returns the u component of the wind speed (x direction)

        Returns
        -------
        np.ndarray
             m x 1 vector of the u wind speed in m/s
        """
        return self.states[:, 0] * np.cos(ot_deg2rad(self.states[:, 1]))

    def get_wind_speed_v(self) -> np.ndarray:
        """
        Returns the v component of the wind speed (y direction)

        Returns
        -------
        np.ndarray
             m x 1 vector of the v wind speed in m/s
        """
        return self.states[:, 0] * np.sin(ot_deg2rad(self.states[:, 1]))

    def get_wind_dir(self) -> np.ndarray:
        """
        Returns all stored wind directions

        :return: m x 1 vector of wind direction states in deg
        """
        return self.states[:, 1]

    def get_turbine_wind_dir(self) -> float:
        """
        Returns all wind directions

        :return: float of wind direction state at the turbine location in deg
        """

        if self.n_time_steps > 1:
            return self.states[0, 1]
        else:
            return self.states[1]

    def get_wind_dir_ind(self, ind: int):
        """
        Returns wind directions at an index

        Parameters
        ----------
        ind: int
            Index

        Returns
        -------
        np.ndarray
            m x 1 vector of wind direction states in deg
        """
        return self.states[ind, 1]

    def create_interpolated_state(self, index1: int, index2: int, w1, w2):
        """
        Creates an AmbientStates object of its own kind with only one state entry, based on two weighted states.
        The returned object then still has access to functions such as get_turbine_wind_dir()

        Parameters
        ----------
        index1 : int
            Index of the first state
        index2 : int
            Index of the second state
        w1 : float
            Weight for first index (has to be w1 = 1 - w2, and [0,1])
        w2 : float
            Weight for second index (has to be w2 = 1 - w1, and [0,1])

        Returns
        -------
        AmbientStates
            ambient state object with single entry
        """
        # TODO create check for weights
        a_s = FLORIDynAmbient(1)
        a_s.set_all_states(self.states[index1, :]*w1 + self.states[index2, :]*w2)
        return a_s

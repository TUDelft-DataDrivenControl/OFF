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

import logging
from typing import List
lg = logging.getLogger(__name__)

import numpy as np
from abc import ABC, abstractmethod
from off.states import States
from off.utils import ot_deg2rad

from windfarm import WindFarm
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


FIELD_MAP = {'Abs. wind speed (m/s)':            'wind_speeds',
             'Wind direction (deg)':             'wind_directions',
             'Ambient turbulence intensity (%)': 'turbulence_intensities',
             'Wind shear (-)':                   'wind_shear',
             'Wind veer (-)':                    'wind_veer'}

def AmbientCorrector_Filter(ABC):
    settings: dict

    @abstractmethod
    def __init__(self, settings_cor: dict, wf: WindFarm):        
        """
        Abstract base class for the ambient state filtering

        Parameters
        ----------
        settings_cor : dict
            A dictionary containing the parameters of the consensus filter
        """   
        pass

    @abstractmethod
    def update(self, time:float):
        """
        Updates the internal state of the filter

        Parameters
        ----------
        t : float
            current time in s
        """      
        pass

    @abstractmethod
    def __call__(self, states_turbines: np.array, states_turbines_prev: np.array):
        """
        Apply the Consensus filter to states_turbines

        Parameters
        ----------
        states_turbines : np.array
            [nT] array, new ambient state
        states_turbines_prev : np.array
            [nT] array, previous ambient state

        Returns
        -------
        np.array
            [nT] array, new filtered ambient state
        """
        pass

def Consensus(AmbientCorrector_Filter):

    def __init__(self, settings_cor: dict, wf: WindFarm):
        """
        Consensus-based ambient flow state filtering: WTi's measurements of the ambient state are corrected
        in order to integrate information from neighboring WTs

        Parameters
        ----------
        settings_cor : dict
            A dictionary containing the parameters of the consensus filter
        """        

        # -- Parameters initialization
        self.settings = settings_cor['consensus_spatial_filtering']

        self.sigma = self.settings['sigma'] # spacial weighting factor used to determine the weight of adacents WTs based on their relative distance 
        self.alpha = self.settings['alpha'] # weighting factor (>=0) determining how much measurements of adjacent WTs should be used 

        n_wts = wf.nT
        x_wts = wf.get_layout()[:,:3]
        self.wt_filter = self.settings.setdefault('wt_filter', np.ones(n_wts)) # nT long array: wt_filter[i] = 0 if measurements of WTi should be ignored.

        self.weights = np.zeros((self.n_wts, self.n_wts))

        # -- Weight of adjacent nodes
        dx = [np.subtract.outer(x, x)**2 for x in x_wts.T]
        w = np.fill_diagonal( np.exp(sum(dx) / (2 * self.sigma ** 2) ), 0 )
    
        self.weights = -2 * self.alpha * w

        # -- Weight of current node
        np.fill_diagonal(self.weights , self.wt_filter + np.sum(w, axis=0))

    def update(self, time:float):
        """
        Updates the internal state of the filter

        Parameters
        ----------
        t : float
            current time in s
        """        
        pass
    
    def __call__(self, states_turbines: np.array, states_turbines_prev: np.array):
        """
        Apply the Consensus filter to states_turbines

        Parameters
        ----------
        states_turbines : np.array
            [nT] array, new ambient state
        states_turbines_prev : np.array
            [nT] array, previous ambient state

        Returns
        -------
        np.array
            [nT] array, new filtered ambient state
        """
        if self.alpha != 0: # if alpha = 0, neighboring nodes ar not used
            return np.linalg.solve(self.weights, states_turbines * self.wt_filter)
        return states_turbines
    
def ExponentialMovingAverage(AmbientCorrector_Filter):

    def __init__(self, settings_cor: dict, wf: WindFarm):
        """
        Exponential Moving Average filter

        Parameters
        ----------
        settings_cor : dict
            A dictionary containing the parameters of the consensus filter
        """    
        self.settings = settings_cor['ema_temporal_filtering']
        self.time_prev = -np.inf
        self.alpha = np.nan

    def update(self, time:float):
        """
        Updates the internal state of the filter

        Parameters
        ----------
        t : float
            current time in s
        """     
        self.alpha = 1.0 - np.exp((time-self.time_prev)/self.settings['tau'])
        self.time_prev = time
    
    def __call__(self, states_turbines: np.array, states_turbines_prev: np.array):
        """
        Apply the Consensus filter to states_turbines

        Parameters
        ----------
        states_turbines : np.array
            [nT] array, new ambient state
        states_turbines_prev : np.array
            [nT] array, previous ambient state

        Returns
        -------
        np.array
            [nT] array, new filtered ambient state
        """
        return self.alpha * states_turbines + (1-self.alpha) * states_turbines_prev

AVAILABLE_FILTERS = {'consensus_spatial_filtering': Consensus,
                     'ema_temporal_filtering'     : ExponentialMovingAverage }

class AmbientCorrector():
    filters: AmbientCorrector_Filter

    def __init__(self, settings_cor: dict, wf: WindFarm, states_name: List[str]):
        """ Feeds the ambient flow parameters to the simulation

        Parameters
        ----------
        settings_cor : dict
            A dictionary containing the inflow temporal and / or spacial discretization
        nT : int
            number of wind turbines in the farm
        states_name : List[str]
            name of the AmbientStates states

        Raises
        ------
        ValueError
            If no value is specified for one of the AmbientStates states
        ValueError
            If the time discretization used is not consistent
        ValueError
            If the number of values provided and the number of wind turbines does not match
        """
        # -- Initializing ambient data feed
        self.state_id = [ FIELD_MAP[n] for n in states_name ]
        self.settings_cor = settings_cor
        self.values    = [None] * len(self.state_id) # Values vector associated to each state [nT x nS]
        self.time      = [None] * len(self.state_id) # Time vector associated to each state
        self.wt_flag   = [None] * len(self.state_id) # True if a distinct flow state is associate to each turbine
        self.time_flag = [None] * len(self.state_id) # True if state is varying with time

        for i_s, s in enumerate(self.state_id):
            if s not in settings_cor:
                raise ValueError(f'No value provided for state {s}')

            self.values[i_s] = np.array(settings_cor[s])
            self.time[i_s]   = settings_cor.get(f'{s}_t', [0.0])

            self.time_flag[i_s] = len(self.time[i_s]) > 1
            if not len(self.values[i_s]) == len(self.time[i_s]):
                raise ValueError(f'Time discretization for state {s} not consistent')

            self.wt_flag[i_s]   = hasattr(self.values[i_s][0], '__len__')
            if self.wt_flag[i_s]:
                if not len(self.values[i_s]) == wf.nT:
                    raise ValueError(f'Mismatch between the number of values provided and the number of wind turbines '
                                     f'for {s}.')

        self.buffer = np.zeros((wf.nT, len(self.state_id)))

        # -- Initializing filters
        self.filters = [AVAILABLE_FILTERS[key](self.settings_cor[key], wf) for key in self.settings_cor if key in AVAILABLE_FILTERS]

        # -- Finalizing initialization
        self._init  = True

    def update(self, t: float):
        """ updates the corrector buffer

        Parameters
        ----------
        t : float
            current time in s
        """

        for f in self.filters: f.update(t)

        # -- Retrieving all states recursively
        for i_s, s in enumerate(self.state_id):
            if self.time_flag[i_s] or self._init:
                # -- Loading states
                if self.wt_flag[i_s]:
                    buffer = [np.interp(t, self.time[i_s], v) for v in self.values[i_s].T]
                else:
                    buffer = np.interp(t, self.time[i_s], self.values[i_s])
                
                # -- Filtering states
                for f in self.filters: buffer = f(buffer, self.buffer[:, i_s])

                # -- Exporting states
                self.buffer[:, i_s] = buffer
                
        self._init = False

    def __call__(self, idx: int, states: AmbientStates):
        """Feeds in the buffer values into the states of the first particles

        Parameters
        ----------
        idx : int
            current wind turbine index
        states : AmbientStates
            ambient states of the selected wind turbine
        """
        if self.settings_cor["corr_overwrite_direction"]:
            states.init_all_states(self.buffer[idx, :])
        else:
            states.set_ind_state(0, self.buffer[idx, :])


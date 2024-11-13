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
# from off.states import States
# from off.utils import ot_deg2rad
from off.windfarm import WindFarm
from off.ambient import AmbientStates


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


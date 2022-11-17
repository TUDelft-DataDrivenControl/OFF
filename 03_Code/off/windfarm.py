import logging
from typing import List
from off.turbine import Turbine
import numpy as np
lg = logging.getLogger(__name__)


class WindFarm:
    """
    Wind Farm Attributes

    turbines : List[Turbine]
        List of turbine objects which form the wind farm
    dependencies : np.ndarray (with boolean entries)
        Row i describes which turbines influence turbine i. The main diagonal should always be 'True'.
    nT : int
        Number of turbines in the wind farm
    """
    turbines: List[Turbine]
    dependencies: np.ndarray
    nT: int

    def __init__(self, turbines: List[Turbine]):
        """
        Object which hosts the turbine array as well as parameters, constants & variables important to the simulation.

        Parameters
        ----------
        turbines : Turbine object list
            List of turbines in the wind farm
        """
        lg.info(f'Wind farm created with {len(turbines)} turbines.')
        self.nT = len(turbines)
        self.turbines = turbines
        self.dependencies = np.full((self.nT, self.nT), True)

    def get_sub_windfarm(self, indices):
        """
        Creates a subset of the wind farm with the turbines at the given indices

        Parameters
        ----------
        indices: int[]

        Returns
        -------
        turbines array
        """
        return self.turbines[indices]

    def set_dependencies(self, dependencies: np.ndarray):
        """

        Parameters
        ----------
        dependencies: np.ndarray
            boolean array with dependencies - true if there is a dependency, false if not
        """
        self.dependencies = dependencies

    def get_layout(self) -> np.ndarray:
        """
        Gets the current wind farm layout and diameters

        Returns
        -------
        np.ndarray:
            [n_t x 4] matrix with wind farm layout in the world coordinate system and turbine diameter
        """
        layout = np.zeros((len(self.turbines), 4))
        for idx, trb in enumerate(self.turbines):
            layout[idx, :] = np.append(trb.get_rotor_pos(), trb.diameter)

        lg.info(f'Wind farm layout:')
        lg.info(layout)

        return layout

    def get_current_turbine_states(self) -> np.ndarray:
        """
        Collects and returns the current turbine states of the turbines

        Returns
        -------
        np.ndarray:
            [n_t x m] matrix with current turbine states at the turbine locations
        """
        tmp_t_state = self.turbines[0].turbine_states.get_ind_state(0)
        t_states = np.zeros((len(self.turbines), len(tmp_t_state)))
        for idx, trb in enumerate(self.turbines):
            t_states[idx, :] = trb.turbine_states.get_ind_state(0)

        lg.info(f'Current wind farm states:')
        lg.info(t_states)

        return t_states


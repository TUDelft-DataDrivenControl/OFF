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

    def add_turbine(self, turb: Turbine) -> int:
        """
        Adds another turbine to the list

        Parameters
        ----------
        turb: Turbine
            Turbine class object which describes the new turbine

        Returns
        -------
        int
            index of the newly added turbine object
        """
        # Add turbine to the list
        self.turbines.append(turb)
        lg.info(f'New turbine added to the wind farm, base located at:')
        lg.info(self.turbines[-1].base_location)

        # Update the dependencies array
        # Add row of ones
        row_of_ones = np.ones((1, self.dependencies.shape[1]), dtype=self.dependencies.dtype)
        self.dependencies = np.append(self.dependencies, row_of_ones, axis=0)

        # Add column of ones
        column_of_ones = np.ones((self.dependencies.shape[0], 1), dtype=self.dependencies.dtype)
        self.dependencies = np.append(self.dependencies, column_of_ones, axis=1)

        # Update turbine count
        self.nT = self.turbines.__len__()
        lg.info(f'Number of turbines now: {self.nT}')

        return self.nT-1

    def rmv_turbine(self, ind: int) -> Turbine:
        """
        Removes a turbine from the wind farm

        Parameters
        ----------
        ind: int
            Index of the turbine to remove

        Returns
        -------
        Turbine
            removed turbine object
        """
        lg.info(f'Turbine {ind} is being removed, base located at:')
        lg.info(self.turbines[ind].base_location)

        # Remove dependencies
        self.dependencies = np.delete(self.dependencies, ind, axis=0)  # Remove row
        self.dependencies = np.delete(self.dependencies, ind, axis=1)  # Remove column

        # Reduce turbine count
        self.nT = self.nT - 1
        lg.info(f'Number of turbines now: {self.nT}')

        # Remove and return turbine
        return self.turbines.pop(ind)

    def get_op_world_coordinates(self) -> np.ndarray:
        """
        Collects and returns all OP world locations

        Returns
        -------
        np.ndarray
            x,y,z coordiates of the OPs (in m)
        """
        collected_coordinates = []

        for turbine in self.turbines:
            collected_coordinates.append(turbine.observation_points.get_world_coord())

        return np.array(collected_coordinates)

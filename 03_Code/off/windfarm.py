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
            [n_t x 2] matrix with axial induction factor and yaw angle for each turbine
        """
        t_states = np.zeros((len(self.turbines), 2))
        for idx, trb in enumerate(self.turbines):
            t_states[idx, :] = np.array([
                trb.turbine_states.get_current_ax_ind(), trb.turbine_states.get_current_yaw()])

        lg.info(f'Wind farm states (axial Induction, yaw):')
        lg.info(t_states)

        return t_states


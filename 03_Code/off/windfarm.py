import logging
from typing import List
from off.turbine import Turbine
import numpy as np
lg = logging.getLogger(__name__)


class WindFarm:
    turbines: List[Turbine]

    def __init__(self, turbines: List[Turbine]):
        """
        Object which hosts the turbine array as well as parameters, constants & variables important to the simulation.

        Parameters
        ----------
        turbines : Turbine object list
            List of turbines in the wind farm
        """
        self.turbines = turbines

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

        return t_states


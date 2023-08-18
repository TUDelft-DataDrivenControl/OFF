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
from abc import ABC, abstractmethod
import off.Turbine as Turbine
import logging

lg = logging.getLogger(__name__)


class Controller(ABC):
    settings: dict

    def __init__(self, settings: dict):
        """
        Class to decide the turbine states for the next time step based on the current states of turbines,
        ambient states and/or external input.

        Parameters
        ----------
        settings
        """

        self.settings = settings

    @abstractmethod
    def set_turbine_states(self, turbine: Turbine, i_t: int, time_step: float) -> Turbine:
        """
        Reads the given turbine and sets its turbine states

        Parameters
        ----------
        turbine: Turbine
            Turbine object with turbine states, ambient states and observation points
        i_t:
            Index of the turbine (for look-up tables)
        time_step:
            current time step (for look-up tables or counters)

        Returns
        -------
        Turbine object with updated turbine states
        """
        pass


class IdealGreedyBaselineController(Controller):
    """
    Follows the main wind direction, disregarding yaw travel costs or other turbines
    """
    def __init__(self, settings: dict):
        super(IdealGreedyBaselineController, self).__init__(settings)
        lg.info('Ideal greedy baseline controller created.')

    def set_turbine_states(self, turbine: Turbine, i_t: int, time_step: float) -> Turbine:
        """
        Reads the given turbine and sets its turbine states

        Parameters
        ----------
        turbine: Turbine
            Turbine object with turbine states, ambient states and observation points
        i_t:
            Index of the turbine (for look-up tables)
        time_step:
            current time step (for look-up tables or counters)

        Returns
        -------
        Turbine object with updated turbine states
        """
        pass


class RealisticGreedyBaselineController(Controller):
    """
    Follows the main wind direction, only corrects if offset to the averaged wind direction is larger then a given
    offset.
    """

    def __init__(self, settings: dict):
        super(RealisticGreedyBaselineController, self).__init__(settings)
        lg.info('Realistic greedy baseline controller created.')

    def set_turbine_states(self, turbine: Turbine, i_t: int, time_step: float) -> Turbine:
        """
        Reads the given turbine and sets its turbine states

        Parameters
        ----------
        turbine: Turbine
            Turbine object with turbine states, ambient states and observation points
        i_t:
            Index of the turbine (for look-up tables)
        time_step:
            current time step (for look-up tables or counters)

        Returns
        -------
        Turbine object with updated turbine states
        """
        pass


class YawSteeringLUTController(Controller):
    """
    Chooses yaw angles based on the current main wind direction and a lut for it. Only corrects if offset to the
    averaged wind direction is larger then a given offset.
    """

    def __init__(self, settings: dict):
        super(YawSteeringLUTController, self).__init__(settings)
        lg.info('Yaw steering LUT controller created.')

    def set_turbine_states(self, turbine: Turbine, i_t: int, time_step: float) -> Turbine:
        """
        Reads the given turbine and sets its turbine states

        Parameters
        ----------
        turbine: Turbine
            Turbine object with turbine states, ambient states and observation points
        i_t:
            Index of the turbine (for look-up tables)
        time_step:
            current time step (for look-up tables or counters)

        Returns
        -------
        Turbine object with updated turbine states
        """
        pass


class YawSteeringPrescribedMotionController(Controller):
    """
    Applies prescribed yaw trajectories to the turbines
    """

    def __init__(self, settings: dict):
        super(YawSteeringPrescribedMotionController, self).__init__(settings)
        lg.info('Prescribed yaw motion controller created.')

    def set_turbine_states(self, turbine: Turbine, i_t: int, time_step: float) -> Turbine:
        """
        Reads the given turbine and sets its turbine states

        Parameters
        ----------
        turbine: Turbine
            Turbine object with turbine states, ambient states and observation points
        i_t:
            Index of the turbine (for look-up tables)
        time_step:
            current time step (for look-up tables or counters)

        Returns
        -------
        Turbine object with updated turbine states
        """
        pass


class YawSteeringFilteredPrescribedMotionController(Controller):
    """
    Applies prescribed yaw trajectories to the turbines, but filters and applies the hysteresis control
    """

    def __init__(self, settings: dict):
        super(YawSteeringFilteredPrescribedMotionController, self).__init__(settings)
        lg.info('Prescribed yaw motion controller created.')

    def set_turbine_states(self, turbine: Turbine, i_t: int, time_step: float) -> Turbine:
        """
        Reads the given turbine and sets its turbine states

        Parameters
        ----------
        turbine: Turbine
            Turbine object with turbine states, ambient states and observation points
        i_t:
            Index of the turbine (for look-up tables)
        time_step:
            current time step (for look-up tables or counters)

        Returns
        -------
        Turbine object with updated turbine states
        """
        pass

# ============== Tickets ================
# [ ] implement 1st controller -> ideal greedy baseline
# [ ] Add switch to run file
#       [ ] Select input data
# [ ] Add controller to initialization
#       [ ] Switch between which controller is activated
# [ ] implement controller into simulation loop
#       [ ] include ideal greedy baseline and see what issues arise
#       [ ] check if yaw motions have the desired effect
#       [ ] output the applied motion / cost (maybe not needed)
#       [ ] reserve output for controller info (controller states, counters, errors, etc.)
# [ ] implement realistic yaw controller
#       [ ] Hysteresis
#       [ ] Integration behaviour
# [ ] implement LUT yaw controller
#       [ ] Hysteresis
#       [ ] Integration behaviour
# [ ] implement filtered prescribed motion yaw controller
# [ ] implement prescribed motion yaw controller
#       -> extends the prescribed motion yaw controller with 0° tolerance to the set point

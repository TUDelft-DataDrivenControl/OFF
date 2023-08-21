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
import pandas as pd
import off.turbine as tur
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
    def __call__(self, turbine: tur, i_t: int, time_step: float) -> tur:
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

    @abstractmethod
    def get_applied_settings(self, turbine: tur, i_t: int, time_step: float):
        """
        Extracts the settings that this controller is setting. Does NOT write new settings.

        Parameters
        ----------
        turbine
        i_t
        time_step

        Returns
        -------
        pd.Dataframe
        """
        pass

    @abstractmethod
    def update(self, t: float):
        """
        Updates the controller state (if needed). Called once BEFORE the control settings are being set

        Parameters
        ----------
        t : Simulation time
        """
        pass


class IdealGreedyBaselineController(Controller):
    """
    Follows the main wind direction, disregarding yaw travel costs or other turbines
    """
    def __init__(self, settings: dict):
        super(IdealGreedyBaselineController, self).__init__(settings)
        lg.info('Ideal greedy baseline controller created.')

    def __call__(self, turbine: tur, i_t: int, time_step: float) -> tur:
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
        turbine.set_yaw(turbine.ambient_states.get_wind_dir_ind(0), 0.0)

    def get_applied_settings(self, turbine: tur, i_t: int, time_step: float) -> pd.DataFrame:
        """
        Extracts the settings that this controller is setting. Does NOT write new settings.

        Parameters
        ----------
        turbine
        i_t
        time_step

        Returns
        -------
        pd.Dataframe
        """
        control_settings = pd.DataFrame(
            [[
                i_t,
                turbine.calc_yaw(turbine.ambient_states.get_wind_dir_ind(0)),
                turbine.orientation[0],
                time_step
            ]],
            columns=['t_idx', 'yaw', 'orientation', 'time']
        )

        return control_settings

    def update(self, t: float):
        """
        Updates the controller state (if needed). Called once BEFORE the control settings are being set

        Parameters
        ----------
        t : Simulation time
        """
        pass


class RealisticGreedyBaselineController(Controller):
    """
    Follows the main wind direction, only corrects if offset to the averaged wind direction is larger then a given
    offset.
    """

    def __init__(self, settings: dict):
        super(RealisticGreedyBaselineController, self).__init__(settings)
        self.moving = np.full((settings['number of turbines'], 1), False)
        self.moved = np.full((settings['number of turbines'], 1), False)
        self.time_steps_since_update = 0
        self.run_controller = True
        lg.info('Realistic greedy baseline controller created.')

    def __call__(self, turbine: tur, i_t: int, time_step: float) -> tur:
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
        wind_dir = turbine.ambient_states.get_wind_dir_ind(0)
        yaw = turbine.calc_yaw(wind_dir)
        self.moved[i_t] = self.moving[i_t]

        # check if difference is larger than threshold or if the turbine is moving already
        if np.abs(yaw) >= self.settings['misalignment_thresh'] and self.run_controller:
            # Turbine can move
            self.moving[i_t] = True

        if self.moving[i_t]:
            yaw_delta_t = np.sign(yaw) * self.settings['time step'] * turbine.yaw_rate_lim
            i_correction = np.argmin(
                np.abs([yaw, yaw_delta_t]))

            if i_correction == 0:
                # yaw angle 0.0 deg can be achieved
                turbine.set_yaw(wind_dir, 0.0)
                self.moved[i_t] = True  # in case the yaw misalignment was so small that it could be reached immediately
                self.moving[i_t] = False
            elif i_correction == 1:
                # yaw angle 0.0 deg can not be achieved, turbine keeps correcting
                turbine.set_yaw(wind_dir, yaw - yaw_delta_t)
        else:
            turbine.set_yaw(wind_dir, yaw)

        self.run_controller = False

    def get_applied_settings(self, turbine: tur, i_t: int, time_step: float):
        """
        Extracts the settings that this controller is setting. Does NOT write new settings.

        Parameters
        ----------
        turbine
        i_t
        time_step

        Returns
        -------
        pd.Dataframe
        """
        control_settings = pd.DataFrame(
            [[
                i_t,
                turbine.calc_yaw(turbine.ambient_states.get_wind_dir_ind(0)),
                turbine.orientation[0],
                self.moving[i_t],
                self.moved[i_t],
                time_step
            ]],
            columns=['t_idx', 'yaw', 'orientation', 'moving', 'moved', 'time']
        )

        return control_settings

    def update(self, t: float):
        """
        Updates the controller state (if needed). Called once BEFORE the control settings are being set

        Parameters
        ----------
        t : Simulation time
        """
        self.time_steps_since_update = self.time_steps_since_update + 1
        if self.time_steps_since_update >= self.settings["apply_frequency"]:
            self.run_controller = True
            self.time_steps_since_update = 0


class YawSteeringLUTController(Controller):
    """
    Chooses yaw angles based on the current main wind direction and a lut for it. Only corrects if offset to the
    averaged wind direction is larger then a given offset.
    """

    def __init__(self, settings: dict):
        super(YawSteeringLUTController, self).__init__(settings)
        lg.info('Yaw steering LUT controller created.')

    def __call__(self, turbine: tur, i_t: int, time_step: float) -> tur:
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

    def get_applied_settings(self, turbine: tur, i_t: int, time_step: float):
        """
        Extracts the settings that this controller is setting. Does NOT write new settings.

        Parameters
        ----------
        turbine
        i_t
        time_step

        Returns
        -------
        pd.Dataframe
        """
        pass

    def update(self, t: float):
        """
        Updates the controller state (if needed). Called once BEFORE the control settings are being set

        Parameters
        ----------
        t : Simulation time
        """
        pass


class YawSteeringPrescribedMotionController(Controller):
    """
    Applies prescribed yaw trajectories to the turbines
    """

    def __init__(self, settings: dict):
        super(YawSteeringPrescribedMotionController, self).__init__(settings)
        lg.info('Prescribed yaw motion controller created.')

    def __call__(self, turbine: tur, i_t: int, time_step: float) -> tur:
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

    def get_applied_settings(self, turbine: tur, i_t: int, time_step: float):
        """
        Extracts the settings that this controller is setting. Does NOT write new settings.

        Parameters
        ----------
        turbine
        i_t
        time_step

        Returns
        -------
        pd.Dataframe
        """
        pass

    def update(self, t: float):
        """
        Updates the controller state (if needed). Called once BEFORE the control settings are being set

        Parameters
        ----------
        t : Simulation time
        """
        pass


class YawSteeringFilteredPrescribedMotionController(Controller):
    """
    Applies prescribed yaw trajectories to the turbines, but filters and applies the hysteresis control
    """

    def __init__(self, settings: dict):
        super(YawSteeringFilteredPrescribedMotionController, self).__init__(settings)
        lg.info('Prescribed yaw motion controller created.')

    def __call__(self, turbine: tur, i_t: int, time_step: float) -> tur:
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

    def get_applied_settings(self, turbine: tur, i_t: int, time_step: float):
        """
        Extracts the settings that this controller is setting. Does NOT write new settings.

        Parameters
        ----------
        turbine
        i_t
        time_step

        Returns
        -------
        pd.Dataframe
        """
        pass

    def update(self, t: float):
        """
        Updates the controller state (if needed). Called once BEFORE the control settings are being set

        Parameters
        ----------
        t : Simulation time
        """
        pass

# ============== Tickets ================
# [x] implement 1st controller -> ideal greedy baseline
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


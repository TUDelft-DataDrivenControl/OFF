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
from off.observation_points import ObservationPoints
from off.ambient import AmbientStates
from off.states import States
import off.utils as ot
import logging
lg = logging.getLogger(__name__)


class TurbineStates(States, ABC):

    def __init__(self, number_of_time_steps: int, number_of_states: int, state_names: list):
        """
        Abstract base class for the turbine states, inherits from abstract States class. This class determines how many
        turbine states are stored and how they are used to calculate the Cp and Ct coefficient. The states class provides
        the get, set, init & iterate methods

        Parameters
        ----------
        number_of_time_steps : int
            number of time steps the states should go back / chain length
        number_of_states : int
            number of states per time step
        state_names : list
            name and unit of the states
        """
        super(TurbineStates, self).__init__(number_of_time_steps, number_of_states, state_names)
        lg.info('Turbine states chain created with %s time steps and %s states' %
                (number_of_time_steps, number_of_states))
        lg.info(state_names)

    @abstractmethod
    def get_current_cp(self) -> float:
        """
        get_current_cp returns the current power coefficient of the turbine

        Returns
        -------
        float:
            Power coefficient (-)
        """
        pass

    @abstractmethod
    def get_current_ct(self) -> float:
        """
        get_current_ct returns the current thrust coefficient of the turbine

        Returns
        -------
        float:
            Thrust coefficient (-)
        """
        pass

    @abstractmethod
    def get_current_yaw(self) -> float:
        """
        get_current_yaw returns the current yaw misalignment at the turbine location

        Returns
        -------
        float:
            yaw misalignment at the turbine location (deg)
        """
        pass

    @abstractmethod
    def get_current_ax_ind(self) -> float:
        """
        get_current_axInd returns the current axial induction factor of the turbine

        Returns
        -------
        float:
            Axial induction factor (-)
        """
        pass

    @abstractmethod
    def get_ct(self, index: int) -> float:
        """
        get_ct(index) returns the Ct coefficient at a requested index of the turbine state chain

        Parameters
        ----------
        index : int
            Turbine state list index at which Ct should be calculated
        Returns
        -------
        float:
            Thrust coefficient
        """
        pass

    @abstractmethod
    def get_ax_ind(self, index: int) -> float:
        """
        get_ax_ind(index) returns the axial induction factor at a requested index of the turbine state chain

        Parameters
        ----------
        index : int
            Turbine state list index at which a should be calculated
        Returns
        -------
        float:
            Axial induction factor
        """
        pass

    @abstractmethod
    def set_ax_ind(self, ax_ind):
        """
        Stores the axial induction factor of the turbine in the states.

        Parameters
        ----------
        ax_ind: axial induction factor
        """
        pass

    @abstractmethod
    def get_yaw(self, index: int) -> float:
        """
        get_yaw(index) returns the yaw misalignment at a requested index

        Parameters
        ----------
        index : int

        Returns
        -------
        float:
            yaw misalignment in deg
        """
        pass

    @abstractmethod
    def set_yaw(self, yaw_angle):
        """
        Stores the yaw angle of the turbine in the states.

        Parameters
        ----------
        yaw_angle: Difference between turbine orientation and wind direction
        """
        pass

    @abstractmethod
    def get_all_ct(self) -> np.ndarray:
        """
        get_all_ct(index) returns the Ct coefficients for all turbine states.

        Returns
        -------
        np.ndarray:
            Thrust coefficient at all turbine states (-)
        """
        pass

    @abstractmethod
    def get_all_yaw(self) -> np.ndarray:
        """
        get_all_ct(index) returns the yaw misalignment for all turbine states.

        Returns
        -------
        np.ndarray:
            Yaw misalignment at all turbine states (deg)
        """
        pass

    @abstractmethod
    def get_all_ax_ind(self) -> np.ndarray:
        """
        get_all_axInd returns the all axial induction factors of the saved turbine states

        Returns
        -------
        np.ndarray:
            Axial induction factor (-)
        """
        pass

    @abstractmethod
    def create_interpolated_state(self, index1: int, index2: int, w1, w2):
        """
        Creates a TurbineStates object of its own kind with only one state entry, based on two weighted states.
        The returned object then still has access to functions such as get_current_yaw()

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
        TurbineStates
            turbine state object with single entry
        """
        pass


class Turbine(ABC):
    # Attributes
    diameter = 1  # in Meter
    nacellePos = np.array([0, 0, 1])            # in Meter
    turbine_type = "base"
    orientation = np.array([0, 0])              # yaw, tilt in Degree

    def __init__(self, base_location: np.ndarray, orientation: np.ndarray, turbine_states: TurbineStates,
                 observation_points: ObservationPoints, ambient_states: AmbientStates):
        """
        Turbine abstract base class

        Parameters
        ----------
        base_location : np.ndarray
            1 x 3 vector with x, y, z position of the turbine base (m)
        orientation : np.ndarray
            1 x 2 vector with yaw, tilt orientation of the turbine
        turbine_states : TurbineStates
            Turbine state object
        observation_points : ObservationPoints
            Observation Points object
        ambient_states : AmbientStates
            Ambient states object
        """
        self.base_location = base_location
        self.orientation = orientation
        self.turbine_states = turbine_states
        self.observation_points = observation_points
        self.ambient_states = ambient_states

    def calc_yaw(self, wind_direction: float) -> float:
        """
        Get the yaw misalignment of the turbine

        Parameters
        ----------
        wind_direction : number
            Wind direction (deg)

        Returns
        -------
        float:
            yaw misalignment (deg)
        """
        return ot.ot_get_yaw(wind_direction, self.orientation[0])

    def set_yaw(self, wind_direction: float, yaw: float):
        """
        Sets the orientation based on the given wind direction and yaw angle

        Parameters
        ----------
        wind_direction : float
            Wind direction in degrees
        yaw : float
            Turbine yaw misalignment angle in degrees

        """
        self.orientation[0] = ot.ot_get_orientation(wind_direction, yaw)
        self.turbine_states.set_yaw(yaw)

    def calc_tilt(self):
        """
            Get the tilt of the turbine

            Returns
            -------
            float:
                tilt (deg)
            """
        return self.orientation[1]

    @abstractmethod
    def calc_power(self, wind_speed, air_den):
        """
        Calculate the power based on turbine, ambient and OP states

        Parameters
        ----------
        wind_speed : float
            Wind speed (m/s)
        air_den : float
            air density

        Returns
        -------
        float :
            Power generated (W)
        """
        # TODO this function should either rely on its own states or be more descriptive with what inputs are need
        pass

    def get_rotor_pos(self) -> np.float_:
        """
        Calculates the rotor position based on the current yaw and tilt

        Returns
        -------
        np.ndarray:
            1 x 3 vector with x,y,z location of the rotor in the world coordinate system
        """
        # TODO add tilt to offset calculation
        lg.info('Orientation [0] %s ' % self.orientation[0])

        yaw = ot.ot_deg2rad(self.orientation[0])
        offset = np.array([np.cos(yaw), np.sin(yaw), 1]) * \
            self.nacellePos

        return self.base_location + offset

    def set_rotor_pos(self, pos_rot: np.ndarray):
        """
        Sets the base location, based on a rotor location and the nacelle position
        Parameters
        ----------
        pos_rot
        """
        # TODO should pay attention to orientation of the turbine
        self.base_location = pos_rot - self.nacellePos


class HAWT_ADM(Turbine):
    # Attributes
    diameter = 178.4  # Meter
    nacellePos = np.array([0, 0, 119])  # in Meter
    turbine_type = "name"

    def __init__(self, base_location: np.ndarray, orientation: np.ndarray, turbine_states: TurbineStates,
                 observation_points: ObservationPoints, ambient_states: AmbientStates, turbine_data: dict):
        """
        HAWT extends the base turbine class and specifies a generic horizontal axis wind turbine

        Parameters
        ----------
        base_location : np.ndarray
            1 x 3 vector with x, y, z position of the turbine base (m)
        orientation : np.ndarray
            1 x 2 vector with yaw, tilt orientation of the turbine
        turbine_states : TurbineStates
            Turbine state object
        observation_points : ObservationPoints
            Observation Points object
        ambient_states : AmbientStates
            Ambient states object
        """
        self.diameter = turbine_data["rotor_diameter"]
        self.nacellePos = np.array([0, 0, turbine_data["hub_height"]])
        self.power_calc_method = "cp-u lut"  # alternative to "axial induction", "cp-bpa-tsr"  TODO: Set later in input
        self.thrust_calc_method = "ct-u lut"  # alternative to "axial induction", "ct-bpa-tsr" TODO: Set later in input
        self.yaw_power_coeff = "pP" # alternative to "none" TODO: Set later in input
        self.yaw_thrust_coeff ="pT" # alternative to "none" TODO: Set later in input

        if "rotor_overhang" in turbine_data:
            self.nacellePos[0] = turbine_data["rotor_overhang"]

        if "Cp_curve" in turbine_data["performance"]:
            self.Cp_u_values = turbine_data["performance"]["Cp_curve"]["Cp_u_values"]
            self.Cp_u_wind_speeds = turbine_data["performance"]["Cp_curve"]["Cp_u_wind_speeds"]

        if "Ct_curve" in turbine_data:
            self.Ct_u_values = turbine_data["performance"]["Ct_curve"]["Ct_u_values"]
            self.Ct_u_wind_speeds = turbine_data["performance"]["Ct_curve"]["Ct_u_values"]

        if "pP" in turbine_data:
            self.Cp_pP = turbine_data["pP"]

        if "pT" in turbine_data:
            self.Cp_pT = turbine_data["pT"]

        if "yaw_rate_lim" in turbine_data:
            self.yaw_rate_lim = turbine_data["yaw_rate_lim"]

        super().__init__(base_location, orientation, turbine_states, observation_points, ambient_states)
        lg.info("HAWT turbine of type " + turbine_data["name"] + "created")
        lg.info('Turbine base location: %s' % base_location)
        lg.info('Power calculation method: %s' % self.power_calc_method)
        lg.info('Thrust calculation method: %s' % self.thrust_calc_method)
        lg.info('Power yaw coefficient: %s' % self.yaw_power_coeff)
        lg.info('Thrust yaw coefficient: %s' % self.yaw_thrust_coeff)

    def calc_power(self, wind_speed, air_den = 1.225):
        """
        Calculate the power based on turbine, ambient and OP states

        Parameters
        ----------
        wind_speed : float
            Wind speed (m/s)
        air_den : float
            air density

        Returns
        -------
        float :
            Power generated (W)
        """
        if self.yaw_power_coeff == "pP":
            yaw = np.deg2rad(self.turbine_states.get_current_yaw())
            yaw_coef = np.cos(yaw) ** self.Cp_pP
        elif self.yaw_power_coeff == "none":
            yaw_coef = 1.0
        else:
            raise Exception("Only cos(yaw) ** pP yaw coefficient supported (or none)")

        if self.power_calc_method == "axial induction":
            axi = self.turbine_states.get_current_ax_ind()
            cp = 4 * axi * (1 - axi) ** 2
            p = 0.5 * np.pi * (self.diameter / 2) ** 2 * wind_speed ** 3 * cp * yaw_coef * air_den
        elif self.power_calc_method == "cp-u lut":
            cp = np.interp(wind_speed, self.Cp_u_wind_speeds, self.Cp_u_values)
            p = 0.5 * np.pi * (self.diameter / 2) ** 2 * wind_speed ** 3 * cp * yaw_coef * air_den
        elif self.power_calc_method == "cp-bpa-tsr":
            cp = 0
            p = 0.5 * np.pi * (self.diameter / 2) ** 2 * wind_speed ** 3 * cp * yaw_coef * air_den
            raise Exception("Cp calculation based on cp-bpa-tsr not implemented yet.")
        else:
            raise Exception("The power calculation method %s is unkown. Try cp-u lut, cp-bpa-tsr, axial induction "
                            "instead." % self.power_calc_method)

        return p


class TurbineStatesFLORIDyn(TurbineStates):

    def __init__(self, number_of_time_steps: int):
        """
        TurbineStatesFLORIDyn includes the axial induction factor, the yaw misalignment and the added turbulence
        intensity.

        Parameters
        ----------
        number_of_time_steps : int
            number of time steps the states should go back / chain length
        """
        super().__init__(number_of_time_steps, 3, ['axial induction (-), yaw (deg), added turbulence intensity (%)'])

    def get_current_cp(self) -> float:  # TODO: Remove! This has been moved to the turbine model
        """
        get_current_cp returns the current power coefficient of the turbine

        Returns
        -------
        float:
            Power coefficient (-)
        """
        return 4 * self.states[0, 0] * (1 - self.states[0, 0]) ** 2 * \
               np.cos(self.states[0, 1]) ** 2.2  # TODO Double check correct Cp calculation

    def get_current_ct(self) -> float:  # TODO: Remove! This has been moved to the turbine model
        """
        get_current_ct returns the current thrust coefficient of the turbine

        Returns
        -------
        float:
            Thrust coefficient (-)
        """
        return self.get_ct(0)

    def get_current_ax_ind(self) -> float:
        """
        get_all_axInd returns the all axial induction factors of the saved turbine states

        Returns
        -------
        np.ndarray:
            Axial induction factor (-)
        """
        if self.n_time_steps > 1:
            return self.states[0, 0]
        else:
            return self.states[0]

    def get_current_yaw(self) -> float:
        """
        get_current_yaw returns the current yaw misalignment at the turbine location

        Returns
        -------
        float:
            yaw misalignment at the turbine location (deg)
        """
        return self.get_yaw(0)

    def set_yaw(self, yaw_angle: float):
        """
        Sets the yaw misalignment with the wind direction

        Parameters
        ----------
        yaw_angle
        """
        if self.n_time_steps > 1:
            self.states[0, 1] = yaw_angle
        else:
            self.states[1] = yaw_angle

    def get_ct(self, index: int) -> float:
        """
        get_ct(index) returns the Ct coefficient at a requested index of the turbine state chain

        Parameters
        ----------
        index : int
            Turbine state list index at which Ct should be calculated
        Returns
        -------
        float:
            Thrust coefficient
        """
        ax_i = self.get_ax_ind(index)
        yaw = np.deg2rad(self.get_yaw(index))
        return 4 * ax_i * (1 - ax_i) * \
               np.cos(yaw) ** 2.2  # TODO Insert correct Ct calculation

    def get_ax_ind(self, index: int) -> np.ndarray:
        """
        get_all_axInd returns the all axial induction factors of the saved turbine states

        Returns
        -------
        np.ndarray:
            Axial induction factor (-)
        """
        if self.n_time_steps > 1:
            return self.states[index, 0]
        else:
            return self.states[0]

    def set_ax_ind(self, ax_ind):
        """
        Stores the axial induction factor of the turbine in the states.

        Parameters
        ----------
        ax_ind: axial induction factor
        """
        if self.n_time_steps > 1:
            self.states[0, 0] = ax_ind
        else:
            self.states[0] = ax_ind

    def get_yaw(self, index: int) -> float:
        """
        get_yaw(index) returns the yaw misalignment at a requested index

        Parameters
        ----------
        index : int

        Returns
        -------
        float:
            yaw misalignment in deg
        """
        if self.n_time_steps > 1:
            return self.states[index, 1]
        else:
            return self.states[1]

    def get_all_ct(self) -> np.ndarray:
        """
        get_all_ct(index) returns the Ct coefficients for all turbine states.

        Returns
        -------
        np.ndarray:
            Thrust coefficient at all turbine states (-)
        """
        # TODO vectorized calculation of Ct
        pass

    def get_all_ax_ind(self) -> np.ndarray:
        """
        get_all_axInd returns the all axial induction factors of the saved turbine states

        Returns
        -------
        np.ndarray:
            Axial induction factor (-)
        """
        return self.states[:, 1]

    def get_all_yaw(self) -> np.ndarray:
        """
        returns the yaw misalignment for all turbine states.

        Returns
        -------
        np.ndarray:
            n x 1 vector with all yaw angles
        """
        return self.states[:, 1]

    def create_interpolated_state(self, index1: int, index2: int, w1, w2):
        """
        Creates a TurbineStates object of its own kind with only one state entry, based on two weighted states.
        The returned object then still has access to functions such as get_current_yaw()

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
        TurbineStates
            turbine state object with single entry
        """
        # TODO create check for weights
        t_s = TurbineStatesFLORIDyn(1)
        t_s.set_all_states(self.states[index1, :]*w1 + self.states[index2, :]*w2)
        return t_s


# SOURCES
# [1] The Dtu 10-Mw Reference Wind Turbine, Bak et al., 2013
#       https://orbit.dtu.dk/en/publications/the-dtu-10-mw-reference-wind-turbine

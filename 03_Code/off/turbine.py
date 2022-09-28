import numpy as np
from abc import ABC, abstractmethod
from off.observation_points import ObservationPoints
from off.ambient import AmbientStates
from off.states import States

from off.utils import OFFTools as OT

class TurbineStates(States, ABC):
    """
    Abstract base class for the turbine states, inherits from abstract States class. This class determines how many
    turbine states are stored and how they are used to calculate the Cp and Ct coefficient. The states class provides
    the get, set, init & iterate methods
    """

    def __init__(self, number_of_time_steps: int, number_of_states: int):
        super(TurbineStates, self).__init__(number_of_time_steps, number_of_states)

    @abstractmethod
    def get_current_cp(self) -> float:
        """
        get_current_cp returns the current power coefficient of the turbine location
        :return:
        """
        pass

    @abstractmethod
    def get_current_ct(self) -> float:
        """
        get_current_ct returns the current yaw misalignment at the turbine location
        :return:
        """
        pass

    @abstractmethod
    def get_current_yaw(self) -> float:
        """
        get_current_yaw returns the current yaw misalignment at the turbine location
        :return:
        """
        pass

    @abstractmethod
    def get_ct(self, index: int) -> float:
        """
        get_ct(index) returns the Ct coefficient at a requested index.
        :param index: Turbine state list index at which Ct should be calculated
        :return: Ct coefficient
        """

    @abstractmethod
    def get_yaw(self, index: int) -> float:
        """
        get_yaw(index) returns the yaw misalignment at a requested index
        :param index: Turbine state list index at which yaw should be returned
        :return: yaw misalignment in deg
        """
        pass

    @abstractmethod
    def get_all_ct(self) -> np.array:
        """
        get_all_ct(index) returns the Ct coefficients for all turbine states.
        :return:
        """
        pass

    @abstractmethod
    def get_all_yaw(self) -> np.array:
        """
        get_all_yaw(index) returns the yaw misalignment for all turbine states.
        :return:
        """
        pass


class Turbine(ABC):
    """
    Turbine abstract base class, specifies a turbine with a diameter of 1m and a nacelle height of 1m.

    Args:
        base_location (`numpy.ndarray`): x, y, z position of the turbine base in meter
        orientation (`numpy.ndarray`): yaw, tilt of the rotor in deg, yaw in world orientation, NOT yaw misalignment
    """
    # Attributes
    diameter = 1  # in Meter
    nacellePos = np.array([0, 0, 1])  # in Meter
    turbine_type = "base"
    orientation = np.array([0, 0])  # yaw, tilt in Degree

    def __init__(self, base_location: np.ndarray, orientation: np.ndarray, turbine_states: TurbineStates,
                 observation_points: ObservationPoints, ambient_states: AmbientStates):
        self.base_location = base_location
        self.orientation = orientation
        self.turbine_states = turbine_states
        self.observation_points = observation_points
        self.ambient_states = ambient_states

    def calc_yaw(self, wind_direction):
        return self.orientation[0] - wind_direction

    def calc_tilt(self):
        return self.orientation[1]

    @abstractmethod
    def calc_power(self, wind_speed, air_den):
        pass

    def get_rotor_pos(self) -> np.float_:
        """
        Calculates the rotor position based on the current yaw and tilt

        :return:
        """
        # TODO add tilt to offset calculation
        print("Orientation [0] ", self.orientation[0])

        ot = OT()
        yaw = ot.deg2rad(self.orientation[0])
        offset = np.array([np.cos(yaw), np.sin(yaw), 0]) * \
            self.nacellePos

        return self.base_location + offset


class DTU10MW(Turbine):
    """
    DTU10MW extends the base turbine class and specifies the 10 MW turbine based on
    SOURCE

    Args:
        base_location (`numpy.ndarray`): x, y, z position of the turbine base in meter
        orientation (`numpy.ndarray`): yaw, tilt of the rotor in deg, yaw in world orientation, NOT yaw misalignment
    """
    diameter = 178.4  # Meter
    nacellePos = np.array([0, 0, 119])  # in Meter
    turbine_type = "DTU10MW"

    def __init__(self, base_location: np.ndarray, orientation: np.ndarray, turbine_states: TurbineStates,
                 observation_points: ObservationPoints, ambient_states: AmbientStates):
        super().__init__(base_location, orientation, turbine_states, observation_points, ambient_states)
        print("DTU10MW turbine created")

    def calc_power(self, wind_speed, air_den):
        return 0.5 * np.pi * (self.diameter / 2) ** 2 * wind_speed ** 3  # TODO link with turbine state Cp calculation


class TurbineStatesFLORIDyn(TurbineStates):
    """
    TurbineStatesFLORIDyn includes the axial induction factor, the yaw misalignment and the added turbulence intensity.
    """

    def __init__(self, list_length: int):
        super().__init__(list_length, 3)

    def get_current_cp(self) -> float:
        """
        get_current_cp returns the current power coefficient of the turbine location
        :return:
        """
        return 4 * self.states[0, 0] * (1 - self.states[0, 0]) ** 2 * \
               np.cos(self.states[0, 1]) ** 2.2  # TODO Double check correct Cp calculation

    def get_current_ct(self) -> float:
        """
        get_current_ct returns the current yaw misalignment at the turbine location
        :return:
        """
        return self.get_ct(0)

    def get_current_yaw(self) -> float:
        """
        get_current_yaw returns the current yaw misalignment at the turbine location
        :return:
        """
        return self.get_yaw(0)

    def get_ct(self, index: int) -> float:
        """
        get_ct(index) returns the Ct coefficient at a requested index.
        :param index: Turbine state list index at which Ct should be calculated
        :return: Ct coefficient
        """
        return 4 * self.states[index, 0] * (1 - self.states[index, 0]) * \
               np.cos(self.states[index, 1]) ** 2.2  # TODO Insert correct Ct calculation

    def get_yaw(self, index: int) -> float:
        """
        get_yaw(index) returns the yaw misalignment at a requested index
        :param index: Turbine state list index at which yaw should be returned
        :return: yaw misalignment in deg
        """
        return self.states[index, 1]

    def get_all_ct(self) -> np.ndarray:
        """
        get_all_ct(index) returns the Ct coefficients for all turbine states.
        :return:
        """
        # TODO vectorized calculation of Ct
        pass

    def get_all_yaw(self) -> np.ndarray:
        """
        get_all_yaw(index) returns the yaw misalignment for all turbine states.
        :return:
        """
        return self.states[:, 1]

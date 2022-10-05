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
        get_ax_ind(index) returns the axial induction coefficient at a requested index of the turbine state chain

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


class Turbine(ABC):
    # Attributes
    diameter = 1  # in Meter
    nacellePos = np.array([0, 0, 1])  # in Meter
    turbine_type = "base"
    orientation = np.array([0, 0])  # yaw, tilt in Degree

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

    def calc_yaw(self, wind_direction):
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
        return ot.get_yaw(wind_direction, self.orientation[0])

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
        lg.info(f'Orientation [0] {self.orientation[0]}')

        yaw = ot.ot_deg2rad(self.orientation[0])
        offset = np.array([np.cos(yaw), np.sin(yaw), 1]) * \
            self.nacellePos

        return self.base_location + offset


class DTU10MW(Turbine):
    # Attributes
    diameter = 178.4  # Meter
    nacellePos = np.array([0, 0, 119])  # in Meter
    turbine_type = "DTU10MW"

    def __init__(self, base_location: np.ndarray, orientation: np.ndarray, turbine_states: TurbineStates,
                 observation_points: ObservationPoints, ambient_states: AmbientStates):
        """
        DTU10MW extends the base turbine class and specifies the 10 MW turbine based on [1]

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
        super().__init__(base_location, orientation, turbine_states, observation_points, ambient_states)
        lg.info("DTU10MW turbine created")

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

        # TODO probably needs to be rewritten
        return 0.5 * np.pi * (self.diameter / 2) ** 2 * wind_speed ** 3  # TODO link with turbine state Cp calculation


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

    def get_current_cp(self) -> float:
        """
        get_current_cp returns the current power coefficient of the turbine

        Returns
        -------
        float:
            Power coefficient (-)
        """
        return 4 * self.states[0, 0] * (1 - self.states[0, 0]) ** 2 * \
               np.cos(self.states[0, 1]) ** 2.2  # TODO Double check correct Cp calculation

    def get_current_ct(self) -> float:
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
        return self.states[0, 0]

    def get_current_yaw(self) -> float:
        """
        get_current_yaw returns the current yaw misalignment at the turbine location

        Returns
        -------
        float:
            yaw misalignment at the turbine location (deg)
        """
        return self.get_yaw(0)

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
        return 4 * self.states[index, 0] * (1 - self.states[index, 0]) * \
               np.cos(self.states[index, 1]) ** 2.2  # TODO Insert correct Ct calculation

    def get_ax_ind(self, index: int) -> np.ndarray:
        """
        get_all_axInd returns the all axial induction factors of the saved turbine states

        Returns
        -------
        np.ndarray:
            Axial induction factor (-)
        """
        return self.states[index, 0]

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
        return self.states[index, 1]

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

# SOURCES
# [1] The Dtu 10-Mw Reference Wind Turbine, Bak et al., 2013
#       https://orbit.dtu.dk/en/publications/the-dtu-10-mw-reference-wind-turbine

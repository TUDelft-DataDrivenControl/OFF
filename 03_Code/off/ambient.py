import numpy as np
from abc import ABC, abstractmethod
from off.states import States


class AmbientStates(States, ABC):
    """
    Abstract base class for the ambient states, such as wind speed and direction.
    """

    def __init__(self, number_of_time_steps: int, number_of_states: int):
        super(AmbientStates, self).__init__(number_of_time_steps, number_of_states)

    @abstractmethod
    def get_wind_speed(self) -> np.ndarray:
        """
        Returns all wind speeds

        :return: m x 1 vector of wind speed states in m/s
        """
        pass

    @abstractmethod
    def get_wind_dir(self) -> np.ndarray:
        """
        Returns all wind directions

        :return: m x 1 vector of wind direction states in deg
        """
        pass

    @abstractmethod
    def get_turbine_wind_speed(self) -> np.float_:
        """
        Returns wind at the turbine location (first entry)

        :return: float of wind speed state at the turbine location in m/s
        """
        pass

    @abstractmethod
    def get_turbine_wind_dir(self) -> np.float_:
        """
        Returns all wind directions

        :return: float of wind direction state at the turbine location in deg
        """
        pass


class FLORIDynAmbient(AmbientStates):
    """
    Ambient flow field based on the FLORIDyn formulation.
    The states are tied to the OP locations.
    The states are wind speed, wind direction and ambient turbulence intensity.
    """
    def __init__(self, list_length: int):
        super(FLORIDynAmbient, self).__init__(list_length, 3)

    def get_wind_speed_at(self, location: np.ndarray, op_coord: np.ndarray) -> np.ndarray:
        """
        Returns the wind speed at a requested location

        :param location: m x 3 matrix where the columns are [x,y,z] locations in m
        :param op_coord: n x 3 matrix of the OP world coordinate states in m
        :return: m x 1 vector wind speeds in m/s
        """

        # TODO
        pass

    def get_wind_direction_at(self, location: np.ndarray, op_coord: np.ndarray):
        """
        Returns the wind direction at a requested location

        :param location: m x 3 matrix where the columns are [x,y,z] locations in m
        :param op_coord: n x 3 matrix of the OP world coordinate states in m
        :return: m x 1 vector wind direction in deg
        """

        # TODO
        pass

    def get_turbine_wind_speed(self) -> np.float_:
        """
        Returns wind at the turbine location (first entry)

        :return: float of wind speed state at the turbine location in m/s
        """
        return self.states[0, 0]

    def get_turbine_wind_dir(self) -> np.float_:
        """
        Returns all wind directions

        :return: float of wind direction state at the turbine location in deg
        """
        return self.states[0, 1]

    def get_wind_speed(self) -> np.ndarray:
        """
        Returns all stored wind speeds

        :return: m x 1 vector of wind speed states in m/s
        """
        return self.states[:, 0]

    def get_wind_dir(self) -> np.ndarray:
        """
        Returns all stored wind directions

        :return: m x 1 vector of wind direction states in deg
        """
        return self.states[:, 1]


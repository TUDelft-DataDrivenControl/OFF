import numpy as np
from abc import ABC, abstractmethod
from off.states import States
from off.utils import OFFTools as OT


class ObservationPoints(States, ABC):
    """
    ObservationPoints is the abstract base class for a list of wake tracers / particles
    The class inherits get, set & iterate methods from the abstract States class, init is overwritten
    """

    def __init__(self, number_of_time_steps: int, number_of_states: int):
        super(ObservationPoints, self).__init__(number_of_time_steps, number_of_states)

    @abstractmethod
    def get_world_coord(self) -> np.ndarray:
        """
        Returns the x, y, z coordinates of all OPs

        :return: m x 3 matrix where the columns are the x,y,z coordinates
        """
        pass

    @abstractmethod
    def init_all_states(self, wind_speed: float, wind_direction: float, rotor_pos: np.ndarray, time_step: float):
        """
        Creates a downstream chain of OPs
        Overwrites the base method of the States class

        :param wind_speed: Wind speed in m/s
        :param wind_direction: Wind direction in deg
        :param rotor_pos: 1 x 3 vector with x,y,z location of the rotor in the world coordinate system
        :param time_step: simulation time step in s
        :return:
        """
        pass


class FLORIDynOPs4(ObservationPoints):
    """
    FLORIDynOPs have four states, three in the world coordinate system (x,y,z) and one in the wake coordinate system
    (downstream).
    """
    def __init__(self, number_of_time_steps: int):
        super(FLORIDynOPs4, self).__init__(number_of_time_steps, 4)

    def get_world_coord(self) -> np.ndarray:
        """
        Returns the world coordinates of the OPs
        :return:
        """
        return self.states[:, 0:3]

    def init_all_states(self, wind_speed: float, wind_direction: float, rotor_pos: np.ndarray, time_step: float):
        """
        Creates a downstream chain of OPs
        Overwrites the base method of the States class

        :param wind_speed: Wind speed in m/s
        :param wind_direction: Wind direction in deg
        :param rotor_pos: 1 x 3 vector with x,y,z location of the rotor in the world coordinate system
        :param time_step: simulation time step in s
        :return:
        """
        ot = OT()

        dw = np.arange(self.n_time_steps) * wind_speed
        self.states[:, 0] = np.cos(ot.deg2rad(wind_direction)) * dw + rotor_pos[0]
        self.states[:, 1] = np.sin(ot.deg2rad(wind_direction)) * dw + rotor_pos[1]
        self.states[:, 2] = rotor_pos[2]
        self.states[:, 3] = dw


class FLORIDynOPs6(ObservationPoints):
    """
    FLORIDynOPs have six states, three in the world coordinate system (x,y,z) and one in the wake coordinate system
    (x,y,z). This method requires more memory but less calculations at runtime.

    Args:
        list_length (`int`): length of the OP list
    """

    def __init__(self, number_of_time_steps: int):
        super(FLORIDynOPs6, self).__init__(number_of_time_steps, 6)

    def get_world_coord(self) -> np.ndarray:
        """
        Returns the world coordinates of the OPs
        :return:
        """
        return self.op_list[:, 0:3]

    def init_all_states(self, wind_speed: float, wind_direction: float, rotor_pos: np.ndarray, time_step: float):
        """
        Creates a downstream chain of OPs
        Overwrites the base method of the States class

        :param wind_speed: Wind speed in m/s
        :param wind_direction: Wind direction in deg
        :param rotor_pos: 1 x 3 vector with x,y,z location of the rotor in the world coordinate system
        :param time_step: simulation time step in s
        :return:
        """
        ot = OT()

        dw = np.arange(self.n_time_steps) * wind_speed
        self.states[:, 0] = np.cos(ot.deg2rad(wind_direction)) * dw + rotor_pos[0]
        self.states[:, 1] = np.sin(ot.deg2rad(wind_direction)) * dw + rotor_pos[1]
        self.states[:, 2] = rotor_pos[2]
        self.states[:, 3] = dw

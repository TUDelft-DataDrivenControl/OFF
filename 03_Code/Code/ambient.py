import numpy as np
from abc import ABC, abstractmethod


class AmbientStates(ABC):
    """
    Abstract base class for the ambient states, such as wind speed and direction.
    """
    # Attributes
    ambient_states = np.array([])

    def __init__(self, list_length: int, number_of_states: int):
        self.ambient_states = np.zeros((list_length, number_of_states))

    @abstractmethod
    def iterate_ambient(self, new_ambient: np.ndarray):
        pass

    def get_ambient(self) -> np.ndarray:
        return self.ambient_states


class FLORIDynAmbient(AmbientStates):
    """
    Ambient flow field based on the FLORIDyn formulation.
    The states are tied to the OP locations.
    The states are Wind speed, wind direction and ambient turbulence intensity.
    """
    def __init__(self, list_length: int):
        super().__init__(list_length, 3)

    def iterate_ambient(self, new_ambient: np.ndarray):
        self.ambient_states = np.roll(self.ambient_states, 1, axis=0)
        self.ambient_states[0, :] = new_ambient

    def get_wind_speed_at(self, location: np.ndarray, op_coord: np.ndarray):
        # TODO
        pass

    def get_wind_direction_at(self, location: np.ndarray, op_coord: np.ndarray):
        # TODO
        pass

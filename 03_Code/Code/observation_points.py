import numpy as np
from abc import ABC, abstractmethod


class ObservationPoints(ABC):
    """
    ObservationPoints is the abstract base class for a list of wake tracers / particles

    Args:
        list_length (`int`): length of the OP list
    """

    # Attributes
    op_list = np.array([])

    def __init__(self, list_length: int, number_of_states: int):
        self.op_list = np.zeros((list_length, number_of_states))

    @abstractmethod
    def iterate_ops(self, new_op: np.ndarray):
        pass

    def get_ops(self) -> np.ndarray:
        return self.op_list

    @abstractmethod
    def get_world_coord(self) -> np.ndarray:
        pass


class FLORIDynOPs4(ObservationPoints):
    """
    FLORIDynOPs have four states, three in the world coordinate system (x,y,z) and one in the wake coordinate system
    (downstream).

    Args:
        list_length (`int`): length of the OP list
    """
    def __init__(self, list_length: int):
        super().__init__(list_length, 4)

    def iterate_ops(self, new_op: np.ndarray):
        self.op_list = np.roll(self.op_list, 1, axis=0)
        self.op_list[0, :] = new_op

    def get_world_coord(self) -> np.ndarray:
        """
        Returns the world coordinates of the OPs
        :return:
        """
        return self.op_list[:, 0:3]


class FLORIDynOPs6(ObservationPoints):
    """
    FLORIDynOPs have six states, three in the world coordinate system (x,y,z) and one in the wake coordinate system
    (x,y,z). This method requires more memory but less calculations at runtime.

    Args:
        list_length (`int`): length of the OP list
    """

    def __init__(self, list_length: int):
        super().__init__(list_length, 4)

    def iterate_ops(self, new_op: np.ndarray):
        self.op_list = np.roll(self.op_list, 1, axis=0)
        self.op_list[0, :] = new_op

    def get_world_coord(self) -> np.ndarray:
        """
        Returns the world coordinates of the OPs
        :return:
        """
        return self.op_list[:, 0:3]

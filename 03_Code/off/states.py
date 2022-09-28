import numpy as np
from abc import ABC


class States(ABC):
    """
    Abstract state class
    Provides basic functions for state lists, such as get, set, initialize and iterate
    """

    # Attributes
    states = np.array([])
    n_time_steps = 0
    n_states = 0

    def __init__(self, number_of_time_steps: int, number_of_states: int):
        """
        Creates a new state object. States are organized as a que with young states at low indexes and old states with
        high indexes. The rows mark the progressing time, the columns the different states at each time step

        :param number_of_time_steps: Number of time steps the list should store
        :param number_of_states: Number of states per time step
        """
        self.states = np.zeros((number_of_time_steps, number_of_states))
        self.n_time_steps = number_of_time_steps
        self.n_states = number_of_states

    def get_all_states(self) -> np.ndarray:
        """
        Returns state matrix

        :return: m x n matrix, columns mark different states, rows time steps
        """
        return self.states

    def set_all_states(self, new_states: np.array):
        """
        Overwrites the states with the given matrix.

        :param new_states: m x n matrix with new states, columns mark different states, rows time steps
        :return: none
        """
        # TODO check matrix size, log if size of state array changed, update self.n_time_steps & .n_states
        self.states = new_states

    def iterate_states(self, new_state: np.ndarray):
        """
        shift_states shifts all states and adds a new entry in first place

        :param new_state: 1 x n vector
        :return: none
        """
        # TODO check vector size
        self.states = np.roll(self.states, 1, axis=0)
        self.states[0, :] = new_state

    def get_ind_state(self, index: int) -> np.ndarray:
        """
        Returns the state at a given index
        :param index:
        :return: state at the requested index
        """
        # TODO check for out of bounds request
        return self.states[index, :]

    def set_ind_state(self, index: int, new_state: np.ndarray):
        """
        Overwrites a state at the given index

        :param index: index of state to overwrite
        :param new_state: 1 x n vector which overwrites the state
        :return: none
        """
        # TODO check vector size
        # TODO check for out of bounds request
        self.states[index, :] = new_state

    def init_all_states(self, init_state: np.ndarray):
        """
        Copies a given state across all state entries as initialisation.
        For more advanced initialisation, use set_all_states()

        :param init_state: 1 x n vector of init state
        :return: none
        """
        # TODO check vector size
        self.states[:, :] = init_state
import numpy as np
from abc import ABC, abstractmethod


class States(ABC):
    """
    Abstract state class
    Provides basic functions for state lists, such as get, set, initialize and iterate
    """

    # Attributes
    states = np.array([])
    n_time_steps = 0
    n_states = 0
    state_names = []

    def __init__(self, number_of_time_steps: int, number_of_states: int, state_names: list):
        """
        Creates a new state object. States are organized as a que with young states at low indexes and old states with
        high indexes. The rows mark the progressing time, the columns the different states at each time step

        Parameters
        ----------
        number_of_time_steps : int
            number of time steps the states should go back / chain length
        number_of_states : int
            number of states per time step
        state_names : list
            name and unit of the states
        """
        self.states = np.zeros((number_of_time_steps, number_of_states))
        self.n_time_steps = number_of_time_steps
        self.n_states = number_of_states
        self.state_names = state_names

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

        Parameters
        ----------
        index : int
            index of the state to return
        Returns
        -------
        state : np.ndarray
            1 x n state vector
        """
        # TODO check for out of bounds request
        return self.states[index, :]

    def set_ind_state(self, index: int, new_state: np.ndarray):
        """
        Overwrites a state at the given index

        Parameters
        ----------
        index : int
            index of state to overwrite
        new_state : np.ndarray
            1 x n vector which overwrites the state

        Returns
        -------

        """
        # TODO check vector size
        # TODO check for out of bounds request
        self.states[index, :] = new_state

    def init_all_states(self, init_state: np.ndarray):
        """
        Copies a given state across all state entries as initialisation.
        For more advanced initialisation, use set_all_states()

        Parameters
        ----------
        init_state : np.ndarray
            1 x n vector of init state

        Returns
        -------

        """
        # TODO check vector size
        self.states[:, :] = init_state

    def get_state_names(self) -> list:
        """
        List with names of the stored states

        Returns
        -------
        list
            List with the names of the stored states in the corresponding order
        """
        return self.state_names

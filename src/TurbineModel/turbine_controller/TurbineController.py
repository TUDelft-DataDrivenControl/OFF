import numpy as np
from __future__ import annotations

from abc import ABC, abstractmethod

from Utils import *

class TurbineController(ABC):
    """Base interface for local (per-turbine) controller."""
    
    @abstractmethod
    def step(self, it: int) -> None:
        """ Advances the atmospheric model by a given number of iterations.

        Args:
            it (int): Current iteration of the simulation. The current real time since simulation start is it * dt, where dt is the global time step.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def get_citation(self) -> str:
        """ Returns a citation string for the turbine controller. Default implementation returns a generic citation.

        Returns:
            str: Citation string for the turbine controller.
        """
        return (
            "@software{floridyn_off_2026,\n"
            "  author       = {Becker, Marcus and Lejeune, Maxime and van Straalen, Ivo},\n"
            "  title        = {OFF wind farm simulation toolbox},\n"
            "  year         = {2026},\n"
            "  version      = {1.0.0},\n"
            "  publisher    = {GitHub},\n"
            "  url          = {https://github.com/TUDelft-DataDrivenControl/OFF}\n"
            "}\n\n"
            "@article{becker2025ADynamicModel,\n"
            "  author       = {Becker, Marcus and Lejeune, Maxime and Chatelain, Philippe and Allaerts, Dries and Mudafort, Rafael and van Wingerden, Jan-Willem},\n"
            "  title        = {A dynamic open-source model to investigate wake dynamics in response to wind farm flow control strategies},\n"
            "  journal      = {Wind Energy Science},\n"
            "  year         = {2025},\n"
            "  volume       = {10},\n"
            "  pages        = {1055--1075},\n"
            "  doi          = {10.5194/wes-10-1055-2025}\n"
            "}"
        )
    
    def req_describe(self) -> dict[str, SupportType]:
        """ Returns a dictionary describing the atmospheric model. Default implementation returns an empty dictionary.

        Returns:
            dict[str, Any]: Dictionary describing the atmospheric model.
        """
        return {
            "obs_power_setpoint_w": SupportType.NOT_SUPPORTED,
            "obs_yaw_setpoint_deg": SupportType.NOT_SUPPORTED,
            "obs_collective_pitch_setpoint_deg": SupportType.NOT_SUPPORTED,
            "obs_individual_pitch_setpoints_deg": SupportType.NOT_SUPPORTED,
            "obs_generator_torque_setpoint_nm": SupportType.NOT_SUPPORTED,
            "obs_rotor_torque_setpoint_nm": SupportType.NOT_SUPPORTED,
            "obs_rotor_speed_setpoint_radps": SupportType.NOT_SUPPORTED,
            "obs_rotor_speed_setpoint_rpm": SupportType.NOT_SUPPORTED,
            "obs_curtailment_factor": SupportType.NOT_SUPPORTED,
            "obs_control_mode": SupportType.NOT_SUPPORTED
        }
    
    def req_check_component(self, component) -> bool:
        """ Checks if the given component is compatible with the turbine controller. Default implementation returns False.

        Args:
            component: Component to check for compatibility.

        Returns:
            bool: True if the component is compatible, False otherwise.
        """
        return False

    """ 
    ---------------------------------------
    Observables 
    --------------------------------------- 
    """
    
    def obs_power_setpoint_w(self, t_s: np.float64) -> np.float64:
        """ Abstract method to observe the current power setpoint of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.float64: Current power setpoint of the turbine (W).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_yaw_setpoint_deg(self, t_s: np.float64) -> np.float64:
        """ Observes the current yaw setpoint of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.float64: Current yaw setpoint of the turbine (deg).
        """
        raise NotImplementedError
    
    def obs_collective_pitch_setpoint_deg(self, t_s: np.float64) -> np.float64:
        """ Observes the current collective pitch setpoint of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.float64: Current collective pitch setpoint of the turbine (deg).
        """
        raise NotImplementedError
    
    def obs_individual_pitch_setpoints_deg(self, t_s: np.float64) -> np.ndarray:
        """ Observes the current individual pitch setpoints of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.ndarray: Current individual pitch setpoints of the turbine (deg).
        """
        raise NotImplementedError
    
    def obs_generator_torque_setpoint_nm(self, t_s: np.float64) -> np.float64:
        """ Observes the current generator torque setpoint of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.float64: Current generator torque setpoint of the turbine (Nm).
        """
        raise NotImplementedError
    
    def obs_rotor_torque_setpoint_nm(self, t_s: np.float64) -> np.float64:
        """ Observes the current rotor torque setpoint of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.float64: Current rotor torque setpoint of the turbine (Nm).
        """
        raise NotImplementedError
    
    def obs_rotor_speed_setpoint_radps(self, t_s: np.float64) -> np.float64:
        """ Observes the current rotor speed setpoint of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.float64: Current rotor speed setpoint of the turbine (rad/s).
        """
        raise NotImplementedError
    
    def obs_rotor_speed_setpoint_rpm(self, t_s: np.float64) -> np.float64:
        """ Observes the current rotor speed setpoint of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.float64: Current rotor speed setpoint of the turbine (rpm).
        """
        return self.obs_rotor_speed_setpoint_radps(t_s) * 60 / (2 * np.pi)

    def obs_curtailment_factor(self, t_s: np.float64) -> np.float64:
        """ Observes the current curtailment factor of the turbine. 0 means no curtailment, 1 means full curtailment.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.float64: Current curtailment factor of the turbine (dimensionless).
        """
        raise NotImplementedError
    
    def obs_control_mode(self, t_s: np.float64) -> str:
        """ Observes the current control mode of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            str: Current control mode of the turbine.
        """
        raise NotImplementedError

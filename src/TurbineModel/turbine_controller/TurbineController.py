import numpy as np
from __future__ import annotations

from abc import ABC, abstractmethod


class TurbineController(ABC):
    """Base interface for local (per-turbine) controller."""
    
    def get_citation(self) -> str:
        """ Returns a citation string for the turbine controller. Default implementation returns a generic citation.

        Returns:
            str: Citation string for the turbine controller.
        """
        return "No specific Turbine Controller citation available."

    """ 
    ---------------------------------------
    Observables 
    --------------------------------------- 
    """

    @abstractmethod
    def obs_power_setpoint_w(self, t_s: np.float64) -> float:
        """ Abstract method to observe the current power setpoint of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            float: Current power setpoint of the turbine (W).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_yaw_setpoint_deg(self, t_s: np.float64) -> float:
        """ Observes the current yaw setpoint of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            float: Current yaw setpoint of the turbine (deg).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_pitch_setpoint_deg(self, t_s: np.float64) -> float:
        """ Observes the current pitch setpoint of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            float: Current pitch setpoint of the turbine (deg).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_torque_setpoint_nm(self, t_s: np.float64) -> float:
        """ Observes the current torque setpoint of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            float: Current torque setpoint of the turbine (Nm).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_rotor_speed_setpoint_radps(self, t_s: np.float64) -> float:
        """ Observes the current rotor speed setpoint of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            float: Current rotor speed setpoint of the turbine (rpm).
        """
        raise NotImplementedError
    
    def obs_rotor_speed_setpoint_rpm(self, t_s: np.float64) -> float:
        """ Observes the current rotor speed setpoint of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            float: Current rotor speed setpoint of the turbine (rpm).
        """
        return self.obs_rotor_speed_setpoint_radps(t_s) * 60 / (2 * np.pi)

    @abstractmethod
    def obs_curtailment_factor(self, t_s: np.float64) -> float:
        """ Observes the current curtailment factor of the turbine. 0 means no curtailment, 1 means full curtailment.

        Args:
            t_s (np.float64): Current simulation time in seconds.
            
        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            float: Current curtailment factor of the turbine (dimensionless).
        """
        raise NotImplementedError

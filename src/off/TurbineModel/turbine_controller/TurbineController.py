from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from off.OFFModule import *

class TurbineController(ABC):
    """Base interface for local (per-turbine) controller."""
    
    MODULE_TYPE = "TurbineController"
    
    """ 
    ---------------------------------------
    Observables 
    --------------------------------------- 
    """

    @compatibility(CompatibilityLevel.NONE)    
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
    @compatibility(CompatibilityLevel.NONE)
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
    
    @compatibility(CompatibilityLevel.NONE)   
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
    
    @compatibility(CompatibilityLevel.NONE)   
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
    
    @compatibility(CompatibilityLevel.NONE)   
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
    
    @compatibility(CompatibilityLevel.NONE)   
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
    
    @compatibility(CompatibilityLevel.NONE)   
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
    
    @compatibility(CompatibilityLevel.OPTIONAL)   
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

    @compatibility(CompatibilityLevel.NONE)   
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
    
    @compatibility(CompatibilityLevel.NONE)   
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

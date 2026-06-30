import numpy as np
from __future__ import annotations

from abc import ABC, abstractmethod

from Utils import *

class FarmController_Base(ABC):
    """Base interface for farm-level controllers."""

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
        """ Returns a citation string for the farm controller.

        Default implementation returns example BibTeX entries for software and paper citations.

        Returns:
            str: Citation string for the farm controller.
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
            "obs_turbine_yaw_setpoint_deg": SupportType.NOT_SUPPORTED,
            "obs_turbine_power_setpoint_w": SupportType.NOT_SUPPORTED,
            "obs_curtailment_setpoint_percent": SupportType.NOT_SUPPORTED,
            "obs_curtailment_setpoint_W": SupportType.NOT_SUPPORTED,
            "obs_farm_wind_direction_deg": SupportType.NOT_SUPPORTED,
            "obs_farm_wind_direction_at_turbine_deg": SupportType.NOT_SUPPORTED,
            "obs_farm_wind_speed_mps": SupportType.NOT_SUPPORTED,
            "obs_farm_wind_speed_at_turbine_mps": SupportType.NOT_SUPPORTED
        }
    
    """
    ---------------------------------------
    Observables
    ---------------------------------------
    """
    
    @abstractmethod
    def obs_turbine_yaw_setpoint_deg(self, turbine_id: int, t_s: np.float64) -> np.float64:
        """ Abstract method to observe the current yaw setpoint of a specific turbine.

        Args:
            turbine_id (int): ID of the turbine for which to observe the yaw setpoint.
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.float64: Current yaw setpoint of the specified turbine (degrees).
        """
        raise NotImplementedError
    
    def obs_turbine_power_setpoint_w(self, turbine_id: int, t_s: np.float64) -> np.float64:
        """ Abstract method to observe the current power setpoint of a specific turbine.

        Args:
            turbine_id (int): ID of the turbine for which to observe the power setpoint.
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.float64: Current power setpoint of the specified turbine (W).
        """
        raise NotImplementedError

    def obs_curtailment_setpoint_percent(self, turbine_id: int, t_s: np.float64) -> np.float64:
        """ Abstract method to observe the current curtailment setpoint of a specific turbine.

        Args:
            turbine_id (int): ID of the turbine for which to observe the curtailment setpoint.
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.float64: Current curtailment setpoint of the specified turbine (percent).
        """
        raise NotImplementedError
    
    def obs_curtailment_setpoint_W(self, turbine_id: int, t_s: np.float64) -> np.float64:
        """ Abstract method to observe the current curtailment setpoint of a specific turbine in Watts.

        Args:
            turbine_id (int): ID of the turbine for which to observe the curtailment setpoint.
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.float64: Current curtailment setpoint of the specified turbine (W).
        """
        raise NotImplementedError

    def obs_farm_wind_direction_deg(self, t_s: np.float64) -> np.float64:
        """ Abstract method to observe the current wind direction at the farm level.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.float64: Current wind direction at the farm level (degrees).
        """
        raise NotImplementedError
    
    def obs_farm_wind_direction_at_turbine_deg(self, turbine_id: int, t_s: np.float64) -> np.float64:
        """ Abstract method to observe the current wind direction at the farm level, localized to a specific turbine.

        Args:
            turbine_id (int): ID of the turbine for which to observe the wind direction.
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.float64: Current wind direction at the farm level (degrees).
        """
        raise NotImplementedError
    
    def obs_farm_wind_speed_mps(self, t_s: np.float64) -> np.float64:
        """ Abstract method to observe the current wind speed at the farm level.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.float64: Current wind speed at the farm level (m/s).
        """
        raise NotImplementedError
    
    def obs_farm_wind_speed_at_turbine_mps(self, turbine_id: int, t_s: np.float64) -> np.float64:
        """ Abstract method to observe the current wind speed at the farm level, localized to a specific turbine.

        Args:
            turbine_id (int): ID of the turbine for which to observe the wind speed.
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.float64: Current wind speed at the farm level (m/s).
        """
        raise NotImplementedError


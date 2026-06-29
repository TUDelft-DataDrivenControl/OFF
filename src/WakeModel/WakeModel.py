from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from AtmosphericModel import AtmosphericModel

class WakeModel(ABC):
    """
    Base interface for wake deficit models.
    TODO: Rename to 'TurbineImpactModel' or something similar. 
    TODO: Should we add 'corrected' to observables?   
    """

    def __init__(self, atmospheric_model: AtmosphericModel) -> None:
        self._atmospheric_model = atmospheric_model


    @abstractmethod
    def step(self, dt: np.float64) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
    
    def get_citation(self) -> str:
        """ Returns a citation string for the wake model. Default implementation returns a generic citation.

        Returns:
            str: Citation string for the wake model.
        """
        return "No specific Wake Model citation available."

    """ 
    ---------------------------------------
    Observables 
    --------------------------------------- 
    """

    @abstractmethod
    def obs_u_mps(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: np.float64) -> np.ndarray:
        """ Abstract method to observe the u-component of the flow field at given positions.

        Args:
            x (np.ndarray): Positions in the x-direction (m)
            y (np.ndarray): Positions in the y-direction (m)
            z (np.ndarray): Positions in the z-direction (m)
            t (np.float64): Time (s)

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the u-component of the flow field at the corresponding position (m/s).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_v_mps(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: np.float64) -> np.ndarray:
        """ Abstract method to observe the v-component of the flow field at given positions.

        Args:
            x (np.ndarray): Positions in the x-direction (m)
            y (np.ndarray): Positions in the y-direction (m)
            z (np.ndarray): Positions in the z-direction (m)
            t (np.float64): Time (s)

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the v-component of the flow field at the corresponding position (m/s).
        """
        raise NotImplementedError

    def obs_w_mps(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: np.float64) -> np.ndarray:
        """ Observes the w-component of the flow field (vertical wind speed) at specified locations. Default implementation returns zeros.

        Args:
            x (np.ndarray): Positions in the x-direction (m)
            y (np.ndarray): Positions in the y-direction (m)
            z (np.ndarray): Positions in the z-direction (m)
            t (np.float64): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the w-component of the flow field at the corresponding position (m/s).
        """
        assert x.shape == y.shape == z.shape, "x, y, z must have the same shape"
        return np.zeros_like(x)

    def obs_horizontal_wind_speed_mps(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: np.float64) -> np.ndarray:
        """ Observes the horizontal wind speed at given positions. Default implementation calculates from internal (u,v) values. 

        Args:
            x (np.ndarray): Positions in the x-direction (m)
            y (np.ndarray): Positions in the y-direction (m)
            z (np.ndarray): Positions in the z-direction (m)
            t (np.float64): Time (s)

        Returns:
            np.ndarray: Horizontal wind speed at the given positions (m/s)
        """
        return np.sqrt(self.obs_u_mps(x, y, z, t)**2 + self.obs_v_mps(x, y, z, t)**2)

    def obs_horizontal_wind_dir_deg(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: np.float64) -> np.ndarray:
        """ Observes the horizontal wind direction at given positions. Default implementation calculates from internal (u,v) values.

        Args:
            x (np.ndarray): Positions in the x-direction (m)
            y (np.ndarray): Positions in the y-direction (m)
            z (np.ndarray): Positions in the z-direction (m)
            t (np.float64): Time (s)

        Returns:
            np.ndarray: Horizontal wind direction at the given positions (degrees)
        """
        return 270 - np.degrees(np.arctan2(self.obs_v_mps(x, y, z, t), self.obs_u_mps(x, y, z, t)))

    def obs_rotor_averaged_wind_speed_mps(self, x:np.array, y:np.array, z:np.array, R: np.ndarray, yaw_orientation: np.float64, t:np.float64, tilt: np.float64 = 0.0) -> np.ndarray:
        """ Observes the rotor-averaged wind speed at given positions. Default implementation calculates from internal ws values.

        Args:
            x (np.ndarray): Rotor center position in the x-direction (m)
            y (np.ndarray): Rotor center position in the y-direction (m)
            z (np.ndarray): Rotor center position in the z-direction (m)
            R (np.ndarray): Rotor radius at each position (m)
            yaw_orientation (np.float64): Yaw orientation (degrees)
            t (np.float64): Time (s)
            tilt (np.float64): Tilt angle (degrees) (optional, default=0.0)

        Returns:
            np.ndarray: Rotor-averaged wind speed at the given positions (m/s)
        """
        return self.obs_horizontal_wind_speed_mps(x, y, z, t)
    
    @abstractmethod
    def obs_blades_wind_speed_mps(self, x:np.array, y:np.array, z:np.array, R: np.ndarray, azimuth: np.ndarray, yaw_orientation: np.float64, t:np.float64, tilt: np.float64 = 0.0) -> np.ndarray:
        """ Observes the wind speed at the blades at given positions. Default implementation calculates from internal ws values.

        Args:
            x (np.ndarray): Rotor center position in the x-direction (m)
            y (np.ndarray): Rotor center position in the y-direction (m)
            z (np.ndarray): Rotor center position in the z-direction (m)
            R (np.ndarray): Rotor radius at each position (m)
            azimuth (np.ndarray): Azimuth angle of the blades at each position (degrees)
            yaw_orientation (np.float64): Yaw orientation (degrees)
            t (np.float64): Time (s)
            tilt (np.float64): Tilt angle (degrees) (optional, default=0.0)
        Returns:
            np.ndarray: Wind speed at the blades at the given positions (m/s)
        """
        raise NotImplementedError

    def obs_rotor_averaged_wind_dir_deg(self, x:np.array, y:np.array, z:np.array, R: np.ndarray, yaw_orientation: np.float64, t:np.float64, tilt: np.float64 = 0.0) -> np.ndarray:
        """ Observes the rotor-averaged wind direction at given positions. Default implementation calculates from internal wd values.

        Args:
            x (np.ndarray): Rotor center position in the x-direction (m)
            y (np.ndarray): Rotor center position in the y-direction (m)
            z (np.ndarray): Rotor center position in the z-direction (m)
            R (np.ndarray): Rotor radius at each position (m)
            yaw_orientation (np.float64): Yaw orientation (degrees)
            t (np.float64): Time (s)
            tilt (np.float64): Tilt angle (degrees) (optional, default=0.0)

        Returns:
            np.ndarray: Rotor-averaged wind direction at the given positions (degrees)
        """
        return self.obs_horizontal_wind_dir_deg(x, y, z, t)

    def obs_rotor_averaged_u_mps(self, x:np.array, y:np.array, z:np.array, R: np.float64, yaw_orientation: np.float64, t:np.float64, tilt: np.float64 = 0.0) -> np.ndarray:
        """ Observes the rotor-averaged u-component of the flow field at the given positions. Default implementation calculates from internal u values.

        Args:
            x (np.ndarray): Rotor center position in the x-direction (m)
            y (np.ndarray): Rotor center position in the y-direction (m)
            z (np.ndarray): Rotor center position in the z-direction (m)
            R (np.float64): Rotor radius (m)
            yaw_orientation (np.float64): Yaw orientation (degrees)
            t (np.float64): Time (s)
            tilt (np.float64): Tilt angle (degrees) (optional, default=0.0)

        Returns:
            np.ndarray: Rotor-averaged u-component of the flow field at the given positions (m/s)
        """
        return self.obs_u_mps(x, y, z, t)
    
    def obs_rotor_averaged_v_mps(self, x:np.array, y:np.array, z:np.array, R: np.float64, yaw_orientation: np.float64, t:np.float64, tilt: np.float64 = 0.0) -> np.ndarray:
        """ Observes the rotor-averaged v-component of the flow field at the given positions. Default implementation calculates from internal v values.

        Args:
            x (np.ndarray): Rotor center position in the x-direction (m)
            y (np.ndarray): Rotor center position in the y-direction (m)
            z (np.ndarray): Rotor center position in the z-direction (m)
            R (np.float64): Rotor radius (m)
            yaw_orientation (np.float64): Yaw orientation (degrees)
            t (np.float64): Time (s)
            tilt (np.float64): Tilt angle (degrees) (optional, default=0.0)

        Returns:
            np.ndarray: Rotor-averaged v-component of the flow field at the given positions (m/s)
        """
        return self.obs_v_mps(x, y, z, t)
    
    def obs_rotor_averaged_w_mps(self, x:np.array, y:np.array, z:np.array, R: np.float64, yaw_orientation: np.float64, t:np.float64, tilt: np.float64 = 0.0) -> np.ndarray:
        """ Observes the rotor-averaged w-component of the flow field at the given positions. Default implementation calculates from internal w values.

        Args:
            x (np.ndarray): Rotor center position in the x-direction (m)
            y (np.ndarray): Rotor center position in the y-direction (m)
            z (np.ndarray): Rotor center position in the z-direction (m)
            R (np.float64): Rotor radius (m)
            yaw_orientation (np.float64): Yaw orientation (degrees)
            t (np.float64): Time (s)
            tilt (np.float64): Tilt angle (degrees) (optional, default=0.0)

        Returns:
            np.ndarray: Rotor-averaged w-component of the flow field at the given positions (m/s)
        """
        return self.obs_w_mps(x, y, z, t)
    
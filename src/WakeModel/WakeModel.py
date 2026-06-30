from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from AtmosphericModel import AtmosphericModel
from Utils import *

class WakeModel(ABC):
    """
    Base interface for wake deficit models.
    TODO: Rename to 'TurbineImpactModel' or something similar. 
    TODO: Should we add 'corrected' to observables?   
    """

    def __init__(self, atmospheric_model: AtmosphericModel) -> None:
        self._atmospheric_model = atmospheric_model


    @abstractmethod
    def step(self, it: int) -> None:
        """ Advances the wake model by a given number of iterations.

        Args:
            it (int): Current iteration of the simulation. The current real time since simulation start is it * dt, where dt is the global time step.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
    
    def req_describe(self) -> dict[str, SupportType]:
        """ Returns a dictionary describing the wake model. Default implementation returns an empty dictionary.

        Returns:
            dict[str, Any]: Dictionary describing the wake model.
        """
        return {
            "obs uwv mps":                                  SupportType.NOT_SUPPORTED,
            "obs uv mps":                                   SupportType.NOT_SUPPORTED,
            "obs horizontal wind speed mps":                SupportType.NOT_SUPPORTED,
            "obs horizontal wind direction deg":            SupportType.NOT_SUPPORTED,
            "obs horizontal wind speed and direction mps deg": SupportType.NOT_SUPPORTED,
            "obs rotor-averaged wind speed mps":            SupportType.NOT_SUPPORTED,
            "obs rotor-averaged wind direction deg":        SupportType.NOT_SUPPORTED,
            "obs rotor-averaged uvw mps":                   SupportType.NOT_SUPPORTED,
            "obs rotor-averaged uv mps":                    SupportType.NOT_SUPPORTED,
            "obs blades uvw wind speed mps":                SupportType.NOT_SUPPORTED,
            "obs rotor averaged wind dir deg":              SupportType.NOT_SUPPORTED,
            "obs rotor averaged uvw mps":                   SupportType.NOT_SUPPORTED,
            "obs rotor averaged uv mps":                    SupportType.NOT_SUPPORTED,
        }
    
    def req_check_component(self, component) -> bool:
        """ Checks if the given component is compatible with the wake model. Default implementation returns False.

        Args:
            component: Component to check for compatibility.

        Returns:
            bool: True if the component is compatible, False otherwise.
        """
        return False
    
    def get_citation(self) -> str:
        """ Returns a citation string for the wake model. Default implementation returns a generic citation.

        Returns:
            str: Citation string for the wake model.
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

    """ 
    ---------------------------------------
    Observables 
    --------------------------------------- 
    """

    @abstractmethod
    def obs_uvw_mps(self, xyz: np.ndarray, t_s: np.float64) -> np.ndarray:
        """ Abstract method to observe the u-, v-, and w-components of the flow field at given positions.

        Args:
            xyz (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and columns [x, y, z].
            t_s (np.float64): Time (s)

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.ndarray: Array of shape (3, N) where N is the number of positions. Each row contains the u-, v-, and w-components of the flow field at the corresponding position (m/s).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_uv_mps(self, xyz: np.ndarray, t_s: np.float64) -> np.ndarray:
        """ Abstract method to observe the u- and v-components of the flow field at given positions.

        Args:
            xyz (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (np.float64): Time (s)

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.ndarray: Array of shape (2, N) where N is the number of positions. Each row contains the u- and v-components of the flow field at the corresponding position (m/s).
        """
        raise NotImplementedError

    def obs_horizontal_wind_speed_mps(self, xyz: np.ndarray, t_s: np.float64) -> np.ndarray:
        """ Observes the horizontal wind speed at given positions. Default implementation calculates from internal (u,v) values. 

        Args:
            xyz (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (np.float64): Time (s)

        Returns:
            np.ndarray: Horizontal wind speed at the given positions (m/s)
        """
        
        return np.sqrt(np.sum(self.obs_uv_mps(xyz, t_s)**2, axis=1))

    def obs_horizontal_wind_dir_deg(self, xyz: np.ndarray, t_s: np.float64) -> np.ndarray:
        """ Observes the horizontal wind direction at given positions. Default implementation calculates from internal (u,v) values.

        Args:
            xyz (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (np.float64): Time (s)

        Returns:
            np.ndarray: Horizontal wind direction at the given positions (degrees)
        """
        uv = self.obs_uv_mps(xyz, t_s)
        return 270 - np.degrees(np.arctan2(uv[1,:], uv[0,:]))

    def obs_horizontal_wind_speed_and_dir_mps_deg(self, xyz: np.ndarray, t_s: np.float64) -> np.ndarray:
        """ Observes the horizontal wind speed and direction at given positions. Default implementation calculates from internal (u,v) values.

        Args:
            xyz (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (np.float64): Time (s)

        Returns:
            np.ndarray: Array of shape (2, N) where N is the number of positions. The first row contains the horizontal wind speed at the corresponding position (m/s), and the second row contains the horizontal wind direction at the corresponding position (degrees).
        """
        uv = self.obs_uv_mps(xyz, t_s)
        return np.vstack((np.sqrt(np.sum(uv**2, axis=1)), 270 - np.degrees(np.arctan2(uv[1,:], uv[0,:]))))

    def obs_rotor_averaged_wind_speed_mps(self, xyz: np.ndarray, R_m: np.float64, yaw_orientation_deg: np.float64, t_s: np.float64, tilt_deg: np.float64 = 0.0) -> np.ndarray:
        """ Observes the rotor-averaged wind speed at given positions. Default implementation calculates from internal ws values.

        Args:
            xyz (np.ndarray): Rotor center positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            R_m (np.float64): Rotor radius (m)
            yaw_orientation_deg (np.float64): Yaw orientation (degrees)
            t_s (np.float64): Time (s)
            tilt_deg (np.float64): Tilt angle (degrees) (optional, default=0.0)

        Returns:
            np.ndarray: Rotor-averaged wind speed at the given positions (m/s)
        """
        raise NotImplementedError

    @abstractmethod
    def obs_blades_uvw_wind_speed_mps(self, xyz: np.ndarray, R_m: np.ndarray, azimuth_deg: np.ndarray, yaw_orientation_deg: np.float64, t_s: np.float64, tilt_deg: np.float64 = 0.0, n_blades: int = 3) -> np.ndarray:
        """ Observes the wind speed at the blades at given positions. Default implementation calculates from internal ws values.

        Args:
            xyz (np.ndarray): Rotor center positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            R_m (np.ndarray): Rotor radii to calculate uvw for (m)
            azimuth_deg (np.ndarray): Azimuth angle of the blades at each position (degrees)
            yaw_orientation_deg (np.float64): Yaw orientation (degrees)
            t_s (np.float64): Time (s)
            tilt_deg (np.float64): Tilt angle (degrees) (optional, default=0.0)
            n_blades (int): Number of blades (optional, default=3)
        Returns:
            np.ndarray: Wind speed at the blades at the given positions (m/s)
        """
        
        # The azimuth angle is defined clockwise from the positive y-axis, as viewed from the front, with 0 degrees meaning that blade 0 is pointing in the positive y-direction.
        # The yaw orientation is defined clockwise from the positive x-axis, with 0 degrees meaning that the rotor plane is facing the positive x-direction.
        # Tilt is defined as the angle between the rotor plane and the horizontal plane, with 0 degrees meaning that the rotor plane is horizontal, >0 degrees meaning that the rotor plane is tilted backwards.
        raise NotImplementedError
        
    def obs_rotor_averaged_wind_dir_deg(self, xyz: np.ndarray, R_m: np.ndarray, yaw_orientation_deg: np.float64, t_s: np.float64, tilt_deg: np.float64 = 0.0) -> np.ndarray:
        """ Observes the rotor-averaged wind direction at given positions. Default implementation calculates from internal wd values.

        Args:
            xyz (np.ndarray): Rotor center positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            R_m (np.ndarray): Rotor radius at each position (m)
            yaw_orientation_deg (np.float64): Yaw orientation (degrees)
            t_s (np.float64): Time (s)
            tilt_deg (np.float64): Tilt angle (degrees) (optional, default=0.0)

        Returns:
            np.ndarray: Rotor-averaged wind direction at the given positions (degrees)
        """
        return self.obs_horizontal_wind_dir_deg(xyz, t_s)

    def obs_rotor_averaged_uvw_mps(self, xyz: np.ndarray, R_m: np.float64, yaw_orientation_deg: np.float64, t_s: np.float64, tilt_deg: np.float64 = 0.0) -> np.ndarray:
        """ Observes the rotor-averaged u, v, and w components of the flow field at the given positions. Default implementation calculates from internal u, v, and w values.

        Args:
            xyz (np.ndarray): Rotor center positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            R_m (np.float64): Rotor radius (m)
            yaw_orientation_deg (np.float64): Yaw orientation (degrees)
            t_s (np.float64): Time (s)
            tilt_deg (np.float64): Tilt angle (degrees) (optional, default=0.0)

        Returns:
            np.ndarray: Rotor-averaged u, v, and w components of the flow field at the given positions (m/s)
        """
        return self.obs_uvw_mps(xyz, t_s)

    def obs_rotor_averaged_uv_mps(self, xyz: np.ndarray, R_m: np.float64, yaw_orientation_deg: np.float64, t_s: np.float64, tilt_deg: np.float64 = 0.0) -> np.ndarray:
        """ Observes the rotor-averaged u and v components of the flow field at the given positions. Default implementation calculates from internal u and v values.

        Args:
            xyz (np.ndarray): Rotor center positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            R_m (np.float64): Rotor radius (m)
            yaw_orientation_deg (np.float64): Yaw orientation (degrees)
            t_s (np.float64): Time (s)
            tilt_deg (np.float64): Tilt angle (degrees) (optional, default=0.0)

        Returns:
            np.ndarray: Rotor-averaged u and v components of the flow field at the given positions (m/s)
        """
        return self.obs_uv_mps(xyz, t_s)
    
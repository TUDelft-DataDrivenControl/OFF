from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from off.AtmosphericModel import AtmosphericModel
from off.OFFModule import *

class WakeModel(OFFModule):
    """
    Base interface for wake deficit models.
    TODO: Rename to 'TurbineImpactModel' or something similar. 
    TODO: Should we add 'corrected' to observables?   
    """

    MODULE_TYPE = "WakeModel"

    """ 
    ---------------------------------------
    Observables 
    --------------------------------------- 
    """

    @abstractmethod
    @compatibility(CompatibilityLevel.NONE)
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
    @compatibility(CompatibilityLevel.NONE)
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

    @compatibility(CompatibilityLevel.OPTIONAL)
    def obs_horizontal_wind_speed_mps(self, xyz: np.ndarray, t_s: np.float64) -> np.ndarray:
        """ Observes the horizontal wind speed at given positions. Default implementation calculates from internal (u,v) values. 

        Args:
            xyz (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (np.float64): Time (s)

        Returns:
            np.ndarray: Horizontal wind speed at the given positions (m/s)
        """
        
        return np.sqrt(np.sum(self.obs_uv_mps(xyz, t_s)**2, axis=1))

    @compatibility(CompatibilityLevel.OPTIONAL)
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

    @compatibility(CompatibilityLevel.OPTIONAL)
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

    @compatibility(CompatibilityLevel.NONE)
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
    @compatibility(CompatibilityLevel.NONE)
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
        
    @compatibility(CompatibilityLevel.OPTIONAL)
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

    @compatibility(CompatibilityLevel.OPTIONAL)
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

    @compatibility(CompatibilityLevel.OPTIONAL)
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
    

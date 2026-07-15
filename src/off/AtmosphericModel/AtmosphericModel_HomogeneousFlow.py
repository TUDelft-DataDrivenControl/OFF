from __future__ import annotations
import numpy as np

from .AtmosphericModel import AtmosphericModel
from off.OFFModule import *

class AtmosphericModel_HomogeneousFlow(AtmosphericModel):
    """Steady and spatially uniform atmospheric model."""

    wind_speed_abs_mps: float
    wind_dir_deg: float
    turbulence_intensity_percent: float
    wind_shear_exponent: float
    wind_veer_factor_degpm: float
    atmospheric_stability_L_m: float
    air_density_kgpm3: float
    reference_height_m: float

    _initial_state_wind_speed_abs_mps: float
    _initial_state_wind_dir_deg: float
    _initial_state_turbulence_intensity_percent: float    
    _initial_state_wind_shear_exponent: float
    _initial_state_wind_veer_factor_degpm: float
    _initial_state_atmospheric_stability_L_m: float  
    _initial_state_air_density_kgpm3: float 

    def __init__(self, settings_AtmosphericModel: dict) -> None:
        super().__init__()
        self.wind_speed_abs_mps             = settings_AtmosphericModel["wind_speed_mps"][0]
        self.wind_dir_deg                   = settings_AtmosphericModel["wind_direction_degrees"][0]
        self.turbulence_intensity_percent   = settings_AtmosphericModel["turbulence_intensities_percent"][0]
        self.wind_shear_exponent            = settings_AtmosphericModel["wind_shear_exponent"][0]
        self.wind_veer_factor_degpm         = settings_AtmosphericModel["wind_veer_factors_degpm"][0]
        self.atmospheric_stability_L_m      = settings_AtmosphericModel["atmospheric_stability_L_m"][0]
        self.air_density_kgpm3              = settings_AtmosphericModel["air_density_kgpm3"]

        self.reference_height_m             = settings_AtmosphericModel["reference_height_m"]

        # Store initial state for reset
        self._initial_state_wind_speed_abs_mps             = settings_AtmosphericModel["wind_speed_mps"][0]
        self._initial_state_wind_dir_deg                   = settings_AtmosphericModel["wind_direction_degrees"][0]
        self._initial_state_turbulence_intensity_percent   = settings_AtmosphericModel["turbulence_intensities_percent"][0]
        self._initial_state_wind_shear_exponent            = settings_AtmosphericModel["wind_shear_exponent"][0]
        self._initial_state_wind_veer_factor_degpm         = settings_AtmosphericModel["wind_veer_factors_degpm"][0]
        self._initial_state_atmospheric_stability_L_m      = settings_AtmosphericModel["atmospheric_stability_L_m"][0]
        self._initial_state_air_density_kgpm3              = settings_AtmosphericModel["air_density_kgpm3"]

    def step(self, dt: float) -> None:
        return None

    def reset(self) -> None:
        self.wind_speed_abs_mps             = self._initial_state_wind_speed_abs_mps
        self.wind_dir_deg                   = self._initial_state_wind_dir_deg
        self.turbulence_intensity_percent   = self._initial_state_turbulence_intensity_percent
        self.wind_shear_exponent            = self._initial_state_wind_shear_exponent
        self.wind_veer_factor_degpm         = self._initial_state_wind_veer_factor_degpm
        self.atmospheric_stability_L_m      = self._initial_state_atmospheric_stability_L_m
        self.air_density_kgpm3              = self._initial_state_air_density_kgpm3

    @compatibility(CompatibilityLevel.OPTIONAL)
    def obs_u_mps(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the u-component of the wind velocity at given positions.
        Default implementation calculates from internal velocity, direction, shear and veer values.
        
        Args:
            xyz_m (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (float): Time (s) (not used in the homogeneous model)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the u-component of the wind velocity at the corresponding position (m/s).
        """
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        # Calculate u based on wind speed, direction and veer factor
        u = self.wind_speed_abs_mps * np.cos(np.radians(270.0 - (self.wind_dir_deg + (xyz_m[2, :] - self.reference_height_m) * self.wind_veer_factor_degpm)))
        u *= (xyz_m[2, :] / self.reference_height_m) ** self.wind_shear_exponent
        return u

    @compatibility(CompatibilityLevel.OPTIONAL)
    def obs_v_mps(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the v-component of the wind velocity at given positions.
        Default implementation calculates from internal velocity, direction, shear and veer values.

        Args:
            xyz_m (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (float): Time (s) (not used in the homogeneous model)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the v-component of the wind velocity at the corresponding position (m/s).
        """
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        v = self.wind_speed_abs_mps * np.sin(np.radians(270.0 - (self.wind_dir_deg + (xyz_m[2, :] - self.reference_height_m) * self.wind_veer_factor_degpm)))
        v *= (xyz_m[2, :] / self.reference_height_m) ** self.wind_shear_exponent
        return v
    
    @compatibility(CompatibilityLevel.OPTIONAL)
    def obs_uv_mps(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the u- and v-components of the wind velocity at given positions.
        Default implementation calculates from internal velocity, direction, shear and veer values.

        Args:
            xyz_m (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (float): Time (s) (not used in the homogeneous model)

        Returns:
            np.ndarray: Array of shape (2, N) where N is the number of positions. Each row contains the wind velocity components (u, v) at the corresponding position (m/s).
        """
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        u = self.obs_u_mps(xyz_m, t_s)
        v = self.obs_v_mps(xyz_m, t_s)
        return np.vstack([u, v])
    
    @compatibility(CompatibilityLevel.OPTIONAL)
    def obs_uvw_mps(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the u-, v-, and w-components of the wind velocity at given positions.
        Default implementation calculates from internal velocity, direction, shear and veer values.

        Args:
            xyz_m (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (float): Time (s) (not used in the homogeneous model)

        Returns:
            np.ndarray: Array of shape (3, N) where N is the number of positions. Each row contains the wind velocity components (u, v, w) at the corresponding position (m/s).
        """
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        u = self.obs_u_mps(xyz_m, t_s)
        v = self.obs_v_mps(xyz_m, t_s)
        w = np.zeros(xyz_m.shape[1])
        return np.vstack([u, v, w])

    @compatibility(CompatibilityLevel.FULL)
    def obs_horizontal_wind_dir_deg(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the horizontal wind direction at given positions.
        Takes veer linearly into account, based on a reference height and a veer factor (deg/m).

        Args:
            xyz_m (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (float): Time (s) (not used in the homogeneous model)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the horizontal wind direction at the corresponding position (degrees).
        """
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        return (self.wind_dir_deg + (xyz_m[2, :] - self.reference_height_m) * self.wind_veer_factor_degpm)
    
    @compatibility(CompatibilityLevel.FULL)
    def obs_horizontal_wind_speed_mps(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the horizontal wind speed at given positions.
        Takes shear with the power law into account.

        Args:
            xyz_m (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (float): Time (s) (not used in the homogeneous model)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the horizontal wind speed at the corresponding position (m/s).
        """
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        return self.wind_speed_abs_mps * (xyz_m[2, :] / self.reference_height_m) ** self.wind_shear_exponent

    @compatibility(CompatibilityLevel.FULL)
    def obs_turbulence_intensity_percent(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the turbulence intensity at given positions.
        Args:
            xyz_m (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (float): Time (s) (not used in the homogeneous model)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the turbulence intensity at the corresponding position (percent).
        """
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        return np.full(xyz_m.shape[1], self.turbulence_intensity_percent)
    
    @compatibility(CompatibilityLevel.FULL)
    def obs_atmospheric_stability_L_m(self, xyz_m: np.ndarray, t_s: float) -> float:
        """ Returns the atmospheric stability parameter (Monin-Obukhov length) in meters.
        Args:
            xyz_m (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (float): Time (s) (not used in the homogeneous model)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the atmospheric stability parameter at the corresponding position (m).
        """
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        return np.ones(xyz_m.shape[1]) * self.atmospheric_stability_L_m
    
    @compatibility(CompatibilityLevel.FULL)
    def obs_air_density_kgpm3(self, xyz_m: np.ndarray, t_s: float) -> float:
        """ Returns the air density at given positions.
        Args:
            xyz_m (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (float): Time (s) (not used in the homogeneous model)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the air density at the corresponding position (kg/m^3).
        """
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        return np.ones(xyz_m.shape[1]) * self.air_density_kgpm3

    @compatibility(CompatibilityLevel.FULL)
    def obs_pressure_Pa(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the atmospheric pressure at given positions.

        Args:
            xyz_m (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (float): Time (s) (not used in the homogeneous model)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the atmospheric pressure at the corresponding position (Pa).
        """
        return self.surface_pressure_Pa * np.ones(xyz_m.shape[1])
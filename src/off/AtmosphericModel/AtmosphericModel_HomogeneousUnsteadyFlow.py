from __future__ import annotations
import numpy as np

from .AtmosphericModel import AtmosphericModel
from off.OFFModule import *

class AtmosphericModel_HomogeneousFlow(AtmosphericModel):
    """Spatially uniform atmospheric model.
    Allows for time-varying atmospheric conditions, but assumes that the atmospheric state is uniform across the spatial domain.
    The resulting wind field is homogeneous and setpoints are linearly interpolated based on the provided time steps and values.
    If the requested time is outside the provided time steps, the first or last value is used, depending on whether the time is before or after the provided range.
    """

    wind_speed_abs_mps: np.ndarray
    wind_speed_timestep_s: np.ndarray

    wind_dir_deg: np.ndarray
    wind_dir_timestep_s: np.ndarray

    turbulence_intensities_percent: np.ndarray
    turbulence_intensities_timestep_s: np.ndarray

    wind_shear_exponent: np.ndarray
    wind_shear_timestep_s: np.ndarray

    wind_veer_factor_degpm: np.ndarray
    wind_veer_timestep_s: np.ndarray

    atmospheric_stability_L_m: np.ndarray
    atmospheric_stability_timestep_s: np.ndarray

    air_density_kgpm3: float
    reference_height_m: float

    _initial_state_wind_speed_abs_mps: float
    _initial_state_wind_speed_timestep_s: float

    _initial_state_wind_dir_deg: float
    _initial_state_wind_dir_timestep_s: float

    _initial_state_turbulence_intensities_percent: float    
    _initial_state_turbulence_intensities_timestep_s: float

    _initial_state_wind_shear_exponent: float
    _initial_state_wind_shear_timestep_s: float

    _initial_state_wind_veer_factor_degpm: float
    _initial_state_wind_veer_timestep_s: float

    _initial_state_atmospheric_stability_L_m: float  
    _initial_state_air_density_kgpm3: float 

    def __init__(self, settings_AtmosphericModel: dict) -> None:
        super().__init__()
        self.wind_speed_abs_mps                 = np.array(settings_AtmosphericModel["wind_speed_mps"])
        self.wind_speed_timestep_s              = np.array(settings_AtmosphericModel["wind_speed_timestep_s"])
        self.wind_dir_deg                       = np.array(settings_AtmosphericModel["wind_direction_degrees"])
        self.wind_dir_timestep_s                = np.array(settings_AtmosphericModel["wind_direction_timestep_s"])
        self.turbulence_intensities_percent     = np.array(settings_AtmosphericModel["turbulence_intensities_percent"])
        self.turbulence_intensities_timestep_s  = np.array(settings_AtmosphericModel["turbulence_intensities_timestep_s"])
        self.wind_shear_exponent                = np.array(settings_AtmosphericModel["wind_shear_exponent"])
        self.wind_shear_timestep_s              = np.array(settings_AtmosphericModel["wind_shear_timestep_s"])
        self.wind_veer_factor_degpm             = np.array(settings_AtmosphericModel["wind_veer_factors_degpm"])
        self.wind_veer_timestep_s               = np.array(settings_AtmosphericModel["wind_veer_timestep_s"])
        self.atmospheric_stability_L_m          = np.array(settings_AtmosphericModel["atmospheric_stability_L_m"])
        self.atmospheric_stability_timestep_s   = np.array(settings_AtmosphericModel["atmospheric_stability_timestep_s"])

        self.air_density_kgpm3              = settings_AtmosphericModel["air_density_kgpm3"]
        self.reference_height_m             = settings_AtmosphericModel["reference_height_m"]

        # Store initial state for reset
        self._initial_state_wind_speed_abs_mps                  = np.array(settings_AtmosphericModel["wind_speed_mps"])
        self._initial_state_wind_speed_timestep_s               = np.array(settings_AtmosphericModel["wind_speed_timestep_s"])
        self._initial_state_wind_dir_deg                        = np.array(settings_AtmosphericModel["wind_direction_degrees"])
        self._initial_state_wind_dir_timestep_s                 = np.array(settings_AtmosphericModel["wind_direction_timestep_s"])
        self._initial_state_turbulence_intensities_percent      = np.array(settings_AtmosphericModel["turbulence_intensities_percent"])
        self._initial_state_turbulence_intensities_timestep_s   = np.array(settings_AtmosphericModel["turbulence_intensities_timestep_s"])
        self._initial_state_wind_shear_exponent                 = np.array(settings_AtmosphericModel["wind_shear_exponent"])
        self._initial_state_wind_shear_timestep_s               = np.array(settings_AtmosphericModel["wind_shear_timestep_s"])
        self._initial_state_wind_veer_factor_degpm              = np.array(settings_AtmosphericModel["wind_veer_factors_degpm"])
        self._initial_state_wind_veer_timestep_s                = np.array(settings_AtmosphericModel["wind_veer_timestep_s"])
        self._initial_state_atmospheric_stability_L_m           = np.array(settings_AtmosphericModel["atmospheric_stability_L_m"])
        self._initial_state_atmospheric_stability_timestep_s    = np.array(settings_AtmosphericModel["atmospheric_stability_timestep_s"])
        self._initial_state_air_density_kgpm3                   = np.array(settings_AtmosphericModel["air_density_kgpm3"])

    def step(self, dt: float) -> None:
        return None

    def reset(self) -> None:
        self.wind_speed_abs_mps             = self._initial_state_wind_speed_abs_mps
        self.wind_dir_deg                   = self._initial_state_wind_dir_deg
        self.turbulence_intensities_percent = self._initial_state_turbulence_intensities_percent
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
            t_s (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the u-component of the wind velocity at the corresponding position (m/s).
        """
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"

        # Linearly interpolate wind speed and direction based on time t_s
        if t_s < self.wind_speed_timestep_s[0]:
            wind_speed = self.wind_speed_abs_mps[0]
            wind_dir   = self.wind_dir_deg[0]
            wind_veer  = self.wind_veer_factor_degpm[0]
            wind_shear = self.wind_shear_exponent[0]
        elif t_s > self.wind_speed_timestep_s[-1]:
            wind_speed = self.wind_speed_abs_mps[-1]
            wind_dir   = self.wind_dir_deg[-1]
            wind_veer  = self.wind_veer_factor_degpm[-1]
            wind_shear = self.wind_shear_exponent[-1]
        else:
            wind_speed = np.interp(t_s, self.wind_speed_timestep_s, self.wind_speed_abs_mps)
            wind_dir   = np.interp(t_s, self.wind_dir_timestep_s, self.wind_dir_deg)
            wind_veer  = np.interp(t_s, self.wind_veer_timestep_s, self.wind_veer_factor_degpm)
            wind_shear = np.interp(t_s, self.wind_shear_timestep_s, self.wind_shear_exponent)

        # Calculate u based on wind speed, direction and veer factor
        u = wind_speed * np.cos(np.radians(270.0 - (wind_dir + (xyz_m[2, :] - self.reference_height_m) * wind_veer)))
        u *= (xyz_m[2, :] / self.reference_height_m) ** wind_shear
        return u

    @compatibility(CompatibilityLevel.OPTIONAL)
    def obs_v_mps(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the v-component of the wind velocity at given positions.
        Default implementation calculates from internal velocity, direction, shear and veer values.

        Args:
            xyz_m (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the v-component of the wind velocity at the corresponding position (m/s).
        """
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        # Linearly interpolate wind speed and direction based on time t_s
        if t_s < self.wind_speed_timestep_s[0]:
            wind_speed = self.wind_speed_abs_mps[0]
            wind_dir   = self.wind_dir_deg[0]
            wind_veer  = self.wind_veer_factor_degpm[0]
            wind_shear = self.wind_shear_exponent[0]
        elif t_s > self.wind_speed_timestep_s[-1]:
            wind_speed = self.wind_speed_abs_mps[-1]
            wind_dir   = self.wind_dir_deg[-1]
            wind_veer  = self.wind_veer_factor_degpm[-1]
            wind_shear = self.wind_shear_exponent[-1]
        else:
            wind_speed = np.interp(t_s, self.wind_speed_timestep_s, self.wind_speed_abs_mps)
            wind_dir   = np.interp(t_s, self.wind_dir_timestep_s, self.wind_dir_deg)
            wind_veer  = np.interp(t_s, self.wind_veer_timestep_s, self.wind_veer_factor_degpm)
            wind_shear = np.interp(t_s, self.wind_shear_timestep_s, self.wind_shear_exponent)

        v = wind_speed * np.sin(np.radians(270.0 - (wind_dir + (xyz_m[2, :] - self.reference_height_m) * wind_veer)))
        v *= (xyz_m[2, :] / self.reference_height_m) ** wind_shear
        return v
    
    @compatibility(CompatibilityLevel.FULL)
    def obs_uv_mps(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the u- and v-components of the wind velocity at given positions.
        Default implementation calculates from internal velocity, direction, shear and veer values.

        Args:
            xyz_m (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (float): Time (s)

        Returns:
            np.ndarray: Array of shape (2, N) where N is the number of positions. Each row contains the wind velocity components (u, v) at the corresponding position (m/s).
        """
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        # Linearly interpolate wind speed and direction based on time t_s
        if t_s < self.wind_speed_timestep_s[0]:
            wind_speed = self.wind_speed_abs_mps[0]
            wind_dir   = self.wind_dir_deg[0]
            wind_veer  = self.wind_veer_factor_degpm[0]
            wind_shear = self.wind_shear_exponent[0]
        elif t_s > self.wind_speed_timestep_s[-1]:
            wind_speed = self.wind_speed_abs_mps[-1]
            wind_dir   = self.wind_dir_deg[-1]
            wind_veer  = self.wind_veer_factor_degpm[-1]
            wind_shear = self.wind_shear_exponent[-1]
        else:
            wind_speed = np.interp(t_s, self.wind_speed_timestep_s, self.wind_speed_abs_mps)
            wind_dir   = np.interp(t_s, self.wind_dir_timestep_s, self.wind_dir_deg)
            wind_veer  = np.interp(t_s, self.wind_veer_timestep_s, self.wind_veer_factor_degpm)
            wind_shear = np.interp(t_s, self.wind_shear_timestep_s, self.wind_shear_exponent)

        u = wind_speed * np.cos(np.radians(270.0 - (wind_dir + (xyz_m[2, :] - self.reference_height_m) * wind_veer)))
        u *= (xyz_m[2, :] / self.reference_height_m) ** wind_shear

        v = wind_speed * np.sin(np.radians(270.0 - (wind_dir + (xyz_m[2, :] - self.reference_height_m) * wind_veer)))
        v *= (xyz_m[2, :] / self.reference_height_m) ** wind_shear

        return np.vstack([u, v])
    
    @compatibility(CompatibilityLevel.FULL)
    def obs_uvw_mps(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the u-, v-, and w-components of the wind velocity at given positions.
        Default implementation calculates from internal velocity, direction, shear and veer values.

        Args:
            xyz_m (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (float): Time (s)

        Returns:
            np.ndarray: Array of shape (3, N) where N is the number of positions. Each row contains the wind velocity components (u, v, w) at the corresponding position (m/s).
        """
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"

        # Linearly interpolate wind speed and direction based on time t_s
        if t_s < self.wind_speed_timestep_s[0]:
            wind_speed = self.wind_speed_abs_mps[0]
            wind_dir   = self.wind_dir_deg[0]
            wind_veer  = self.wind_veer_factor_degpm[0]
            wind_shear = self.wind_shear_exponent[0]
        elif t_s > self.wind_speed_timestep_s[-1]:
            wind_speed = self.wind_speed_abs_mps[-1]
            wind_dir   = self.wind_dir_deg[-1]
            wind_veer  = self.wind_veer_factor_degpm[-1]
            wind_shear = self.wind_shear_exponent[-1]
        else:
            wind_speed = np.interp(t_s, self.wind_speed_timestep_s, self.wind_speed_abs_mps)
            wind_dir   = np.interp(t_s, self.wind_dir_timestep_s, self.wind_dir_deg)
            wind_veer  = np.interp(t_s, self.wind_veer_timestep_s, self.wind_veer_factor_degpm)
            wind_shear = np.interp(t_s, self.wind_shear_timestep_s, self.wind_shear_exponent)

        u = wind_speed * np.cos(np.radians(270.0 - (wind_dir + (xyz_m[2, :] - self.reference_height_m) * wind_veer)))
        u *= (xyz_m[2, :] / self.reference_height_m) ** wind_shear

        v = wind_speed * np.sin(np.radians(270.0 - (wind_dir + (xyz_m[2, :] - self.reference_height_m) * wind_veer)))
        v *= (xyz_m[2, :] / self.reference_height_m) ** wind_shear

        w = np.zeros(xyz_m.shape[1])

        return np.vstack([u, v, w])

    @compatibility(CompatibilityLevel.FULL)
    def obs_horizontal_wind_dir_deg(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the horizontal wind direction at given positions.
        Takes veer linearly into account, based on a reference height and a veer factor (deg/m).

        Args:
            xyz_m (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the horizontal wind direction at the corresponding position (degrees).
        """
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"

        if t_s < self.wind_dir_timestep_s[0]:
            wind_dir = self.wind_dir_deg[0]
            wind_veer = self.wind_veer_factor_degpm[0]
        elif t_s > self.wind_dir_timestep_s[-1]:
            wind_dir = self.wind_dir_deg[-1]
            wind_veer = self.wind_veer_factor_degpm[-1]
        else:
            wind_dir  = np.interp(t_s, self.wind_dir_timestep_s, self.wind_dir_deg)
            wind_veer = np.interp(t_s, self.wind_veer_timestep_s, self.wind_veer_factor_degpm)

        return wind_dir + (xyz_m[2, :] - self.reference_height_m) * wind_veer
    
    @compatibility(CompatibilityLevel.FULL)
    def obs_horizontal_wind_speed_mps(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the horizontal wind speed at given positions.
        Takes shear with the power law into account.

        Args:
            xyz_m (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the horizontal wind speed at the corresponding position (m/s).
        """
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"

        if t_s < self.wind_speed_timestep_s[0]:
            wind_speed = self.wind_speed_abs_mps[0]
            wind_shear = self.wind_shear_exponent[0]
        elif t_s > self.wind_speed_timestep_s[-1]:
            wind_speed = self.wind_speed_abs_mps[-1]
            wind_shear = self.wind_shear_exponent[-1]
        else:
            wind_speed = np.interp(t_s, self.wind_speed_timestep_s, self.wind_speed_abs_mps)
            wind_shear = np.interp(t_s, self.wind_shear_timestep_s, self.wind_shear_exponent)
        
        return wind_speed * (xyz_m[2, :] / self.reference_height_m) ** wind_shear

    @compatibility(CompatibilityLevel.FULL)
    def obs_turbulence_intensity_percent(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the turbulence intensity at given positions.
        Args:
            xyz_m (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the turbulence intensity at the corresponding position (percent).
        """
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        if t_s < self.turbulence_intensities_timestep_s[0]:
            return np.full(xyz_m.shape[1], self.turbulence_intensities_percent[0])
        elif t_s > self.turbulence_intensities_timestep_s[-1]:
            return np.full(xyz_m.shape[1], self.turbulence_intensities_percent[-1])
        else:
            return np.full(xyz_m.shape[1], np.interp(t_s, self.turbulence_intensities_timestep_s, self.turbulence_intensities_percent))
    
    @compatibility(CompatibilityLevel.FULL)
    def obs_atmospheric_stability_L_m(self, xyz_m: np.ndarray, t_s: float) -> float:
        """ Returns the atmospheric stability parameter (Monin-Obukhov length) in meters.
        Args:
            xyz_m (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the atmospheric stability parameter at the corresponding position (m).
        """
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"

        # Linearly interpolate atmospheric stability parameter based on time t_s
        if t_s < self.atmospheric_stability_timestep_s[0]:
            return np.ones(xyz_m.shape[1]) * self.atmospheric_stability_L_m[0]
        elif t_s > self.atmospheric_stability_timestep_s[-1]:
            return np.ones(xyz_m.shape[1]) * self.atmospheric_stability_L_m[-1]
        else:
            return np.ones(xyz_m.shape[1]) * np.interp(t_s, self.atmospheric_stability_timestep_s, self.atmospheric_stability_L_m)

    @compatibility(CompatibilityLevel.FULL)
    def obs_air_density_kgpm3(self, xyz_m: np.ndarray, t_s: float) -> float:
        """ Returns the air density at given positions.
        Args:
            xyz_m (np.ndarray): Positions in Cartesian coordinates (m) with shape (3, N) and rows [x, y, z].
            t_s (float): Time (s)

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
            t_s (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the atmospheric pressure at the corresponding position (Pa).
        """
        return self.surface_pressure_Pa * np.ones(xyz_m.shape[1])
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class AtmosphericModel(ABC):
    """Base interface for atmospheric state providers."""

    @abstractmethod
    def step(self, dt: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
    
    def get_citation(self) -> str:
        """ Returns a citation string for the atmospheric model. Default implementation returns a generic citation.

        Returns:
            str: Citation string for the atmospheric model.
        """
        return "No specific Atmospheric Model citation available."
    
    """ 
    ---------------------------------------
    Observables 
    --------------------------------------- 
    """
    @abstractmethod
    def obs_uvw_mps(self, x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Abstract method to observe the wind velocity components (u, v, w) at given positions.

        Args:
            x_m (np.ndarray): Positions in the x-direction (m)
            y_m (np.ndarray): Positions in the y-direction (m)
            z_m (np.ndarray): Positions in the z-direction (m)
            t_s (float): Time (s)

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.ndarray: Array of shape (3, N) where N is the number of positions. Each row contains the wind velocity components (u, v, w) at the corresponding position (m/s).
        """
        raise NotImplementedError

    @abstractmethod
    def obs_uv_mps(self, x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Abstract method to observe the wind velocity components (u, v) at given positions.

        Args:
            x_m (np.ndarray): Positions in the x-direction (m)
            y_m (np.ndarray): Positions in the y-direction (m)
            z_m (np.ndarray): Positions in the z-direction (m)
            t_s (float): Time (s)

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.ndarray: Array of shape (2, N) where N is the number of positions. Each row contains the wind velocity components (u, v) at the corresponding position (m/s).
        """
        raise NotImplementedError

    def obs_horizontal_wind_speed_mps(self, x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the horizontal wind speed at given positions.
        Default implementation calculates from internal (u,v) values. 

        Args:
            x_m (np.ndarray): Positions in the x-direction (m)
            y_m (np.ndarray): Positions in the y-direction (m)
            z_m (np.ndarray): Positions in the z-direction (m)
            t_s (float): Time (s)

        Returns:
            np.ndarray: Horizontal wind speed at the given positions (m/s)
        """
        return np.sqrt(np.sum(self.obs_uv_mps(x_m, y_m, z_m, t_s)**2, axis=1))

    def obs_horizontal_wind_dir_deg(self, x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the horizontal wind direction at given positions. Default implementation calculates from internal (u,v) values.

        Args:
            x_m (np.ndarray): Positions in the x-direction (m)
            y_m (np.ndarray): Positions in the y-direction (m)
            z_m (np.ndarray): Positions in the z-direction (m)
            t_s (float): Time (s)

        Returns:
            np.ndarray: Horizontal wind direction at the given positions (degrees)
        """
        uv = self.obs_uv_mps(x_m, y_m, z_m, t_s)
        return 270 - np.degrees(np.arctan2(uv[:,1], uv[:,0]))
    
    def obs_horizontal_wind_speed_and_dir_mps_deg(self, x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the horizontal wind speed and direction at given positions. Default implementation calculates from internal (u,v) values.

        Args:
            x_m (np.ndarray): Positions in the x-direction (m)
            y_m (np.ndarray): Positions in the y-direction (m)
            z_m (np.ndarray): Positions in the z-direction (m)
            t_s (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N, 2) where N is the number of positions. Each row contains [wind_speed, wind_direction] for the corresponding position.
        """
        uv = self.obs_uv_mps(x_m, y_m, z_m, t_s)
        return np.column_stack((np.sqrt(np.sum(uv**2, axis=1)), 270 - np.degrees(np.arctan2(uv[:,1], uv[:,0]))))
    
    def obs_wind_shear_coefficient(self, t_s: float) -> float:
        """ Returns the wind shear coefficient $\alpha$, used in the equation $v = v_0 \times \left(\frac{h}{h_0}\right)^\alpha$ to calculate the wind speed at height $h$ given 
        reference wind speed $v_0$ and reference height $h_0$. Default implementation returns 1/7 (industry standard for open water).

        Args:
            t_s (float): Time (s)

        Returns:
            float: Wind shear coefficient $\alpha$ (dimensionless)
        """
        return 1.0/7.0
    
    def obs_wind_shear_loglaw_tuple_m_mps(self, t_s: float) -> tuple[float, float]:
        """ Observes the wind shear roughness length and friction velocity, as used in the equation $\frac{u_\star}{\kappa}\log \left(\frac{z-d}{z_0}\right)$ to describe the wind shear profile. 
        Default returns [0.0002 (m), 0.1 (m/s)] (Open water). $\kappa$ is the von Karman constant (approximately 0.4).

        Args:
            t_s (float): Time(s)

        Returns:
            tuple[float, float]: Wind shear roughness length (m) and friction velocity (m/s)
        """
        return 0.0002, 0.25

    def obs_wind_veer_factor_degpm(self, x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the wind veer factor $\beta$, used in the equation $\theta = \theta_0 + \beta \times (h - h_0)$ to calculate the wind direction at height $h$ given 
        reference wind direction $\theta_0$ and reference height $h_0$. Default implementation returns z,0.

        Args:
            x_m (np.ndarray): Positions in the x-direction (m)
            y_m (np.ndarray): Positions in the y-direction (m)
            z_m (np.ndarray): Positions in the z-direction (m)
            t_s (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N, 2) where N is the number of positions. Each row contains [z, beta] for the corresponding position.
        """
        return np.column_stack((z_m, np.zeros_like(z_m)))

    def obs_temperature_K(self, x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the temperature at given positions. Default implementation returns 288.15 K (15°C).

        Args:
            x_m (np.ndarray): Positions in the x-direction (m)
            y_m (np.ndarray): Positions in the y-direction (m)
            z_m (np.ndarray): Positions in the z-direction (m)
            t_s (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the temperature at the corresponding position (K).
        """
        return 288.15 * np.ones_like(x_m)  # Default to 15°C in Kelvin
    
    def obs_air_density_kgpm3(self, x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the air density at given positions. Default implementation returns 1.225 kg/m³ (standard sea level density).

        Args:
            x_m (np.ndarray): Positions in the x-direction (m)
            y_m (np.ndarray): Positions in the y-direction (m)
            z_m (np.ndarray): Positions in the z-direction (m)
            t_s (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the air density at the corresponding position (kg/m³).
        """
        return 1.225 * np.ones_like(x_m)  # Default to standard sea level density

    def obs_turbulence_intensity_percent(self, x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the turbulence intensity at given positions. Default implementation returns 0 (0%).

        Args:
            x_m (np.ndarray): Positions in the x-direction (m)
            y_m (np.ndarray): Positions in the y-direction (m)
            z_m (np.ndarray): Positions in the z-direction (m)
            t_s (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the turbulence intensity at the corresponding position (dimensionless).
        """
        return np.zeros_like(x_m)  # Default to 0% turbulence intensity

    def obs_pressure_Pa(self, x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the atmospheric pressure at given positions. Default implementation returns 101325 Pa (standard sea level pressure).

        Args:
            x_m (np.ndarray): Positions in the x-direction (m)
            y_m (np.ndarray): Positions in the y-direction (m)
            z_m (np.ndarray): Positions in the z-direction (m)
            t_s (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the atmospheric pressure at the corresponding position (Pa).
        """
        return 101325 * np.ones_like(x_m)  # Default to standard sea level pressure

    def obs_atmospheric_boundary_layer_height_m(self, x_m: np.ndarray, y_m: np.ndarray, t_s: float) -> np.ndarray:
        """ Returns the atmospheric boundary layer height at given positions. Default implementation returns 1000 m.

        Args:
            x_m (np.ndarray): Positions in the x-direction (m)
            y_m (np.ndarray): Positions in the y-direction (m)
            t_s (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the atmospheric boundary layer height at the corresponding position (m).
        """
        return 1000 * np.ones_like(x_m)  # Default to 1000 m boundary layer height
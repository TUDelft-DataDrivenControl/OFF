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
    
    """ 
    ---------------------------------------
    Observables 
    --------------------------------------- 
    """
    @abstractmethod
    def obs_u_mps(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        """ Abstract method to observe the u-component of the flow field at given positions.

        Args:
            x (np.ndarray): Positions in the x-direction (m)
            y (np.ndarray): Positions in the y-direction (m)
            z (np.ndarray): Positions in the z-direction (m)
            t (float): Time (s)

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the u-component of the flow field at the corresponding position (m/s).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_v_mps(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        """ Abstract method to observe the v-component of the flow field at given positions.

        Args:
            x (np.ndarray): Positions in the x-direction (m)
            y (np.ndarray): Positions in the y-direction (m)
            z (np.ndarray): Positions in the z-direction (m)
            t (float): Time (s)

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the v-component of the flow field at the corresponding position (m/s).
        """
        raise NotImplementedError

    def obs_w_mps(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        """ Observes the w-component of the flow field (vertical wind speed) at specified locations. Default implementation returns zeros.

        Args:
            x (np.ndarray): Positions in the x-direction (m)
            y (np.ndarray): Positions in the y-direction (m)
            z (np.ndarray): Positions in the z-direction (m)
            t (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the w-component of the flow field at the corresponding position (m/s).
        """
        assert x.shape == y.shape == z.shape, "x, y, z must have the same shape"
        return np.zeros_like(x)

    def obs_horizontal_wind_speed_mps(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        """ Returns the horizontal wind speed at given positions.
        Default implementation calculates from internal (u,v) values. 

        Args:
            x (np.ndarray): Positions in the x-direction (m)
            y (np.ndarray): Positions in the y-direction (m)
            z (np.ndarray): Positions in the z-direction (m)
            t (float): Time (s)

        Returns:
            np.ndarray: Horizontal wind speed at the given positions (m/s)
        """
        return np.sqrt(self.obs_u_mps(x, y, z, t)**2 + self.obs_v_mps(x, y, z, t)**2)

    def obs_horizontal_wind_dir_deg(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        """ Returns the horizontal wind direction at given positions. Default implementation calculates from internal (u,v) values.

        Args:
            x (np.ndarray): Positions in the x-direction (m)
            y (np.ndarray): Positions in the y-direction (m)
            z (np.ndarray): Positions in the z-direction (m)
            t (float): Time (s)

        Returns:
            np.ndarray: Horizontal wind direction at the given positions (degrees)
        """
        return 270 - np.degrees(np.arctan2(self.obs_v_mps(x, y, z, t), self.obs_u_mps(x, y, z, t)))
    
    def obs_wind_shear_coefficient(self, t: float) -> float:
        """ Returns the wind shear coefficient $\alpha$, used in the equation $v = v_0 \times \left(\frac{h}{h_0}\right)^\alpha$ to calculate the wind speed at height $h$ given 
        reference wind speed $v_0$ and reference height $h_0$. Default implementation returns 1/7 (industry standard for open water).

        Args:
            t (float): Time (s)

        Returns:
            float: Wind shear coefficient $\alpha$ (dimensionless)
        """
        return 1.0/7.0
    
    def obs_wind_shear_loglaw_tuple_m_mps(self, t: float) -> tuple[float, float]:
        """ Observes the wind shear roughness length and friction velocity, as used in the equation $\frac{u_\star}{\kappa}\log \left(\frac{z-d}{z_0}\right)$ to describe the wind shear profile. 
        Default returns [0.0002 (m), 0.1 (m/s)] (Open water). $\kappa$ is the von Karman constant (approximately 0.4).

        Args:
            t (float): Time(s)

        Returns:
            tuple[float, float]: Wind shear roughness length (m) and friction velocity (m/s)
        """
        return 0.0002, 0.25

    def obs_wind_veer_factor_degpm(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        """ Returns the wind veer factor $\beta$, used in the equation $\theta = \theta_0 + \beta \times (h - h_0)$ to calculate the wind direction at height $h$ given 
        reference wind direction $\theta_0$ and reference height $h_0$. Default implementation returns z,0.

        Args:
            x (np.ndarray): Positions in the x-direction (m)
            y (np.ndarray): Positions in the y-direction (m)
            z (np.ndarray): Positions in the z-direction (m)
            t (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N, 2) where N is the number of positions. Each row contains [z, beta] for the corresponding position.
        """
        return np.column_stack((z, np.zeros_like(z)))

    def obs_temperature_K(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        """ Returns the temperature at given positions. Default implementation returns 288.15 K (15°C).

        Args:
            x (np.ndarray): Positions in the x-direction (m)
            y (np.ndarray): Positions in the y-direction (m)
            z (np.ndarray): Positions in the z-direction (m)
            t (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the temperature at the corresponding position (K).
        """
        return 288.15 * np.ones_like(x)  # Default to 15°C in Kelvin
    
    def obs_air_density_kgpm3(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        """ Returns the air density at given positions. Default implementation returns 1.225 kg/m³ (standard sea level density).

        Args:
            x (np.ndarray): Positions in the x-direction (m)
            y (np.ndarray): Positions in the y-direction (m)
            z (np.ndarray): Positions in the z-direction (m)
            t (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the air density at the corresponding position (kg/m³).
        """
        return 1.225 * np.ones_like(x)  # Default to standard sea level density

    def obs_turbulence_intensity_percent(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        """ Returns the turbulence intensity at given positions. Default implementation returns 0 (0%).

        Args:
            x (np.ndarray): Positions in the x-direction (m)
            y (np.ndarray): Positions in the y-direction (m)
            z (np.ndarray): Positions in the z-direction (m)
            t (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the turbulence intensity at the corresponding position (dimensionless).
        """
        return np.zeros_like(x)  # Default to 0% turbulence intensity

    def obs_pressure_Pa(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        """ Returns the atmospheric pressure at given positions. Default implementation returns 101325 Pa (standard sea level pressure).

        Args:
            x (np.ndarray): Positions in the x-direction (m)
            y (np.ndarray): Positions in the y-direction (m)
            z (np.ndarray): Positions in the z-direction (m)
            t (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the atmospheric pressure at the corresponding position (Pa).
        """
        return 101325 * np.ones_like(x)  # Default to standard sea level pressure
    
    def obs_atmospheric_boundary_layer_height_m(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """ Returns the atmospheric boundary layer height at given positions. Default implementation returns 1000 m.

        Args:
            x (np.ndarray): Positions in the x-direction (m)
            y (np.ndarray): Positions in the y-direction (m)
            t (float): Time (s)

        Returns:
            np.ndarray: Array of shape (N,) where N is the number of positions. Each element contains the atmospheric boundary layer height at the corresponding position (m).
        """
        return 1000 * np.ones_like(x)  # Default to 1000 m boundary layer height
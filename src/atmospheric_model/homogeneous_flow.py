from __future__ import annotations
import numpy as np

from atmospheric_model.base import AtmosphericModel_Base


class AtmosphericModel_HomogeneousFlow(AtmosphericModel_Base):
    """Steady and spatially uniform atmospheric model."""

    wind_speed_abs_mps: float
    wind_dir_deg: float
    turbulence_intensity_percent: float
    _initial_state_wind_speed_abs_mps: float
    _initial_state_wind_dir_deg: float
    _initial_state_turbulence_intensity_percent: float

    def __init__(self, wind_speed_abs_mps: float = 8.0, wind_dir_deg: float = 270.0, turbulence_intensity_percent: float = 6.0,) -> None:
        self.wind_speed_abs_mps = wind_speed_abs_mps
        self.wind_dir_deg = wind_dir_deg
        self.turbulence_intensity_percent = turbulence_intensity_percent

        # Store initial state for reset
        self._initial_state_wind_speed_abs_mps = wind_speed_abs_mps
        self._initial_state_wind_dir_deg = wind_dir_deg
        self._initial_state_turbulence_intensity_percent = turbulence_intensity_percent

    def step(self, dt: float) -> None:
        return None

    def reset(self) -> None:
        self.wind_speed_abs_mps = self._initial_state_wind_speed_abs_mps
        self.wind_dir_deg = self._initial_state_wind_dir_deg
        self.turbulence_intensity_percent = self._initial_state_turbulence_intensity_percent

    def get_u_mps(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        assert x.shape == y.shape == z.shape, "x, y, z must have the same shape"
        u = self.wind_speed_abs_mps * np.cos(np.radians(self.wind_dir_deg))
        return np.full_like(x, u)

    def get_v_mps(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        assert x.shape == y.shape == z.shape, "x, y, z must have the same shape"
        v = self.wind_speed_abs_mps * np.sin(np.radians(self.wind_dir_deg))
        return np.full_like(x, v)

    def get_horizontal_wind_dir_deg(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        assert x.shape == y.shape == z.shape, "x, y, z must have the same shape"
        return np.full_like(x, self.wind_dir_deg)
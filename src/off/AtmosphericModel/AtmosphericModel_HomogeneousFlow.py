from __future__ import annotations
import numpy as np

from .AtmosphericModel import AtmosphericModel
from off.OFFModule import *

class AtmosphericModel_HomogeneousFlow(AtmosphericModel):
    """Steady and spatially uniform atmospheric model."""

    wind_speed_abs_mps: float
    wind_dir_deg: float
    turbulence_intensity_percent: float
    _initial_state_wind_speed_abs_mps: float
    _initial_state_wind_dir_deg: float
    _initial_state_turbulence_intensity_percent: float       

    def __init__(self, wind_speed_abs_mps: float = 8.0, wind_dir_deg: float = 270.0, turbulence_intensity_percent: float = 6.0) -> None:
        super().__init__()
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

    @compatibility(CompatibilityLevel.OPTIONAL)
    def obs_u_mps(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        u = self.wind_speed_abs_mps * np.cos(np.radians(self.wind_dir_deg))
        return np.full(xyz_m.shape[1], u)

    @compatibility(CompatibilityLevel.OPTIONAL)
    def obs_v_mps(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        v = self.wind_speed_abs_mps * np.sin(np.radians(self.wind_dir_deg))
        return np.full(xyz_m.shape[1], v)
    
    @compatibility(CompatibilityLevel.OPTIONAL)
    def obs_uv_mps(self, xyz_m, t_s):
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        u = self.wind_speed_abs_mps * np.cos(np.radians(self.wind_dir_deg))
        v = self.wind_speed_abs_mps * np.sin(np.radians(self.wind_dir_deg))
        return np.vstack([np.full(xyz_m.shape[1], u), np.full(xyz_m.shape[1], v)])
    
    @compatibility(CompatibilityLevel.OPTIONAL)
    def obs_uvw_mps(self, xyz_m, t_s):
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        u = self.wind_speed_abs_mps * np.cos(np.radians(self.wind_dir_deg))
        v = self.wind_speed_abs_mps * np.sin(np.radians(self.wind_dir_deg))
        return np.vstack([np.full(xyz_m.shape[1], u), np.full(xyz_m.shape[1], v), np.zeros(xyz_m.shape[1])])

    @compatibility(CompatibilityLevel.FULL)
    def obs_horizontal_wind_dir_deg(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        return np.full(xyz_m.shape[1], self.wind_dir_deg)
    
    @compatibility(CompatibilityLevel.FULL)
    def obs_horizontal_wind_speed_mps(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        return np.full(xyz_m.shape[1], self.wind_speed_abs_mps)

    @compatibility(CompatibilityLevel.FULL)
    def obs_turbulence_intensity_percent(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
        assert xyz_m.shape[0] == 3, "xyz_m must have shape (3, N)"
        return np.full(xyz_m.shape[1], self.turbulence_intensity_percent)
    
if __name__ == '__main__':
    # Example usage and compatibility check
    print(AtmosphericModel_HomogeneousFlow.compatibility)

    AtmosphericModel_HomogeneousFlow.describe_compatibility()
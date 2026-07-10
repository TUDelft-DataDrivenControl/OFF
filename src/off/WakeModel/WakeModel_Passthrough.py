import numpy as np

from off.AtmosphericModel import AtmosphericModel
from off.OFFModule import *
from .WakeModel import WakeModel

class WakeModel_Passthrough(WakeModel):
    """
    A wake model that simply passes through the input flow field without any modifications.
    This model is useful for testing and validation purposes.
    """

    REQUIRES = {
        'AtmosphericModel': {
            'obs_u_mps': CompatibilityLevel.OPTIONAL,
            'obs_v_mps': CompatibilityLevel.OPTIONAL,
            'obs_w_mps': CompatibilityLevel.OPTIONAL,
            'obs_uv_mps': CompatibilityLevel.OPTIONAL,
            'obs_uvw_mps': CompatibilityLevel.OPTIONAL,
            'obs_blades_uvw_wind_speed_mps': CompatibilityLevel.OPTIONAL,
            'obs_horizontal_wind_speed_mps': CompatibilityLevel.OPTIONAL,
            'obs_horizontal_wind_dir_deg': CompatibilityLevel.OPTIONAL
        }
    }

    def __init__(self) -> None:
        super().__init__()

    def step(self, dt: float) -> None:
        # No state to update in a passthrough model
        pass

    def reset(self) -> None:
        # No state to reset in a passthrough model
        pass

    @compatibility(CompatibilityLevel.FULL)
    def obs_u_mps(self, xyz: np.ndarray, t_s: float) -> np.ndarray:
        # Return the u-component of the flow field as is
        return self._atmospheric_model.obs_u_mps(xyz, t_s)

    @compatibility(CompatibilityLevel.FULL)
    def obs_v_mps(self, xyz: np.ndarray, t_s: float) -> np.ndarray:
        # Return the v-component of the flow field as is
        return self._atmospheric_model.obs_v_mps(xyz, t_s)

    @compatibility(CompatibilityLevel.FULL)
    def obs_w_mps(self, xyz: np.ndarray, t_s: float) -> np.ndarray:
        # Return the w-component of the flow field as is
        return self._atmospheric_model.obs_w_mps(xyz, t_s)

    @compatibility(CompatibilityLevel.FULL)
    def obs_uvw_mps(self, xyz: np.ndarray, t_s: float) -> np.ndarray:
        return self._atmospheric_model.obs_uvw_mps(xyz, t_s)
    
    @compatibility(CompatibilityLevel.FULL)
    def obs_uv_mps(self, xyz: np.ndarray, t_s: float) -> np.ndarray:
        return self._atmospheric_model.obs_uv_mps(xyz, t_s)
    
    @compatibility(CompatibilityLevel.FULL)
    def obs_blades_uvw_wind_speed_mps(self, xyz: np.ndarray, R_m: float, azimuth_deg: float, yaw_orientation_deg: float, t_s: float, tilt_deg: float = 0, n_blades: int = 3) -> np.ndarray:
        return self._atmospheric_model.obs_uvw_mps(xyz, t_s)

    @compatibility(CompatibilityLevel.FULL)
    def obs_horizontal_wind_speed_mps(self, xyz: np.ndarray, t_s: float) -> np.ndarray:
        # Return the horizontal wind speed as is
        return self._atmospheric_model.obs_horizontal_wind_speed_mps(xyz, t_s)
    
    @compatibility(CompatibilityLevel.FULL)
    def obs_horizontal_wind_dir_deg(self, xyz: np.ndarray, t_s: float) -> np.ndarray:
        # Return the horizontal wind direction as is
        return self._atmospheric_model.obs_horizontal_wind_dir_deg(xyz, t_s)

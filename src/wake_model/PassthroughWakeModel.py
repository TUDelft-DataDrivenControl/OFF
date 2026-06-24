import numpy as np

from atmospheric_model import AtmosphericModel
from wake_model.WakeModel import WakeModel

class PassthroughWakeModel(WakeModel):
    """
    A wake model that simply passes through the input flow field without any modifications.
    This model is useful for testing and validation purposes.
    """
    def __init__(self, atmospheric_model: AtmosphericModel) -> None:
        super().__init__(atmospheric_model)

    def step(self, dt: float) -> None:
        # No state to update in a passthrough model
        pass

    def reset(self) -> None:
        # No state to reset in a passthrough model
        pass

    def obs_u_mps(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        # Return the u-component of the flow field as is
        return self._atmospheric_model.obs_u_mps(x, y, z, t)

    def obs_v_mps(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        # Return the v-component of the flow field as is
        return self._atmospheric_model.obs_v_mps(x, y, z, t)

    def obs_w_mps(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        # Return the w-component of the flow field as is
        return self._atmospheric_model.obs_w_mps(x, y, z, t)
    
    def obs_horizontal_wind_speed_mps(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        # Return the horizontal wind speed as is
        return self._atmospheric_model.obs_horizontal_wind_speed_mps(x, y, z, t)
    
    def obs_horizontal_wind_dir_deg(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        # Return the horizontal wind direction as is
        return self._atmospheric_model.obs_horizontal_wind_dir_deg(x, y, z, t)
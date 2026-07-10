from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod

from off.OFFModule import*
from .TurbineModel import TurbineModel

class TurbineModelStatic(TurbineModel):
    """ 
        Temporary Placeholder for a static turbine that maximizes power extraction from the wind.
    """

    REQUIRES = {
        'WakeModel': {
            'obs_horizontal_wind_speed_mps': CompatibilityLevel.OPTIONAL
        }
    }
    
    def __init__(self, radius: float = 128):
        super().__init__()
        self.radius = radius

    @compatibility(CompatibilityLevel.FULL)
    def obs_generator_power_w(self, t_s: np.float64) -> np.float64:
        """ Observes the current generator power of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            np.float64: Current generator power of the turbine (W).
        """
        return 10e6

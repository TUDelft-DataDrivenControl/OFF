from __future__ import annotations

from abc import ABC, abstractmethod

class TurbineModel(ABC):
    """Base interface for turbine aerodynamics/dynamics."""

    @abstractmethod
    def step(self, dt: float, effective_wind_speed_mps: float) -> TurbineOutput:
        raise NotImplementedError

    @abstractmethod
    def set_yaw(self, yaw_deg: float) -> None:
        raise NotImplementedError

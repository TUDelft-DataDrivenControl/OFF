from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TurbineOutput:
    power_w: float = 0.0
    thrust_n: float = 0.0
    cp: float = 0.0
    ct: float = 0.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0


class TurbineStates(ABC):
    """Base interface for turbine state history."""

    @abstractmethod
    def get_current_yaw(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def set_yaw(self, yaw_deg: float) -> None:
        raise NotImplementedError


class TurbineModel(ABC):
    """Base interface for turbine aerodynamics/dynamics."""

    @abstractmethod
    def step(self, dt: float, effective_wind_speed_mps: float) -> TurbineOutput:
        raise NotImplementedError

    @abstractmethod
    def set_yaw(self, yaw_deg: float) -> None:
        raise NotImplementedError


class DummyTurbineStates(TurbineStates):
    """Minimal in-memory state container."""

    def __init__(self) -> None:
        self._yaw_deg = 0.0

    def get_current_yaw(self) -> float:
        return self._yaw_deg

    def set_yaw(self, yaw_deg: float) -> None:
        self._yaw_deg = yaw_deg


class DummyTurbineModel(TurbineModel):
    """Minimal turbine model for integration scaffolding."""

    def __init__(self, states: TurbineStates | None = None) -> None:
        self.states = states or DummyTurbineStates()

    def step(self, dt: float, effective_wind_speed_mps: float) -> TurbineOutput:
        return TurbineOutput(yaw_deg=self.states.get_current_yaw())

    def set_yaw(self, yaw_deg: float) -> None:
        self.states.set_yaw(yaw_deg)

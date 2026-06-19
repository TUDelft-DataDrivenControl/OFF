from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class AmbientState:
    wind_speed_abs_mps: float = 0.0
    wind_dir_deg: float = 270.0
    turbulence_intensity: float = 0.0


class AtmosphericModel_Base(ABC):
    """Base interface for atmospheric state providers."""

    @abstractmethod
    def get_state_at_turbine(self, turbine_id: int) -> AmbientState:
        raise NotImplementedError

    @abstractmethod
    def step(self, dt: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

class AmbientCorrector(ABC):
    """Base interface for ambient correction / filtering."""

    @abstractmethod
    def update(self, t: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, state: AmbientState) -> AmbientState:
        raise NotImplementedError


class AtmosphericModel_Dummy(AtmosphericModel_Base):
    """Minimal atmospheric model for integration scaffolding."""

    def __init__(self) -> None:
        self._state = AmbientState()

    def get_state_at_turbine(self, turbine_id: int) -> AmbientState:
        return self._state

    def step(self, dt: float) -> None:
        return None

    def reset(self) -> None:
        self._state = AmbientState()


class DummyAmbientCorrector(AmbientCorrector):
    """No-op ambient corrector."""

    def update(self, t: float) -> None:
        return None

    def __call__(self, state: AmbientState) -> AmbientState:
        return state

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class FarmControlCommand:
    yaw_target_deg: float | None = None


class FarmController(ABC):
    """Base interface for farm-level controllers."""

    @abstractmethod
    def compute_command_for_turbine(self, turbine_id: int, t: float, dt: float) -> FarmControlCommand:
        raise NotImplementedError


class DummyFarmController(FarmController):
    """Minimal farm-level controller for scaffolding."""

    def compute_command_for_turbine(self, turbine_id: int, t: float, dt: float) -> FarmControlCommand:
        return FarmControlCommand(yaw_target_deg=0.0)

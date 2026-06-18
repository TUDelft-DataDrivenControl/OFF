from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class WakeMeasurement:
    u_mps: float = 0.0
    v_mps: float = 0.0
    added_ti: float = 0.0


class WakeModel(ABC):
    """Base interface for wake deficit models."""

    @abstractmethod
    def get_measurements(self, turbine_id: int) -> WakeMeasurement:
        raise NotImplementedError


class WakeSolver(ABC):
    """Base interface for wake solver/orchestrator layer."""

    @abstractmethod
    def solve_for_turbine(self, turbine_id: int) -> WakeMeasurement:
        raise NotImplementedError


class WindFarm(ABC):
    """Base interface for turbine collection and dependency graph."""

    @abstractmethod
    def turbine_count(self) -> int:
        raise NotImplementedError


class DummyWakeModel(WakeModel):
    """No-op wake model used for wiring tests."""

    def get_measurements(self, turbine_id: int) -> WakeMeasurement:
        return WakeMeasurement()


class DummyWakeSolver(WakeSolver):
    """No-op wake solver used for wiring tests."""

    def __init__(self, wake_model: WakeModel | None = None) -> None:
        self.wake_model = wake_model or DummyWakeModel()

    def solve_for_turbine(self, turbine_id: int) -> WakeMeasurement:
        return self.wake_model.get_measurements(turbine_id)


class DummyWindFarm(WindFarm):
    """Simple in-memory wind farm container."""

    def __init__(self, n_turbines: int = 1) -> None:
        self._n_turbines = n_turbines

    def turbine_count(self) -> int:
        return self._n_turbines

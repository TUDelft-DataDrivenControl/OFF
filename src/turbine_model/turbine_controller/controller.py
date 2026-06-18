from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class FarmCommand:
    yaw_target_deg: float | None = None
    power_reference_w: float | None = None


@dataclass
class TurbineMeasurement:
    wind_speed_abs_mps: float = 0.0
    wind_dir_deg: float = 270.0
    power_w: float = 0.0
    yaw_deg: float = 0.0


@dataclass
class TurbineSetpoints:
    yaw_setpoint_deg: float | None = None
    pitch_setpoint_deg: float | None = None
    torque_setpoint_nm: float | None = None


class TurbineController(ABC):
    """Base interface for local (per-turbine) controllers."""

    @abstractmethod
    def update_from_farm_command(self, cmd: FarmCommand | None) -> None:
        raise NotImplementedError

    @abstractmethod
    def observe(self, measurement: TurbineMeasurement) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_setpoints(self, t: float, dt: float) -> TurbineSetpoints:
        raise NotImplementedError


class DummyTurbineController(TurbineController):
    """Minimal turbine controller that forwards a yaw target if present."""

    def __init__(self) -> None:
        self._cmd: FarmCommand | None = None
        self._meas = TurbineMeasurement()

    def update_from_farm_command(self, cmd: FarmCommand | None) -> None:
        self._cmd = cmd

    def observe(self, measurement: TurbineMeasurement) -> None:
        self._meas = measurement

    def compute_setpoints(self, t: float, dt: float) -> TurbineSetpoints:
        if self._cmd and self._cmd.yaw_target_deg is not None:
            return TurbineSetpoints(yaw_setpoint_deg=self._cmd.yaw_target_deg)
        return TurbineSetpoints(yaw_setpoint_deg=self._meas.yaw_deg)

from __future__ import annotations

import math

from off.OFFModule import *
from .AtmosphericModel import  AtmosphericModel

class AtmosphericModel_UnsteadyBackgroundFlow(AtmosphericModel):
    """Simple time-varying background flow model."""

    def __init__(
        self,
        base_wind_speed_abs_mps: float = 8.0,
        speed_amplitude_mps: float = 1.0,
        speed_period_s: float = 60.0,
        base_wind_dir_deg: float = 270.0,
        dir_amplitude_deg: float = 5.0,
        dir_period_s: float = 90.0,
        turbulence_intensity: float = 0.08,
    ) -> None:
        self._t_s = 0.0
        self._base_wind_speed_abs_mps = base_wind_speed_abs_mps
        self._speed_amplitude_mps = speed_amplitude_mps
        self._speed_period_s = max(speed_period_s, 1e-9)
        self._base_wind_dir_deg = base_wind_dir_deg
        self._dir_amplitude_deg = dir_amplitude_deg
        self._dir_period_s = max(dir_period_s, 1e-9)
        self._turbulence_intensity = turbulence_intensity
        # self._state = AmbientState(
        #     wind_speed_abs_mps=base_wind_speed_abs_mps,
        #     wind_dir_deg=base_wind_dir_deg,
        #     turbulence_intensity=turbulence_intensity,
        # )

    # def get_state_at_turbine(self, turbine_id: int) -> AmbientState:
    #     return self._state

    def step(self, dt: float) -> None:
        self._t_s += dt
        speed_phase = 2.0 * math.pi * self._t_s / self._speed_period_s
        dir_phase = 2.0 * math.pi * self._t_s / self._dir_period_s

        # self._state = AmbientState(
        #     wind_speed_abs_mps=self._base_wind_speed_abs_mps
        #     + self._speed_amplitude_mps * math.sin(speed_phase),
        #     wind_dir_deg=self._base_wind_dir_deg + self._dir_amplitude_deg * math.sin(dir_phase),
        #     turbulence_intensity=self._turbulence_intensity,
        # )

    def reset(self) -> None:
        self._t_s = 0.0
        # self._state = AmbientState(
        #     wind_speed_abs_mps=self._base_wind_speed_abs_mps,
        #     wind_dir_deg=self._base_wind_dir_deg,
        #     turbulence_intensity=self._turbulence_intensity,
        # )

from __future__ import annotations

from atmospheric_model.base import AmbientState, AtmosphericModel


class HomogeneousFlow(AtmosphericModel):
    """Steady and spatially uniform atmospheric model."""

    def __init__(
        self,
        wind_speed_abs_mps: float = 8.0,
        wind_dir_deg: float = 270.0,
        turbulence_intensity: float = 0.06,
    ) -> None:
        self._initial_state = AmbientState(
            wind_speed_abs_mps=wind_speed_abs_mps,
            wind_dir_deg=wind_dir_deg,
            turbulence_intensity=turbulence_intensity,
        )
        self._state = AmbientState(
            wind_speed_abs_mps=wind_speed_abs_mps,
            wind_dir_deg=wind_dir_deg,
            turbulence_intensity=turbulence_intensity,
        )

    def get_state_at_turbine(self, turbine_id: int) -> AmbientState:
        return self._state

    def step(self, dt: float) -> None:
        return None

    def reset(self) -> None:
        self._state = AmbientState(
            wind_speed_abs_mps=self._initial_state.wind_speed_abs_mps,
            wind_dir_deg=self._initial_state.wind_dir_deg,
            turbulence_intensity=self._initial_state.turbulence_intensity,
        )

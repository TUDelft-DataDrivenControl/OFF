from __future__ import annotations

from atmospheric_model import DummyAmbientCorrector, DummyAtmosphericModel
from farm_controller import DummyFarmController
from turbine_model import DummyTurbineModel
from turbine_model.turbine_controller import (
	DummyTurbineController,
	FarmCommand,
	TurbineMeasurement,
)
from utils import SimulationClock
from wake_model import DummyWakeSolver, DummyWindFarm


class OFFOrchestrator:
	"""Minimal simulation orchestrator wiring all dummy module components."""

	def __init__(self) -> None:
		self.clock = SimulationClock()
		self.wind_farm = DummyWindFarm(n_turbines=1)
		self.atmospheric_model = DummyAtmosphericModel()
		self.ambient_corrector = DummyAmbientCorrector()
		self.wake_solver = DummyWakeSolver()
		self.farm_controller = DummyFarmController()
		self.turbine_controller = DummyTurbineController()
		self.turbine_model = DummyTurbineModel()

	def step(self) -> None:
		t = self.clock.t_s
		dt = self.clock.dt_s

		self.atmospheric_model.step(dt)
		self.ambient_corrector.update(t)

		for turbine_id in range(self.wind_farm.turbine_count()):
			_ = self.wake_solver.solve_for_turbine(turbine_id)
			farm_cmd = self.farm_controller.compute_command_for_turbine(turbine_id, t, dt)
			self.turbine_controller.update_from_farm_command(
				FarmCommand(yaw_target_deg=farm_cmd.yaw_target_deg)
			)

			meas = TurbineMeasurement(yaw_deg=self.turbine_model.states.get_current_yaw())
			self.turbine_controller.observe(meas)
			setpoints = self.turbine_controller.compute_setpoints(t, dt)

			if setpoints.yaw_setpoint_deg is not None:
				self.turbine_model.set_yaw(setpoints.yaw_setpoint_deg)

			self.turbine_model.step(dt=dt, effective_wind_speed_mps=0.0)

		self.clock.t_s += dt


def create_default_simulation() -> OFFOrchestrator:
	"""Create a default dummy simulation instance."""
	return OFFOrchestrator()


if __name__ == "__main__":
	sim = create_default_simulation()
	sim.step()

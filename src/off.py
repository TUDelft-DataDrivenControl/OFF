from __future__ import annotations

from AtmosphericModel import (
	AtmosphericModel,
	DummyAmbientCorrector,
	HomogeneousFlow,
	unsteadyBackgroundFlow,
)
from FarmController import DummyFarmController
from TurbineModel import DummyTurbineModel
from TurbineModel.turbine_controller import (
	DummyTurbineController,
	FarmCommand,
	TurbineMeasurement,
)
from utils import SimulationClock
from WakeModel import DummyWakeSolver, DummyWindFarm


class OFFOrchestrator:
	"""Minimal simulation orchestrator wiring all dummy module components."""

	def __init__(self, atmospheric_model_version: str = "HomogeneousFlow") -> None:
		self.clock = SimulationClock()
		self.wind_farm = DummyWindFarm(n_turbines=1)
		self.atmospheric_model = self._build_atmospheric_model(atmospheric_model_version)
		self.ambient_corrector = DummyAmbientCorrector()
		self.wake_solver = DummyWakeSolver()
		self.farm_controller = DummyFarmController()
		self.turbine_controller = DummyTurbineController()
		self.turbine_model = DummyTurbineModel()

	def _build_atmospheric_model(self, version: str) -> AtmosphericModel:
		if version == "HomogeneousFlow":
			return HomogeneousFlow()
		if version == "unsteadyBackgroundFlow":
			return unsteadyBackgroundFlow()
		raise ValueError(
			"Unknown atmospheric_model_version. Use 'HomogeneousFlow' or 'unsteadyBackgroundFlow'."
		)

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


		def get_citation(self) -> dict[str, str]:
			""" Returns a collection of citation strings used in the simulation.

			Returns:
				dict[str, str]: Dictionary containing citation strings for each component.
			"""

			# Create an array of all citation strings
			citation_strings = [
				self.atmospheric_model.get_citation(),
				self.ambient_corrector.get_citation(),
				self.wake_solver.get_citation(),
				self.wind_farm.get_citation(),
				self.farm_controller.get_citation(),
				self.turbine_controller.get_citation(),
				self.turbine_model.get_citation(),
			]

			# Remove duplicates by converting to a set and back to a list
			unique_citations = list(set(citation_strings))

			return unique_citations


def create_default_simulation(
	atmospheric_model_version: str = "HomogeneousFlow",
) -> OFFOrchestrator:
	"""Create a default dummy simulation instance."""
	return OFFOrchestrator(atmospheric_model_version=atmospheric_model_version)


if __name__ == "__main__":
	sim = create_default_simulation(atmospheric_model_version="HomogeneousFlow")
	sim.step()

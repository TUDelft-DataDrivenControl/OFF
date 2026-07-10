from __future__ import annotations

from off.AtmosphericModel import AtmosphericModel_HomogeneousFlow
from off.utils import SimulationClock
from off.WakeModel import WakeModel_Passthrough


class OFFOrchestrator:
	"""Minimal simulation orchestrator wiring all dummy module components."""

	def __init__(self) -> None:
		self.clock = SimulationClock()
		# self.atmospheric_model = self._build_atmospheric_model(atmospheric_model_version)
		self.atmospheric_model = AtmosphericModel_HomogeneousFlow()
		self.wake_solver = WakeModel_Passthrough(self.atmospheric_model)
		# self.farm_controller = DummyFarmController()
		# self.turbine_controller = DummyTurbineController()
		# self.turbine_model = DummyTurbineModel()

	def step(self) -> None:
		""" 
		TODO: Fix this function to incorporate all modules
		TODO: Time is iteration based
		"""
		t = self.clock.t_s
		dt = self.clock.dt_s

		self.atmospheric_model.step(t)
		self.wake_solver.step(t)

		# for turbine_id in range(self.wind_farm.turbine_count()):
		# 	self.turbine_model.step(turbine_id, t)

		self.clock.t_s += dt


	def get_citation(self) -> dict[str, str]:
		""" Returns a collection of citation strings used in the simulation.

		Returns:
			dict[str, str]: Dictionary containing citation strings for each component.
		"""

		# Create an array of all citation strings
		citation_strings = [
			self.atmospheric_model.get_citation(),
			self.wake_solver.get_citation(),
			# self.farm_controller.get_citation(),
			# self.turbine_controller.get_citation(),
			# self.turbine_model.get_citation(),
		]

		# Remove duplicates by converting to a set and back to a list
		unique_citations = list(set(citation_strings))

		return unique_citations


def create_default_simulation() -> OFFOrchestrator:
	"""Create a default dummy simulation instance."""
	return OFFOrchestrator()


if __name__ == "__main__":
	sim = create_default_simulation()
	sim.step()

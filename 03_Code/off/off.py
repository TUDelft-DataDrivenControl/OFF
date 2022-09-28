import off.turbine as tur
import off.windfarm as wfm
import off.observation_points as ops
import off.ambient as amb
import numpy as np


class OFF:
    """
    OFF is the central object which initializes the wind farm and runs the simulation
    """

    def __init__(self, wind_farm: wfm.WindFarm):
        self.wind_farm = wind_farm

    def init_sim(self, start_ambient: np.ndarray, start_turbine: np.ndarray):
        """
        Function which initializes the states within the self.wind_farm object. Assigns turbine & ambient states and
        distributes the OPs downstream. OP locations are not necessarily correct but the wakes are "unrolled" and do not
        have to first develop.
        :param start_ambient: 1 x n vector with initial ambient state
        :param start_turbine: 1 x n vector with initial turbine state
        :return: none
        """

        for t in self.wind_farm.turbines:
            t.ambient_states.init_all_states(start_ambient)
            t.turbine_states.init_all_states(start_turbine)
            t.observation_points.init_all_states(t.ambient_states.get_turbine_wind_speed(),
                                                 t.ambient_states.get_turbine_wind_dir(),
                                                 t.get_rotor_pos(), self.wind_farm.time_step)

    def run_sim(self):
        """
        Central function which executes the simulation and manipulates the self.wind_farm object
        :return:
        """
        pass

    def set_wind_farm(self, new_wf: wfm.WindFarm):
        self.wind_farm = new_wf

    def get_wind_farm(self) -> wfm.WindFarm:
        return self.wind_farm




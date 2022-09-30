import off.turbine as tur
import off.windfarm as wfm
import off.observation_points as ops
import off.ambient as amb
import numpy as np
import off.wake_solver as ws


class OFF:
    """
    OFF is the central object which initializes the wind farm and runs the simulation
    """
    settings_sim = dict()

    def __init__(self, wind_farm: wfm.WindFarm, settings_sim: dict):
        self.wind_farm = wind_farm
        self.settings_sim = settings_sim
        self.wake_solver = ws.FLORIDynTWFWakeSolver()

    def init_sim(self, start_ambient: np.ndarray, start_turbine: np.ndarray):
        """
        Function which initializes the states within the ``self.wind_farm`` object. 
        Assigns turbine & ambient states and distributes the OPs downstream. OP 
        locations are not necessarily correct but the wakes are "unrolled" and 
        do not have to first develop.

        Parameters
        ----------
        start_ambient : np.ndarray
            1 x n vector with initial ambient state
        start_turbine : np.ndarray
            1 x n vector with initial turbine state
        """        

        for t in self.wind_farm.turbines:
            t.ambient_states.init_all_states(start_ambient)
            t.turbine_states.init_all_states(start_turbine)
            t.observation_points.init_all_states(t.ambient_states.get_turbine_wind_speed_u(),
                                                 t.ambient_states.get_turbine_wind_speed_v(),
                                                 t.get_rotor_pos(), self.settings_sim['time step'])

    def run_sim(self):
        """
        Central function which executes the simulation and manipulates the 
        ``self.wind_farm object``
        """

        for t in np.arange(self.settings_sim['time start'],
                           self.settings_sim['time end'],
                           self.settings_sim['time step']):

            # Predict
            #   Get all wind speeds
            for idx, tur in enumerate(self.wind_farm.turbines):
                uv_r, uv_op = self.wake_solver.get_wind_speeds(idx, self.wind_farm)
                # TODO get turbine measurements (reduction, added turbulence, ...) in a generic way
                # print('Wind speed turbine ', idx)
                # print(uv_r)

            # TODO store uv_r and uv_op for all turbines
            # iterate all states
            for idx, tur in enumerate(self.wind_farm.turbines):
                tur.ambient_states.iterate_states_and_keep()
                tur.turbine_states.iterate_states_and_keep()
                tur.observation_points.propagate_ops(uv_op, self.settings_sim['time step'])
                print(tur.observation_points.get_world_coord())
                # TODO probably bug with the OP propagation, should not change under steady state

            # Correct

            # Control

            # Visualize

            print(t)
        pass

    def set_wind_farm(self, new_wf: wfm.WindFarm):
        """
        Overwrite wind farm object with a new wind farm object. Can be used to restart the simulation from a given state

        Parameters
        ----------
        new_wf :  windfarm.WindFarm
            Wind farm object with turbines and states
        -------
        """
        self.wind_farm = new_wf

    def get_wind_farm(self) -> wfm.WindFarm:
        """
        Get the current wind farm object which equals the simulation state

        Returns
        -------
        windfarm.WindFarm :
            Wind farm object with turbines and states
        """
        return self.wind_farm




import numpy as np
from abc import ABC, abstractmethod
import off.windfarm as wfm
import off.wake_model as wm
import off.utils as ot
import off.wake_solver as ws
import logging
lg = logging.getLogger(__name__)


class TWFSolver(ws.WakeSolver):
    def __init__(self, settings_wke: dict, settings_sol: dict):
        """
        FLORIDyn temporary wind farm wake solver, based on [1].

        Parameters
        ----------
        settings_wke: dict
            Wake settings, including all parameters the wake needs to run
        settings_sol: dict
            Wake solver settings
        """
        super(TWFSolver, self).__init__(settings_sol)
        lg.info('FLORIDyn wake solver created.')

        # TODO Init FLORIS model
        pass

    def get_measurements(self, i_t: int, wind_farm: wfm.WindFarm) -> tuple:
        """
        Get the wind speed at the location of all OPs and the rotor plane of turbine with index i_t

        Parameters
        ----------
        i_t : int
            index of the turbine to find the wind speeds for
        wind_farm
            influencing wind farm

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            [u,v] wind speeds at the rotor plane (entry 1) and OPs (entry 2)
        """
        u_rp, m = self._get_wind_speeds_rp(i_t, wind_farm)
        u_op = self._get_wind_speeds_op(i_t, wind_farm)
        return u_rp, u_op, m

    def _get_wind_speeds_rp(self, i_t: int, wind_farm: wfm.WindFarm) -> tuple:
        """
        Calculates the effective wind speed at the rotor plane of turbine i_t
        Parameters
        ----------
        i_t : int
            index of the turbine to find the wind speeds for
        wind_farm
            influencing wind farm

        Returns
        -------
        nd.array
            [u,v] wind speeds at the rotor plane
        """
        # Go through dependencies
        for idx in np.arange(wind_farm.nT)[wind_farm.dependencies[i_t, :]]:
            # Interpolation of turbine states
            #   Step 1 retrieve closest up and downstream OPs

            #   Step 2 calculate interpolation weights

            #   Reconstruct turbine location

            # Based on settings either apply weighted retreval of flow field state or interpolation
            if self.settings_sol("Weighted"):
                lg.debug('Ambient states: Weighted interpolation')
            else:
                lg.debug('Ambient states: Two OP interpolation')
                # Use same weights as OP calculation

        wind_farm_layout = wind_farm.get_layout()
        turbine_states = wind_farm.get_current_turbine_states()
        ambient_states = np.array([wind_farm.turbines[i_t].ambient_states.get_turbine_wind_speed_abs(),
                                   wind_farm.turbines[i_t].ambient_states.get_turbine_wind_dir()])
        self.dummy_wake.set_wind_farm(wind_farm_layout, turbine_states, ambient_states)
        ueff, m = self.dummy_wake.get_measurements_i_t(i_t)
        [u_eff, v_eff] = ot.ot_abs2uv(ueff, ambient_states[1])
        return np.array([u_eff, v_eff]), m

    def _get_wind_speeds_op(self, i_t: int, wind_farm: wfm.WindFarm) -> np.ndarray:
        """
        Calculates the free wind speeds for the OPs of turbine i_t
        Parameters
        ----------
        i_t : int
            index of the turbine to find the wind speeds for
        wind_farm
            influencing wind farm

        Returns
        -------
        nd.array
            [u,v] wind speeds at the OP locations
        """
        # TODO combine the influence of different wakes
        return wind_farm.turbines[i_t].ambient_states.get_wind_speed()





# [1] FLORIDyn - A dynamic and flexible framework for real - time wind farm control,
# Becker et al., 2022


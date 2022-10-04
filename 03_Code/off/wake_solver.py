import numpy as np
from abc import ABC, abstractmethod
import off.windfarm as wfm
import off.wake_model as wm
import off.utils as ot
import logging
lg = logging.getLogger(__name__)


class WakeSolver(ABC):
    settings_sol: dict()

    def __init__(self, settings_sol: dict):
        """
        Object to connect OFF to the wake model.
        The common interface is the get_wind_speeds function.

        Parameters
        ----------
        settings_sol: dict
            Wake solver settings
        """
        self.settings_sol = settings_sol
        pass

    @abstractmethod
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
        tuple(np.ndarray, np.ndarray, nd.array)
            [u,v] wind speeds at the rotor plane (entry 1) and OPs (entry 2)
            m further measurements, depending on the used wake model
        """
        pass


class FLORIDynTWFWakeSolver(WakeSolver):
    dummy_wake: wm.DummyWake

    def __init__(self, settings_wke: dict, settings_sol: dict):
        """
        FLORIDyn temporary wind farm wake

        Parameters
        ----------
        settings_wke: dict
            Wake settings, including all parameters the wake needs to run
        settings_sol: dict
            Wake solver settings
        """
        super(FLORIDynTWFWakeSolver, self).__init__(settings_sol)
        self.dummy_wake = wm.DummyWake(settings_wke, np.array([]), np.array([]), np.array([]))

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
        u_rp = self._get_wind_speeds_rp(i_t, wind_farm)
        u_op = self._get_wind_speeds_op(i_t, wind_farm)
        return u_rp, u_op

    def _get_wind_speeds_rp(self, i_t: int, wind_farm: wfm.WindFarm) -> np.ndarray:
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
        # TODO call wake model with rotor points & wind farm
        wind_farm_layout = wind_farm.get_layout()
        turbine_states = wind_farm.get_current_turbine_states()
        ambient_states = np.array([wind_farm.turbines[i_t].ambient_states.get_turbine_wind_speed(),
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

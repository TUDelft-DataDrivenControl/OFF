import numpy as np
from abc import ABC, abstractmethod
import off.utils as ot
import logging
lg = logging.getLogger(__name__)


class WakeModel(ABC):
    settings = dict()
    wind_farm_layout = np.array([])
    turbine_states = np.array([])
    ambient_states = np.array([])
    rp_s = np.array([])
    rp_w = float

    def __init__(self, settings: dict, wind_farm_layout: np.ndarray, turbine_states: np.ndarray,
                 ambient_states: np.ndarray):
        """
        Class to calculate the wake influence onto a certain turbine,
        """
        self.settings = settings
        self.wind_farm_layout = wind_farm_layout
        self.turbine_states = turbine_states
        self.ambient_states = ambient_states
        self.rp_s, self.rp_w = ot.ot_isocell(settings['nRP'])

    @abstractmethod
    def get_measurements_i_t(self, i_t: int) -> tuple:
        """
        Returns the measurements of the wake model including the effective wind speed at the turbine i_t

        Parameters
        ----------
        i_t : int
            Index of the turbine of interest
        wind_farm : off.windfarm
            Wind farm object to retrieve the wakes from

        Returns
        -------
        tuple:
            float: u_eff at turbine i_t
            pandas.dataframe: m other measurements (Power gen, added turbulence, etc.)
        """
        pass


class DummyWake(WakeModel):
    """
    Dummy wake with funky shape for testing
    """

    def __init__(self, settings: dict, wind_farm_layout: np.ndarray, turbine_states: np.ndarray,
                 ambient_states: np.ndarray):
        """
        Wake with funky shape for testing

        Parameters
        ----------
        settings : dict
            .dw down wind wave number
            .cw cross wind wave number
            .sig dw down wind weight
            .sig cw cross wind weight
        wind_farm_layout : np.ndarray
            n_t x 4 array with [x,y,z,D] - world coordinates of the rotor center & diameter
        turbine_states : np.ndarray
            n_t x 2 array with [axial ind., yaw]
        ambient_states : np.ndarray
            1 x 2 : [u_abs, phi] - absolute background wind speed and direction
        """
        super(DummyWake, self).__init__(settings, wind_farm_layout, turbine_states, ambient_states)

    def get_measurements_i_t(self, i_t: int) -> tuple:
        """
        Returns the measurements of the wake model including the effective wind speed at the turbine i_t

        Parameters
        ----------
        i_t : int
            Index of the turbine of interest

        Returns
        -------
        tuple:
            float: u_eff at turbine i_t
            pandas.dataframe: m other measurements (Power gen, added turbulence, etc.)
        """
        #  Iterate over all turbines besides the relevant one and calculate reduction at the rotor points and
        #  subsequently across the plane
        for idx, tur in enumerate(self.wind_farm_layout):
            if idx == i_t:
                continue

            # calculate wake influence at rotor points

            # average

        #  multiply with background wind speed and return

        pass


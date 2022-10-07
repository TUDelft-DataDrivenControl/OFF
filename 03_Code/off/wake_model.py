import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import off.utils as ot
import logging
from floris.tools import FlorisInterface

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

    def set_wind_farm(self, wind_farm_layout: np.ndarray, turbine_states: np.ndarray, ambient_states: np.ndarray):
        """
        Changes the states of the stored wind farm

        Parameters
        ----------
        wind_farm_layout: np.ndarray
            n_t x 4 array with [x,y,z,D] - world coordinates of the rotor center & diameter
        turbine_states: np.ndarray
            n_t x 2 array with [axial ind., yaw]
        ambient_states: np.ndarray
            1 x 2 : [u_abs, phi] - absolute background wind speed and direction
        """
        self.wind_farm_layout = wind_farm_layout
        self.turbine_states = turbine_states
        self.ambient_states = ambient_states


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
            .sig r  radial weight
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
        # Adapt basic rotor points to location and orientation of turbine i_t
        n_rps = self.rp_s.shape[0]
        rps = np.zeros((n_rps, 3))
        phi = ot.ot_deg2rad(
            ot.ot_get_orientation(self.ambient_states[1], self.turbine_states[i_t, 1]))
        phi_u = ot.ot_deg2rad(self.ambient_states[1])

        rps[:, 0] = self.wind_farm_layout[i_t, 0] - np.sin(phi) * self.rp_s[:, 0] * self.wind_farm_layout[i_t, 3]
        rps[:, 1] = self.wind_farm_layout[i_t, 1] + np.cos(phi) * self.rp_s[:, 0] * self.wind_farm_layout[i_t, 3]
        rps[:, 2] = self.wind_farm_layout[i_t, 2] + self.rp_s[:, 1] * self.wind_farm_layout[i_t, 3]

        #  Iterate over all turbines besides the relevant one and calculate reduction at the rotor points and
        #  subsequently across the plane
        red = np.ones((self.wind_farm_layout.shape[0], 1))

        for idx, tur in enumerate(self.wind_farm_layout):
            if idx == i_t:
                # Wind shear
                red[idx] = np.sum((rps[:, 2] / self.wind_farm_layout[i_t, 3]) ** 0.2) / n_rps
                continue

            # calculate wake influence at rotor points
            #   calculate down, crosswind distance and difference in height
            dist_wc = np.transpose(np.array([self.wind_farm_layout[idx, 0] - rps[:, 0],
                                             np.transpose(self.wind_farm_layout[idx, 1] - rps[:, 1])]))
            dist_dw = np.cos(phi_u) * dist_wc[:, 0] + np.sin(phi_u) * dist_wc[:, 1]

            if np.average(dist_dw) > -.1:
                lg.debug(f'Turbine {idx} has no influence on turbine {i_t}')
                lg.debug(f'Location Turbine {idx}: {self.wind_farm_layout[idx, 0:3]}')
                lg.debug(f'Location Turbine {i_t}: {self.wind_farm_layout[i_t, 0:3]}')
                lg.debug(f'Wind direction: {ot.ot_get_orientation(self.ambient_states[1], self.turbine_states[i_t, 1])} deg')
                continue

            dist_cw = -np.sin(phi_u) * dist_wc[:, 0] + np.cos(phi_u) * dist_wc[:, 1]
            dist_h = self.wind_farm_layout[idx, 2] - rps[:, 2]
            dist_r = np.sqrt(dist_cw ** 2 + dist_h ** 2)

            #   calculate resulting reduction
            r = (0.7 + 0.3 * np.cos(dist_dw * np.pi / self.settings['dw'])) * \
                (0.7 + 0.3 * np.sin(dist_r * np.pi / self.settings['cw'])) * \
                (1 - np.exp(-.5 * (dist_dw / self.settings['sig dw']) ** 2) *
                 np.exp(-.5 * (dist_r / self.settings['sig r']) ** 2))

            # average
            red[idx] = np.sum(r) / n_rps

        # Multiply with background wind speed and return
        m = pd.DataFrame([[i_t, self.ambient_states[0] * np.prod(red), np.prod(red)]],
                         columns=['t_idx', 'u_abs_eff', 'red'])
        return self.ambient_states[0] * np.prod(red), m


class FlorisGaussianWake(WakeModel):
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
            .sig r  radial weight
        wind_farm_layout : np.ndarray
            n_t x 4 array with [x,y,z,D] - world coordinates of the rotor center & diameter
        turbine_states : np.ndarray
            n_t x 2 array with [axial ind., yaw]
        ambient_states : np.ndarray
            1 x 2 : [u_abs, phi] - absolute background wind speed and direction
        """
        super(FlorisGaussianWake, self).__init__(settings, wind_farm_layout, turbine_states, ambient_states)

    def set_wind_farm(self, wind_farm_layout: np.ndarray, turbine_states: np.ndarray, ambient_states: np.ndarray):
        """
        Changes the states of the stored wind farm

        Parameters
        ----------
        wind_farm_layout: np.ndarray
            n_t x 4 array with [x,y,z,D] - world coordinates of the rotor center & diameter
        turbine_states: np.ndarray
            n_t x 2 array with [axial ind., yaw]
        ambient_states: np.ndarray
            1 x 2 : [u_abs, phi] - absolute background wind speed and direction
        """
        self.wind_farm_layout = wind_farm_layout
        self.turbine_states = turbine_states
        self.ambient_states = ambient_states

        self.fi = FlorisInterface('/home/cbay/floris_v3/examples/inputs/gch.yaml')
        self.fi.reinitialize(
            layout_x=wind_farm_layout[:,0],
            layout_y=wind_farm_layout[:,1],
            wind_directions=[ambient_states[1]],
            wind_speeds=[ambient_states[0]],
        )

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

        self.fi.calculate_wake(yaw_angles=self.turbine_states[:,1])
        
        avg_vel = self.fi.get_turbine_average_velocities()
        Cts = self.fi.get_turbine_Cts()
        AIs = self.fi.get_turbine_ais()
        TIs = self.fi.get_turbine_TIs()

        m = pd.DataFrame(
            [[
                i_t,
                avg_vel[:,:,i_t].flatten()[0],
                Cts[:,:,i_t].flatten()[0],
                AIs[:,:,i_t].flatten()[0],
                TIs[:,:,i_t].flatten()[0],
            ]],
            columns=['t_idx', 'u_abs_eff', 'Ct', 'AI', 'TI']
        )

        return avg_vel[:,:,i_t].flatten()[0], m

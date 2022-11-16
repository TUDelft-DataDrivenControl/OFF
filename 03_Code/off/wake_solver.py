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
        lg.info('Wake solver settings:')
        lg.info(settings_sol)

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
        lg.info('FLORIDyn wake solver created.')

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


class FLORIDynFlorisWakeSolver(WakeSolver):
    floris_wake: wm.FlorisGaussianWake

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
        super(FLORIDynFlorisWakeSolver, self).__init__(settings_sol)
        lg.info('FLORIDyn FLORIS wake solver created.')

        self.floris_wake = wm.FlorisGaussianWake(settings_wke, np.array([]), np.array([]), np.array([]))

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
        print('u_rp: ', u_rp)
        print('shape u_rp: ', np.shape(u_rp))
        print( 'm: ', m)
        print('u_op: ', u_op)
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
        wind_farm_layout = wind_farm.get_layout()
        turbine_states = wind_farm.get_current_turbine_states()
        ambient_states = np.array([wind_farm.turbines[i_t].ambient_states.get_turbine_wind_speed_abs(),
                                   wind_farm.turbines[i_t].ambient_states.get_turbine_wind_dir()])
        self.floris_wake.set_wind_farm(wind_farm_layout, turbine_states, ambient_states)
        ueff, m = self.floris_wake.get_measurements_i_t(i_t)
        print('ueff: ', ueff)
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


class TWFSolver(WakeSolver):
    floris_wake: wm.FlorisGaussianWake

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

        self.floris_wake = wm.FlorisGaussianWake(settings_wke, np.array([]), np.array([]), np.array([]))

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
        # Load current data
        wind_farm_layout = wind_farm.get_layout()
        turbine_states = wind_farm.get_current_turbine_states()
        ambient_states = np.array([wind_farm.turbines[i_t].ambient_states.get_turbine_wind_speed_abs(),
                                   wind_farm.turbines[i_t].ambient_states.get_turbine_wind_dir()])  # TODO

        # Create an index range over all nT turbines and only select the ones saved in the dependencies
        inf_turbines = np.arange(wind_farm.nT)[wind_farm.dependencies[i_t, :]]
        # index i_t is not correct anymore as only a subset of turbines are considered
        i_t_tmp = np.sum(wind_farm.dependencies[i_t, 0:i_t])

        twf_layout = np.zeros((inf_turbines.shape[0], 3))  # Allocation of x, y, z coordinates of the turbines # TODO D
        twf_t_states = np.zeros((inf_turbines.shape[0], 3))  # TODO fix second dimension
        twf_a_states = np.zeros((inf_turbines.shape[0], 3))  # TODO fix second dimension

        # TODO ========== BIGGER TOPIC ==============
        # Wake models need to receive the states not as vectors but as objects with the state relevant methods.

        # Get reference point of main wind turbine
        rotor_center_i_t = wind_farm.turbines[i_t].get_rotor_pos()

        # Go through dependencies
        for idx in np.arange(inf_turbines.shape[0]):
            if idx == i_t_tmp:
                # Turbine itself
                twf_layout[idx, :] = wind_farm_layout[i_t_tmp, 0:3]
                twf_t_states[idx, :] = wind_farm.turbines[i_t_tmp].turbine_states.get_ind_state(0)
                twf_a_states[idx, :] = wind_farm.turbines[i_t_tmp].ambient_states.get_ind_state(0)
                continue

            lg.debug('Ambient states: Two OP interpolation')
            # Interpolation of turbine states
            #   Step 1 retrieve closest up and downstream OPs
            op_locations = wind_farm.turbines[inf_turbines[idx]].observation_points.get_world_coord()
            ind_op = ot.ot_get_closest_2_points_3d_sorted(rotor_center_i_t, op_locations)

            #   Step 2 calculate interpolation weights
            a = op_locations[ind_op[0], 0:2].transpose()
            b = op_locations[ind_op[1], 0:2].transpose()
            c = rotor_center_i_t[0:2].transpose()

            # d = ((b - a).transpose() * (c - a)) / ((b - a).transpose() * (b - a))
            d = ((b - a) @ (c - a)) / ((b - a) @ (b - a))

            lg.info(f'TWF - OP interpolation weight (should be between 0 and 1): {d} ')
            d = np.fmin(np.fmax(d, 0), 1)
            lg.info(f'TWF - Used OP interpolation weight: {d}')

            r0 = 1 - d
            r1 = d

            #   Interpolate states
            #       1. OP location
            tmp_op = op_locations[ind_op[0], 0:3] * r0 + op_locations[ind_op[1], 0:3] * r1
            #       2. Ambient
            twf_a_states[idx, :] = wind_farm.turbines[inf_turbines[idx]].ambient_states.get_ind_state(ind_op[0]) * r0 \
                + wind_farm.turbines[inf_turbines[idx]].ambient_states.get_ind_state(ind_op[1]) * r1
            #       3. Turbine state
            twf_t_states[idx, :] = wind_farm.turbines[inf_turbines[idx]].turbine_states.get_ind_state(ind_op[0]) * r0 \
                + wind_farm.turbines[inf_turbines[idx]].turbine_states.get_ind_state(ind_op[1]) * r1
            #   Reconstruct turbine location
            tmp_phi = wind_farm.turbines[inf_turbines[idx]].ambient_states.get_wind_dir_ind(ind_op[0]) * r0 \
                + wind_farm.turbines[inf_turbines[idx]].ambient_states.get_wind_dir_ind(ind_op[1]) * r1
            tmp_phi = ot.ot_deg2rad(tmp_phi)
            #       1. Get vector from OP to related turbine
            vec_op2t = wind_farm.turbines[inf_turbines[idx]].observation_points.get_vec_op_to_turbine(ind_op[0]) * r0 \
                + wind_farm.turbines[inf_turbines[idx]].observation_points.get_vec_op_to_turbine(ind_op[1]) * r1
            #       2. Set turbine location
            twf_layout[idx, :] = tmp_op - np.array([[np.cos(tmp_phi), -np.sin(tmp_phi), 0],
                                                    [np.sin(tmp_phi), np.cos(tmp_phi),  0],
                                                    [0, 0, 1]]) @ vec_op2t

        # TODO Debug plot of effective wind farm layout
        # Set wind farm in the wake model
        self.floris_wake.set_wind_farm(twf_layout, twf_t_states, twf_a_states)
        # Get the measurements
        ueff, m = self.floris_wake.get_measurements_i_t(i_t_tmp)
        lg.info(f'Effective wind speed of turbine {i_t} : {ueff} m/s')
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

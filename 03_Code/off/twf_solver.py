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
        # Load current data
        wind_farm_layout = wind_farm.get_layout()
        turbine_states = wind_farm.get_current_turbine_states()
        ambient_states = np.array([wind_farm.turbines[i_t].ambient_states.get_turbine_wind_speed_abs(),
                                   wind_farm.turbines[i_t].ambient_states.get_turbine_wind_dir()])

        # Create an index range over all nT turbines and only select the ones saved in the dependencies
        inf_turbines = np.arange(wind_farm.nT)[wind_farm.dependencies[i_t, :]]
        twf_layout = np.zeros(inf_turbines.shape[0], 2)  # Allocation of x & y coordinates of the turbines
        twf_t_states = np.zeros(inf_turbines.shape[0], turbine_states.shape[1])
        twf_a_states = np.zeros(inf_turbines.shape[0], ambient_states.shape[1])

        rotor_center_i_t = wind_farm.turbines[i_t].get_rotor_pos()

        # Create the tmp wind farm based on the influencing turbines
        # TODO This should create a wind farm "light" object. It would have the same structure, but its turbines would
        # TODO only have one state instread of the many OPs, Ambient & turbine states.
        tmp_wf = wfm.WindFarm(wind_farm.get_sub_windfarm(inf_turbines))

        # Go through dependencies
        for idx in np.arange(inf_turbines.shape[0]):
            if idx == i_t:
                continue

            # Interpolation of turbine states
            #   Step 1 retrieve closest up and downstream OPs
            op_locations = wind_farm.turbines[inf_turbines[idx]].observation_points.get_world_coord()
            ind_op = ot.ot_get_closest_2_points_3d_sorted(rotor_center_i_t, op_locations)

            #   Step 2 calculate interpolation weights
            a = op_locations[ind_op[0], :].transpose()
            b = op_locations[ind_op[1], :].transpose()
            c = rotor_center_i_t[0:2].transpose()

            d = ((b - a).transpose() * (c - a)) / ((b - a).transpose() * (b - a))

            d = np.min(np.max(d, 0), 1)
            lg.info(f'TWF - OP interpolation weight (should be between 0 and 1): {d} ')
            r0 = 1 - d
            lg.info(f'TWF - Used OP interpolation weight: {d}')
            r1 = d

            #   Interpolate states
            #       1. OP location
            tmp_op = op_locations[ind_op[0], 0:2] * r0 + op_locations[ind_op[1], 0:2] * r1
            #       2. Ambient
            twf_a_states[idx, :] = wind_farm.turbines[inf_turbines[idx]].ambient_states.get_ind_state(ind_op[0]) * r0 \
                + wind_farm.turbines[inf_turbines[idx]].ambient_states.get_ind_state(ind_op[1]) * r1
            #       3. Turbine state
            twf_t_states[idx, :] = wind_farm.turbines[inf_turbines[idx]].ambient_states.get_ind_state(ind_op[0]) * r0 \
                + wind_farm.turbines[inf_turbines[idx]].ambient_states.get_ind_state(ind_op[1]) * r1
            #   Reconstruct turbine location
            tmp_phi = wind_farm.turbines[inf_turbines[idx]].ambient_states.get_wind_dir_ind(ind_op[0]) * r0 \
                + wind_farm.turbines[inf_turbines[idx]].ambient_states.get_wind_dir_ind(ind_op[1]) * r1
            tmp_phi = ot.ot_deg2rad(tmp_phi)
            #       1. Get vector from OP to related turbine
            vec_op2t = wind_farm.turbines[inf_turbines[idx]].observation_points.get_vec_op_to_turbine(ind_op[0]) * r0 \
                + wind_farm.turbines[inf_turbines[idx]].observation_points.get_vec_op_to_turbine(ind_op[1]) * r1
            #       2. Set turbine location
            tmp_wf.turbines[idx].set_rotor_pos = tmp_op - np.array([[np.cos(tmp_phi), -np.sin(tmp_phi), 0],
                                                                    [np.sin(tmp_phi), np.cos(tmp_phi),  0],
                                                                    [0, 0, 1]])*np.transpose(vec_op2t)

            # Based on settings either apply weighted retreval of flow field state or interpolation
            if self.settings_sol("Weighted"):
                lg.debug('Ambient states: Weighted interpolation')
            else:
                lg.debug('Ambient states: Two OP interpolation')
                # Use same weights as OP calculation

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

# Copyright (C) <2023>, M Becker (TUDelft), M Lejeune (UCLouvain)

# List of the contributors to the development of OFF: see LICENSE file.
# Description and complete License: see LICENSE file.
	
# This program (OFF) is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program (see COPYING file).  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import off.utils as ot
import logging
# from floris.tools import FlorisInterface
# from floris.tools.visualization import visualize_cut_plane
from floris import FlorisModel, TimeSeries
import matplotlib.pyplot as plt
import yaml

from typing import List
from .turbine import TurbineStates, AmbientStates

lg = logging.getLogger(__name__)


class WakeModel(ABC):
    settings : dict
    wind_farm_layout : np.ndarray
    turbine_states : List[TurbineStates]
    ambient_states : List[AmbientStates]
    rp_s = np.ndarray
    rp_w : float

    def __init__(self, settings: dict, wind_farm_layout: np.ndarray, turbine_states, ambient_states):
        """
        Class to calculate the impact of a wind farm onto a specific turbine

        Parameters
        ----------
        settings: dict
            wake settings, such as parameters
        wind_farm_layout : np.ndarray
            [n_t x 4] array with x,y,z,D entries for each turbine rotor (D = diameter)
        turbine_states : array of TurbineStates objects
            array with n_t TurbineStates objects with each one state
            Obejcts have access to methods such as get_current_Ct()
        ambient_states : array of AmbientStates objects
            array with n_t AmbientStates objects with each one state
            Obejcts have access to methods such as get_turbine_wind_dir()
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

    def set_wind_farm(self, wind_farm_layout: np.ndarray, turbine_states, ambient_states):
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

    def __init__(self, settings: dict, wind_farm_layout: np.ndarray, turbine_states, ambient_states):
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
            ot.ot_get_orientation(self.ambient_states[i_t].get_turbine_wind_dir(),
                                  self.turbine_states[i_t].get_current_yaw()))
        phi_u = ot.ot_deg2rad(self.ambient_states[i_t].get_turbine_wind_dir())

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
                lg.debug('Turbine %s has no influence on turbine %s' % (idx, i_t))
                lg.debug('Location Turbine %s: %s' % (idx, self.wind_farm_layout[idx, 0:3]))
                lg.debug('Location Turbine %s: %s' % (i_t, self.wind_farm_layout[i_t, 0:3]))
                lg.debug('Wind direction: %s deg' % ot.ot_get_orientation(self.ambient_states[i_t].get_turbine_wind_dir()))
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
        m = pd.DataFrame([[i_t, self.ambient_states[i_t].get_turbine_wind_speed_abs() * np.prod(red), np.prod(red)]],
                         columns=['t_idx', 'u_abs_eff', 'red'])
        return self.ambient_states[i_t].get_turbine_wind_speed_abs() * np.prod(red), m


class FlorisGaussianWake(WakeModel):
    """
    Interface to the FLORIS wake models
    """

    def __init__(self, settings: dict, wind_farm_layout: np.ndarray, turbine_states, ambient_states):
        """
        Interface to the FLORIS wake models

        Parameters
        ----------
        settings : dict
            .["yaml_path"] path to settings, e.g. gch.yaml
                example files can be found at https://github.com/NREL/floris/tree/main/examples/inputs
        wind_farm_layout : np.ndarray
            n_t x 4 array with [x,y,z,D] - world coordinates of the rotor center & diameter
        turbine_states : array of TurbineStates objects
            array with n_t TurbineStates objects with each one state
            Obejcts have access to methods such as get_current_Ct()
        ambient_states : array of AmbientStates objects
            array with n_t AmbientStates objects with each one state
            Obejcts have access to methods such as get_turbine_wind_dir()
        turbine_states : np.ndarray
            n_t x 2 array with [axial ind., yaw]
        ambient_states : np.ndarray
            1 x 2 : [u_abs, phi] - absolute background wind speed and direction
        """
        super(FlorisGaussianWake, self).__init__(settings, wind_farm_layout, turbine_states, ambient_states)
        lg.info('Interface for the ' + ' model initialized')  # TODO: Add which model has been initialized
        self.fi = FlorisInterface(self.settings['yaml_path'])
        lg.info('FLORIS object created.')

    def set_wind_farm(self, wind_farm_layout: np.ndarray, turbine_states, ambient_states):
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

        self.fi.reinitialize(
            layout_x=wind_farm_layout[:, 0],
            layout_y=wind_farm_layout[:, 1],
            wind_directions=[ambient_states[0].get_turbine_wind_dir()],  # TODO Assign wind vel from main turbine
            wind_speeds=[ambient_states[0].get_turbine_wind_speed_abs()],    # TODO Assign wind dir from main turbine
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
            float: u_eff
                effective wind speed at turbine i_t
            pandas.dataframe: measurements
                all measurements (Power gen, added turbulence, etc.)
        """
        n_t = len(self.turbine_states)
        yaw_ang = np.zeros([1, 1, n_t])

        for ii_t in np.arange(n_t):
            yaw_ang[0, 0, ii_t] = self.turbine_states[ii_t].get_current_yaw()

        self.fi.calculate_wake(yaw_angles=yaw_ang)
        
        avg_vel = self.fi.turbine_average_velocities
        Cts = self.fi.get_turbine_Cts()
        # AIs = self.fi.get_turbine_ais()               # TODO: Fix since FLORIS v3.4 has a bug causing this line to crash
        TIs = self.fi.get_turbine_TIs()
        Pows = self.fi.get_turbine_powers()

        measurements = pd.DataFrame(
            [[
                i_t,
                avg_vel[:, :, i_t].flatten()[0],
                Cts[:, :, i_t].flatten()[0],
                # AIs[:, :, i_t].flatten()[0],          # TODO: FLORIS v3.4 fix
                TIs[:, :, i_t].flatten()[0],
                Pows[:, :, i_t].flatten()[0]
            ]],
            # columns=['t_idx', 'u_abs_eff', 'Ct', 'AI', 'TI']  # TODO: FLORIS v3.4 fix
            columns=['t_idx', 'u_abs_eff', 'Ct', 'TI', 'PowFLORIS']
        )

        return avg_vel[:, :, i_t].flatten()[0], measurements

    def vis_flow_field(self):
        """
        Creates a plot of the wind farm applied to the given turbine using the FLORIS interface
        """

        n_t = len(self.turbine_states)
        yaw_ang = np.zeros([1, 1, n_t])

        for ii_t in np.arange(n_t):
            yaw_ang[0, 0, ii_t] = self.turbine_states[ii_t].get_current_yaw()

        # Don't know if the calculate_wake is needed, but probably for yaw angles
        self.fi.calculate_wake(yaw_angles=yaw_ang)
        horizontal_plane = self.fi.calculate_horizontal_plane(height=self.wind_farm_layout[0, 2])

        fig, ax_horo_plane = plt.subplots()
        # visualize_cut_plane(horizontal_plane, ax=ax_horo_plane, title="Horizontal") # Belongs to FLORIS 3.4, update needed for FLORIS 4
        plt.show()


class PythonGaussianWake(WakeModel):
    """
    Python interface for the Gaussian Curl Hybrid model
    """

    def __init__(self, settings: dict, wind_farm_layout: np.ndarray, turbine_states, ambient_states):
        """
        FLORIS interface for the Gaussian Curl Hybrid model

        Parameters
        ----------
        settings : dict
            .["gch_yaml_path"] path to settings gch.yaml
                example file can be found at https://github.com/NREL/floris/tree/main/examples/inputs
        wind_farm_layout : np.ndarray
            n_t x 4 array with [x,y,z,D] - world coordinates of the rotor center & diameter
        turbine_states : array of TurbineStates objects
            array with n_t TurbineStates objects with each one state
            Obejcts have access to methods such as get_current_Ct()
        ambient_states : array of AmbientStates objects
            array with n_t AmbientStates objects with each one state
            Obejcts have access to methods such as get_turbine_wind_dir()
        turbine_states : np.ndarray
            n_t x 2 array with [axial ind., yaw]
        ambient_states : np.ndarray
            1 x 2 : [u_abs, phi] - absolute background wind speed and direction
        """
        super(PythonGaussianWake, self).__init__(settings, wind_farm_layout, turbine_states, ambient_states)
        lg.info('Loading input file for Gaussian Wake model (Build in)')

        stream = open(self.settings['sim_dir'] + self.settings['gch_yaml_path'], 'r')
        sim_info = yaml.safe_load(stream)

        # TODO: parameters should be loaded from the yaml file
        self.param = {}
        self.param['ka_ti']  = sim_info["wake"]["wake_deflection_parameters"]["gauss"]["ka"]
        self.param['kb_ti']  = sim_info["wake"]["wake_deflection_parameters"]["gauss"]["kb"]
        
        self.param["ky"]     = 1
        self.param["kz"]     = 1

        self.param['ka_ti']  = 0.03
        self.param['kb_ti']  = 0
        self.param['epsfac'] = 0.2

        # sim_info["wake"]["wake_deflection_parameters"]["gauss"]["ad"]

        lg.info('Gaussian Wake model created.')

    def set_wind_farm(self, wind_farm_layout: np.ndarray, turbine_states: List[TurbineStates], ambient_states:List[AmbientStates]):
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

        # self.fi.reinitialize(
        #     layout_x=wind_farm_layout[:, 0],
        #     layout_y=wind_farm_layout[:, 1],
        #     wind_directions=[ambient_states[0].get_turbine_wind_dir()],  # TODO Assign wind vel from main turbine
        #     wind_speeds=[ambient_states[0].get_turbine_wind_speed_abs()],    # TODO Assign wind dir from main turbine
        # )

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
            float: u_eff
                effective wind speed at turbine i_t
            pandas.dataframe: measurements
                all measurements (Power gen, added turbulence, etc.)
        """
        n_t = len(self.turbine_states)

        yaw_angle = np.array([s.get_current_yaw() for s in self.turbine_states])
        ct      = np.array([s.get_current_ct()  for s in self.turbine_states])
        ti      = 0.06
        avg_vel = self._u_rotors(ct, yaw_angle, ti, i_t=[i_t])[0]
        # avg_vel = self._u(self.wind_farm_layout[:2,:], ct, yaw_angle, ti)

        measurements = pd.DataFrame(
            [[
                i_t,
                avg_vel,
            ]],
            columns=['t_idx', 'u_abs_eff']
        )

        return avg_vel, measurements

    def _u_rotors(self, ct: np.ndarray, yaw_angle: np.ndarray, ti: np.ndarray, i_t: List[int]=None) -> np.ndarray: 
        """ computes the local velocities at the rotor plane

        Parameters
        ----------
        ct : np.ndarray
            n_wts long array containing the local thrust coefficients of the wind turbines
        yaw_angle : np.ndarray
            n_wts long array containing the yaw angles of the wind turbines in deg
        ti : np.ndarray
            n_wts long array containing the turbulence intensity at the location of the wind turbines 
        i_t : List[int], optional
            indexes of the wind turbine where the wind field should be evaluated, by default None (computed at all the wind turbines)

        Returns
        -------
        np.np_array
            array containing the effective wind speed at turbine [i_t]
        """
        # TODO: effective wind speed should be averaged over the rotor
        if i_t is None:
            return self._u(self.wind_farm_layout, ct, yaw_angle, ti)
        else:
            return self._u(self.wind_farm_layout[i_t,:], ct, yaw_angle, ti)

    def _u(self, pos: np.ndarray, ct: np.ndarray, yaw_angle: np.ndarray, ti: np.ndarray) -> np.ndarray: # for now, only velocity at the center
        """ Computes the effective wind speed at pos

        Parameters
        ----------
        pos : np.ndarray
            n_pos x 3 array containing the position the wind field is evaluated at
        ct : np.ndarray
            n_wts long array containing the local thrust coefficients of the wind turbines
        yaw_angle : np.ndarray
            n_wts long array containing the yaw angles of the wind turbines in deg
        ti : np.ndarray
            n_wts long array containing the turbulence intensity at the location of the wind turbines 

        Returns
        -------
        np.ndarray:
            n_pos long array containing the effective wind speed at pos
        """        
        # TODO: is it actually the correct way to superpose those fields
        return self.ambient_states[0].get_turbine_wind_speed_abs() - self._du(pos, ct, yaw_angle, ti)

    def _du(self, pos: np.ndarray, ct: np.ndarray, yaw_angle: np.ndarray, ti: np.ndarray) -> np.ndarray:
        """ Computes the velocity deficit at pos 

        Parameters
        ----------
        pos : np.ndarray
            n_pos x 3 array containing the position the wind field is evaluated at
        ct : np.ndarray
            n_wts long array containing the local thrust coefficients of the wind turbines
        yaw_angle : np.ndarray
            n_wts long array containing the yaw angles of the wind turbines in deg
        ti : np.ndarray
            n_wts long array containing the turbulence intensity at the location of the wind turbines 

        Returns
        -------
        np.ndarray:
            n_pos long array containing the wake deficit field at pos
        """        
                
        dx = np.array([np.subtract.outer(pos[:,comp], self.wind_farm_layout[:,comp]) for comp in range(3)])
        
        streamwise_axis = np.array([*self.ambient_states[0].get_turbine_wind_speed(), 0]) / self.ambient_states[0].get_turbine_wind_speed_abs()
        crosswind_axis  = np.array([-streamwise_axis[1], streamwise_axis[0], 0])
        vertical_axis   = np.array([0, 0, 1])

        xi  = np.moveaxis(dx, 0, -1)@streamwise_axis 
        r_h = np.moveaxis(dx, 0, -1)@crosswind_axis 
        r_v = np.moveaxis(dx, 0, -1)@vertical_axis 
        
        du = self._du_xi_r(xi, r_h, r_v, ct, yaw_angle, ti)

        return np.sqrt( np.sum( du ** 2 , axis=1) )

    def _du_xi_r(self, xi: np.ndarray, r_h: np.ndarray, r_v: np.ndarray, ct: np.ndarray, yaw_angle: np.ndarray, ti: np.ndarray) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        xi : np.ndarray
            n_pos x n_wts array containing the streamwise position of pos_i in the reference frame of wind turbine j
        r_h : np.ndarray
            n_pos x n_wts array containing the horizontal position of pos_i in the reference frame of wind turbine j
        r_v : np.ndarray
            n_pos x n_wts array containing the vertical position of pos_i in the reference frame of wind turbine j
        ct : np.ndarray
            n_wts long array containing the local thrust coefficients of the wind turbines
        yaw_angle : np.ndarray
            n_wts long array containing the yaw angles of the wind turbines in deg
        ti : np.ndarray
            n_wts long array containing the turbulence intensity at the location of the wind turbines 

        Returns
        -------
        np.ndarray
            n_pos x n_wts containing the velocity deficit of wind turbine j at location pos i
        """

        # TODO: speed deficit should be 2D and account for yaw deflection

        kstar = self.param['ka_ti'] + self.param['kb_ti'] * ti
        
        _tmp = np.sqrt(1-ct)
        eps = self.param['epsfac'] * np.sqrt(0.5*(1+_tmp)/_tmp)
        xi0 = (np.sqrt(.125) - eps) * self.wind_farm_layout[:,3]/kstar
            
        sig_o_d_sqr = ( kstar * xi/self.wind_farm_layout[:,3] + eps ) ** 2
        rad = 1 - ct / (8.*sig_o_d_sqr)

        u = self.ambient_states[0].get_turbine_wind_speed_abs()
        du   =  u * (1 - np.sqrt(rad * (rad > 0)))* np.exp( -1./(2*sig_o_d_sqr) * (r_h/self.wind_farm_layout[:,3])**2. )  
        du_nw = u * (1 - (np.sqrt(1-ct)))

        idx_bounded = du>du_nw
        idx_cone    = np.abs(r_h) < self.wind_farm_layout[:,3]/2*(1. - xi/xi0) * (xi < xi0) * (xi>0)
        idx_us      = xi>0

        du = du_nw * idx_bounded + du * (~idx_bounded)
        du = du_nw * idx_cone + du * (~idx_cone)
        du = idx_us * du

        return du



    def _deflection_xi(self, xi, x0, sigy, sigz, ct, yaw_ang, D):
        """
        Wake deflection following Bastankhah and Porté-Agel 2016

        Parameters
        ----------
        xi : [nT x nT] np.ndarray 
            Matrix with downstream distance of different points (col) respective to each turbine (row)
        x0 : [1 x nT] np.ndarray 
            potential core length
        sigy : [1 x nT] np.ndarray 
            y axis gaussian spread
        sigz : [1 x nT] np.ndarray 
            z axis gaussian spread
        ct : [1 x nT] np.ndarray 
            Thrust coefficient
        yaw_ang : [1 x nT] np.ndarray 
            misalignment with the main wind direction
        D : [1 x nT] np.ndarray 
            turbine diameter
        """
        delta = np.zeros_like(xi)

        # Create matrix from inputs for calculation
        #   The xi matrix needs a mask and the other inputs can not be masked with the same if they are vectors
        yaw_ang_r = np.tile(np.deg2rad(yaw_ang), (len(yaw_ang),1))
        cos_yaw = np.cos(yaw_ang_r)
        
        ct      = np.tile(ct,(len(ct),1))
        sigy    = np.tile(sigy,(len(sigy),1))
        sigz    = np.tile(sigz,(len(sigz),1))
        yaw_ang = np.tile(yaw_ang,(len(yaw_ang),1))
        D       = np.tile(D,(len(D),1))

        # Eq. 6.12
        theta = 0.3 * yaw_ang_r / cos_yaw * (1 - np.sqrt(1 - ct * cos_yaw))

        # Near wake deflection for all entries (Eq. 7.4)
        delta = theta * xi
        delta[xi < 0.0] = 0.0  # No delfection for upstream turbines

        # Correct wake defelction for far wake entries
        mask_far_wake = xi > x0
        #   Far wake fraction 1 & 2
        fwf1 = (1.6 + np.sqrt(ct[mask_far_wake]))*(1.6* np.sqrt((8*sigy[mask_far_wake]*sigz[mask_far_wake])/(D[mask_far_wake]**2 * cos_yaw[mask_far_wake])) - np.sqrt(ct[mask_far_wake]))
        fwf2 = (1.6 - np.sqrt(ct[mask_far_wake]))*(1.6* np.sqrt((8*sigy[mask_far_wake]*sigz[mask_far_wake])/(D[mask_far_wake]**2 * cos_yaw[mask_far_wake])) + np.sqrt(ct[mask_far_wake]))
        #   Calculate displacement additional to the near wake deflection (Eq. 7.4)
        delta[mask_far_wake] += theta[mask_far_wake]/14.7 * np.sqrt(cos_yaw[mask_far_wake]/(self.param["ky"] * self.param["kz"] * ct[mask_far_wake])) * \
            (2.9 + 1.3*np.sqrt(1 - ct[mask_far_wake]) - ct[mask_far_wake]) * np.log(fwf1 / fwf2) * D[mask_far_wake]
        
        return delta

    def vis_flow_field(self):
        """
        Creates a plot of the wind farm applied to the given turbine using the FLORIS interface
        """

        n_t = len(self.turbine_states)
        yaw_ang = np.zeros([1, 1, n_t])

        for ii_t in np.arange(n_t):
            yaw_ang[0, 0, ii_t] = self.turbine_states[ii_t].get_current_yaw()

        # Don't know if the calculate_wake is needed, but probably for yaw angles
        self.fi.calculate_wake(yaw_angles=yaw_ang)
        horizontal_plane = self.fi.calculate_horizontal_plane(height=self.wind_farm_layout[0, 2])

        fig, ax_horo_plane = plt.subplots()
        # visualize_cut_plane(horizontal_plane, ax=ax_horo_plane, title="Horizontal") # Belongs to FLORIS 3.4, update needed for FLORIS 4
        plt.show()



class Floris4Wake(WakeModel):
    """
    Interface to the FLORIS 4 wake models
    """

    def __init__(self, settings: dict, wind_farm_layout: np.ndarray, turbine_states, ambient_states):
        """

        Parameters
        ----------
        settings : dict
            .["yaml_path"] path to settings, e.g. gch.yaml
                example files can be found at https://github.com/NREL/floris/tree/main/examples/inputs
        wind_farm_layout : np.ndarray
            n_t x 4 array with [x,y,z,D] - world coordinates of the rotor center & diameter
        turbine_states : array of TurbineStates objects
            array with n_t TurbineStates objects with each one state
            Obejcts have access to methods such as get_current_Ct()
        ambient_states : array of AmbientStates objects
            array with n_t AmbientStates objects with each one state
            Obejcts have access to methods such as get_turbine_wind_dir()
        turbine_states : np.ndarray
            n_t x 2 array with [axial ind., yaw]
        ambient_states : np.ndarray
            1 x 2 : [u_abs, phi] - absolute background wind speed and direction
        """
        super(Floris4Wake, self).__init__(settings, wind_farm_layout, turbine_states, ambient_states)
        lg.info('Interface for the ' + ' model initialized')  # TODO: Add which model has been initialized
        self.fmodel = FlorisModel(self.settings['yaml_path'])
        lg.info('FLORIS 4 object created.')

    def set_wind_farm(self, wind_farm_layout: np.ndarray, turbine_states, ambient_states):
        """
        Changes the states of the stored wind farm

        Parameters
        ----------
        wind_farm_layout: np.ndarray
            n_t x 4 array with [x,y,z,D] - world coordinates of the rotor center & diameter
        turbine_states: TurbineStates object with n_t entries for yaw
        ambient_states: AmbientStates object with 1 entry for wind speed and direction
        """
        self.wind_farm_layout = wind_farm_layout
        self.turbine_states = turbine_states
        self.ambient_states = ambient_states

        # retrieve yaw angles
        n_t = len(self.turbine_states)
        yaw_ang = np.zeros([1,n_t])

        for ii_t in np.arange(n_t):
            yaw_ang[0, ii_t] = self.turbine_states[ii_t].get_current_yaw()

        # Set ambient conditions 
        # TODO: Replace turbulence intensity with actual value + index of turbine
        time_series = TimeSeries(
            wind_directions=np.ones(1)*ambient_states[0].get_turbine_wind_dir(),
            wind_speeds=np.ones(1)*ambient_states[0].get_turbine_wind_speed_abs(),
            turbulence_intensities=np.ones(1)*0.06,
        )
        
        # Set the wind farm conditions
        self.fmodel.set(
            layout_x=wind_farm_layout[:, 0],
            layout_y=wind_farm_layout[:, 1],
            wind_data=time_series,
            yaw_angles=yaw_ang,
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
            float: u_eff
                effective wind speed at turbine i_t
            pandas.dataframe: measurements
                all measurements (Power gen, added turbulence, etc.)
        """
        
        # Solve the wind farm
        self.fmodel.run()

        # Retrieve the measurements
        avg_vel = self.fmodel.turbine_average_velocities
        Cts = self.fmodel.get_turbine_thrust_coefficients()
        AIs = self.fmodel.get_turbine_ais() 
        TIs = self.fmodel.get_turbine_TIs()
        Pows = self.fmodel.get_turbine_powers()

        # Store measurements in a pandas dataframe
        measurements = pd.DataFrame(
            [[
                i_t,
                avg_vel.flatten()[i_t],
                Cts.flatten()[i_t],
                AIs.flatten()[i_t],
                TIs.flatten()[i_t],
                Pows.flatten()[i_t]
            ]],
            columns=['t_idx', 'u_abs_eff', 'Ct', 'AI', 'TI','Power']
        )

        # Return the effective wind speed and the measurements
        return avg_vel.flatten()[i_t], measurements

    def vis_flow_field(self):
        """
        Creates a plot of the wind farm applied to the given turbine using the FLORIS interface
        """

        n_t = len(self.turbine_states)
        yaw_ang = np.zeros([1, 1, n_t])

        for ii_t in np.arange(n_t):
            yaw_ang[0, 0, ii_t] = self.turbine_states[ii_t].get_current_yaw()

        # Don't know if the calculate_wake is needed, but probably for yaw angles
        self.fi.calculate_wake(yaw_angles=yaw_ang)
        horizontal_plane = self.fi.calculate_horizontal_plane(height=self.wind_farm_layout[0, 2])

        fig, ax_horo_plane = plt.subplots()
        # visualize_cut_plane(horizontal_plane, ax=ax_horo_plane, title="Horizontal") # Belongs to FLORIS 3.4, update needed for FLORIS 4
        plt.show()

    def vis_tile(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Get datapoints to visualize the turbine wake field

        Parameters
        ----------
        x : np.ndarray
            x coordinates of the points to visualize
        y : np.ndarray
            y coordinates of the points to visualize
        z : np.ndarray
            z coordinates of the points to visualize

        Returns
        -------
        np.ndarray
            effective velocity at the points
        """
        # TODO z might be a single value, extend to array of length x
        
        self.fmodel.run() # TODO check if this is necessary
        return self.fmodel.sample_flow_at_points(x, y, z)
                
        

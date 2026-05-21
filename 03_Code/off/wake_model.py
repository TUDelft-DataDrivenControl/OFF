# Copyright (C) <2024>, M Becker (TUDelft), M Lejeune (UCLouvain)

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
from floris.flow_visualization import visualize_cut_plane
from floris import FlorisModel, TimeSeries
import matplotlib.pyplot as plt
import yaml
import os

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
        # TODO: Find way to combine wind speed and direction of all turbines into one input for FLORIS
        time_series = TimeSeries(
            wind_directions =np.ones(1)*ambient_states[0].get_turbine_wind_dir(),
            wind_speeds     =np.ones(1)*ambient_states[0].get_turbine_wind_speed_abs(),
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
            columns=['t_idx', 'u_abs_eff_FLORIS', 'Ct_FLORIS', 'AI_FLORIS', 'TI_FLORIS','Power_FLORIS']
        )

        # Return the effective wind speed and the measurements
        return avg_vel.flatten()[i_t], measurements

    def vis_flow_field(self):
        """
        Creates a plot of the wind farm applied to the given turbine using the FLORIS interface
        """
        fig, ax_horo_plane = plt.subplots()
        horizontal_plane = self.fmodel.calculate_horizontal_plane(height=self.wind_farm_layout[0, 2])
        visualize_cut_plane(horizontal_plane, ax=ax_horo_plane, title="Horizontal", minSpeed=0, maxSpeed=10)
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
        # Check if z is of the same size as x
        if z.shape != x.shape:
            # Add z points for every height stored in z
            Z = np.ones(x.shape) * z
            return self.fmodel.sample_flow_at_points(x, y, Z)
        else:
            return self.fmodel.sample_flow_at_points(x, y, z)
        
    def get_point_vel(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Get datapoints to visualize the turbine wake field
        Difference to vis_tile is that the FLORIS model is solved here.

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
        # Solve the wind farm
        #self.fmodel.run()

        # Check if z is of the same size as x
        if z.shape != x.shape:
            # Add z points for every height stored in z
            Z = np.ones(x.shape) * z
            return self.fmodel.sample_flow_at_points(x, y, Z)
        elif not isinstance(x,np.ndarray): # Scalar values, not passed as array need to be converted
            return self.fmodel.sample_flow_at_points([x], [y], [z])
        else:
            return self.fmodel.sample_flow_at_points(x, y, z)


class PyWakeModel(WakeModel):
    """
    Interface to PyWake wake models
    """

    def __init__(self, settings: dict, wind_farm_layout: np.ndarray, turbine_states, ambient_states):
        """
        Initialize PyWake model interface

        Parameters
        ----------
        settings : dict
            Configuration including deficit_model, turbulence_model, wind_farm_model, 
            site_model, superposition_model, rotor_avg_model, site_ti, site_shear,
            and optionally floris_wake which points to a separate YAML config file
        wind_farm_layout : np.ndarray
            n_t x 4 array with [x,y,z,D] - world coordinates of rotor center & diameter
        turbine_states : array of TurbineStates objects
        ambient_states : array of AmbientStates objects
        """
        super(PyWakeModel, self).__init__(settings, wind_farm_layout, turbine_states, ambient_states)
        
        # Import PyWake modules
        try:
            from py_wake.site._site import UniformSite
            from py_wake.wind_turbines import WindTurbines
            from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
        except ImportError as e:
            raise ImportError(f"PyWake not installed or import failed: {e}")
        
        # Load model settings from external YAML file if specified
        if 'floris_wake' in settings and settings['floris_wake']:
            import off.off as off
            yaml_path = os.path.join(off.OFF_PATH, settings['floris_wake'])
            lg.info(f'Loading PyWake configuration from: {yaml_path}')
            
            try:
                with open(yaml_path, 'r') as f:
                    wake_config = yaml.safe_load(f)
                
                # Extract model_strings from the wake configuration
                if 'wake' in wake_config and 'model_strings' in wake_config['wake']:
                    model_strings = wake_config['wake']['model_strings']
                    
                    # Override settings with values from YAML file
                    settings = settings.copy()  # Don't modify original
                    settings['wind_farm_model'] = model_strings.get('wind_farm_model', settings.get('wind_farm_model'))
                    settings['deficit_model'] = model_strings.get('deficit_model', settings.get('deficit_model'))
                    settings['deflection_model'] = model_strings.get('deflection_model', settings.get('deflection_model'))
                    settings['turbulence_model'] = model_strings.get('turbulence_model', settings.get('turbulence_model'))
                    settings['superposition_model'] = model_strings.get('superposition_model', settings.get('superposition_model'))
                    settings['rotor_avg_model'] = model_strings.get('rotor_avg_model', settings.get('rotor_avg_model'))
                    settings['site_model'] = model_strings.get('site', settings.get('site_model'))
                    
                    lg.info(f'Loaded PyWake models from YAML: deficit={settings["deficit_model"]}, '
                           f'deflection={settings.get("deflection_model")}, '
                           f'turbulence={settings.get("turbulence_model")}')
                else:
                    lg.warning(f'No wake>model_strings found in {yaml_path}, using settings from main config')
            
            except FileNotFoundError:
                lg.error(f'PyWake config file not found: {yaml_path}')
                lg.warning('Falling back to settings from main configuration')
            except Exception as e:
                lg.error(f'Error loading PyWake config from {yaml_path}: {e}')
                lg.warning('Falling back to settings from main configuration')
        
        lg.info(f'PyWake model initialized with deficit model: {settings.get("deficit_model", "default")}')
        
        # Store settings
        self.site_ti = settings.get('site_ti', 0.06)
        self.site_shear = settings.get('site_shear', 0.0)
        self.site_model_name = settings.get('site_model', 'UniformSite')
        self.deficit_model_name = settings.get('deficit_model', 'BastankhahGaussianDeficit')
        self.turbulence_model_name = settings.get('turbulence_model', 'CrespoHernandez')
        self.wind_farm_model_name = settings.get('wind_farm_model', 'PropagateDownwind')
        self.deflection_model_name = settings.get('deflection_model', 'JimenezWakeDeflection')
        self.superposition_model_name = settings.get('superposition_model', 'SquaredSum')
        self.rotor_avg_model_name = settings.get('rotor_avg_model', 'RotorCenter')
        
        # Ensure turbine_library exists in self.settings (the original, not the copy)
        # This is critical because OFF Turbine objects read from self.settings['turbine_library']
        if 'turbine_library' not in self.settings:
            self.settings['turbine_library'] = {}
        self.turbine_library = self.settings['turbine_library']
        
        # Support loading turbines from PyWake's built-in library
        # Read from self.settings (original) not the local settings copy
        self.use_pywake_turbine_library = self.settings.get('use_pywake_turbine_library', False)
        # Note: pywake_turbine_name is now specified per turbine type in turbine_library
        
        # Initialize deficit model
        self.deficit_model_class = self._get_deficit_model_class(self.deficit_model_name)
        # Initialize turbulence model if specified
        self.turbulence_model = self._get_turbulence_model_class(self.turbulence_model_name)
        # Initialize deflection model if specified
        self.deflection_model = self._get_deflection_model_class(self.deflection_model_name)
        # Initialize superposition model
        self.superposition_model = self._get_superposition_model_class(self.superposition_model_name)
        # Initialize rotor averaging model
        self.rotor_avg_model = self._get_rotor_avg_model(self.rotor_avg_model_name)
        # Create site using the specified site model
        site_class = self._get_site_class(self.site_model_name)
        self.site = site_class(ti=self.site_ti, shear=self.site_shear)
        
        # Initialize wind turbine model (will be updated with actual turbine data)
        self.wind_turbines = None
        self.wake_model = None
        self.shaft_tilt = 5.0  # Default shaft tilt, will be updated from turbine data
        
        # Note: If using PyWake turbine library, Cp/Ct curves are populated by OFFInterface
        # before this __init__() is called, so they're already available in turbine_library
        
        lg.info('PyWake model interface created.')

    def _import_pywake_component(self, component_type: str, model_name: str, return_instance: bool = True):
        """Dynamically import PyWake component by searching common module patterns"""
        # Module search patterns for each component type
        search_patterns = {
            'deficit': ['py_wake.deficit_models', 'py_wake.literature.gaussian_models'],
            'deflection': ['py_wake.deflection_models'],
            'turbulence': ['py_wake.turbulence_models'],
            'superposition': ['py_wake.superposition_models'],
            'wind_farm': ['py_wake.wind_farm_models'],
            'site': ['py_wake.site._site', 'py_wake.site']
        }
        
        if component_type not in search_patterns:
            raise ValueError(f"Unknown component type: {component_type}")
        
        # Try to dynamically find and import the model
        for base_module in search_patterns[component_type]:
            try:
                # Import base module and search for the model
                module = __import__(base_module, fromlist=[''])
                if hasattr(module, model_name):
                    component_class = getattr(module, model_name)
                    return component_class() if return_instance else component_class
                
                # For structured modules, search submodules
                for submodule_name in dir(module):
                    if not submodule_name.startswith('_'):
                        try:
                            submodule = getattr(module, submodule_name)
                            if hasattr(submodule, model_name):
                                component_class = getattr(submodule, model_name)
                                return component_class() if return_instance else component_class
                        except (AttributeError, TypeError):
                            continue
            except ImportError:
                continue
        
        # If not found, raise clear error
        raise ImportError(
            f"Could not find {component_type} model '{model_name}' in PyWake. "
            f"Searched in: {', '.join(search_patterns[component_type])}. "
            f"Please check your YAML configuration."
        )

    def _get_rotor_avg_model(self, model_spec: str):
        """Parse and instantiate PyWake rotor averaging model (e.g., 'CGIRotorAvg(9)')"""
        import re
        
        try:
            # Parse model specification: ModelName(param1, param2, ...)
            match = re.match(r'([A-Za-z0-9_]+)(?:\((.*)\))?', str(model_spec).strip())
            if not match:
                raise ValueError(f"Could not parse '{model_spec}'")
            
            model_name, params_str = match.group(1), match.group(2)
            
            # Import from py_wake.rotor_avg_models
            from py_wake.rotor_avg_models import rotor_avg_model
            if not hasattr(rotor_avg_model, model_name):
                raise AttributeError(f"Model '{model_name}' not found")
            
            model_class = getattr(rotor_avg_model, model_name)
            
            # Parse and convert parameters if provided
            if params_str:
                args = []
                for p in params_str.split(','):
                    try:
                        args.append(int(p.strip()))
                    except ValueError:
                        try:
                            args.append(float(p.strip()))
                        except ValueError:
                            args.append(p.strip())
                return model_class(*args)
            return model_class()
                
        except (ImportError, AttributeError, TypeError, ValueError) as e:
            lg.warning(f"Failed to instantiate rotor averaging model '{model_spec}': {e}, using RotorCenter")
            from py_wake.rotor_avg_models import RotorCenter
            return RotorCenter()

    def _get_superposition_model_class(self, model_name: str):
        """Get PyWake superposition model instance"""
        return self._import_pywake_component('superposition', model_name, return_instance=True)

    def _get_site_class(self, model_name: str):
        """Get PyWake site class"""
        return self._import_pywake_component('site', model_name, return_instance=False)

    def _get_deficit_model_class(self, model_name: str):
        """Get PyWake deficit model class"""
        return self._import_pywake_component('deficit', model_name, return_instance=False)

    def _get_deflection_model_class(self, model_name: str):
        """Get PyWake deflection model instance (optional, returns None if not found)"""
        try:
            return self._import_pywake_component('deflection', model_name, return_instance=True)
        except ImportError as e:
            lg.warning(f"Deflection model '{model_name}' not found: {e}. Continuing without deflection.")
            return None

    def _get_turbulence_model_class(self, model_name: str):
        """Get PyWake turbulence model instance (optional, returns None if not found)"""
        try:
            return self._import_pywake_component('turbulence', model_name, return_instance=True)
        except ImportError as e:
            lg.warning(f"Turbulence model '{model_name}' not found: {e}. Continuing without turbulence.")
            return None

    def _get_wind_farm_model_class(self, model_name: str):
        """Get PyWake wind farm model class"""
        return self._import_pywake_component('wind_farm', model_name, return_instance=False)

    def _load_turbine_from_pywake_library(self, turbine_name: str):
        """
        Load turbine from PyWake's built-in library for use with PyWake wake models.
        Note: Cp/Ct curves should already be populated in turbine_library by OFFInterface
        
        Returns
        -------
        WindTurbine
            PyWake WindTurbine object ready to use with wake models
        """
        # turbine_library was already populated in __init__, just load the turbine object
        lg.info(f'Loading PyWake turbine object: {turbine_name}')
        
        import inspect
        from py_wake.wind_turbines import WindTurbine
        
        # Generate possible module names (lowercase variations)
        # "DTU10MW" -> ["dtu10mw"]
        # "IEA_22MW_280_RWT" -> ["iea22mw", "iea22mw280rwt", ...]
        import re
        module_name_lower = turbine_name.lower()
        module_name_no_underscore = turbine_name.replace('_', '').lower()
        # Extract base name (e.g., "IEA_22MW" from "IEA_22MW_280_RWT")
        match = re.match(r'([a-zA-Z]+_?\d+(?:mw|kw)?)', turbine_name, re.IGNORECASE)
        module_name_base = match.group(1).lower().replace('_', '') if match else module_name_no_underscore
        
        # Try possible module names in order of likelihood
        module_names = [module_name_base, module_name_no_underscore, module_name_lower]
        
        turbine_obj = None
        for module_name in module_names:
            try:
                # Import the module package
                module = __import__(f'py_wake.examples.data.{module_name}', fromlist=[''])
                
                # Find all WindTurbine classes in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, WindTurbine) and obj is not WindTurbine:
                        # Check if class name matches (case-insensitive)
                        if name.lower() == turbine_name.lower() or name.lower().replace('_', '') == turbine_name.lower().replace('_', ''):
                            turbine_obj = obj()
                            lg.info(f'Loaded turbine "{turbine_name}" from py_wake.examples.data.{module_name}.{name}')
                            break
                
                if turbine_obj is not None:
                    break
            except (ImportError, AttributeError) as e:
                lg.debug(f'Module py_wake.examples.data.{module_name} not found or has no matching turbine')
                continue
        
        if turbine_obj is None:
            raise ImportError(f'Cannot load turbine "{turbine_name}" from PyWake library. Tried modules: {", ".join(module_names)}')
        
        # Read shaft_tilt from turbine_library (YAML value)
        if self.turbine_library:
            turbine_key = list(self.turbine_library.keys())[0]
            turbine_data = self.turbine_library[turbine_key]
            self.shaft_tilt = turbine_data.get('shaft_tilt', 5.0)  # Default to 5 degrees if not specified
        
        return turbine_obj

    def _create_wind_turbine_from_yaml(self):
        """Create PyWake wind turbine from YAML turbine library"""
        from py_wake.wind_turbines import WindTurbines
        from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

        # Get turbine type from first turbine (assuming homogeneous farm)
        # TODO: Support heterogeneous farms with different turbine types
        turbine_type = list(self.turbine_library.keys())[0]
        turbine_data = self.turbine_library[turbine_type]
        
        # Extract parameters from YAML
        D = self.wind_farm_layout[0, 3]
        hub_height = turbine_data['hub_height']
        self.shaft_tilt = turbine_data.get('shaft_tilt', 5.0)  # Default to 5 degrees if not specified
        ws = np.array(turbine_data['performance']['Ct_curve']['Ct_u_wind_speeds'])
        ct = np.array(turbine_data['performance']['Ct_curve']['Ct_u_values'])
        cp = np.array(turbine_data['performance']['Cp_curve']['Cp_u_values'])
        
        # Calculate power curve from Cp
        power_curve = 0.5 * 1.225 * (np.pi * (D/2)**2) * ws**3 * cp / 1e6  # Power in MW
        
        self.wind_turbines = WindTurbines(
            names=[turbine_type],
            diameters=[D],
            hub_heights=[hub_height],
            powerCtFunctions=[PowerCtTabular(ws, power_curve, 'MW', ct)]
        )
        return self.wind_turbines

    def set_wind_farm(self, wind_farm_layout: np.ndarray, turbine_states, ambient_states):
        """Update wind farm layout and turbine/ambient states"""
        self.wind_farm_layout = wind_farm_layout
        self.turbine_states = turbine_states
        self.ambient_states = ambient_states
        
        # Create wind turbine model
        # Check if we should use PyWake's built-in library or YAML definitions
        if self.use_pywake_turbine_library:
            # Find a turbine type that has pywake_turbine_name defined
            # (currently assumes all turbines are the same type)
            pywake_turbine_name = None
            turbine_name = None
            
            for t_name, t_data in self.turbine_library.items():
                if 'pywake_turbine_name' in t_data:
                    pywake_turbine_name = t_data['pywake_turbine_name']
                    turbine_name = t_name
                    break
            
            if pywake_turbine_name:
                lg.info(f'Loading turbine from PyWake library: {turbine_name} -> {pywake_turbine_name}')
                self.wind_turbines = self._load_turbine_from_pywake_library(pywake_turbine_name)
            else:
                lg.warning('No pywake_turbine_name found in turbine_library. Using YAML definitions.')
                self._create_wind_turbine_from_yaml()
        else:
            lg.info('Creating turbine from YAML definitions')
            self._create_wind_turbine_from_yaml()
        
        # Initialize wake model
        # PyWake distinguishes between literature models (complete wind farm models)
        # and deficit models (components that need wrapping in a wind farm model)
        # Detect this by checking if the model is from the literature module
        
        is_literature_model = 'literature' in self.deficit_model_class.__module__
        
        if is_literature_model:
            # Literature model - complete wake model
            kwargs = {
                'site': self.site,
                'windTurbines': self.wind_turbines
            }
            if self.turbulence_model:
                kwargs['turbulenceModel'] = self.turbulence_model
            if self.deflection_model:
                kwargs['deflectionModel'] = self.deflection_model
            self.wake_model = self.deficit_model_class(**kwargs)
        else:
            # Deficit model component - wrap in wind farm model
            self.wake_model = self._get_wind_farm_model_class(self.wind_farm_model_name)(
                site=self.site,
                windTurbines=self.wind_turbines,
                wake_deficitModel=self.deficit_model_class(),
                superpositionModel=self.superposition_model,
                rotorAvgModel=self.rotor_avg_model,
                turbulenceModel=self.turbulence_model,
                deflectionModel=self.deflection_model
            )

    def get_measurements_i_t(self, i_t: int) -> tuple:
        """Get effective wind speed and measurements for turbine i_t"""
        if self.wake_model is None:
            raise RuntimeError("Wake model not initialized. Call set_wind_farm first.")
        
        # Run wake simulation
        sim_res = self._run_wake_simulation()
        
        # Extract results for turbine i_t
        WS_eff = sim_res.WS_eff.values.flatten()[i_t]
        TI_eff = sim_res.TI_eff.values.flatten()[i_t]
        Power = sim_res.Power.values.flatten()[i_t]
        CT = sim_res.CT.values.flatten()[i_t]
        
        # Calculate axial induction factor from thrust coefficient
        # Using momentum theory: CT = 4a(1-a)
        # Solving for a: a = 0.5 * (1 - sqrt(1 - CT))
        # For CT > 1 (Glauert region), use empirical correction
        if CT < 0.96:
            AI = 0.5 * (1 - np.sqrt(1 - CT))
        else:
            # Empirical correction for high thrust (CT > 0.96)
            AI = 1.0 / (2.0 - CT)
        
        # Store measurements in pandas dataframe
        measurements = pd.DataFrame(
            [[
                i_t,
                WS_eff,
                CT,
                AI,
                TI_eff,
                Power
            ]],
            columns=['t_idx', 'u_abs_eff_PyWake', 'Ct_PyWake', 'AI_PyWake', 'TI_PyWake', 'Power_PyWake']
        )
        
        return WS_eff, measurements

    def vis_flow_field(self):
        """Visualize wind farm flow field using PyWake"""
        if self.wake_model is None:
            lg.warning("Wake model not initialized, cannot visualize")
            return
        
        # Run wake simulation
        sim_res = self._run_wake_simulation()
        
        # Get ambient conditions for plotting
        wind_speed = self.ambient_states[0].get_turbine_wind_speed_abs()
        wind_direction = self.ambient_states[0].get_turbine_wind_dir()
        
        # Plot horizontal plane at hub height
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sim_res.flow_map(wd=wind_direction, ws=wind_speed).plot_wake_map(ax=ax)
        plt.title("PyWake Flow Field")
        plt.show()

    def compute_wake_flow_map(self, x_grid: np.ndarray, y_grid: np.ndarray):
        """Compute PyWake flow map on horizontal grid"""
        if self.wake_model is None:
            raise RuntimeError("Wake model not initialized. Call set_wind_farm first.")

        # Run wake simulation
        sim_res = self._run_wake_simulation()

        try:
            from py_wake import HorizontalGrid
            # Don't pass wd and ws parameters - they cause PyWake to recalculate
            # without the yaw angles, losing deflection effects!
            fm = sim_res.flow_map(grid=HorizontalGrid(x=x_grid, y=y_grid))
            return fm
        except Exception:
            lg.exception("Failed to compute PyWake flow_map on provided grid")
            raise

    def _run_wake_simulation(self):
        """Run PyWake simulation with current turbine states and ambient conditions"""
        # Get ambient conditions
        wind_speed = self.ambient_states[0].get_turbine_wind_speed_abs()
        wind_direction = self.ambient_states[0].get_turbine_wind_dir()
        
        # Get yaw angles from turbine states
        n_t = len(self.turbine_states)
        yaw_angles = np.array([self.turbine_states[ii_t].get_current_yaw() for ii_t in range(n_t)])
        
        # Reshape yaw and tilt for PyWake (n_turbines, n_wd, n_ws) = (n_t, 1, 1)
        yaw_in = yaw_angles.reshape(n_t, 1, 1)
        tilt_in = self.shaft_tilt * np.ones(n_t).reshape(n_t, 1, 1)
        
        # Run wake model simulation
        return self.wake_model(
            x=self.wind_farm_layout[:, 0],
            y=self.wind_farm_layout[:, 1],
            wd=wind_direction,
            ws=wind_speed,
            yaw=yaw_in,
            tilt=tilt_in
        )

    def _sample_flow_at_points(self, x, y, z):
        """Sample PyWake flow at specified points (internal helper method)"""
        # Run wake simulation
        sim_res = self._run_wake_simulation()
        
        # Convert inputs to arrays and handle z broadcasting
        x_arr = np.atleast_1d(np.array(x, dtype=float))
        y_arr = np.atleast_1d(np.array(y, dtype=float))
        z_arr = np.atleast_1d(np.array(z, dtype=float))
        if z_arr.size == 1:
            z_arr = np.full_like(x_arr, z_arr[0])
        
        # Sample using PyWake Points
        from py_wake.flow_map import Points
        grid = Points(x_arr, y_arr, z_arr)
        fm = sim_res.flow_map(grid)
        ws_eff = fm.WS_eff.values.flatten()
        
        # Return in [3 x n_points] format: [u, v, w]
        result = np.zeros((3, x_arr.size))
        result[0, :] = ws_eff
        return result

    def get_point_vel(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Get velocity at specific points in the flow field"""
        if self.wake_model is None:
            lg.warning("Wake model not initialized, returning ambient wind speed")
            n_points = np.size(x)
            result = np.ones((3, n_points)) * self.ambient_states[0].get_turbine_wind_speed_abs() if self.ambient_states else np.zeros((3, n_points))
            result[1:, :] = 0
            return result
        return self._sample_flow_at_points(x, y, z)

    def vis_tile(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Get datapoints to visualize the turbine wake field"""
        if self.wake_model is None:
            lg.warning("Wake model not initialized, returning zeros")
            return np.zeros((3, np.size(x)))
        
        try:
            return self._sample_flow_at_points(x, y, z)
        except Exception:
            lg.exception("vis_tile PyWake flow_map sampling failed; using ambient fallback")
            wind_speed = self.ambient_states[0].get_turbine_wind_speed_abs()
            result = np.zeros((3, np.size(x)))
            result[0, :] = wind_speed
            return result
        

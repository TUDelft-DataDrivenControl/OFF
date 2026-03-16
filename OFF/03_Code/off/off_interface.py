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

import os, logging
# logging.basicConfig(level=logging.DEBUG)

import off.turbine as tur
import off.windfarm as wfm
import off.observation_points as ops
import off.ambient as amb
import numpy as np
import off.off as off
import importlib.util
import sys
import yaml
import pandas as pd
import shutil
import os
import datetime
from pathlib import Path

class OFFInterface:
    """
    The OFF Interface enables automated and simplified access to the OFF framework.
    New users should start here before extending the code itself.
    """
    off_sim: off.OFF
    ready_to_run: bool
    measurements: pd.DataFrame
    control_applied: pd.DataFrame

    def __init__(self) -> None:
        self.ready_to_run = False

        # ===== Check package requirements =====
        pkg_missing = False
        floris_pkg = 'floris'

        if floris_pkg in sys.modules:
            print(f"{floris_pkg!r} is installed")
        elif (spec := importlib.util.find_spec(floris_pkg)) is not None:
            # Perform import if possible
            module = importlib.util.module_from_spec(spec)
            sys.modules[floris_pkg] = module
            spec.loader.exec_module(module)
            print(f"{floris_pkg!r} has been imported")
        else:
            print(f"can't find the {floris_pkg!r} module / package")
            print("To solve, run 'pip install floris'")
            pkg_missing = True

        if pkg_missing:
            raise ImportError('At least one required package can not be imported.')
        # ===== end check requirements =====

    def init_simulation_by_path(self, path_to_yaml: str):
        """
        Initialize the simulation using a .yaml file

        Parameters
        ----------
        path_to_yaml: str
            Path to the simulation yaml file
        """
        stream = open(path_to_yaml, 'r')
        sim_info = yaml.safe_load(stream)
        stream.close()

        # Convert run data into settings and wind farm object
        self.settings_sim, self.settings_sol, self.settings_wke, self.settings_cor, self.settings_ctr = self._run_yaml_to_dict(sim_info)
        self.settings_sim['path_to_yaml'] = path_to_yaml
        
        # CRITICAL: If using PyWake turbine library, populate curves NOW
        # (BEFORE creating wind_farm, so Turbine objects can read the curves)
        if (self.settings_sol.get("wake_model", "").startswith("PyWake") and 
            self.settings_wke.get('use_pywake_turbine_library', False)):
            
            # Get unique turbine types from the wind farm layout
            turbine_types = set(sim_info["wind_farm"]["farm"]["turbine_type"])
            
            # Populate curves for each turbine type that has pywake_turbine_name
            for turbine_type in turbine_types:
                turbine_def = self.settings_wke['turbine_library'].get(turbine_type, {})
                pywake_name = turbine_def.get('pywake_turbine_name')
                
                if pywake_name:
                    self._populate_pywake_curves(
                        turbine_type,
                        pywake_name,
                        self.settings_wke['turbine_library'],
                        sim_info["wind_farm"]["farm"]
                    )
        
        self.wind_farm = self._run_yaml_to_wind_farm(sim_info)

        # Generate an input file for FLORIS (only if using FLORIS wake model)
        if self.settings_sol["wake_model"].startswith("FLORIS"):
            tmp_yaml_path = self._gen_FLORIS_yaml(self.settings_wke,
                                                  sim_info["wind_farm"],
                                                  sim_info["ambient"],
                                                  path_to_yaml.parent)  # This might not work on Windows
            self.settings_wke.update(dict([('tmp_yaml_path', tmp_yaml_path)]))
        elif self.settings_sol["wake_model"] == "PyWake":
            # PyWake doesn't need a FLORIS yaml file
            pass
        # PythonGaussianWake also needs FLORIS yaml for parameters
        elif self.settings_sol["wake_model"] == "PythonGaussianWake":
            tmp_yaml_path = self._gen_FLORIS_yaml(self.settings_wke,
                                                  sim_info["wind_farm"],
                                                  sim_info["ambient"],
                                                  path_to_yaml.parent)
            self.settings_wke.update(dict([('tmp_yaml_path', tmp_yaml_path)]))

        # Visualization settings
        self.vis = sim_info["vis"]

        # Create OFF simulation object
        self.create_off_simulation()

    def init_simulation_by_dicts(self, 
                                 settings_sim: dict = None, 
                                 settings_sol: dict = None, 
                                 settings_wke: dict = None, 
                                 settings_cor: dict = None, 
                                 settings_ctr: dict = None,
                                 wind_farm: wfm.WindFarm = None):
        """
        Initialize the simulation using dictionaries

        Parameters
        ----------
        settings_sim: dict
            Simulation settings
        settings_sol: dict
            Solver settings
        settings_wke: dict
            Wake settings
        settings_cor: dict
            Corrector settings
        settings_ctr: dict
            Controller settings
        wind_farm: wfm.WindFarm
            Wind farm object
        """
        # Change root directory to previous simulation folder
        if settings_sim is not None:
            self.settings_sim = settings_sim
        else:
            # This setting prevents the simulation from creating a new folder with every simulation
            self.settings_sim['simulation folder'] = self.off_sim.get_sim_dir()

        self.settings_sol = settings_sol if settings_sol is not None else self.settings_sol
        self.settings_wke = settings_wke if settings_wke is not None else self.settings_wke
        self.settings_cor = settings_cor if settings_cor is not None else self.settings_cor
        self.settings_ctr = settings_ctr if settings_ctr is not None else self.settings_ctr
        self.wind_farm = wind_farm if wind_farm is not None else self.wind_farm

        # Create OFF simulation object
        #self.create_off_simulation()
        self.ready_to_run = False

    def create_off_simulation(self, yaw: float = 0.0, axind: float = 1/3):
        """
        Creates the OFF simulation object based on the initialized settings.

        Parameters
        ----------
        yaw: float
            Initial yaw angle for the wind farm (default is 0.0 deg)
        axind: float
            Initial axial induction factor for the wind farm (default is the Betzlimit at 1/3)
        """
        # Create OFF simulation object
        self.off_sim = off.OFF(self.wind_farm, 
                               self.settings_sim, 
                               self.settings_wke, 
                               self.settings_sol, 
                               self.settings_cor, 
                               self.settings_ctr, 
                               self.vis)

        # TODO init based on sim_info inputs & used ambient state model / turbine state model
        self.off_sim.init_sim(
            np.array([self.settings_cor["ambient"]["wind_speeds"][0],
                      self.settings_cor["ambient"]["wind_directions"][0],
                      self.settings_cor["ambient"]["turbulence_intensities"][0]]),
            np.array([axind, yaw, 0]))

        self.ready_to_run = True

    def run_sim(self):
        """
        Runs the initialized simulation.
        """
        if self.ready_to_run:
            self.measurements, self.control_applied = self.off_sim.run_sim()
        else:
            # Throw an error or warning that the simulation is not ready to run yet, possibly it has not yet been initialized
            logging.error('The simulation is not ready to run yet. Possibly it has not yet been initialized.')
            

    def store_measurements(self, path_to_csv=""):
        """
        Stores the measurements as a csv in the run folder or at a given path.
        """
        if len(path_to_csv) == 0:
            path_to_csv = self.off_sim.sim_dir + "/measurements.csv"

        self.measurements.to_csv(path_or_buf=path_to_csv)

    def get_measurements(self) -> pd.DataFrame:
        """
        Returns the measurements generated by the simulation
        """
        return self.measurements

    def store_run_file(self):
        """
        Stores the yaml file used to run the simulation
        """
        # shutil.copyfile(self.off_sim.settings_sim['path_to_yaml'],
        #                 self.off_sim.sim_dir + '/' + self.off_sim.settings_sim['path_to_yaml'].rsplit('/', 1)[-1])
        shutil.copyfile(self.off_sim.settings_sim['path_to_yaml'],
                        Path(self.off_sim.sim_dir) / self.off_sim.settings_sim['path_to_yaml'].name)

    def store_applied_control(self, path_to_csv=""):
        """
        Stores the measurements as a csv in the run folder or at a given path.
        """
        if len(path_to_csv) == 0:
            path_to_csv = Path(self.off_sim.sim_dir) / "applied_control.csv"

        self.control_applied.to_csv(path_or_buf=path_to_csv)

    def get_state(self) -> off.OFF:
        """
        Return the entire simulation state as OFF object
        """
        return self.off_sim

    def set_state(self, off_sim: off.OFF) -> None:
        """
        Set the entire simulation state as OFF object
        """
        self.off_sim = off_sim

    def move_output_to(self, path: str):
        """
        Moves the files in the run folder to a given location.
        Parameters
        ----------
        path: string of the new folder, does not need to end with a delimiter

        Returns
        -------

        """
        # gather all files
        allfiles = os.listdir(self.off_sim.sim_dir)

        # iterate on all files to move them to destination folder
        for f in allfiles:
            src_path = os.path.join(self.off_sim.sim_dir, f)
            dst_path = os.path.join(path, f)
            os.rename(src_path, dst_path)

    def get_ambient_conditions(self) -> None:
        pass

    def set_ambient_conditions(self) -> None:
        pass

    def get_control_parameters(self) -> None:
        pass

    def set_control_parameters(self) -> None:
        pass

    def set_wind_farm(self) -> None:
        pass

    # ================================================================
    # ///////////////// PRIVATE FUNCTIONS \\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # ================================================================
    def _check_requirements(self) -> bool:
        """
        Checks if the required packages are installed, mainly FLORIS.
        But should also check numpy, pandas, ...

        Returns
        -------
        pkg_missing : bool
            If true, there is at least one required package missing
        """
        # TODO add a check for all packages
        pkg_missing = False
        floris_pkg = 'floris'

        if floris_pkg in sys.modules:
            print(f"{floris_pkg!r} is installed")
        elif (spec := importlib.util.find_spec(floris_pkg)) is not None:
            # If you choose to perform the actual import ...
            module = importlib.util.module_from_spec(spec)
            sys.modules[floris_pkg] = module
            spec.loader.exec_module(module)
            print(f"{floris_pkg!r} has been imported")
        else:
            print(f"can't find the {floris_pkg!r} module / package")
            print("To solve, run 'pip install floris'")
            pkg_missing = True

        return pkg_missing

    def _run_yaml_to_dict(self, sim_info: dict) -> tuple:
        """
        Returns settings dicts for simulation, wake, solver, corrector and controller (sim, wke, sol, cor, ctr)
        Parameters
        ----------
        sim_info:
            read directory from a run .yaml file

        Returns
        -------
        (settings_sim, settings_wke, settings_sol):
            Tuple of different directories for the OFF simulation
        """
        # Import data and create Simulation Dict
        settings_sim = dict([('time step', sim_info["sim"]["sim"]["time step"]),
                             ('time start', sim_info["sim"]["sim"]["time start"]),
                             ('time end', sim_info["sim"]["sim"]["time end"]),
                             ('simulation folder', sim_info["io"]["simulation folder"]),
                             ('data folder', sim_info["io"]["data folder"]),
                             ('log console enable', sim_info["sim"]["logging"]["console"]["enable"]),
                             ('log console lvl', sim_info["sim"]["logging"]["console"]["level"]),
                             ('log file enable', sim_info["sim"]["logging"]["file"]["enable"]),
                             ('log file lvl', sim_info["sim"]["logging"]["file"]["level"])])

        settings_sol = sim_info["solver"]["settings"]

        settings_wke = sim_info["wake"]["settings"]
        settings_wke['turbine_library'] = sim_info["turbine"]  # Pass turbine definitions to wake model

        settings_cor = {'ambient': sim_info["ambient"].get('flow_field', False),
                        'turbine': sim_info["turbine"].get('feed', False),
                        'wake': sim_info["wake"].get('feed', False)}

        settings_ctr = sim_info["controller"]["settings"]
        settings_ctr['time step'] = sim_info["sim"]["sim"]["time step"]
        settings_ctr['number of turbines'] = len(sim_info["wind_farm"]["farm"]["layout_x"])

        return settings_sim, settings_sol, settings_wke, settings_cor, settings_ctr

    def _populate_pywake_curves(self, turbine_type: str, pywake_turbine_name: str, turbine_library: dict, wind_farm_info: dict):
        """
        Populate Cp/Ct curves from PyWake library BEFORE Turbine objects are created.
        This is called early in initialization to ensure curves exist when Turbine.__init__ reads them.
        
        Parameters
        ----------
        turbine_type : str
            The turbine type key in the YAML (e.g., 'iea22mw')
        pywake_turbine_name : str
            The PyWake class name to import (e.g., 'IEA_22MW_280_RWT')
        """
        import logging as lg
        import inspect
        from py_wake.wind_turbines import WindTurbine
        
        lg.info(f'Pre-loading PyWake turbine curves for: {turbine_type} (using {pywake_turbine_name})')
        
        # Generate possible module names (lowercase variations)
        # "DTU10MW" -> ["dtu10mw", "dtu10mw"]
        # "IEA_22MW_280_RWT" -> ["iea_22mw_280_rwt", "iea22mw280rwt", "iea22mw", ...]
        import re
        module_name_lower = pywake_turbine_name.lower()
        module_name_no_underscore = pywake_turbine_name.replace('_', '').lower()
        # Extract base name (e.g., "IEA_22MW" from "IEA_22MW_280_RWT")
        match = re.match(r'([a-zA-Z]+_?\d+(?:mw|kw)?)', pywake_turbine_name, re.IGNORECASE)
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
                        if name.lower() == pywake_turbine_name.lower() or name.lower().replace('_', '') == pywake_turbine_name.lower().replace('_', ''):
                            turbine_obj = obj()
                            lg.info(f'Loaded turbine "{pywake_turbine_name}" from py_wake.examples.data.{module_name}.{name}')
                            break
                
                if turbine_obj is not None:
                    break
            except (ImportError, AttributeError) as e:
                lg.debug(f'Module py_wake.examples.data.{module_name} not found or has no matching turbine')
                continue
        
        if turbine_obj is None:
            raise ImportError(f'Cannot load turbine "{pywake_turbine_name}" from PyWake library. Tried modules: {", ".join(module_names)}')
        
        # Get diameter - use YAML value if available
        if turbine_type in turbine_library and 'rotor_diameter' in turbine_library[turbine_type]:
            diameter = turbine_library[turbine_type]['rotor_diameter']
        else:
            # Get from PyWake or wind farm layout
            try:
                diameter = turbine_obj.diameter() if callable(getattr(turbine_obj, 'diameter', None)) else turbine_obj.diameter
            except Exception:
                diameter = wind_farm_info.get('diameter', [178.4])[0]  # Default to DTU10MW diameter
        
        # Extract curves
        ws = np.arange(3, 26)
        power_w = turbine_obj.power(ws)
        ct_values = turbine_obj.ct(ws)
        
        # Convert to Cp
        rho = 1.225
        rotor_area = np.pi * (diameter / 2) ** 2
        cp_values = power_w / (0.5 * rho * rotor_area * ws ** 3)
        cp_values = np.clip(cp_values, 0, 0.59)
        ct_values = np.clip(ct_values, 0, 1.2)
        
        # Populate turbine_library
        if turbine_type not in turbine_library:
            turbine_library[turbine_type] = {}
        if 'performance' not in turbine_library[turbine_type]:
            turbine_library[turbine_type]['performance'] = {}
        
        perf = turbine_library[turbine_type]['performance']
        perf['Cp_curve'] = {
            'Cp_u_values': cp_values.tolist(),
            'Cp_u_wind_speeds': ws.tolist()
        }
        perf['Ct_curve'] = {
            'Ct_u_values': ct_values.tolist(),
            'Ct_u_wind_speeds': ws.tolist()
        }
        
        lg.info(f'Pre-loaded Cp/Ct curves from PyWake: {len(ws)} points')

    def _run_yaml_to_wind_farm(self, sim_info: dict) -> wfm.WindFarm:
        """
        Generates wind farm based on loaded yaml information

        Parameters
        ----------
        sim_info

        Returns
        -------
        wfm.WindFarm:
            Object you can call OFF with.
        """
        # Create turbines
        #   Turbines are created with
        #       - base location (x,y,z)     -> np.ndarray,
        #       - orientation (yaw,tilt)    -> np.ndarray,
        #       - Turbine states            -> TurbineStates,
        #       - Observation Points        -> ObservationPoints,
        #       - Ambient states            -> AmbientStates
        #       - Turbine data              -> Turbine type specific data
        turbines = []

        # TODO The current turbine creation does not select a specific OP class/ turbine state class / ambient class
        for idx in range(len(sim_info["wind_farm"]["farm"]["turbine_type"])):
            t = sim_info["wind_farm"]["farm"]["turbine_type"][idx]

            if sim_info["wind_farm"]["farm"]["unit"][0] == 'D':
                dist_factor = sim_info["wind_farm"]["farm"]["diameter"][0]
            else:
                dist_factor = 1

            turbines.append(tur.HAWT_ADM(np.array([sim_info["wind_farm"]["farm"]["layout_x"][idx] * dist_factor,
                                                   sim_info["wind_farm"]["farm"]["layout_y"][idx] * dist_factor,
                                                   sim_info["wind_farm"]["farm"]["layout_z"][idx] * dist_factor]),
                                         np.array(
                                             [sim_info["ambient"]["flow_field"]["wind_directions"][0],  # orientation
                                              sim_info["turbine"][t]["shaft_tilt"]]),  # tilt
                                         tur.TurbineStatesFLORIDyn(sim_info["solver"]["settings"]["n_op"]),
                                         # Turb. states
                                         ops.FLORIDynOPs4(sim_info["solver"]["settings"]["n_op"]),  # OP model
                                         amb.FLORIDynAmbient(sim_info["solver"]["settings"]["n_op"]),  # Ambient model
                                         sim_info["turbine"][t]))  # Turbine data

        wind_farm = wfm.WindFarm(turbines)
        return wind_farm

    def _gen_FLORIS_yaml(self, settings_wke: dict, settings_wf: dict, settings_amb: dict, path_to_tmp: str) -> str:
        """
        Combines the OFF simulation settings into the initial FLORIS input file

        Parameters
        ----------
        settings_wke
        settings_wf
        settings_amb

        Returns
        -------
        path to the FLORIS file
        """
        floris_file = dict([('name', "OFF FLORIS Init Simulation"),
                            ('description', "File to initialize the FLORIS simulation with the correct turbine types "
                                            "and settings"),
                            ('floris_version', "v4")])

        # Read yaml data for Logging, Solver and Wake settings
        stream = open(off.OFF_PATH + '/' + settings_wke["floris_logging"], 'r')
        f_logging = yaml.safe_load(stream)
        floris_file.update(f_logging)
        stream.close()

        stream = open(off.OFF_PATH + '/' + settings_wke["floris_solver"], 'r')
        f_solver = yaml.safe_load(stream)
        floris_file.update(f_solver)
        stream.close()

        stream = open(off.OFF_PATH + '/' + settings_wke["floris_wake"], 'r')
        f_wake = yaml.safe_load(stream)
        floris_file.update(f_wake)
        stream.close()

        # Write yaml data for farm and flow field
        farm = dict([('layout_x', settings_wf['farm']['layout_x']),  # Will be overwritten by twf solver
                     ('layout_y', settings_wf['farm']['layout_y']),  # Will be overwritten by twf solver
                     ('turbine_type', settings_wf['farm']['turbine_type'])])

        floris_file.update(dict([('farm', farm)]))

        # The code below does not take time into account, but rather initializes with a given wind dir / vel
        flow_field = dict([('air_density', settings_amb['flow_field']['air_density']),
                           ('reference_wind_height', settings_amb['flow_field']['reference_wind_height']),
                           ('turbulence_intensities', [settings_amb['flow_field']['turbulence_intensities'][0]]),
                           ('wind_directions', [settings_amb['flow_field']['wind_directions'][0]]),
                           ('wind_shear', settings_amb['flow_field']['wind_shear']),
                           ('wind_speeds', [settings_amb['flow_field']['wind_speeds'][0]]),
                           ('wind_veer', settings_amb['flow_field']['wind_veer'])])
        floris_file.update(dict([('flow_field', flow_field)]))

        current_time = datetime.datetime.now()
        
        path_out = path_to_tmp / ('tmp_floris_input' + current_time.strftime("%Y%m%d%H%M%S%f") + '.yaml')
        with open(path_out, "w") as yaml_file:
            yaml.dump(floris_file, yaml_file)

        return path_out

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

        # Convert run data into settings and wind farm object
        settings_sim, settings_sol, settings_wke, settings_cor, settings_ctr = self._run_yaml_to_dict(sim_info)
        settings_sim['path_to_yaml'] = path_to_yaml
        wind_farm = self._run_yaml_to_wind_farm(sim_info)

        # Generate an input file for FLORIS
        tmp_yaml_path = self._gen_FLORIS_yaml(settings_wke,
                                              sim_info["wind_farm"],
                                              sim_info["ambient"],
                                              path_to_yaml.rsplit('/', 1)[0])  # This might not work on Windows
        settings_wke.update(dict([('tmp_yaml_path', tmp_yaml_path)]))

        # Visualization settings
        vis = sim_info["vis"]

        # Create OFF simulation object
        self.off_sim = off.OFF(wind_farm, settings_sim, settings_wke, settings_sol, settings_cor, settings_ctr, vis)

        # TODO init based on sim_info inputs & used ambient state model / turbine state model
        self.off_sim.init_sim(
            np.array([sim_info["ambient"]["flow_field"]["wind_speeds"][0],
                      sim_info["ambient"]["flow_field"]["wind_directions"][0],
                      sim_info["ambient"]["flow_field"]["turbulence_intensities"][0]]),
            np.array([1 / 3, 0, 0]))

        self.ready_to_run = True

    def run_sim(self):
        """
        Runs the initialized simulation.
        """
        if self.ready_to_run:
            self.measurements, self.control_applied = self.off_sim.run_sim()
        else:
            print('The simulation is not ready to run yet. Possibly it has not yet been initialized.')

    def store_measurements(self, path_to_csv=""):
        """
        Stores the measurements as a csv in the run folder or at a given path.
        """
        if len(path_to_csv) == 0:
            path_to_csv = self.off_sim.sim_dir + "/measurements.csv"

        self.measurements.to_csv(path_or_buf=path_to_csv)

    def store_run_file(self):
        """
        Stores the yaml file used to run the simulation
        """
        shutil.copyfile(self.off_sim.settings_sim['path_to_yaml'],
                        self.off_sim.sim_dir + '/' + self.off_sim.settings_sim['path_to_yaml'].rsplit('/', 1)[-1])

    def store_applied_control(self, path_to_csv=""):
        """
        Stores the measurements as a csv in the run folder or at a given path.
        """
        if len(path_to_csv) == 0:
            path_to_csv = self.off_sim.sim_dir + "/applied_control.csv"

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

        settings_cor = {'ambient': sim_info["ambient"].get('flow_field', False),
                        'turbine': sim_info["turbine"].get('feed', False),
                        'wake': sim_info["wake"].get('feed', False)}

        settings_ctr = sim_info["controller"]["settings"]
        settings_ctr['time step'] = sim_info["sim"]["sim"]["time step"]
        settings_ctr['number of turbines'] = len(sim_info["wind_farm"]["farm"]["layout_x"])

        return settings_sim, settings_sol, settings_wke, settings_cor, settings_ctr

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

        stream = open(off.OFF_PATH + '/' + settings_wke["floris_solver"], 'r')
        f_solver = yaml.safe_load(stream)
        floris_file.update(f_solver)

        stream = open(off.OFF_PATH + '/' + settings_wke["floris_wake"], 'r')
        f_wake = yaml.safe_load(stream)
        floris_file.update(f_wake)

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

        path_out = path_to_tmp + '/tmp_floris_input.yaml'
        with open(path_out, "w") as yaml_file:
            yaml.dump(floris_file, yaml_file)

        return path_out

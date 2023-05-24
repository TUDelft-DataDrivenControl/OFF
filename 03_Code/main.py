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


def main():
    if _check_requirements():
        print("Not all required packages installed, see terminal output for more info.")
        return 1

    # Call the run .yaml
    stream = open(f'{off.OFF_PATH}/02_Examples_and_Cases/02_Example_Cases/run_example.yaml', 'r')
    sim_info = yaml.safe_load(stream)

    # Convert run data into settings and wind farm object
    settings_sim, settings_sol, settings_wke, settings_cor = _run_yaml_to_dict(sim_info)
    wind_farm = _run_yaml_to_wind_farm(sim_info)

    # Visualization settings
    vis = sim_info["vis"]

    # Create OFF simulation object
    off_sim = off.OFF(wind_farm, settings_sim, settings_wke, settings_sol, settings_cor, vis)

    # TODO init based on sim_info inputs & used ambient state model / turbine state model
    off_sim.init_sim(
        np.array([sim_info["ambient"]["flow_field"]["wind_speeds"][0],
                  sim_info["ambient"]["flow_field"]["wind_directions"][0],
                  sim_info["ambient"]["flow_field"]["turbulence_intensity"][0]]),
        np.array([1 / 3, 0, 0]))

    # Run simulation
    measurements = off_sim.run_sim()

    # Store output
    if vis["turbine"]["store_data"]:
        measurements.to_csv(path_or_buf=off_sim.sim_dir + "/measurements.csv")


def _check_requirements() -> bool:
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


def _run_yaml_to_dict(sim_info: dict) -> tuple:
    """
    Returns settings dicts for simulation, wake and solver (sim, wke, sol)
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
    settings_sim = dict([('time step',          sim_info["sim"]["sim"]["time step"]),
                         ('time start',         sim_info["sim"]["sim"]["time start"]),
                         ('time end',           sim_info["sim"]["sim"]["time end"]),
                         ('simulation folder',  sim_info["io"]["simulation folder"]),
                         ('data folder',        sim_info["io"]["data folder"]),
                         ('log console enable', sim_info["sim"]["logging"]["console"]["enable"]),
                         ('log console lvl',    sim_info["sim"]["logging"]["console"]["level"]),
                         ('log file enable',    sim_info["sim"]["logging"]["file"]["enable"]),
                         ('log file lvl',       sim_info["sim"]["logging"]["file"]["level"])])

    settings_sol = sim_info["solver"]["settings"]

    settings_wke = sim_info["wake"]["settings"]

    settings_cor = { 'ambient': sim_info["ambient"].get('flow_field', False),
                     'turbine': sim_info["turbine"].get('feed', False),
                     'wake': sim_info["wake"].get('feed', False) }

    return settings_sim, settings_sol, settings_wke, settings_cor


def _run_yaml_to_wind_farm(sim_info: dict) -> wfm.WindFarm:
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
                                     np.array([0,                                                   # yaw
                                               sim_info["turbine"][t]["shaft_tilt"]]),            # tilt
                                     tur.TurbineStatesFLORIDyn(sim_info["solver"]["settings"]["n_op"]),  # Turb. states
                                     ops.FLORIDynOPs4(sim_info["solver"]["settings"]["n_op"]),           # OP model
                                     amb.FLORIDynAmbient(sim_info["solver"]["settings"]["n_op"]),        # Ambient model
                                     sim_info["turbine"][t]))                                     # Turbine data

    wind_farm = wfm.WindFarm(turbines)
    return wind_farm


if __name__ == "__main__":
    main()

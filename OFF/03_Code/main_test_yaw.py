# //////////////////////////////////////////////////////////////////// #
#    ____  ______ ______ 
#   / __ \|  ____|  ____|
#  | |  | | |__  | |__   
#  | |  | |  __| |  __|  
#  | |__| | |    | |     
#   \____/|_|    |_|     
# //////////////////////////////////////////////////////////////////// #

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

# //////////////////////////////////////////////////////////////////// #
# Welcome to the example OFF main file. This showcases how to run a simulation using the OFF framework.
# The settings are defined in the run_example.yaml file, have a look to see what is possible.
# If you experience issues, create a new issue on the GitHub page https://github.com/TUDelft-DataDrivenControl/OFF
# //////////////////////////////////////////////////////////////////// #

import os, logging
logging.basicConfig(level=logging.ERROR)

import off.off as off
import off.off_interface as offi
import time
import matplotlib.pyplot as plt
from pathlib import Path
OFF_PATH: Path = Path(off.OFF_PATH)

def main():
    start_time = time.time()

    # Create an interface object
    #   The interface object does mot yet know the simulation environment, it only checks requirements
    oi = offi.OFFInterface()
    
    # Tell the simulation what to run
    #   The run file needs to contain everything, the wake model, the ambient conditions etc.
    # Example case
    oi.init_simulation_by_path( OFF_PATH / "02_Examples_and_Cases" / "03_Cases" / "yaw_angle_test.yaml" )
    
    # Get and modify controller setpoints
    control_data = oi.settings_ctr
    control_data["orientation_deg"] = [[270, 270], [270, 270], [270, 270], [270, 270]]
    control_data["orientation_t"]   = [0.0, 100.0, 300.0, 2000.0]
    oi.init_simulation_by_dicts(settings_ctr=control_data)
    oi.create_off_simulation()

    # Run the simulation 1 without yaw
    oi.run_sim()
    results_run1 = oi.get_measurements()


    # Get and modify controller setpoints
    control_data = oi.settings_ctr
    control_data["orientation_deg"] = [[270, 270], [270, 270], [240, 270], [240, 270]]
    control_data["orientation_t"]   = [0.0, 100.0, 300.0, 2000.0]
    oi.init_simulation_by_dicts(settings_ctr=control_data)
    oi.create_off_simulation()

    # Run the simulation 2 with yaw
    oi.run_sim()
    results_run2 = oi.get_measurements()

    print("---OFF Simulation 2 took %s seconds ---" % (time.time() - start_time))


    # Extract time and power data
    time_run1 = results_run1['time']
    power_run1 = results_run1['Power_FLORIS']
    power_run2 = results_run2['Power_FLORIS']

    # A plot comparing the power output of the two runs
    plt.plot(time_run1[0::2], power_run1[0::2], label='Run 1, T1', color='green')
    plt.plot(time_run1[1::2], power_run1[1::2], label='Run 1, T2', color='green', linestyle='dashed')
    plt.plot(time_run1[0::2], power_run2[0::2], label='Run 2, T1', color='blue')
    plt.plot(time_run1[1::2], power_run2[1::2], label='Run 2, T2', color='blue', linestyle='dashed')
    plt.xlabel('Time (s)')
    plt.ylabel('Power [W]')
    plt.title('Power over time, calculated using FLORIDyn with FLORIS')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()


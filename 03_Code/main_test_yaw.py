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

def main():
    start_time = time.time()

    # Create an interface object
    #   The interface object does mot yet know the simulation environment, it only checks requirements
    oi = offi.OFFInterface()
    
    # Tell the simulation what to run
    #   The run file needs to contain everything, the wake model, the ambient conditions etc.
    # Example case
    oi.init_simulation_by_path(f'{off.OFF_PATH}/02_Examples_and_Cases/03_Cases/yaw_angle_test.yaml')
    
    # Get and modify controller setpoints
    control_data = oi.settings_ctr
    control_data["orientation_deg"] = [[270, 270], [270, 270], [270, 270], [270, 270]]
    control_data["orientation_t"]   = [0.0, 100.0, 300.0, 2000.0]
    oi.init_simulation_by_dicts(settings_ctr=control_data)

    # One case used for the publication "A dynamic open-source model to investigate wake dynamics in response to wind farm flow control strategies" Becker, Lejeune et al. 2024
    #oi.init_simulation_by_path(f'{off.OFF_PATH}/02_Examples_and_Cases/03_Cases/nawea_grid_zp_ki0-02_th5_LuT.yaml')
    
    # Run the simulation
    oi.run_sim()
    results_run1 = oi.get_measurements()


    # Get and modify controller setpoints
    control_data = oi.settings_ctr
    control_data["orientation_deg"] = [[270, 270], [270, 270], [240, 270], [240, 270]]
    control_data["orientation_t"]   = [0.0, 100.0, 300.0, 2000.0]
    oi.init_simulation_by_dicts(settings_ctr=control_data)
    
    # ambient_data = oi.settings_cor
    # ambient_data["ambient"]['wind_directions']   = [270, 270, 260, 270]
    # ambient_data["ambient"]['wind_directions_t'] = [0, 100, 300, 1200]
    # oi.init_simulation_by_dicts(settings_cor=ambient_data)

    oi.run_sim()
    results_run2 = oi.get_measurements()

    print("---OFF Simulation took %s seconds ---" % (time.time() - start_time))


    # Extract time and power data
    time_run1 = results_run1['time']
    power_run1 = results_run1['Power_FLORIS']
    power_run2 = results_run2['Power_FLORIS']

    # A plot comparing the power output of the two runs
    #fig, ax = plt.subplots()
    plt.plot(time_run1[0::2], power_run1[0::2], label='Run 1, T1', color='green')
    plt.plot(time_run1[1::2], power_run1[1::2], label='Run 1, T2', color='green', linestyle='dashed')
    plt.plot(time_run1[0::2], power_run2[0::2], label='Run 2, T1', color='blue')
    plt.plot(time_run1[1::2], power_run2[1::2], label='Run 2, T2', color='blue', linestyle='dashed')
    plt.xlabel('Time (s)')
    plt.ylabel('Power FLORIS')
    plt.title('Power FLORIS vs Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Store output
    # oi.store_measurements()
    # oi.store_applied_control()
    # oi.store_run_file()


if __name__ == "__main__":
    main()


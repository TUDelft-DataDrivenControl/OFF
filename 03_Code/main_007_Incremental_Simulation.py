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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import off.off as off
import off.off_interface as offi
import time
from pathlib import Path
OFF_PATH: Path = Path(off.OFF_PATH)

# This example shows how to run a simulation using the OFF framework in an incremental manner.

def main():
    # ======================================================= #
    # Get continous reference simulation
    # ======================================================= #

    start_time = time.time()
    # Create an interface object
    oi = offi.OFFInterface()
    
    # Tell the simulation what to run
    #   The run file needs to contain everything, the wake model, the ambient conditions etc.
    # Example case with PyWake
    oi.init_simulation_by_path( OFF_PATH / "02_Examples_and_Cases" / "02_Example_Cases" / "001_two_turbines_yaw_step_pywake.yaml")
    
    # Run the simulation
    oi.run_sim()

    print("\n---OFF Simulation took %s seconds ---" % (time.time() - start_time))

    # Get output
    measurements_long = oi.get_measurements()
    control_long      = oi.get_applied_control()



    # ======================================================= #
    # Run incremental simulation
    # ======================================================= #
    start_time = time.time()

    Delta_t = 200.0 # seconds
    # Create an interface object
    oi_inc = offi.OFFInterface()
    
    # Initialize the simulation with the same settings as the continous simulation
    oi_inc.init_simulation_by_path( OFF_PATH / "02_Examples_and_Cases" / "02_Example_Cases" / "001_two_turbines_yaw_step_pywake.yaml")
    
    # Retrieve the simulation end time from the settings of the continous simulation
    sim_end_time = oi_inc.off_sim.settings_sim['time end']

    # Create placeholder for incremental measurements and control
    measurements_inc = pd.DataFrame()
    control_inc      = pd.DataFrame()

    for step in np.arange(0.0, sim_end_time, Delta_t):
        # Update the total number of iterations for the incremental simulation (only needed for progress bar)
        oi_inc.off_sim.iterations_total = int(Delta_t / oi_inc.off_sim.settings_sim['time step']) + 1

        # Run the incremental simulation for Delta_t seconds
        oi_inc.increment_sim(Delta_t-oi_inc.off_sim.settings_sim['time step'], start_time=step)

        # Get output and concatenate to the previous outputs
        measurements_inc = pd.concat([measurements_inc, oi_inc.get_measurements()])
        control_inc      = pd.concat([control_inc, oi_inc.get_applied_control()])

    print("\n---OFF Incremental Simulation took %s seconds ---" % (time.time() - start_time))




    # ======================================================= #
    # Plot comparison 
    # ======================================================= #
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axs[0].plot(measurements_long[(measurements_long['t_idx'] == 0)]['time'],
                 measurements_long[(measurements_long['t_idx'] == 0)]['Power_PyWake']/1e6, 
                 label='Continuous Simulation', color='blue')
    axs[0].plot(measurements_inc[(measurements_inc['t_idx'] == 0)]['time'],
                 measurements_inc[(measurements_inc['t_idx'] == 0)]['Power_PyWake']/1e6, 
                 color='orange', linestyle='--', label='Incremental Simulation')
    axs[0].vlines(np.arange(0, sim_end_time, Delta_t),
                color='gray', linestyle=':', alpha=0.5,
                ymin=0, ymax=1.1*np.max(measurements_long['Power_PyWake'])/1e6,
                label='Incremental Simulation Steps')
    axs[0].set_ylim(0, 1.05*np.max(measurements_long['Power_PyWake'])/1e6)
    axs[0].set_ylabel('Power (MW)')
    axs[0].set_title('Turbine 0 Power Comparison')
    axs[0].legend()

    axs[1].plot(measurements_long[(measurements_long['t_idx'] == 1)]['time'],
                 measurements_long[(measurements_long['t_idx'] == 1)]['Power_PyWake']/1e6, 
                 label='Continuous Simulation', color='blue')
    axs[1].plot(measurements_inc[(measurements_inc['t_idx'] == 1)]['time'],
                 measurements_inc[(measurements_inc['t_idx'] == 1)]['Power_PyWake']/1e6, 
                 color='orange', linestyle='--', label='Incremental Simulation')
    axs[1].vlines(np.arange(0, sim_end_time, Delta_t),
                color='gray', linestyle=':', alpha=0.5,
                ymin=0, ymax=1.1*np.max(measurements_long['Power_PyWake'])/1e6,
                label='Incremental Simulation Steps')
    axs[1].set_ylim(0, 1.05*np.max(measurements_long['Power_PyWake'])/1e6)
    axs[1].set_ylabel('Power (MW)')
    axs[1].set_title('Turbine 1 Power Comparison')
    axs[1].set_xlabel('Time (s)')
    axs[1].legend()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()


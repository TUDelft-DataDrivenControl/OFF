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
logging.basicConfig(level=logging.ERROR)

import off.off as off
import off.off_interface as offi
import time

def main():
    start_time = time.time()

    # Create an interface object
    #   The interface object does mot yet know the simulation environment, it only checks requirements
    oi = offi.OFFInterface()
    
    # Tell the simulation what to run
    #   The run file needs to contain everything, the wake model, the ambient conditions etc.
    oi.init_simulation_by_path(f'{off.OFF_PATH}/02_Examples_and_Cases/02_Example_Cases/run_full_NAWEA_time_ki0_01_th_8.yaml')

    # Run the simulation
    oi.run_sim()

    print("---OFF Simulation took %s seconds ---" % (time.time() - start_time))

    # Store output
    oi.store_measurements()
    oi.store_applied_control()
    oi.store_run_file()


if __name__ == "__main__":
    main()


# =========== TICKETS ===========
# [ ] implement the yaw controllers
#   Four controllers with varying complexity are suggested: ideal greedy baseline, realistic greedy baseline, yaw
#   steering LUT based, yaw steering prescribed motion. The controller object should take into account that the same
#   input workflow will be required for axial induction and possibly tilt.

"""
Run file for the Torque 2024 experiments on changing wind directions
"""

import os, logging
from datetime import datetime
logging.basicConfig(level=logging.ERROR)

import off.off as off
import off.off_interface as offi


if __name__ == "__main__":
    path_to_input_files = "/Users/marcusbecker/surfdrive/PhD_Surf/02_Communication/04_Conferences/10_Torque2024/" \
                          "Simulations/Input"

    dir_content = os.listdir(path_to_input_files)
    # Loop over all cases
    for f in dir_content:
        if f.startswith("base") or not f.endswith(".yaml"):
            continue

        # Check if case has been run
        if os.path.isdir(path_to_input_files + "/" + f[0:-5]):
            continue

        # If not, run case
        oi = offi.OFFInterface()
        oi.init_simulation_by_path(path_to_input_files + "/" + f)
        oi.run_sim()
        oi.store_measurements()
        oi.store_applied_control()
        oi.store_run_file()

        # Create folder
        os.mkdir(path_to_input_files + "/" + f[0:-5])

        # Move data to folders
        oi.move_output_to(path_to_input_files + "/" + f[0:-5])

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        print("Finished: " + f)
        print("Stored in: " + path_to_input_files + "/" + f[0:-5])

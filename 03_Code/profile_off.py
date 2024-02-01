import os, logging

logging.basicConfig(level=logging.ERROR)

import off.off as off
import off.off_interface as offi
import time
import cProfile


def main():

    #start_time = time.time()

    # Create an interface object
    #   The interface object does mot yet know the simulation environment, it only checks requirements
    oi = offi.OFFInterface()

    # Tell the simulation what to run
    #   The run file needs to contain everything, the wake model, the ambient conditions etc.
    # oi.init_simulation_by_path(f'{off.OFF_PATH}/02_Examples_and_Cases/02_Example_Cases/run_example.yaml')
    oi.init_simulation_by_path(f'{off.OFF_PATH}/02_Examples_and_Cases/02_Example_Cases/run_example.yaml')

    # Run the simulation
    oi.run_sim()

    # Store output
    oi.store_measurements()
    oi.store_applied_control()
    oi.store_run_file()


if __name__ == "__main__":
    cProfile.run('main()')
    #main()
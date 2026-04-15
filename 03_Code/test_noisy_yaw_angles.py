


import os, logging
logging.basicConfig(level=logging.ERROR)

import off.off as off
import off.off_interface as offi
#import time
#import matplotlib.pyplot as plt
import numpy as np

yaw_misalignment_angles_to_test = np.arange(-30, 31, 2.5)
wind_dir = 235.0  # deg
yaw_rate = 0.3  # deg/s
std_600s = 1

path_to_output = '/Users/marcusbecker/surfdrive/PhD_Surf/01_Research/20_max_energy_instead_of_power/01_OFF_results_noise'

# Load random walk data
path_to_random_walk_data = f'/Users/marcusbecker/surfdrive/PhD_Surf/01_Research/20_max_energy_instead_of_power/02_random_walk/random_walk_data_std_{std_600s}_deg.txt'
random_walk_data = np.loadtxt(path_to_random_walk_data, delimiter=',')
time_random = random_walk_data[0]
random_walk_data = random_walk_data[1:]

def main():

    # Create an interface object
    #   The interface object does mot yet know the simulation environment, it only checks requirements
    oi = offi.OFFInterface()
    
    # Tell the simulation what to run
    #   The run file needs to contain everything, the wake model, the ambient conditions etc.
    # Example case
    oi.init_simulation_by_path(f'{off.OFF_PATH}/02_Examples_and_Cases/03_Cases/yaw_angle_test.yaml')

    for row in random_walk_data:

        # Run a simulation for each yaw misalignment angle
        for yaw_misalignment_angle in yaw_misalignment_angles_to_test:
            print(f'Running simulation for yaw misalignment angle {yaw_misalignment_angle} deg')
            delta_t = abs(yaw_misalignment_angle) / yaw_rate

            # Get and modify controller setpoints
            control_data = oi.settings_ctr
            control_data["orientation_deg"] = [[wind_dir, wind_dir], [wind_dir, wind_dir], [wind_dir - yaw_misalignment_angle, wind_dir], [wind_dir - yaw_misalignment_angle, wind_dir]]
            control_data["orientation_t"]   = [0.0, 100.0, 100.0 + delta_t, 2000.0]

            # Get and modify wind direction setpoints
            ambient_data = oi.settings_cor
            ambient_data["ambient"]['wind_directions']   = row + wind_dir
            ambient_data["ambient"]['wind_directions_t'] = time_random
            #ambient_data["ambient"]['wind_directions']   = [wind_dir, wind_dir, wind_dir, wind_dir]
            #ambient_data["ambient"]['wind_directions_t'] = [0, 100, 300, 2000]

            oi.init_simulation_by_dicts(settings_ctr=control_data, settings_cor=ambient_data)
            oi.create_off_simulation()

            # Run the simulation
            oi.run_sim()

            # Get the power output and save it in a file
            results = oi.get_measurements()

            # Append the power output and yaw misalignment angle to a CSV file
            data = np.concatenate((np.array([yaw_misalignment_angle, delta_t]), results['Power_FLORIS']),axis=0)

            output_file = os.path.join(path_to_output, f'{wind_dir}deg_std_{std_600s}.csv')

            if not os.path.isfile(output_file):
                np.savetxt(output_file, data[np.newaxis], delimiter=',', comments='')
            else:
                with open(output_file, 'ab') as f:
                    np.savetxt(f, data[np.newaxis], delimiter=',')


if __name__ == "__main__":
    main()
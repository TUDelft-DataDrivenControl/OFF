


import os, logging
logging.basicConfig(level=logging.ERROR)

import off.off as off
import off.off_interface as offi
#import time
#import matplotlib.pyplot as plt
import numpy as np

yaw_misalignment_angles_to_test = np.arange(-30, 31, 2.5)
yaw_starting_angles = np.concatenate((np.array([-25]), np.arange(-20,21,2.5), np.array([25])))
wind_dir = 235.0  # deg
yaw_rate = 0.3  # deg/s
path_to_output = '/Users/marcusbecker/surfdrive/PhD_Surf/01_Research/20_max_energy_instead_of_power/01_OFF_results_clean_yaw_starting_angles'


def main():

    # Create an interface object
    #   The interface object does mot yet know the simulation environment, it only checks requirements
    oi = offi.OFFInterface()
    
    # Tell the simulation what to run
    #   The run file needs to contain everything, the wake model, the ambient conditions etc.
    # Example case
    oi.init_simulation_by_path(f'{off.OFF_PATH}/02_Examples_and_Cases/03_Cases/yaw_angle_test.yaml')

    print('Created simulation object')

    for yaw_starting_angle in yaw_starting_angles:
        # Run a simulation for each yaw misalignment angle
        for yaw_misalignment_angle in yaw_misalignment_angles_to_test:
            print(f'Running simulation for yaw misalignment angle {yaw_misalignment_angle} deg')
            delta_t = abs(yaw_misalignment_angle - yaw_starting_angle) / yaw_rate

            # Get and modify controller setpoints
            control_data = oi.settings_ctr
            control_data["orientation_deg"] = [[wind_dir - yaw_starting_angle, wind_dir], [wind_dir - yaw_starting_angle, wind_dir], [wind_dir - yaw_misalignment_angle, wind_dir], [wind_dir - yaw_misalignment_angle, wind_dir]]
            control_data["orientation_t"]   = [0.0, 100.0, 100.0 + delta_t, 2000.0]

            # Get and modify wind direction setpoints
            ambient_data = oi.settings_cor
            ambient_data["ambient"]['wind_directions']   = [wind_dir, wind_dir, wind_dir, wind_dir]
            ambient_data["ambient"]['wind_directions_t'] = [0, 100, 300, 2000]

            oi.init_simulation_by_dicts(settings_ctr=control_data, settings_cor=ambient_data)
            oi.create_off_simulation(yaw=yaw_starting_angle) # Initialize both turbines with the misalignment angle, but T2 changes back to 0

            # Run the simulation
            oi.run_sim()

            # Get the power output and save it in a file
            results = oi.get_measurements()

            # Append the power output and yaw misalignment angle to a CSV file
            data = np.concatenate((np.array([yaw_misalignment_angle, delta_t, yaw_starting_angle]), results['Power_FLORIS']),axis=0)

            output_file = os.path.join(path_to_output, f'{wind_dir}deg.csv')

            if not os.path.isfile(output_file):
                np.savetxt(output_file, data[np.newaxis], delimiter=',', comments='')
            else:
                with open(output_file, 'ab') as f:
                    np.savetxt(f, data[np.newaxis], delimiter=',')


if __name__ == "__main__":
    main()



import os, logging
logging.basicConfig(level=logging.ERROR)

import off.off as off
import off.off_interface as offi
#import time
#import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import shutil

yaw_rate        = 0.3  # deg/s
path_to_output  = '/home/marcusbecker/03_Results/2026_Torque/'
#path_to_output  = '/Users/marcusbecker/surfdrive/PhD_Surf/02_Communication/04_Conferences/16_Torque2026/Research/02_OFF_Results/'
path_to_db      = 'tomato_simulations.db'
#path_to_WindDirOffset = '/Users/marcusbecker/surfdrive/PhD_Surf/01_Research/20_max_energy_instead_of_power/07_New_Wind_Model/'
path_to_WindDirOffset = '/home/marcusbecker/01_Data/02_GeneratedData/02_Wind_Dir_Changes_Torque2026/'


# connect to the SQLite database tomato_simulations.db
if not os.path.exists('tomato_simulations.db'):
    print('Database does not exist, aborting')
    exit()


def retrieve_settings_from_db(db_path):
    """
    Retrieve the first entry from the database where sim_done is 0.
    
    Parameters:
    db_path (str): Path to the SQLite database file.
    
    Returns:
    tuple: A tuple containing the wind direction, yaw start, yaw end, and yaw sigma.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Read the variables test_wind_direction, test_yaw_start, test_yaw_end, test_yaw_sigma from the first entry from the database where sim_done is 0
    #cursor.execute('SELECT * FROM tomato_simulations WHERE sim_done = 0 LIMIT 1')
    cursor.execute('SELECT * FROM tomato_simulations WHERE sim_done = 0 ORDER BY RANDOM() LIMIT 1')
    row = cursor.fetchone()

    if row:
        # Set sim_done to 1 for the first entry
        cursor.execute('UPDATE tomato_simulations SET sim_done = 1 WHERE rowid = ?', (row[0],))
        conn.commit()
        conn.close()
        return row[0], row[2], row[3], row[4], row[5]
    else:
        print('No remaining simulation found')
        exit()

    # Close the database connection
    conn.close()
    return None

def main():


    while True:
        # Retrieve the next set of simulation parameters from the database
        simID, test_wind_direction, test_yaw_start, test_yaw_end, test_sigma = retrieve_settings_from_db(path_to_db)

        if test_wind_direction is None:
            print('No remaining simulation found')
            break

        print(f'Running simulation for wind direction {test_wind_direction} deg, yaw start {test_yaw_start} deg, yaw end {test_yaw_end} deg, sigma {test_sigma} deg')

        # Retrieve Wind Dir Offsets
        wind_dir_offset_data = np.loadtxt(os.path.join(path_to_WindDirOffset, f'WindDirOffset_std_{test_sigma:.0f}_len_1800s.txt'), delimiter=',')
        wind_dir_t = wind_dir_offset_data[0, :]  # Time in seconds
        wind_dir_t += 100  # Offset to start at t=100s
        
        for i in range(1, wind_dir_offset_data.shape[0]):
            for ii in range(1, 2):
                wind_dir_sign = 1 if ii == 1 else -1  # Sign for the wind direction offset
                # Create an interface object
                #   The interface object does mot yet know the simulation environment, it only checks requirements
                oi = offi.OFFInterface()

                # Tell the simulation what to run
                #   The run file needs to contain everything, the wake model, the ambient conditions etc.
                # Example case
                oi.init_simulation_by_path(f'{off.OFF_PATH}/02_Examples_and_Cases/02_Example_Cases/001_two_turbines_yaw_step.yaml')

                print('Created simulation object')
                delta_t = abs(test_yaw_end - test_yaw_start) / yaw_rate

                # Get and modify controller setpoints
                control_data = oi.settings_ctr
                control_data["orientation_deg"] = [[test_wind_direction - test_yaw_start, test_wind_direction], [test_wind_direction - test_yaw_start, test_wind_direction], [test_wind_direction - test_yaw_end, test_wind_direction], [test_wind_direction - test_yaw_end, test_wind_direction]]
                control_data["orientation_t"]   = [0.0, 100.0, 100.0 + delta_t, 2000.0]

                # Get and modify wind direction setpoints
                ambient_data = oi.settings_cor
                ambient_data["ambient"]['wind_directions']   = np.hstack((test_wind_direction, wind_dir_sign*wind_dir_offset_data[i, :] + test_wind_direction)).tolist()
                ambient_data["ambient"]['wind_directions_t'] = np.hstack(([0],wind_dir_t)).tolist()

                oi.init_simulation_by_dicts(settings_ctr=control_data, settings_cor=ambient_data)
                oi.create_off_simulation(yaw=test_yaw_start) # Initialize both turbines with the misalignment angle, but T2 changes back to 0

                # Run the simulation
                oi.run_sim()

                # Get the power output and save it in a file
                results = oi.get_measurements()

                # Append the power output and yaw misalignment angle to a CSV file
                data = np.concatenate((np.array([simID, i*wind_dir_sign, test_yaw_end, test_yaw_start, delta_t, test_wind_direction, test_sigma]), results['Power_FLORIS']),axis=0)

                output_file = os.path.join(path_to_output, f'Results_sim_{simID}.csv')

                if not os.path.isfile(output_file):
                    np.savetxt(output_file, data[np.newaxis], delimiter=',', comments='')
                else:
                    with open(output_file, 'ab') as f:
                        np.savetxt(f, data[np.newaxis], delimiter=',')

                # Cleanup the simulation folder
                shutil.rmtree(oi.off_sim.sim_dir)


if __name__ == "__main__":
    main()
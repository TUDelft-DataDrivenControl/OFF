### Example script to create and manage a SQLite database for storing simulation settings.

import sqlite3
import numpy as np
import os

def count_simulations(db_path):
    """
    Count the number of simulations in the database.
    
    Parameters:
    db_path (str): Path to the SQLite database file.
    
    Returns:
    int: Number of simulations in the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM tomato_simulations')
    count = cursor.fetchone()[0]
    
    conn.close()
    return count

def count_remaining_simulations(db_path):
    """
    Count the number of remaining simulations in the database where sim_done is 0.
    
    Parameters:
    db_path (str): Path to the SQLite database file.
    
    Returns:
    int: Number of remaining simulations in the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM tomato_simulations WHERE sim_done = 0')
    count = cursor.fetchone()[0]
    
    conn.close()
    return count

def reset_simulation(db_path, id):
    """
    Reset the simulation with the given ID in the database.
    
    Parameters:
    db_path (str): Path to the SQLite database file.
    id (int): ID of the simulation to reset.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('UPDATE tomato_simulations SET sim_done = 0 WHERE rowid = ?', (id,))
    conn.commit()
    conn.close()

def create_simulations_db(db_path):
    """
    Create a SQLite database for storing simulation results.
    
    Parameters:
    db_path (str): Path to the SQLite database file.
    """
    
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table for simulations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tomato_simulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sim_done BOOLEAN DEFAULT 0,
            test_wind_direction REAL,
            test_yaw_start REAL,
            test_yaw_end REAL,
            test_sigma INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()

def insert_simulation(db_path, wind_direction, yaw_start, yaw_end, sigma):
    """
    Insert a simulation record into the database.
    
    Parameters:
    db_path (str): Path to the SQLite database file.
    wind_direction (float): Wind direction for the simulation.
    yaw_start (float): Starting yaw angle for the simulation.
    yaw_end (float): Ending yaw angle for the simulation.
    sigma (float): Yaw sigma value for the simulation.
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO tomato_simulations (test_wind_direction, test_yaw_start, test_yaw_end, test_sigma)
        VALUES (?, ?, ?, ?)
    ''', (wind_direction, yaw_start, yaw_end, sigma))
    
    conn.commit()
    conn.close()

def check_if_entry_exists(db_path, wind_direction, yaw_start, yaw_end, sigma):
    """
    Check if a simulation entry with the given parameters already exists in the database.
    
    Parameters:
    db_path (str): Path to the SQLite database file.
    wind_direction (float): Wind direction for the simulation.
    yaw_start (float): Starting yaw angle for the simulation.
    yaw_end (float): Ending yaw angle for the simulation.
    sigma (float): Yaw sigma value for the simulation.
    
    Returns:
    bool: True if the entry exists, False otherwise.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT COUNT(*) FROM tomato_simulations 
        WHERE test_wind_direction = ? AND test_yaw_start = ? AND test_yaw_end = ? AND test_sigma = ?
    ''', (wind_direction, yaw_start, yaw_end, sigma))
    
    exists = cursor.fetchone()[0] > 0
    
    conn.close()
    return exists


def main(db_path):
    
    # Create the database if needed
    if not os.path.exists(db_path):
        create_simulations_db(db_path)
    
    # Define test parameters
    test_wind_directions = np.arange(250, 291, 2.5)  # deg
    test_yaw_start       = np.arange(-30, 31, 2.5)   # deg
    test_yaw_end         = np.arange(-30, 31, 2.5)   # deg
    test_sigma           = np.arange(1, 10, 1)  # deg
    
    # Insert test data into the database
    for wind_direction in test_wind_directions:
        for yaw_start in test_yaw_start:
            for yaw_end in test_yaw_end:
                for sigma in test_sigma:
                    if not check_if_entry_exists(db_path, wind_direction, yaw_start, yaw_end, sigma):
                        insert_simulation(db_path, 
                                        float(wind_direction),
                                        float(yaw_start), 
                                        float(yaw_end), 
                                        int(sigma))
    
    print(f"Database '{db_path}' created and populated with simulation data.")


if __name__ == "__main__":
    # Define the path to the database
    #   if the database does not exist, it will be created. If it already exists, it will be used as is.
    db_path = '/home/marcusbecker/02_Code/01_FLORIDyn/OFF/tomato_simulations.db'

    # Reset specific simulations by their IDs, if simulations were not successfully completed. 
    to_reset = [16, 22, 3, 4, 7, 9]  # IDs to reset
    
    main(db_path)

    for id in to_reset:
        reset_simulation(db_path, id)

    print(f'Current number of simulations in the database: {count_simulations(db_path)}')
    print(f'Current number of remaining simulations in the database: {count_remaining_simulations(db_path)}')
    print(f'Percentage of remaining simulations: {count_remaining_simulations(db_path) / count_simulations(db_path) * 100:.2f}%')
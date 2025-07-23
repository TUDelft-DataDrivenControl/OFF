import sqlite3
import numpy as np

def create_simulations_db(db_path):
    """
    Create a SQLite database for storing simulation results.
    
    Parameters:
    db_path (str): Path to the SQLite database file.
    """
    # Remove existing db
    try:
        import os
        if os.path.exists(db_path):
            os.remove(db_path)
    except Exception as e:
        print(f"Error removing existing database: {e}")

    
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


def main():
    db_path = 'tomato_simulations.db'
    
    # Create the database
    create_simulations_db(db_path)

    test_wind_directions = np.arange(250, 290, 2.5)  # deg
    test_yaw_start       = np.arange(-30, 30, 2.5)   # deg
    test_yaw_end         = np.arange(-30, 30, 2.5)   # deg
    test_sigma           = np.arange(1, 10, 1)  # deg
    
    # Insert test data into the database
    for wind_direction in test_wind_directions:
        for yaw_start in test_yaw_start:
            for yaw_end in test_yaw_end:
                for sigma in test_sigma:
                    insert_simulation(db_path, wind_direction, yaw_start, yaw_end, sigma)
    
    print(f"Database '{db_path}' created and populated with simulation data.")


if __name__ == "__main__":
    main()
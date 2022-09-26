import Code.turbine as t
import Code.windfarm as wf
import numpy as np


def main():
    print("Hello OFF")

    # Import data

    # Create objects
    turbines = [t.DTU10MW(np.array([0, 1, 2]), np.array([0, 0])),
                t.DTU10MW(np.array([0, 1, 2]), np.array([0, 0])),
                t.DTU10MW(np.array([0, 1, 2]), np.array([0, 0]))]

    wind_farm = wf.WindFarm(turbines, 'test2')

    print(wind_farm.turbines[0].orientation)
    wind_farm.turbines[0].orientation[0] = 260
    print(wind_farm.turbines[0].orientation)
    # Run simulation


if __name__ == "__main__":
    main()

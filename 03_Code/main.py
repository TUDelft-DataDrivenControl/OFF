import logging
# logging.basicConfig(level=logging.DEBUG)

import off.turbine as tur
import off.windfarm as wfm
import off.observation_points as ops
import off.ambient as amb
import numpy as np
import off.off as off


def main():
    print("Hello OFF")

    # Import data and create Simulation Dict
    settings_sim = dict([('time step', 4),
                         ('time start', 0),
                         ('time end', 40),
                        #  ('simulation folder', '/Users/lejeunemax/Desktop'),
                         ('data folder', ''),
                         ('log console lvl', 'DEBUG'),
                         ('log file lvl', 'INFO')])

    settings_sol = dict([('wake superposition', 'internal'),    # Within the wake model or outside in OFF
                         ('multi wake', False),                 # Enable different wakes per turbine
                         ('wake switch', False),                # Turbine can switch between wakes (modified OP)
                         ('extrapolation', 'pair')])            # Extrapolation method from OP to point of interest

    settings_wke = dict([('rotor discretization', 'isocell'),
                         ('nRP', 50),                           # Number of points to discretetize the rotor plane
                         ('dw', 50),                            # Dummy wake value
                         ('cw', 20),                            # Dummy wake value
                         ('sig dw', 500),                       # Dummy wake value
                         ('sig r', 40)])                        # Dummy wake value

    # Create turbines
    #   Turbines are created with
    #       - base location (x,y,z)     -> np.ndarray,
    #       - orientation (yaw,tilt)    -> np.ndarray,
    #       - Turbine states            -> TurbineStates,
    #       - Observation Points        -> ObservationPoints,
    #       - Ambient states            -> AmbientStates
    turbines = [tur.DTU10MW(np.array([600, 600, 0]), np.array([0, 0]), tur.TurbineStatesFLORIDyn(10),
                            ops.FLORIDynOPs4(10), amb.FLORIDynAmbient(10)),
                tur.DTU10MW(np.array([1200, 600, 0]), np.array([0, 0]), tur.TurbineStatesFLORIDyn(10),
                            ops.FLORIDynOPs4(10), amb.FLORIDynAmbient(10)),
                tur.DTU10MW(np.array([1800, 600, 0]), np.array([0, 0]), tur.TurbineStatesFLORIDyn(10),
                            ops.FLORIDynOPs4(10), amb.FLORIDynAmbient(10))]

    wind_farm = wfm.WindFarm(turbines)

    # Create simulation object
    off_sim = off.OFF(wind_farm, settings_sim, settings_wke, settings_sol)
    off_sim.init_sim(np.array([8, 225, 0]), np.array([1/3, 0, 0]))

    m = off_sim.run_sim()

    print(wind_farm.turbines[0].orientation)
    wind_farm.turbines[0].orientation[0] = 260
    print(wind_farm.turbines[0].orientation)
    # Run simulation


if __name__ == "__main__":
    main()

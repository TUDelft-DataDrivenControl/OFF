import Code.turbine as tur
import Code.windfarm as wfm
import Code.observation_points as ops
import Code.ambient as amb

import numpy as np


def main():
    print("Hello OFF")

    # Import data

    # Create turbines
    #   Turbines are created with
    #       - base location (x,y,z) -> np.ndarray,
    #       - orientation (yaw,tilt) -> np.ndarray,
    #       - Turbine states -> TurbineStates
    #       - Observation Points  -> ObservationPoints
    #       - Ambient states -> AmbientStates
    turbines = [tur.DTU10MW(np.array([600, 600, 0]), np.array([0, 0]), tur.TurbineStatesFLORIDyn(10),
                            ops.FLORIDynOPs4(10), amb.FLORIDynAmbient(10)),
                tur.DTU10MW(np.array([1200, 600, 0]), np.array([0, 0]), tur.TurbineStatesFLORIDyn(10),
                            ops.FLORIDynOPs4(10), amb.FLORIDynAmbient(10)),
                tur.DTU10MW(np.array([1800, 600, 0]), np.array([0, 0]), tur.TurbineStatesFLORIDyn(10),
                            ops.FLORIDynOPs4(10), amb.FLORIDynAmbient(10))]

    wind_farm = wfm.WindFarm(turbines, 'test2')

    print(wind_farm.turbines[0].orientation)
    wind_farm.turbines[0].orientation[0] = 260
    print(wind_farm.turbines[0].orientation)
    # Run simulation


if __name__ == "__main__":
    main()

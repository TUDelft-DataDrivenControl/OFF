import off.turbine as tur
import off.windfarm as wfm
import off.observation_points as ops
import off.ambient as amb
import numpy as np
import off.off as off


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

    wind_farm = wfm.WindFarm(turbines, 4)

    # Create simulation object
    off_sim = off.OFF(wind_farm)
    off_sim.init_sim(np.array([8, 255, 0]), np.array([1/3, 0, 0]))

    print(wind_farm.turbines[0].orientation)
    wind_farm.turbines[0].orientation[0] = 260
    print(wind_farm.turbines[0].orientation)
    # Run simulation


if __name__ == "__main__":
    main()

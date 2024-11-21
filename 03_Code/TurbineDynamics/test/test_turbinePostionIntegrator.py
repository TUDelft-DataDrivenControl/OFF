import numpy as np
import unittest
import matplotlib.pyplot as plt
import casadi as cas

import TurbineDynamics.turbinePositionIntegrator as tpi
import TurbineDynamics.lib.hydroForce as hyf
import TurbineDynamics.lib.aerodynamicForce as aef
import TurbineDynamics.lib.mooringForce as mof

class TestTurbinePositionIntegrator(unittest.TestCase):
    def setUp(self):
        yaw = cas.MX.sym('yaw')
        windvel = cas.MX.sym('winddir', 2, 1)
        x_dot = cas.MX.sym('x_dot')
        y_dot = cas.MX.sym('y_dot')
        x = cas.MX.sym('x')
        y = cas.MX.sym('y')

        self.Integrator = tpi.TuribinePositionIntegrator()
        self.aero_force, self.rel_yaw_f = aef.AerodynamicForce(x_dot, y_dot, yaw, windvel)
        self.mooring_f = mof.MooringForce(x, y)
        self.hydro_f = hyf.HydroForce(x_dot, y_dot)


    def test_equilibrium(self):
        x0 = 10
        y0 = -10
        x_dot0 = 0.1
        y_dot0 = 0.1

        yaw_angle = 0
        x_wind = 8
        y_wind = 0

        u = np.array([yaw_angle, x_wind, y_wind])
        x = [x0, y0, x_dot0, y_dot0]

        N = 10000
        x_store = np.zeros([N, 4])
        hydroforce_store = np.zeros([N, 2])
        aeroforce_store = np.zeros([N, 2])
        mooring_store = np.zeros([N, 2])

        for i in range(N):
            x = self.Integrator(x, u)
            x_store[i, :] = np.array(x.full()).squeeze()

            hydroforce_store[i, :] = np.array([self.hydro_f(x[2], x[3]).full()]).squeeze()
            aeroforce_store[i, :] = np.array([self.aero_force(x[2], x[3], yaw_angle, np.array([x_wind, y_wind])).full()]).T.squeeze()

        # self.assertAlmostEqual(x[2], 0, 5, "turbine does not reach an equilibrium")
        # self.assertAlmostEqual(x[3], 0, 5, "turbine does not reach an equilibrium")
        x1 = x

        k = 1e-5
        plt.figure()
        plt.plot(x_store[:,2])
        # plt.plot(hydroforce_store[:,0]*k)
        plt.show()

        plt.figure()
        plt.plot(x_store[:,3])
        # plt.plot(hydroforce_store[:,1]*k)
        # plt.plot(aeroforce_store[:,1]*k)
        plt.show()

        # x = -[x0, y0, x_dot0, y_dot0]
        #
        # Integrator = tpi.TuribinePositionIntegrator()
        # for i in range(N):
        #     x = Integrator(x, u)
        #
        # self.assertAlmostEqual(x[2], 0, 5, "turbine does not reach an equilibrium")
        # self.assertAlmostEqual(x[3], 0, 5, "turbine does not reach an equilibrium")

if __name__ == '__main__':
    unittest.main()
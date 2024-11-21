import unittest
import numpy as np
import matplotlib.pyplot as plt
import casadi as cas

from TurbineDynamics.lib import aerodynamicForce as aef

class TestCalculations(unittest.TestCase):
    def setUp(self):
        yaw = cas.MX.sym('yaw')
        windvel = cas.MX.sym('winddir', 2, 1)

        x_dot = cas.MX.sym('x_dot')
        y_dot = cas.MX.sym('y_dot')

        self.aero_force, self.rel_yaw_f = aef.AerodynamicForce(x_dot, y_dot, yaw, windvel)


    def test_yawing(self):
        # Quadrant one
        windvel = np.array([8, 0])
        x_dot = 0
        y_dot = 0

        aero_f_min = np.array([self.aero_force(x_dot, y_dot, -1e-2, windvel).full()]).T.squeeze()
        aero_f_0 = np.array([self.aero_force(x_dot, y_dot, 0, windvel).full()]).T.squeeze()
        aero_f_plus = np.array([self.aero_force(x_dot, y_dot, 1e-2, windvel).full()]).T.squeeze()

        relyaw_0 = np.array([self.rel_yaw_f(x_dot, y_dot, 0, windvel).full()]).T.squeeze()

        self.assertAlmostEqual(relyaw_0, 0)
        self.assertTrue((np.linalg.norm(aero_f_0)>np.linalg.norm(aero_f_min) and np.linalg.norm(aero_f_0)>np.linalg.norm(aero_f_plus)), "Thrust force not maximal with zero relative yaw")
        self.assertTrue((aero_f_0[0]>aero_f_min[0] and aero_f_0[0]>aero_f_plus[0]), "Thrust force parallel to wind not maximal with zero relative yaw")
        self.assertTrue((aero_f_0[1]>aero_f_min[1] and aero_f_0[1]<aero_f_plus[1]), "Thrust force in crosswind direction does not change appropriately")

        # Quadrant two
        windvel = np.array([0, 8])
        x_dot = 0
        y_dot = 0

        yaw0 = np.pi/2
        aero_f_min = np.array([self.aero_force(x_dot, y_dot, yaw0-1e-2, windvel).full()]).T.squeeze()
        aero_f_0 = np.array([self.aero_force(x_dot, y_dot, yaw0, windvel).full()]).T.squeeze()
        aero_f_plus = np.array([self.aero_force(x_dot, y_dot, yaw0+1e-2, windvel).full()]).T.squeeze()

        relyaw_0 = np.array([self.rel_yaw_f(x_dot, y_dot, yaw0, windvel).full()]).T.squeeze()

        self.assertAlmostEqual(relyaw_0, 0)
        self.assertTrue((np.linalg.norm(aero_f_0) > np.linalg.norm(aero_f_min) and np.linalg.norm(
            aero_f_0) > np.linalg.norm(aero_f_plus)), "Thrust force not maximal with zero relative yaw")
        self.assertTrue((aero_f_0[0] < aero_f_min[0] and aero_f_0[0] > aero_f_plus[0]),
                        "Thrust force in crosswind direction does not change appropriately")
        self.assertTrue((aero_f_0[1] > aero_f_min[1] and aero_f_0[1] > aero_f_plus[1]),
                        "Thrust force parallel to wind not maximal with zero relative yaw")

        # Quadrant three
        windvel = np.array([-8, 0])
        x_dot = 0
        y_dot = 0

        yaw0 = np.pi
        aero_f_min = np.array([self.aero_force(x_dot, y_dot, yaw0 - 1e-2, windvel).full()]).T.squeeze()
        aero_f_0 = np.array([self.aero_force(x_dot, y_dot, yaw0, windvel).full()]).T.squeeze()
        aero_f_plus = np.array([self.aero_force(x_dot, y_dot, yaw0 + 1e-2, windvel).full()]).T.squeeze()

        relyaw_0 = np.array([self.rel_yaw_f(x_dot, y_dot, yaw0, windvel).full()]).T.squeeze()

        self.assertAlmostEqual(relyaw_0, 0)
        self.assertTrue((np.linalg.norm(aero_f_0) > np.linalg.norm(aero_f_min) and np.linalg.norm(
            aero_f_0) > np.linalg.norm(aero_f_plus)), "Thrust force not maximal with zero relative yaw")
        self.assertTrue((aero_f_0[0] < aero_f_min[0] and aero_f_0[0] < aero_f_plus[0]),
                        "Thrust force parallel to wind not maximal with zero relative yaw")
        self.assertTrue((aero_f_0[1] < aero_f_min[1] and aero_f_0[1] > aero_f_plus[1]),
                        "Thrust force in crosswind direction does not change appropriately")


        # Quadrant four
        windvel = np.array([0, -8])
        x_dot = 0
        y_dot = 0

        yaw0 = -np.pi/2
        aero_f_min = np.array([self.aero_force(x_dot, y_dot, yaw0 - 1e-2, windvel).full()]).T.squeeze()
        aero_f_0 = np.array([self.aero_force(x_dot, y_dot, yaw0, windvel).full()]).T.squeeze()
        aero_f_plus = np.array([self.aero_force(x_dot, y_dot, yaw0 + 1e-2, windvel).full()]).T.squeeze()

        relyaw_0 = np.array([self.rel_yaw_f(x_dot, y_dot, yaw0, windvel).full()]).T.squeeze()

        self.assertAlmostEqual(relyaw_0, 0)
        self.assertTrue((np.linalg.norm(aero_f_0) > np.linalg.norm(aero_f_min) and np.linalg.norm(
            aero_f_0) > np.linalg.norm(aero_f_plus)), "Thrust force not maximal with zero relative yaw")
        self.assertTrue((aero_f_0[0] > aero_f_min[0] and aero_f_0[0] < aero_f_plus[0]),
                        "Thrust force in crosswind direction does not change appropriately")
        self.assertTrue((aero_f_0[1] < aero_f_min[1] and aero_f_0[1] < aero_f_plus[1]),
                        "Thrust force parallel to wind not maximal with zero relative yaw")

    def test_turbine_velocity(self):
        windvel = np.array([0, 0])
        x_dot = -1
        y_dot = 0
        yaw = 0

        aero_f = np.array([self.aero_force(x_dot, y_dot, yaw, windvel).full()]).T.squeeze()
        rel_yaw = self.rel_yaw_f(x_dot, y_dot, yaw, windvel)

        self.assertAlmostEqual(rel_yaw, 0, 'Relative yaw not correct for turbine moving into the wind')
        self.assertAlmostEqual(aero_f[1], 0, 'Crosswind force not correct for moving turbine')
        self.assertTrue(aero_f[0] > 0, 'Thrust force is not opposed to turbine movement')

        windvel = np.array([0, 0])
        x_dot = -1
        y_dot = -1
        yaw = np.pi/4

        aero_f = np.array([self.aero_force(x_dot, y_dot, yaw, windvel).full()]).T.squeeze()
        rel_yaw = self.rel_yaw_f(x_dot, y_dot, yaw, windvel)

        self.assertAlmostEqual(rel_yaw, 0, 8, 'Relative yaw not correct for turbine moving into the wind')
        self.assertAlmostEqual(aero_f[1]-aero_f[0], 0, 8, 'Crosswind force not correct for moving turbine')
        self.assertTrue(aero_f[0] > 0, 'Thrust force is not opposed to turbine movement')
        self.assertTrue(aero_f[1] > 0, 'Thrust force is not opposed to turbine movement')

        windvel = np.array([0, 0])
        x_dot = -1
        y_dot = 1
        yaw = -np.pi/4

        aero_f = np.array([self.aero_force(x_dot, y_dot, yaw, windvel).full()]).T.squeeze()
        rel_yaw = self.rel_yaw_f(x_dot, y_dot, yaw, windvel)

        self.assertAlmostEqual(rel_yaw, 0, 8, 'Relative yaw not correct for turbine moving into the wind')
        self.assertAlmostEqual(aero_f[1]+aero_f[0], 0, 8, 'Crosswind force not correct for moving turbine')
        self.assertTrue(aero_f[0] > 0, 'Thrust force is not opposed to turbine movement')
        self.assertTrue(aero_f[1] < 0, 'Thrust force is not opposed to turbine movement')

    def test_windvel(self):

        x_dot = 0
        y_dot = 0
        yaw = 0

        aero_f_min = np.array([self.aero_force(x_dot, y_dot, yaw,  np.array([8-1e-2, 0])).full()]).T.squeeze()
        aero_f_0 = np.array([self.aero_force(x_dot, y_dot, yaw,  np.array([8, 0])).full()]).T.squeeze()
        aero_f_plus = np.array([self.aero_force(x_dot, y_dot, yaw,  np.array([8+1e-2, 0])).full()]).T.squeeze()

        self.assertTrue((aero_f_0[0]<aero_f_plus[0] and aero_f_0[0]>aero_f_min[0]), 'Thrust force does not increase for increasing wind speed')

        y_dot = 0
        yaw = 0

        aero_f_min = np.array([self.aero_force(-1e-2, y_dot, yaw,  np.array([8, 0])).full()]).T.squeeze()
        aero_f_0 = np.array([self.aero_force(0, y_dot, yaw,  np.array([8, 0])).full()]).T.squeeze()
        aero_f_plus = np.array([self.aero_force(1e-2, y_dot, yaw,  np.array([8, 0])).full()]).T.squeeze()

        self.assertTrue((aero_f_0[0]>aero_f_plus[0] and aero_f_0[0]<aero_f_min[0]), 'Thrust force does not increase for increasing relative wind speed')


if __name__ == '__main__':
    unittest.main()

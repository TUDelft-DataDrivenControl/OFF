import numpy as np
import unittest
import casadi as cas

from TurbineDynamics.lib import hydroForce as hyf

class TestHydroForce(unittest.TestCase):
    def setUp(self):
        x_dot = cas.MX.sym('x_dot')
        y_dot = cas.MX.sym('y_dot')

        self.hydro_f = hyf.HydroForce(x_dot, y_dot)


    def test_direction(self):
        x_dot = 1
        y_dot = 1

        hydro_force = np.array([self.hydro_f(x_dot, y_dot).full()]).T.squeeze()

        self.assertTrue((hydro_force[0]<0 and hydro_force[1] <0))
        self.assertAlmostEqual(hydro_force[0], hydro_force[1])

if __name__ == '__main__':
    unittest.main()
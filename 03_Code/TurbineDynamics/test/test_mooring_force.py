import numpy as np
import unittest
import casadi as cas

from TurbineDynamics.lib import mooringForce as mof

class TestHydroForce(unittest.TestCase):
    def setUp(self):
        x = cas.MX.sym('x')
        y = cas.MX.sym('y')
        self.mooring_f = mof.MooringForce(x, y)

    def test_direction(self):
        x = 0
        y = 0

        mooring_force = np.array([self.mooring_f(x,y)]).T.squeeze()

        self.assertAlmostEqual(mooring_force[0], 0, 8, "Mooring force not zero when at [0,0]")
        self.assertAlmostEqual(mooring_force[1], 0, 8, "Mooring force not zero when at [0,0]")

        n = np.linspace(1e-1, 2*np.pi-1e-1, 50)
        x = 10*np.cos(n)
        y = 10*np.sin(n)

        for xn, yn in zip(x, y):
            mooring_force = np.array([self.mooring_f(xn,yn)]).T.squeeze()
            self.assertTrue(np.sign(mooring_force[0]) == -np.sign(xn))
            self.assertTrue(np.sign(mooring_force[1]) == -np.sign(yn))

        x1 = 10*np.cos(np.pi/3)
        x2 = 10*np.cos(np.pi)
        x3 = 10*np.cos(-np.pi/3)

        y1 = 10*np.sin(np.pi/3)
        y2 = 10*np.sin(np.pi)
        y3 = 10*np.sin(-np.pi/3)

        mooring_force1 = np.array([self.mooring_f(x1,y1)]).T.squeeze()
        mooring_force2 = np.array([self.mooring_f(x2,y2)]).T.squeeze()
        mooring_force3 = np.array([self.mooring_f(x3,y3)]).T.squeeze()

        self.assertAlmostEqual(np.linalg.norm(mooring_force1), np.linalg.norm(mooring_force2), 8,"No geometric equivalence when moving towards an achor")
        self.assertAlmostEqual(np.linalg.norm(mooring_force2), np.linalg.norm(mooring_force3), 8,"No geometric equivalence when moving towards an achor")

if __name__ == '__main__':
    unittest.main()
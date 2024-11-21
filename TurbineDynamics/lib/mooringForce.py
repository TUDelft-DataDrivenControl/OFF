import casadi as cas
import TurbineDynamics.lib.mooringSpline as msp
import numpy as np


def MooringForce(x, y):
    mooringIntp = msp.MooringSpline()

    mooring_len_neutral = 737

    acr1 = [-mooring_len_neutral, 0]
    acr2 = [mooring_len_neutral*np.cos(np.pi/3), mooring_len_neutral*np.sin(np.pi/3)]
    acr3 = [mooring_len_neutral*np.cos(np.pi/3), -mooring_len_neutral*np.sin(np.pi/3)]

    fairlead1 = [x - 25 / np.cos(np.pi / 6), y]
    fairlead2 = [x + 25 * np.tan(np.pi / 6), y + 25]
    fairlead3 = [x + 25 * np.tan(np.pi / 6), y - 25]

    vec1 = cas.vertcat(acr1[0] - fairlead1[0], acr1[1] - fairlead1[1])
    vec2 = cas.vertcat(acr2[0] - fairlead2[0], acr2[1] - fairlead2[1])
    vec3 = cas.vertcat(acr3[0] - fairlead3[0], acr3[1] - fairlead3[1])

    dist1 = cas.norm_2(vec1)
    dist2 = cas.norm_2(vec2)
    dist3 = cas.norm_2(vec3)

    force1 = mooringIntp(dist1)*vec1/dist1
    force2 = mooringIntp(dist2)*vec2/dist2
    force3 = mooringIntp(dist3)*vec3/dist3

    mooring_force = force1 + force2 + force3
    mooring_force_f = cas.Function('mooring_force_f', [x, y], [mooring_force])

    return mooring_force_f
    
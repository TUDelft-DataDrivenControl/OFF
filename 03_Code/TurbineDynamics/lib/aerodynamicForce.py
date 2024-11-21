import casadi as cas
from math import pi

def AerodynamicForce(x_dot, y_dot, yaw, windvel):
    aif = 1/3

    rel_wind_vel = windvel - cas.vertcat(x_dot, y_dot)
    rel_yaw = yaw - cas.atan2(rel_wind_vel[1], rel_wind_vel[0])
    skew_angle = (0.6 * aif + 1) * rel_yaw
    thrust_coeff = 4 * aif * (cas.cos(rel_yaw) + cas.tan(skew_angle / 2) * cas.sin(rel_yaw) - aif * (1 / cas.cos(skew_angle / 2)) ** 2)
    aero_force = (1 / 8 * thrust_coeff * 1.225 * pi * 126 ** 2 * cas.norm_2(rel_wind_vel) ** 2) * cas.vertcat(cas.cos(yaw), cas.sin(yaw))
    aero_force_f = cas.Function('aero_force_f', [x_dot, y_dot, yaw, windvel], [aero_force])
    rel_yaw_f = cas.Function('rel_yaw_f', [x_dot, y_dot, yaw, windvel], [rel_yaw])

    return aero_force_f, rel_yaw_f

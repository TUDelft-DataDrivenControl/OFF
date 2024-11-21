import casadi as cas

def HydroForce(x_dot, y_dot):
    hydro_force = -(1 / 2 * 1025 * 674 * cas.norm_2(cas.vertcat(x_dot, y_dot))) * cas.vertcat(x_dot, y_dot)
    hydro_force_f = cas.Function('aero_force_f', [x_dot, y_dot], [hydro_force])
    return hydro_force_f
import numpy as np
import casadi as cas
import matplotlib.pyplot as plt

from TurbineDynamics.lib import aerodynamicForce as aef, hydroForce as hydf, mooringForce as mof


def TurbinePositionIntegrator():
    dt = 4
    turbine_mass = 1.4e7+8.8e6

    yaw = cas.MX.sym('yaw')
    windvel = cas.MX.sym('winddir', 2, 1)

    x = cas.MX.sym('x')
    y = cas.MX.sym('y')
    x_dot = cas.MX.sym('x_dot')
    y_dot = cas.MX.sym('y_dot')

    states = cas.vertcat(x, y, x_dot, y_dot)
    inputs = cas.vertcat(yaw, windvel)

    aero_force_f, rel_yaw_f = aef.AerodynamicForce(x_dot, y_dot, yaw, windvel)
    hydro_force_f = hydf.HydroForce(x_dot, y_dot)
    mooring_force_f= mof.MooringForce(x, y)

    acceleration = (1/turbine_mass)*(aero_force_f(x_dot, y_dot, yaw, windvel)+hydro_force_f(x_dot, y_dot)+mooring_force_f(x, y))
    acceleration_f = cas.Function('net_force_f',[x, y, x_dot, y_dot, yaw, windvel], [acceleration])

    ode = cas.vertcat(x_dot, y_dot, acceleration_f(x, y, x_dot, y_dot, yaw, windvel))
    ode_f = cas.Function('ode_f',[states, inputs], [ode])

    dae = {'x': states, 'p': inputs, 'ode': ode}
    integrator = cas.integrator('F', 'cvodes', dae, 0, dt)
    res = integrator(x0=states, p=inputs)
    x_next = res['xf']
    F = cas.Function('F',[states, inputs], [x_next])

    return F

if __name__ == '__main__':
    F = TurbinePositionIntegrator()

    N = 400
    dt = 4

    t= np.arange(0, N*dt, dt)

    yaw_angle = np.ones([N, 1])*(2*np.pi/180)
    x_wind = np.ones([N, 1])*8
    y_wind = np.ones([N, 1])*0

    u = np.array([yaw_angle, x_wind, y_wind])
    x = [0,0,0,0]

    x_store = np.zeros([N,4])
    for i in range(N):
        x = F(x, u[:, i])
        x_store[i] = x.full().T

    fig, (ax_x, ax_y, ax_topview) = plt.subplots(3, 1)
    ax_x.plot(t, x_store[:,0])
    ax_x.set_xlabel('Time (m)')
    ax_x.set_ylabel('x (m)')
    ax_x.grid(True)

    ax_y.plot(t, x_store[:,1])
    ax_y.set_xlabel('Time (m)')
    ax_y.set_ylabel('y (m)')
    ax_y.grid(True)

    ax_topview.plot(x_store[:,0], x_store[:,1])
    ax_topview.set_xlabel('x (m)')
    ax_topview.set_ylabel('y (m)')
    ax_topview.grid(True)

    plt.show()
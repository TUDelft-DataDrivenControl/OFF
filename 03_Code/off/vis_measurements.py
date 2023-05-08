""" Functions to plot the measurements generated during an OFF simulation"""
import matplotlib.pyplot as plt
import pandas as pd

from off import __file__ as OFF_PATH
OFF_PATH = OFF_PATH.rsplit('/', 3)[0]


def plot_effective_wind_speed(run_id: int):
    """

    Parameters
    ----------
    run_id : int
        Number referring to the simulation run of interest
    """
    # Load measurements
    measurements = pd.read_csv(filepath_or_buffer=""+str(run_id))
    # plot measurements
    fig, ax = plt.subplots()
    ax.plot(measurements["time"][0::3],
            measurements["u_abs_eff"][0::3],
            linewidth=2.0, label='T0')
    ax.plot(measurements["time"][1::3],
            measurements["u_abs_eff"][1::3],
            linewidth=2.0, label='T1')
    ax.plot(measurements["time"][2::3],
            measurements["u_abs_eff"][2::3],
            linewidth=2.0, label='T2')
    ax.set_xlabel('time (s)')  # Add an x-label to the axes.
    ax.set_ylabel('eff. wind speed (m/s)')  # Add a y-label to the axes.
    ax.legend()
    plt.show()

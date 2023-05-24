# Copyright (C) <2023>, M Becker (TUDelft), M Lejeune (UCLouvain)

# List of the contributors to the development of OFF: see LICENSE file.
# Description and complete License: see LICENSE file.
	
# This program (OFF) is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program (see COPYING file).  If not, see <https://www.gnu.org/licenses/>.

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

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

from typing import List
import logging
lg = logging.getLogger(__name__)

"""
Utilities for the OFF toolbox
functions which are handy in multiple places but do not have a true parent object they could belong to.
"""
import numpy as np


def ot_deg2rad(deg) -> float:   
    # """
    # Function to convert the in LES common degree convention into radians for calculation

    # Parameters
    # ----------
    # deg : number
    #     LES degrees (270 deg pointing along the x-axis, 190 deg along the y axis)

    # Returns
    # -------
    # float
    #     radians (0 rad pointing along the x-axis, pi/2 rad along the y axis)
    # """
    return np.deg2rad(270 - deg)


def ot_uv2deg(u, v) -> float:
    # """
    # Function to convert u and v component to the in LES common degree convention
    
    # Parameters
    # ----------
    # u
    #     wind speed in x direction
    # v
    #     wind speed in y direction

    # Returns
    # -------
    # float
    #     LES degrees (270 deg pointing along the x-axis, 190 deg along the y axis)
    # """
    return 270 - np.arctan2(v, u)*180 / np.pi


def ot_abs_wind_speed(u, v) -> float:
    # """
    # Calculates the magnitude of the wind speed based on u and v component

    # Parameters
    # ----------
    # u:
    #     wind speed in x direction
    # v:
    #     wind speed in y direction

    # Returns
    # -------
    # float
    #     absolute wind speed
    # """

    return np.sqrt(u**2 + v**2)


def ot_abs2uv(wind_speed_abs, wind_dir) -> np.ndarray:
    # """
    # Calculates the u, v components of the wind speed based on the absolute speed and wind direction

    # Parameters
    # ----------
    # wind_speed_abs:
    #     absolute wind speed in x direction
    # wind_dir:
    #     wind direction (deg)

    # Returns
    # -------
    # np.ndarray
    #     u,v component of the wind in [m, 2] matrix
    # """
    phi = ot_deg2rad(wind_dir)
    return np.array([np.cos(phi) * wind_speed_abs, np.sin(phi) * wind_speed_abs])


def ot_uv2abs(u, v) -> float:
    # """
    # Connects the u & v component and returns the absolute wind speed

    # Parameters
    # ----------
    # u:
    #     x component of the wind speed
    # v:
    #     y component of the wind speed

    # Returns
    # -------
    # float
    #     absolute wind speed (in x, y direction)
    # """
    return np.sqrt(u**2 + v**2)


def ot_isocell(n_rp: int) -> tuple:
    """
    Isocell algorithm to discretize the rotor plane (or any circle)
    Masset et al.
    
    https://orbi.uliege.be/bitstream/2268/91953/1/masset_isocell_orbi.pdf
    
    We choose N = 3 here, 4 or 5 are also viable options, 3 is close to optimal

    Parameters
    ----------
    n_rp : int
        desired number of Rotor points (algorithm can not work with all numbers, takes the closest one)

    Returns
    -------
    tuple
        [yRP, zRP] : np.ndarray location of the rotor points with values between -0.5 and 0.5
        w : float weight of the RPs (1/number)
    """
    N = 3
    n = np.round(np.sqrt(n_rp/N)).astype(int)   # Number of rings
    dR = 1/n                        # Radial thickness of each ring
    nC = N * n**2                   # Number of elements

    rp = np.zeros((nC, 2))

    # TODO vectorize
    for idx in range(n):
        nS = (2 * (idx + 1) - 1)*N        # Segments in the ring

        idx_e = np.sum((2 * (np.arange(idx + 1) + 1) - 1) * N)
        idx_s = idx_e - nS       # Start and end index

        phi = np.arange(nS)/nS * 2 * np.pi
        rp[idx_s: idx_e, 0] = 0.5 * np.cos(phi) * dR * (0.5 + idx)
        rp[idx_s: idx_e, 1] = 0.5 * np.sin(phi) * dR * (0.5 + idx)

    return rp, 1/nC


def ot_get_orientation(wind_dir: float, yaw: float) -> float:
    """
    Return the turbine orientation based on the wind direction and the yaw angle

    Parameters
    ----------
    wind_dir : float
        Wind direction in LES degree (270 deg pointing along the x-axis, 190 deg along the y axis)
    yaw : float
        Yaw angle in degree

    Returns
    -------
    float
        Orientation in LES degree
    """
    return wind_dir + yaw


def ot_get_yaw(wind_dir: float, orientation: float) -> float:
    """
    Return the turbine yaw angle based on the wind direction and turbine orientation

    Parameters
    ----------
    wind_dir : float
        Wind direction in LES degree (270 deg pointing along the x-axis, 190 deg along the y axis)
    orientation : float
        Turbine orientation in LES degree (270 deg pointing along the x-axis, 190 deg along the y axis)

    Returns
    -------
    float
        yaw angle in LES degree (clockwise)
    """
    return orientation - wind_dir


def ot_get_closest_point_3d_sorted(ref_loc: np.ndarray, points: np.ndarray) -> int:
    """
    Function to find the index of the closes point to a reference location in 3D.
    The function can expect the list of points to be sorted / trailing each other.

    Parameters
    ----------
    ref_loc:
        [1 x 3] np.ndarray Reference location
    points
        [n x 3] np.ndarray Points

    Returns
    -------
    int
        index
    """
    # Calculate squared distance
    distSqr = (ref_loc[0] - points[:, 0]) ** 2 + (ref_loc[1] - points[:, 1]) ** 2 + (ref_loc[2] - points[:, 2]) ** 2
    return np.argmin(distSqr)


def ot_get_closest_2_points_3d_sorted(ref_loc: np.ndarray, points: np.ndarray) -> List[int]:
    """
    Function to find the index of the closest 2 points to a reference location in 3D.
    The function can expect the list of points to be sorted / trailing each other.

    Parameters
    ----------
    ref_loc:
        [1 x 3] np.ndarray Reference location
    points
        [n x 3] np.ndarray Points

    Returns
    -------
        [1 x 2] int array
    """
    distSqr = (ref_loc[0] - points[:, 0]) ** 2 + (ref_loc[1] - points[:, 1]) ** 2 + (ref_loc[2] - points[:, 2]) ** 2
    i_1 = np.argmin(distSqr)

    if i_1 == 1:
        # First OP
        return [i_1, 2]
    elif i_1 == points.shape[0]-1:
        # Last OP
        return [i_1, i_1-1]

    if distSqr[i_1+1] > distSqr[i_1-1]:
        return [i_1, i_1 - 1]
    else:
        return [i_1, i_1 + 1]



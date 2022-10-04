import logging
lg = logging.getLogger(__name__)

"""
Utilities for the OFF toolbox
    functions which are handy in multiple places but do not have a true parent object they could belong to.
"""
import numpy as np


def ot_deg2rad(deg):
    """
    Function to convert the in LES common degree convention into radians for calculation

    Parameters
    ----------
    deg : number
        LES degrees (270 deg pointing along the x-axis, 190 deg along the y axis)

    Returns
    -------
    number
        radians (0 rad pointing along the x-axis, pi/2 rad along the y axis)
    """
    return np.deg2rad(270 - deg)


def ot_uv2deg(u, v):
    """
    Function to convert u and v component to the in LES common degree convention
    Parameters
    ----------
    u
        wind speed in x direction
    v
        wind speed in y direction

    Returns
    -------
    deg
        LES degrees (270 deg pointing along the x-axis, 190 deg along the y axis)
    """
    return 270 - np.arctan2(v, u)*180 / np.pi


def ot_abs_wind_speed(wind_speed_u, wind_speed_v):
    """
    Calculates the magnitude of the wind speed based on u and v component

    Parameters
    ----------
    wind_speed_u:
        wind speed in x direction
    wind_speed_v:
        wind speed in y direction

    Returns
    -------
    number
        absolute wind speed
    """

    return np.sqrt(wind_speed_u**2 + wind_speed_v**2)

def ot_abs2uv(wind_speed_abs, wind_dir):
    """
    Calculates the u, v components of the wind speed based on the absolute speed and wind direction

    Parameters
    ----------
    wind_speed_abs:
        absolute wind speed in x direction
    wind_dir:
        wind direction (deg)

    Returns
    -------
    np.ndarray
        u,v component of the wind in [m, 2] matrix
    """
    phi = ot_deg2rad(wind_dir)
    return np.array([np.cos(phi) * wind_speed_abs, np.sin(phi) * wind_speed_abs])


def ot_isocell(n_rp: int) -> tuple:
    """
    Isocell algorithm to discretize the rotor plane (or any circle)
     Masset et al.:
        https://orbi.uliege.be/bitstream/2268/91953/1/masset_isocell_orbi.pdf
    We choose N = 3 here, 4 or 5 are also viable options, 3 is close to optimal

    Parameters
    ----------
    n_rp : int
        desired number of Rotor points (algorithm can not work with all numbers, takes the closest one)

    Returns
    -------
    tuple:
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
        idx_s = idx_e - nS       # Start and end index TODO double check because adapted from MATLAB

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
    float:
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
    float:
        yaw angle in LES degree (clockwise)
    """
    return orientation - wind_dir

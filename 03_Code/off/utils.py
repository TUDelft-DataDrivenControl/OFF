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

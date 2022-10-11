"""
File to plot the information retrieved in "Interactive sim assembly"
"""
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools


def _plot_wind_farm(wf_dict: dict, title):
    fig, ax = plt.subplots()
    ax.scatter(wf_dict["farm"]["layout_x"],
               wf_dict["farm"]["layout_y"],
               c=np.arange(len(wf_dict["farm"]["layout_x"])),
               edgecolors='none')

    ax.set_xlabel("Easting (" + wf_dict["farm"]["unit"][0] + ")")
    ax.set_ylabel("Northing (" + wf_dict["farm"]["unit"][0] + ")")
    ax.grid(True)
    ax.set_title(title)
    ax.axis('equal')
    ax.set_xlim(wf_dict["farm"]["boundaries_xyz"][0],
                wf_dict["farm"]["boundaries_xyz"][1])
    ax.set_ylim(wf_dict["farm"]["boundaries_xyz"][2],
                wf_dict["farm"]["boundaries_xyz"][3])

    return ax


def plot_wind_farm(path_to_farms, yaml_file):
    """
    Function for the interactive sim assembly to plot the selected wind farm

    Parameters
    ----------
    path_to_farms : String
        path to wind farms database
    yaml_file : String
        describes the path to the .yaml file with the wind farm layout

    Returns
    -------
    dict :
        The loaded dict of the wind farm
    """
    stream = open(path_to_farms + yaml_file, 'r')
    dictionary = yaml.safe_load(stream)
    title = 'Wind farm layout of the file ' + yaml_file[:-len(".yaml")]
    _plot_wind_farm(dictionary, title)
    return dictionary


def _amb_is_steady(ambient_dict) -> tuple:
    """
    Helper function to determine if the ambient conditions are steady or not

    Parameters
    ----------
    ambient_dict : dict
        ambient dictionary loaded by the function

    Returns
    -------
    Tuple :
        bool :
            True if wind speed is steady state
            False if wind speed is changing
        bool :
            True if wind direction is steady state
            False if wind direction is changing
    """
    u_steady = True
    phi_steady = True

    if len(ambient_dict['flow_field']['wind_directions_t']) > 1:
        phi_steady = False

    if len(ambient_dict['flow_field']['wind_speeds_t']) > 1:
        u_steady = False

    return u_steady, phi_steady


def plot_ambient(path_to_amb, wind_farm, yaml_file):
    """
    Function for the interactive sim assembly to plot the selected wind farm and the (changing) wind conditions

    Parameters
    ----------
    path_to_amb: String
        Path to ambient database
    wind_farm: dict
        wind farm dict (determined previously)
    yaml_file: String
        name of the ambient yaml file used

    Returns
    -------
    dict :
        ambient dict
    """
    stream = open(path_to_amb + yaml_file, 'r')
    dictionary = yaml.safe_load(stream)
    u_s, phi_s = _amb_is_steady(dictionary)

    if u_s and phi_s:
        title = "Wind farm with steady wind speed & direction"
        ax = _plot_wind_farm(wind_farm, title)
        u = dictionary['flow_field']['wind_speeds'][0]
        phi = dictionary['flow_field']['wind_directions'][0]
        phi = np.deg2rad(270 - phi + 180)  # + 180 to fix barbs
        v = np.sin(phi) * u
        u = np.cos(phi) * u
        # https://matplotlib.org/stable/plot_types/arrays/barbs.html
        ax.barbs(wind_farm["farm"]["layout_x"],
                 wind_farm["farm"]["layout_y"],
                 np.ones(len(wind_farm["farm"]["layout_x"])) * u,
                 np.ones(len(wind_farm["farm"]["layout_x"])) * v)
    elif u_s:
        # Wind speed stead, direction not
        title = "Wind farm with changing wind direction"
        ax = _plot_wind_farm(wind_farm, title)
        # TODO
    elif phi_s:
        # Wind direction steady, speed not
        title = "Wind farm with changing wind speed"
        ax = _plot_wind_farm(wind_farm, title)
        # TODO
    else:
        # Neither wind direction nor wind speed steady
        title = "Wind farm with changing wind speed and direction"
        ax = _plot_wind_farm(wind_farm, title)
        # TODO

    return dictionary


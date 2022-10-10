"""
File to plot the information retrieved in "Interactive sim assembly"
"""
import yaml
import numpy as np
import matplotlib.pyplot as plt


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

    """
    stream = open(path_to_farms + yaml_file, 'r')
    dictionary = yaml.safe_load(stream)

    for key, value in dictionary.items():
        print(key + " : " + str(value))

    x = np.linspace(0, 10, 100)

    fig = plt.figure()
    plt.plot(x, np.sin(x), '-')
    plt.plot(x, np.cos(x), '--')



from dis import disco

import casadi
import casadi as cas
import csv
import numpy as np
import matplotlib.pyplot as plt
import pathlib

def MooringSpline():
    print(pathlib.Path(__file__).parent.resolve())
    with open(str(pathlib.Path(__file__).parent.resolve()) + "/mooring_lookuptable_hori_only.csv", newline='') as csvfile:
        mooringdata = csv.reader(csvfile, delimiter=' ', quotechar='|')

        distance_array = []
        force_array = []

        for i, row in enumerate(mooringdata):
            row = row[0].split(',')
            distance_to_anchor = float(row[0])
            force = float(row[1]) # in KN??

            distance_array.append(distance_to_anchor)
            force_array.append(1e3*force)

    mooring_force_f = cas.interpolant('LUT', 'bspline', [distance_array], force_array)
    return mooring_force_f


if __name__ == '__main__':
    mooringforce_f = MooringSpline()

    distances = np.linspace(650, 950, 100)
    forces = np.zeros(distances.shape)
    for i, d in enumerate(distances):
        forces[i] = mooringforce_f(d)

    plt.figure()
    plt.plot(distances, forces)
    plt.show()

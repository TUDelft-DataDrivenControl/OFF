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

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def assign_points_to_nearest(scattered_points, grid_points):
    # Erstellen Sie einen k-dimensionalen Baum aus den verstreuten Punkten
    tree = cKDTree(scattered_points)

    # Finden Sie die Indizes der nächsten verstreuten Punkte für jeden Grid-Punkt
    _, indices = tree.query(grid_points)

    # Erstellen Sie ein Dictionary, das jeden verstreuten Punkt auf eine Liste von Grid-Punkten abbildet, die ihm am nächsten sind
    point_mapping = {}
    for point_index, grid_point in zip(indices, grid_points):
        nearest_scattered_point = tuple(scattered_points[point_index])
        if nearest_scattered_point not in point_mapping:
            point_mapping[nearest_scattered_point] = []
        point_mapping[nearest_scattered_point].append(tuple(grid_point))

    return point_mapping


# Define grid points
x_range = np.arange(0, 50, 1)
y_range = np.arange(0, 30, 1)
x_grid, y_grid = np.meshgrid(x_range, y_range)
grid_points = np.vstack((x_grid.flatten(), y_grid.flatten())).T

# Define scattered points
#scattered_points = np.array([[1, 1], [2, 2], [3, 1], [3, 3]])
scattered_points = np.random.rand(10, 2) * 50

mapping = assign_points_to_nearest(scattered_points, grid_points)

# Plot Voronoi diagram
fig, ax = plt.subplots()

# Plot grid points that belong to the same scattered point in a uniform color
for scattered_point, grid_points in mapping.items():
    x_values = [point[0] for point in grid_points]
    y_values = [point[1] for point in grid_points]
    ax.plot(x_values, y_values, 'o', label=f'Scattered Point {scattered_point}')
# Add legend
ax.legend()

# Show the plot
plt.show()

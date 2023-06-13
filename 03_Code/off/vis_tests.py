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
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Define grid points
x_range = np.arange(0, 5, 1)
y_range = np.arange(0, 3, 1)
x_grid, y_grid = np.meshgrid(x_range, y_range)
grid_points = np.vstack((x_grid.flatten(), y_grid.flatten())).T

print(grid_points)

# Define scattered points
scattered_points = np.array([[1, 1], [2, 2], [3, 1], [3, 3]])

# Combine scattered and grid points
all_points = np.vstack((scattered_points, grid_points))

# Compute Voronoi diagram
vor = Voronoi(all_points)

# Determine Voronoi regions for grid points
region_indices = vor.point_region
grid_regions = region_indices[len(scattered_points):]

# Plot Voronoi diagram
fig, ax = plt.subplots()
voronoi_plot_2d(vor, ax=ax, show_vertices=True, line_colors='gray')

# Plot grid points within their respective Voronoi regions
for i, region_index in enumerate(grid_regions):
    region_vertices = vor.vertices[vor.regions[region_index]]
    ax.fill(*zip(*region_vertices), alpha=0.4)

# Customize plot properties
ax.plot(scattered_points[:, 0], scattered_points[:, 1], 'ko')  # Plot scattered points
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Voronoi Diagram with Grid Points')

# Show the plot
plt.show()

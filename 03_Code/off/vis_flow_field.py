# Copyright (C) <2024>, M Becker (TUDelft), M Lejeune (UCLouvain)

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
import logging


class Visualizer_FlowField:
    settings: dict

    def __init__(self, settings: dict, turbine_locations: np.ndarray):
        """
        Class to visualize the flow field

        Parameters
        ----------
        settings : dict
            Dictionary containing the settings for the visualizer
        """
        self.settings = settings
        self._vis_generate_grid_points()
        self._vis_assign_points_to_nearest(turbine_locations)


    def _vis_assign_points_to_nearest(self, turbine_locations: np.ndarray):
        """
        Function to assign grid points to the nearest turbine locations

        Parameters
        ----------
        turbine_locations : np.ndarray
            Array containing the locations of the turbines
        """
        # Create a k-dimensional tree from the scattered points
        tree = cKDTree(turbine_locations)

        # Find the indices of the nearest scattered points for each grid point
        _, indices = tree.query(self.grid_points)

        # Create a dictionary that maps each scattered point to a list of grid points that are nearest to it
        self.point_mapping = {}
        for point_index, grid_point in zip(indices, self.grid_points):
            nearest_scattered_point = tuple(turbine_locations[point_index])
            if nearest_scattered_point not in self.point_mapping:
                self.point_mapping[nearest_scattered_point] = []
            self.point_mapping[nearest_scattered_point].append(tuple(grid_point))

    def vis_get_grid_points_iT(self, iT:int, turbine_locations: np.ndarray) -> np.ndarray:
        """
        Function to get the grid points for a given turbine location
        
        Parameters
        ----------
        iT : int
            Index of the turbine
        turbine_locations : np.ndarray
            Array containing the locations of the turbines

        Returns
        -------
        np.ndarray
            Grid points that belong to the given turbine
        """
        
        # Get the grid points for the given turbine
        return self.point_mapping[tuple(turbine_locations[iT])]

    def _vis_generate_grid_points(self):
        """
        Function to generate the grid points based on settings
        """

        if self.settings['grid']['unit'] == 'D':
            x_step = ((self.settings['grid']['boundaries'][0][1]*self.settings['grid']['diameter'] -
                        self.settings['grid']['boundaries'][0][0]*self.settings['grid']['diameter']) / 
                            self.settings['grid']['resolution'][0])
            y_step = ((self.settings['grid']['boundaries'][1][1]*self.settings['grid']['diameter'] -
                        self.settings['grid']['boundaries'][1][0]*self.settings['grid']['diameter']) / 
                            self.settings['grid']['resolution'][1])
            
            x_range = np.arange(
                self.settings['grid']['boundaries'][0][0]*self.settings['grid']['diameter'],
                self.settings['grid']['boundaries'][0][1]*self.settings['grid']['diameter'], x_step)
            y_range = np.arange(
                self.settings['grid']['boundaries'][1][0]*self.settings['grid']['diameter'],
                self.settings['grid']['boundaries'][1][1]*self.settings['grid']['diameter'], y_step)
        else:
            x_step = ((self.settings['grid']['boundaries'][0][1] - self.settings['grid']['boundaries'][0][0]) / 
                            self.settings['grid']['resolution'][0])
            y_step = ((self.settings['grid']['boundaries'][1][1] - self.settings['grid']['boundaries'][1][0]) / 
                            self.settings['grid']['resolution'][1])
            
            x_range = np.arange(self.settings['grid']['boundaries'][0][0], self.settings['grid']['boundaries'][0][1], x_step)
            y_range = np.arange(self.settings['grid']['boundaries'][1][0], self.settings['grid']['boundaries'][1][1], y_step)
        
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        self.grid_points = np.vstack((x_grid.flatten(), y_grid.flatten())).T

    def vis_update_grid_point_relations(self, turbine_locations: np.ndarray):
        """
        Function to update the grid point relations based on the new turbine locations

        Parameters
        ----------
        turbine_locations : np.ndarray
            Array containing the locations of the turbines
        """
        self._vis_assign_points_to_nearest(turbine_locations)

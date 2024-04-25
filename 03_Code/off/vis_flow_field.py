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
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import logging


class Visualizer_FlowField:
    settings: dict
    id_matrix: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    u_grid: np.ndarray

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
        self._vis_create_id_matrix(turbine_locations)

    def _vis_create_id_matrix(self, turbine_locations: np.ndarray):
        """
        Function to create the id matrix & generate a color map

        Parameters
        ----------
        turbine_locations : np.ndarray
            Array containing the locations of the turbines
        """
        distances = np.array([np.sqrt(
            (self.x_grid - point[0])**2 +
            (self.y_grid - point[1])**2) for point in turbine_locations])
        
        self.id_matrix = np.argmin(distances, axis=0)

        # Create a graph object where each point is a node
        G = nx.Graph()

        for i in range(len(turbine_locations)):
            G.add_node(i)

        n_neig = np.min([5, len(turbine_locations)])
        # Connect each node with its nearest neighbors to create edges
        nbrs = NearestNeighbors(n_neighbors=n_neig, algorithm='ball_tree').fit(turbine_locations)
        distances, indices = nbrs.kneighbors(turbine_locations)

        for i in range(len(indices)):
            for j in range(len(indices[i])):
                if i != indices[i][j]:
                    G.add_edge(i, indices[i][j])

        # Assign colors to the nodes using a greedy coloring algorithm
        self.color_map = nx.greedy_color(G, strategy="largest_first")
    
    def vis_get_grid_points_iT(self, iT:int) -> np.ndarray:
        """
        Function to get the grid points for a given turbine location
        
        Parameters
        ----------
        iT : int
            Index of the turbine

        Returns
        -------
        np.ndarray
            Grid points that belong to the given turbine
        """
        return np.vstack(
            (self.x_grid[self.id_matrix == iT],self.y_grid[self.id_matrix == iT])).T

    def _vis_generate_grid_points(self):
        """
        Function to generate the grid points based on settings
        """

        if self.settings['grid']['unit'][0] == 'D':
            x_step = ((self.settings['grid']['boundaries'][0][1]*self.settings['grid']['diameter'][0] -
                        self.settings['grid']['boundaries'][0][0]*self.settings['grid']['diameter'][0]) / 
                            self.settings['grid']['resolution'][0])
            y_step = ((self.settings['grid']['boundaries'][1][1]*self.settings['grid']['diameter'][0] -
                        self.settings['grid']['boundaries'][1][0]*self.settings['grid']['diameter'][0]) / 
                            self.settings['grid']['resolution'][1])
            
            x_range = np.arange(
                self.settings['grid']['boundaries'][0][0]*self.settings['grid']['diameter'][0],
                self.settings['grid']['boundaries'][0][1]*self.settings['grid']['diameter'][0], x_step)
            y_range = np.arange(
                self.settings['grid']['boundaries'][1][0]*self.settings['grid']['diameter'][0],
                self.settings['grid']['boundaries'][1][1]*self.settings['grid']['diameter'][0], y_step)
        else:
            x_step = ((self.settings['grid']['boundaries'][0][1] - self.settings['grid']['boundaries'][0][0]) / 
                            self.settings['grid']['resolution'][0])
            y_step = ((self.settings['grid']['boundaries'][1][1] - self.settings['grid']['boundaries'][1][0]) / 
                            self.settings['grid']['resolution'][1])
            
            x_range = np.arange(self.settings['grid']['boundaries'][0][0], self.settings['grid']['boundaries'][0][1], x_step)
            y_range = np.arange(self.settings['grid']['boundaries'][1][0], self.settings['grid']['boundaries'][1][1], y_step)
        
        self.x_grid, self.y_grid = np.meshgrid(x_range, y_range)
        self.u_grid = np.zeros_like(self.x_grid)
        #self.grid_points = np.vstack((x_grid.flatten(), y_grid.flatten())).T

    def vis_update_grid_point_relations(self, turbine_locations: np.ndarray):
        """
        Function to update the grid point relations based on the new turbine locations

        Parameters
        ----------
        turbine_locations : np.ndarray
            Array containing the locations of the turbines
        """
        self._vis_create_id_matrix(turbine_locations)

    def vis_store_u_values(self, u_values: np.ndarray, iT:int):
        """
        Function to store the u values for a given turbine

        Parameters
        ----------
        u_values : np.ndarray
            Array containing the u values
        iT : int
            Index of the turbine
        """
        self.u_grid[self.id_matrix == iT] = u_values

    def vis_save_flow_field(self, path: str):
        """
        Function to save the flow field to a file

        Parameters
        ----------
        path : str
            Path to the file
        """
        if self.settings["debug"]["turbine_effective_wind_speed_store_data"]:
            np.savetxt(path + '.csv', np.column_stack(
                (self.x_grid.flatten(), self.y_grid.flatten(), self.u_grid.flatten())),
                  delimiter=',')
        
        if self.settings["debug"]["turbine_effective_wind_speed_plot"]:
            fig1, ax1 = plt.subplots(layout='constrained')
            for iT, color in self.color_map.items():
                u_values = np.zeros(self.u_grid.shape)
                u_values[self.id_matrix == iT] = self.u_grid[self.id_matrix == iT]
                u_values[self.id_matrix != iT] = np.nan
                CS = ax1.contourf(self.x_grid, self.y_grid, u_values, 10, cmap=plt.cm.bone)
            
            cbar = fig1.colorbar(CS)
            cbar.ax.set_ylabel('Velocity (m/s)')
            ax1.set_aspect('equal', 'box')
            ax1.set_title('Tiled view of the simulation', fontsize=10)
            ax1.set_xlabel('Easting (m)')
            ax1.set_ylabel('Northing (m)')
            #store the plot
            fig1.savefig(path + '.png')
            fig1.clf()

        if (self.settings["debug"]["turbine_effective_wind_speed_plot"] and 
            self.settings["debug"]["effective_wf_tile_5color"]):
            fig1, ax1 = plt.subplots(layout='constrained')
            
            for iT, color in self.color_map.items():
                u_values = np.zeros(self.u_grid.shape)
                u_values[self.id_matrix == iT] = self.u_grid[self.id_matrix == iT]
                u_values[self.id_matrix != iT] = np.nan

                if color == 0:
                    ax1.contourf(self.x_grid, self.y_grid, u_values, 10, cmap=plt.cm.Greys_r)
                elif color == 1:
                    ax1.contourf(self.x_grid, self.y_grid, u_values, 10, cmap=plt.cm.Blues_r)
                elif color == 2:
                    ax1.contourf(self.x_grid, self.y_grid, u_values, 10, cmap=plt.cm.Greens_r)
                elif color == 3:
                    ax1.contourf(self.x_grid, self.y_grid, u_values, 10, cmap=plt.cm.Oranges_r)
                else:
                    ax1.contourf(self.x_grid, self.y_grid, u_values, 10, cmap=plt.cm.Purples_r)

            ax1.set_aspect('equal', 'box')
            ax1.set_title('Colored tiled view of the simulation', fontsize=10)
            ax1.set_xlabel('Easting (m)')
            ax1.set_ylabel('Northing (m)')
            #store the plot
            fig1.savefig(path + '_landscape.png')
            fig1.clf()
    

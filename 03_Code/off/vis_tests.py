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

import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Generiere eine Menge von zufälligen Punkten mit x, y-Koordinaten
points = np.random.rand(50, 2)

# Erstelle ein Graphenobjekt, wobei jeder Punkt ein Knoten ist
G = nx.Graph()

for i in range(len(points)):
    G.add_node(i)

# Verbinde jeden Knoten mit seinen nächsten Nachbarn, um Kanten zu erstellen
nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(points)
distances, indices = nbrs.kneighbors(points)

for i in range(len(indices)):
    for j in range(len(indices[i])):
        if i != indices[i][j]:
            G.add_edge(i, indices[i][j])

# Verwende einen Graphenfärbungsalgorithmus, um jedem Knoten eine von vier Farben zuzuweisen
color_map = nx.greedy_color(G, strategy="largest_first")

# Drucke die Farbzuordnung für jeden Punkt
for node, color in color_map.items():
    print(f'Point {node} is assigned color {color}')


# Verwende einen Graphenfärbungsalgorithmus, um jedem Knoten eine von vier Farben zuzuweisen
color_map = nx.greedy_color(G, strategy="largest_first")

# Erstelle eine Liste von Farben für jeden Knoten
colors = ['red', 'green', 'blue', 'yellow']
node_colors = [colors[color_map[node]] for node in G.nodes()]

# Zeichne den Graphen
nx.draw(G, pos=points, node_color=node_colors, with_labels=True)

# Zeige das Diagramm
plt.show()
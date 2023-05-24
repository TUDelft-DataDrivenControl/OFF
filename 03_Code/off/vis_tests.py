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

import matplotlib.pyplot as plt

# Define the x and y coordinates of the dots
x = [1, 2, 3, 4]
y = [5, 6, 7, 8]

# Define the x and y coordinates of the arrows
dx = [1, 2, -1, -2]
dy = [1, -1, 2, -2]

# Create a scatter plot of the dots
plt.scatter(x, y)

# Add arrows pointing from one dot to another
for i in range(len(x)):
    plt.arrow(x[i], y[i], dx[i], dy[i], length_includes_head=True, head_width=0.2)

# Show the plot
plt.show()
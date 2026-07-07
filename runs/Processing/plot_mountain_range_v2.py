# Import the necessary libraries


# This script is used to plot the mountain range data from a specific run



import numpy as np
import matplotlib.pyplot as plt

# Path to data
path_to_data = '/Users/marcusbecker/Documents/01_Research/01_FLORIDyn/06_FLORIDynCollab/OFF/runs/off_run_2026_07_06-14_32_01.225390'

# Load the csv data
data_x = np.loadtxt(f'{path_to_data}/mountain_plot_x_000656s.csv', delimiter=',')
data_y = np.loadtxt(f'{path_to_data}/mountain_plot_y_000656s.csv', delimiter=',')
data_u = np.loadtxt(f'{path_to_data}/mountain_plot_u_000656s.csv', delimiter=',')
data_v = np.loadtxt(f'{path_to_data}/mountain_plot_v_000656s.csv', delimiter=',')
# find the largest value in the data_u and data_v arrays
max_u = np.max(data_u[:, 1:])
max_v = np.max(data_v[:, 1:])


def line_inerpolation(x, y, u, v, X, Y) -> np.ndarray:
    """
    Interpolates the wind speed data onto a regular grid.

    Parameters:
    - x: n_particles x n_values array of x-coordinates
    - y: n_particles x n_values array of y-coordinates
    - u: n_particles x n_values array of u-component of wind speed
    - v: n_particles x n_values array of v-component of wind speed
    - X: 2D np.ndarray of x-coordinates for interpolation grid
    - Y: 2D np.ndarray of y-coordinates for interpolation grid

    Returns:
    - U: 2D np.ndarray of interpolated u-component of wind speed
    - V: 2D np.ndarray of interpolated v-component of wind speed
    """

    # Create line segments which connect the points in the x and y arrays
    segments = np.array([np.column_stack((x[i, :], y[i, :])) for i in range(x.shape[0])])

    # Calculate the distance from each grid point to each line segment
    dist = np.array([[np.min(np.linalg.norm(np.cross(seg[1] - seg[0], seg[0] - np.array([X[j, k], Y[j, k]])) / np.linalg.norm(seg[1] - seg[0]))) for k in range(X.shape[1])] for j in range(X.shape[0]) for seg in segments])

    dist = dist.reshape(X.shape[0], X.shape[1], segments.shape[0])

    # Find the index of the closest line segment for each grid point
    closest_segment_idx = np.argmin(dist, axis=2)

    # Interpolate the u and v values for each grid point based on the closest line segment
    U = np.array([[u[closest_segment_idx[j, k], np.argmin(np.linalg.norm(np.column_stack((x[closest_segment_idx[j, k], :], y[closest_segment_idx[j, k], :])) - np.array([X[j, k], Y[j, k]]), axis=1))] for k in range(X.shape[1])] for j in range(X.shape[0])])
    V = np.array([[v[closest_segment_idx[j, k], np.argmin(np.linalg.norm(np.column_stack((x[closest_segment_idx[j, k], :], y[closest_segment_idx[j, k], :])) - np.array([X[j, k], Y[j, k]]), axis=1))] for k in range(X.shape[1])] for j in range(X.shape[0])])

    return U, V


def plot_interpolated_mountain_range(ax, data_x, data_y, data_u, data_v, x_grid, y_grid):
    """
    Plots the interpolated mountain range data.

    Parameters:
    - ax: Matplotlib axis object to plot on.
    - data_x: 2D array of x-coordinates of the mountain range.
    - data_y: 2D array of y-coordinates of the mountain range.
    - data_u: 2D array of u-component of wind speed.
    - data_v: 2D array of v-component of wind speed.
    - x_grid: 1D array of x-coordinates for interpolation grid.
    - y_grid: 1D array of y-coordinates for interpolation grid.
    """
    
    n_turbines = np.unique(data_x[:, 0])

    # Create a meshgrid for interpolation
    X, Y = np.meshgrid(x_grid, y_grid)
    # Create nan arrays for U and V to hold the interpolated values
    U = np.full(X.shape, np.nan)
    V = np.full(X.shape, np.nan)
    
    # Go through each turbine 
    for i_t in n_turbines:
        # Filter data for the current turbine
        mask = data_x[:, 0] == i_t
        x_turbine = data_x[mask][:, 1:]
        y_turbine = data_y[mask][:, 1:]
        u_turbine = data_u[mask][:, 1:]
        v_turbine = data_v[mask][:, 1:]

        # Interpolate the wind speed data onto the grid
        U_turbine, V_turbine = line_inerpolation(x_turbine, y_turbine, u_turbine, v_turbine, X, Y)

        # Assign the interpolated values to the overall U and V arrays if they are smaller / not nan
        U = np.where(np.isnan(U) | (np.abs(U_turbine) < np.abs(U)), U_turbine, U)
        V = np.where(np.isnan(V) | (np.abs(V_turbine) < np.abs(V)), V_turbine, V)

    # Plot the interpolated wind speed magnitude
    speed_magnitude = np.sqrt(U**2 + V**2)
    contour = ax.contourf(X, Y, speed_magnitude, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Wind Speed (m/s)')
    ax.set_aspect('equal')

    

fig, ax = plt.subplots()

plot_interpolated_mountain_range(ax, data_x, data_y, data_u, data_v, 
                                 np.linspace(np.min(data_x), np.max(data_x), 100), 
                                 np.linspace(np.min(data_y), np.max(data_y), 100))

ax.plot(data_x[:,(data_x.shape[1]-1)//2],
                data_y[:,(data_x.shape[1]-1)//2], 'o', markersize=1, color="#FFFFFF")

plt.title('Interpolated Wind Speed')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.savefig(path_to_data + "/interpolated_mountain_plot.png")
plt.show()
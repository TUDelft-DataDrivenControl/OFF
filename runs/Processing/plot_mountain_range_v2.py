# Import the necessary libraries


# This script is used to plot the mountain range data from a specific run



import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

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


def line_interpolation(x, y, u, v, X, Y) -> np.ndarray:
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

    U = np.full(X.shape, np.nan)
    V = np.full(X.shape, np.nan)

    # Create a mask to find out if the X,Y grid point is part of the convex hull of the data points
    points  = np.column_stack((x.flatten(), y.flatten()))
    tri     = Delaunay(points)
    mask    = tri.find_simplex(np.column_stack((X.flatten(), Y.flatten()))) >= 0
    mask    = mask.reshape(X.shape)

    # Build segments from the data points based on starting point and vector to the next point
    segments = np.array([])
    for i in range(x.shape[0] - 1):
        for j in range(x.shape[1]):
            if segments.size == 0:
                segments = np.array([x[i, j], y[i, j], x[i + 1, j] - x[i, j], y[i + 1, j] - y[i, j], u[i, j], v[i, j], u[i + 1, j], v[i + 1, j]])
            else:
                segments = np.vstack([segments, [x[i, j], y[i, j], x[i + 1, j] - x[i, j], y[i + 1, j] - y[i, j], u[i, j], v[i, j], u[i + 1, j], v[i + 1, j]]])

    for j in range(X.shape[0]):
        for k in range(X.shape[1]):
            if mask[j, k]:
                # Calculate the downstream, cross-stream distance from the grid point to each segment
                downstream_distance         = np.zeros(segments.shape[0])
                cross_stream_distance       = np.zeros(segments.shape[0])
                percentage_along_segment    = np.zeros(segments.shape[0])
                angle_to_segment            = np.zeros(segments.shape[0])

                for i in range(segments.shape[0]):
                    # Vector from segment start to grid point
                    dx = X[j, k] - segments[i, 0]
                    dy = Y[j, k] - segments[i, 1]

                    # Segment vector
                    seg_dx = segments[i, 2]
                    seg_dy = segments[i, 3]

                    # Calculate the projection of the point onto the segment by dot product
                    seg_length_squared = seg_dx**2 + seg_dy**2
                    t = (dx * seg_dx + dy * seg_dy) / seg_length_squared

                    # Calculate angle between segment and point vector
                    angle = np.arctan2(dy, dx) - np.arctan2(seg_dy, seg_dx)
                    angle = (angle + np.pi) % (2 *np.pi) - np.pi  # Normalize angle to [-pi, pi]
                    angle_to_segment[i] = angle
                    # 

                    if t < 0:
                        # the grid point is before the segment start -> ignore
                        downstream_distance[i] = np.nan
                        cross_stream_distance[i] = np.nan
                        continue
                    elif t > 1:
                        # the grid point is after the segment end -> ignore
                        downstream_distance[i] = np.nan
                        cross_stream_distance[i] = np.nan
                        continue

                    percentage_along_segment[i] = t

                    # Closest point on the segment
                    closest_x = segments[i, 0] + t * seg_dx
                    closest_y = segments[i, 1] + t * seg_dy
                    
                    # This needs to be the distance from the grid point to the closest point on the segment, projected onto the segment direction for downstream and perpendicular for cross-stream
                    dx = X[j, k] - closest_x
                    dy = Y[j, k] - closest_y
                    
                    downstream_distance[i]      = np.cos(angle) * np.sqrt(dx**2 + dy**2)
                    cross_stream_distance[i]    = np.sin(angle) * np.sqrt(dx**2 + dy**2)

                # Find the index of the two closest cross-stream segments
                closest_segment_idxs = np.argsort(np.abs(cross_stream_distance))[0:2]

                frac_closest_seg = np.abs(cross_stream_distance[closest_segment_idxs[0]]) / (np.abs(cross_stream_distance[closest_segment_idxs[0]]) + np.abs(cross_stream_distance[closest_segment_idxs[1]]))

                # same blending as for V: linear along each segment, then cross-stream blend
                denom = np.abs(cross_stream_distance[closest_segment_idxs[0]]) + np.abs(cross_stream_distance[closest_segment_idxs[1]])
                if denom == 0:
                    frac_closest_seg = 0.5  # exactly centered between both segments
                else:
                    frac_closest_seg = np.abs(cross_stream_distance[closest_segment_idxs[0]]) / denom

                U[j, k] = (1 - frac_closest_seg) * (
                    segments[closest_segment_idxs[0], 4]
                    + percentage_along_segment[closest_segment_idxs[0]]
                    * (segments[closest_segment_idxs[0], 6] - segments[closest_segment_idxs[0], 4])
                ) + frac_closest_seg * (
                    segments[closest_segment_idxs[1], 4]
                    + percentage_along_segment[closest_segment_idxs[1]]
                    * (segments[closest_segment_idxs[1], 6] - segments[closest_segment_idxs[1], 4])
                )
                V[j,k] = (1 - frac_closest_seg) * (
                    segments[closest_segment_idxs[0], 5] 
                    + percentage_along_segment[closest_segment_idxs[0]] 
                    * (segments[closest_segment_idxs[0], 7] - segments[closest_segment_idxs[0], 5])
                ) + (frac_closest_seg) * (
                    segments[closest_segment_idxs[1], 5] 
                    + percentage_along_segment[closest_segment_idxs[1]] 
                    * (segments[closest_segment_idxs[1], 7] - segments[closest_segment_idxs[1], 5]))

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
        U_turbine, V_turbine = line_interpolation(x_turbine, y_turbine, u_turbine, v_turbine, X, Y)

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
                                 np.linspace(np.min(data_y), np.max(data_y), 20))

ax.plot(data_x[:,(data_x.shape[1]-1)//2],
                data_y[:,(data_x.shape[1]-1)//2], 'o', markersize=1, color="#FFFFFF")

plt.title('Interpolated Wind Speed')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.savefig(path_to_data + "/interpolated_mountain_plot.png")
plt.show()
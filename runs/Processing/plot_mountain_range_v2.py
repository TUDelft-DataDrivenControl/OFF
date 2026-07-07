# Import the necessary libraries


# This script is used to plot the mountain range data from a specific run



import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

D = 200.0
res_grid = D/10.0

# Path to data
path_to_data = '/Users/marcusbecker/Documents/01_Research/01_FLORIDyn/06_FLORIDynCollab/OFF/runs/off_run_2026_07_07-10_42_12.661434'

# Find unique mountain_plot_x_ files in the directory
import os
files = os.listdir(path_to_data)
mountain_plot_x_files = sorted([f for f in files if f.startswith('mountain_plot_x_') and f.endswith('.csv')])


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
    points = np.column_stack((x.flatten(), y.flatten()))
    tri = Delaunay(points)
    mask = tri.find_simplex(np.column_stack((X.flatten(), Y.flatten()))) >= 0

    # Build segment arrays in one shot: each segment connects row i to i+1 (same column)
    x0 = x[:-1, :].ravel()
    y0 = y[:-1, :].ravel()
    seg_dx = (x[1:, :] - x[:-1, :]).ravel()
    seg_dy = (y[1:, :] - y[:-1, :]).ravel()
    u0 = u[:-1, :].ravel()
    v0 = v[:-1, :].ravel()
    du = (u[1:, :] - u[:-1, :]).ravel()
    dv = (v[1:, :] - v[:-1, :]).ravel()

    seg_len_sq = seg_dx**2 + seg_dy**2
    seg_angle = np.arctan2(seg_dy, seg_dx)
    valid_seg = seg_len_sq > 0.0

    # Interpolate only inside convex hull; process points in chunks for low memory overhead.
    Xf = X.ravel()
    Yf = Y.ravel()
    U_flat = U.ravel()
    V_flat = V.ravel()
    grid_idx = np.flatnonzero(mask)
    if grid_idx.size == 0:
        return U, V

    batch_size = 256
    for i0 in range(0, grid_idx.size, batch_size):
        idx = grid_idx[i0:i0 + batch_size]
        xp = Xf[idx]
        yp = Yf[idx]

        dx = xp[:, None] - x0[None, :]
        dy = yp[:, None] - y0[None, :]

        with np.errstate(divide='ignore', invalid='ignore'):
            t = (dx * seg_dx[None, :] + dy * seg_dy[None, :]) / seg_len_sq[None, :]

        valid = valid_seg[None, :] & (t >= 0.0) & (t <= 1.0)

        closest_x = x0[None, :] + t * seg_dx[None, :]
        closest_y = y0[None, :] + t * seg_dy[None, :]
        rx = xp[:, None] - closest_x
        ry = yp[:, None] - closest_y
        dist = np.sqrt(rx**2 + ry**2)

        angle = np.arctan2(dy, dx) - seg_angle[None, :]
        angle = (angle + np.pi) % (2 * np.pi) - np.pi

        cross = np.where(valid, np.sin(angle) * dist, np.nan)
        abs_cross = np.abs(cross)
        abs_cross = np.where(np.isfinite(abs_cross), abs_cross, np.inf)

        # Need at least two valid segments for cross-stream blending.
        valid_counts = np.sum(np.isfinite(cross), axis=1)
        can_interp = valid_counts >= 2
        if not np.any(can_interp):
            continue

        t_sel = t[can_interp]
        abs_cross_sel = abs_cross[can_interp]
        pair_idx = np.argpartition(abs_cross_sel, kth=1, axis=1)[:, :2]

        rows = np.arange(pair_idx.shape[0])
        s0 = pair_idx[:, 0]
        s1 = pair_idx[:, 1]

        c0 = abs_cross_sel[rows, s0]
        c1 = abs_cross_sel[rows, s1]
        denom = c0 + c1
        frac = np.where(denom == 0.0, 0.5, c0 / denom)

        t0 = t_sel[rows, s0]
        t1 = t_sel[rows, s1]

        u_interp0 = u0[s0] + t0 * du[s0]
        u_interp1 = u0[s1] + t1 * du[s1]
        v_interp0 = v0[s0] + t0 * dv[s0]
        v_interp1 = v0[s1] + t1 * dv[s1]

        out_u = (1.0 - frac) * u_interp0 + frac * u_interp1
        out_v = (1.0 - frac) * v_interp0 + frac * v_interp1

        idx_out = idx[can_interp]
        U_flat[idx_out] = out_u
        V_flat[idx_out] = out_v

    return U, V


def plot_interpolated_mountain_range(ax, data_x, data_y, data_u, data_v, x_grid, y_grid, clims=None):
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
    - clims: Tuple of (vmin, vmax) for color limits.
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
    if clims is not None:
        vmin, vmax = clims
        levels = np.linspace(vmin, vmax, 50)
        contour = ax.contourf(
            X,
            Y,
            speed_magnitude,
            levels=levels,
            cmap='Oranges',
            vmin=vmin,
            vmax=vmax,
            extend='both',
        )
    else:
        contour = ax.contourf(X, Y, speed_magnitude, levels=50, cmap='Oranges')

    cbr = plt.colorbar(contour, ax=ax, label='Wind Speed (m/s)')
    if clims is not None:
        cbr.mappable.set_clim(vmin, vmax)
        cbr.set_ticks(np.arange(vmin, vmax + 1, 2))
    ax.set_aspect('equal')


for i, file in enumerate(mountain_plot_x_files):
    # Load the csv data
    data_x = np.loadtxt(f'{path_to_data}/{file}', delimiter=',')
    data_y = np.loadtxt(f'{path_to_data}/mountain_plot_y_{file.split("_")[-1]}', delimiter=',')
    data_u = np.loadtxt(f'{path_to_data}/mountain_plot_u_{file.split("_")[-1]}', delimiter=',')
    data_v = np.loadtxt(f'{path_to_data}/mountain_plot_v_{file.split("_")[-1]}', delimiter=',')

    fig, ax = plt.subplots(1, 1, figsize=(12, 3), sharex=True)
    plot_interpolated_mountain_range(ax, data_x, data_y, data_u, data_v, 
                                     np.arange(np.min(data_x), np.max(data_x), res_grid), 
                                     np.arange(np.min(data_y), np.max(data_y), res_grid),
                                     clims=(2, 8))
    plt.title(f'Interpolated Wind Speed at {file.split("_")[-1].replace(".csv", "")}')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.tight_layout()
    plt.savefig(path_to_data + f"/interpolated_mountain_plot_{file.split('_')[-1].replace('.csv', '')}.png")
    plt.show()

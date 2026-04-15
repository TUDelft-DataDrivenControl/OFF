# Import the necessary libraries


# This script is used to plot the mountain range data from a specific run



import numpy as np
import matplotlib.pyplot as plt

# Path to data
path_to_data = 'OFF/runs/off_run_20260128104823422709'
amplification_factor = 10.0 #m/(m/s)
# TODO
# change storage type into something like a pickle file? Or a hdf5 file?


# Load the csv data
data_x = np.loadtxt(f'{path_to_data}/mountain_plot_x_000736s.csv', delimiter=',')
data_y = np.loadtxt(f'{path_to_data}/mountain_plot_y_000736s.csv', delimiter=',')
data_u = np.loadtxt(f'{path_to_data}/mountain_plot_u_000736s.csv', delimiter=',')
data_v = np.loadtxt(f'{path_to_data}/mountain_plot_v_000736s.csv', delimiter=',')
# find the largest value in the data_u and data_v arrays
max_u = np.max(data_u)
max_v = np.max(data_v)

t = 100

fig, ax = plt.subplots()
for i in range(0,data_x.shape[0]):
    #ax.plot(data_x[i, :] + data_u[i, :] * amplification_factor, data_y[i, :] + data_v[i, :] * amplification_factor, color='blue', alpha=0.5)
    ax.fill(np.hstack((data_x[i, :] + (max_u - data_u[i, :]) * amplification_factor, data_x[i, ::-1])),
            np.hstack((data_y[i, :] + (max_v - data_v[i, :]) * amplification_factor, data_y[i, ::-1])), 
            color='blue', alpha=0.5, edgecolor='none')


ax.set_aspect('equal')
ax.set_title('Wind speed along OPs')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
#ax.set_xlim(self.settings_vis["grid"]["boundaries"][0][0] * scale_grid, self.settings_vis["grid"]["boundaries"][0][1] * scale_grid)
#ax.set_ylim(self.settings_vis["grid"]["boundaries"][1][0] * scale_grid, self.settings_vis["grid"]["boundaries"][1][1] * scale_grid)
plt.savefig(path_to_data + "/mountain_plot_at_" + str(int(t)).zfill(6) + "s.png")
plt.show()

print("Plot done!")


#fig, ax = plt.subplots()

#ax.plot(data_x.flatten(), data_y.flatten(), 'o', markersize=2, color='lightgrey')
#ax.tricontourf(data_x.flatten(), data_y.flatten(), np.sqrt(data_u.flatten()**2 + data_v.flatten()**2), levels=np.linspace(0, 9, 10))
#ax.set_aspect('equal')

#ax.set(xlim=(-3, 3), ylim=(-3, 3))

#plt.show()
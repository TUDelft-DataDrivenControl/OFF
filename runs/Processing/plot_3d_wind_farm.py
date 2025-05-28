import numpy as np
import matplotlib.pyplot as plt





#HKN
wf_x = [1000.0, 1293.3, 1900.4, 1586.6, 2801.7, 1879.9, 3870.6, 2173.2, 3749.4, 3688.8]
wf_y = [1000.0, 1743.5, 1082.0, 2487.0, 1164.9, 3230.5, 1645.4, 3974.0, 2747.3, 3792.2]
orientation = np.array([270.0, 225.0, 270.0, 225.0, 270.0, 225.0, 270.0, 225.0, 270.0, 225.0])
D   = 178.4
Hh  = 119
phi = np.linspace(0, 2*np.pi)


orientation = np.deg2rad(270.0 - orientation)



# Generate a 3D plot of a wind farm layout
ax = plt.figure().add_subplot(projection='3d')
for iT in np.arange(len(wf_x)):
    # Plot turbine blades
    azimuth = np.arange(3)*2*np.pi/3 + np.deg2rad(np.random.uniform(0, 360, 1))
    for iA in azimuth:
        blade = np.array([0, 0, D/2])
        R_az = np.array(((1, 0, 0),(0, np.cos(iA), -np.sin(iA)), (0, np.sin(iA), np.cos(iA))))
        R_yaw = np.array(((np.cos(orientation[iT]), -np.sin(orientation[iT]), 0),
                       (np.sin(orientation[iT]), np.cos(orientation[iT]), 0), (0, 0, 1)))
        blade = R_yaw @ R_az @ blade
        ax.plot([wf_x[iT], wf_x[iT] + blade[0]], 
                [wf_y[iT], wf_y[iT] + blade[1]], 
                [Hh, Hh + blade[2]], color='k', linewidth=.5)

    # Plot the rotor outline
    ax.plot(wf_x[iT] - D*0.5*np.cos(phi) * np.sin(orientation[iT]),
            wf_y[iT] + D*0.5*np.cos(phi) * np.cos(orientation[iT]), 
            Hh + D*0.5*np.sin(phi), color='k', linewidth=.5)
    
    # Plot the tower
    ax.plot(wf_x[iT]*np.ones(2),
            wf_y[iT]*np.ones(2),
            [0, Hh], color='k', linewidth=.5)
        
ax.set_proj_type('persp', focal_length=0.2)
ax.set_aspect('equal')
plt.savefig('wind_farm_3d_layout.svg', format='svg', dpi=300)
plt.show()






# Relevant Links
# https://matplotlib.org/stable/gallery/mplot3d/polys3d.html

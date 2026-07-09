import os
import numpy as np
import pandas as pd
import xarray as xr
import off.off as off
import off.off_interface as offi
from pathlib import Path

from gpytorch_gp_model import GPyTorchGPSurrogate, HeteroscedasticGPSurrogate, predict_loads_sector_average
import matplotlib.pyplot as plt

# import optimizer
from msr import MSR_optimizer
from msr import ObjFuncComponent
import time

OFF_PATH = Path(off.OFF_PATH)

# ============================================================================
# Configure and load GPyTorch surrogate model

OUTPUT_NAME = ['RootMflp']  # Change to 'TwrBsMyt' or other sensors
USE_HETEROSCEDASTIC = True  # Set to True to use heteroscedastic model

if USE_HETEROSCEDASTIC:
    print("Loading pre-trained HETEROSCEDASTIC GPyTorch surrogate")
    model_path = f'C:\\Users\\daanvanderhoek\\Nextcloud\\PostDoc\\Software\\LoadSurrogate\\Surrogate_models\\heteroscedastic_{OUTPUT_NAME[0]}.h5'
    surrogate = HeteroscedasticGPSurrogate.load_h5(model_path)
    model_type = "Heteroscedastic (Input-Dependent Uncertainty)"
else:
    print("Loading pre-trained HOMOSCEDASTIC GPyTorch surrogate")
    model_path = f'C:\\Users\\daanvanderhoek\\Nextcloud\\PostDoc\\Software\\LoadSurrogate\\Surrogate_models\\gpytorch_surrogate_{OUTPUT_NAME[0]}.h5'
    surrogate = GPyTorchGPSurrogate.load_h5(model_path)
    model_type = "Homoscedastic (Constant Uncertainty)"

# ============================================================================
# Simulation configuration — adjust as needed
run_dir      = '..\\runs\\my_del_assessment'  # user-defined output directory
case_file    = OFF_PATH / '02_Examples_and_Cases' / '02_Example_Cases' / '001_two_turbines_yaw_step_pywake.yaml'

wind_direction  = 272.0   # deg — incoming wind direction
yaw_offset_T1   = 0.0    # deg — yaw misalignment of T1 before the step
yaw_rate        = 0.3     # deg/s — physical yaw drive rate
yaw_step_time   = 600.0   # s   — time at which the yaw step starts

# Read data settings
ws_cols  = ['WS_sec_up_PyWake', 'WS_sec_right_PyWake', 'WS_sec_down_PyWake', 'WS_sec_left_PyWake']
ti_cols  = ['TI_sec_up_PyWake',  'TI_sec_right_PyWake',  'TI_sec_down_PyWake',  'TI_sec_left_PyWake']
P_cols   = ['Power_PyWake']
sectors  = ['up', 'right', 'down', 'left']

# Run simulation and write outputs directly into run_dir
os.makedirs(run_dir, exist_ok=True)
oi = offi.OFFInterface()
oi.init_simulation_by_path(case_file)

# Compute reference power and DELs for normalization
oi.create_off_simulation()
oi.run_sim()
oi.store_measurements(os.path.join(run_dir, 'measurements.csv'))
oi.store_applied_control(os.path.join(run_dir, 'applied_control.csv'))

measurements_bl = oi.get_measurements()
control_bl      = oi.get_applied_control()

df_meas  = pd.read_csv(os.path.join(run_dir, 'measurements.csv'), index_col=0)
# Read power and sector-averaged WS and TI from measurements.csv
# n_wt = int(df_meas['t_idx'].nunique())
# n_time = int(len(df_meas) / n_wt)

n_wt = int(measurements_bl['t_idx'].nunique())
n_time = int(len(measurements_bl) / n_wt)

# sector_data = np.zeros((n_wt, n_time, len(sectors), 2))  # (wt, time, sector, quantity=[WS, TI])
power_data = np.zeros((n_wt, n_time))  # (wt, time)
for wt_i in range(n_wt):
    # wt_df = df_meas[df_meas['t_idx'] == wt_i]
    wt_df = measurements_bl[measurements_bl['t_idx'] == wt_i]
    # sector_data[wt_i, :, :, 0] = wt_df[ws_cols].values
    # sector_data[wt_i, :, :, 1] = wt_df[ti_cols].values
    power_data[wt_i, :] = wt_df[P_cols].values.squeeze()
# time_coords = df_meas[df_meas['t_idx'] == 0]['time'].values
time_coords = measurements_bl[measurements_bl['t_idx'] == 0]['time'].values
time_step = 4
turb_power_ref = np.sum(power_data * time_step, axis=1)  # cumulative power for each turbine
farm_power_ref = np.sum(turb_power_ref) / 1e3 / 3600           # total farm energy in MWh

print(f"Reference farm power: {farm_power_ref:.1f} kWh")

# define objective functions
def calculate_power_OFF(oi, gamma, wPower, refPower):
    # Modify controller: T1 ramps from yaw_offset_T1 to 0 at yaw_rate starting at yaw_step_time.
    # np.interp linearly interpolates between prescribed points, so the ramp duration
    # encodes the physical rate — there is no separate rate-limiter in this controller.
    #   orientation_deg rows: [[T1_orient, T2_orient], ...]  — absolute turbine heading in deg
    gamma_T1 = float(gamma[0])
    gamma_T2 = float(gamma[1])
    delta_t = max(abs(gamma_T1), abs(gamma_T2)) / yaw_rate
    oi.init_simulation_by_path(case_file)
    oi.settings_ctr['orientation_deg'] = [
        [wind_direction,           wind_direction          ],
        [wind_direction,           wind_direction          ],
        [wind_direction + gamma_T1, wind_direction + gamma_T2],
        [wind_direction + gamma_T1, wind_direction + gamma_T2],
    ]
    oi.settings_ctr['orientation_t'] = [0.0, yaw_step_time, yaw_step_time + delta_t, 90000.0]

    oi.init_simulation_by_dicts(settings_ctr=oi.settings_ctr)
    oi.create_off_simulation()
    oi.run_sim()
    # oi.store_measurements(os.path.join(run_dir, 'measurements.csv'))
    # oi.store_applied_control(os.path.join(run_dir, 'applied_control.csv'))

    measurements_opt = oi.get_measurements()
    control_opt      = oi.get_applied_control()
    # ============================================================================
    # Read power and sector-averaged WS and TI from measurements.csv
    df_meas  = pd.read_csv(os.path.join(run_dir, 'measurements.csv'), index_col=0)
    n_wt = int(measurements_opt['t_idx'].nunique())
    n_time = int(len(measurements_opt) / n_wt)
   
    sector_data = np.zeros((n_wt, n_time, len(sectors), 2))  # (wt, time, sector, quantity=[WS, TI])
    power_data = np.zeros((n_wt, n_time))  # (wt, time)
    for wt_i in range(n_wt):
        wt_df = measurements_opt[measurements_opt['t_idx'] == wt_i]
        sector_data[wt_i, :, :, 0] = wt_df[ws_cols].values
        sector_data[wt_i, :, :, 1] = wt_df[ti_cols].values
        power_data[wt_i, :] = wt_df[P_cols].values.squeeze()
    time_coords = measurements_opt[measurements_opt['t_idx'] == 0]['time'].values
    sector_avg = xr.DataArray(
        sector_data,
        dims=['wt', 'time', 'sector', 'quantity'],
        coords={'wt': list(range(n_wt)), 'time': time_coords, 'sector': sectors}
    ).expand_dims({'wd': 1, 'ws': 1})

    # Read yaw angles from applied_control.csv
    # df_ctrl = pd.read_csv(os.path.join(run_dir, 'applied_control.csv'), index_col=0)

    yaw_angles = np.zeros((n_wt, n_time))
    for wt_i in range(n_wt):
        yaw_angles[wt_i, :] = control_opt[control_opt['t_idx'] == wt_i]['yaw'].values
    helix_amplitudes = np.zeros((n_wt, n_time))

    turb_power_opt = np.sum(power_data * time_step, axis=1)  # cumulative power for each turbine
    farm_power_opt = np.sum(turb_power_opt) / 1e3 / 3600           # total farm energy in MWh
    print(f"Farm power for yaw misalignment {gamma}: {farm_power_opt:.1f} kWh")

    # Place holder for load surrogate
    # # Predict loads with uncertainty quantification
    # loads_mean_xr, loads_std_xr = predict_loads_sector_average(
    #     surrogates_dict, 
    #     sector_avg, 
    #     yaw_angles, 
    #     helix_amp,
    #     return_std=True
    # )
    # Extract values for both turbines
    # turb_load_max = loads_mean_xr.mean(dim='wt').values

    wDELs = 1 - wPower
    J_Power = wPower * farm_power_opt/refPower
    # J_DELs = wDELs * np.max([T1_load_mean, T2_load_mean])/(refDELs)
    J_DELs =  0 #wDELs * turb_load_max/(refDELs)
    J = J_Power - J_DELs    
    # print(f"    Helix amp: {helix_amp}, Farm Power: {farm_power:.1f} kW, Max DELs: {turb_load_max/1e3:.1f} kN-m, Objective: {J:.4f}")
    return J

x = np.array([0, 5*284])
y = np.array([0, 0])
# create objective function object
f_obj = ObjFuncComponent(obj_func = calculate_power_OFF,
                        input_keys = ['gamma'],
                        # x = x,          # Can delete x, y, wd, ws
                        # y = y,
                        oi = oi,
                        # wd = wind_direction,
                        # ws = 8,
                        wPower = 1,
                        refPower = farm_power_ref,
                        )

# create optimizer object
optimizer_MSR = MSR_optimizer(x = x,
                              y = y,
                              wd = wind_direction,
                              f_obj = f_obj,
                              n_step = 2,
                              exclusivity = True
                            )

# add strategy (Wake steering - Refine)
optimizer_MSR.add_strategy(str_name = 'Wake steering',
                        var_name = 'gamma',
                        opt_method = 'Refine',
                        n_values = 5,
                        cmin = -30.0,
                        cmax = 30.0)

t = time.time()
optimizer_MSR.optimize()
c_opt = optimizer_MSR.c_opt
yaw_opt = c_opt['gamma']

print(f"Optimal yaw misalignment: T1={yaw_opt[0]:.2f} deg, T2={yaw_opt[1]:.2f} deg")

# ============================================================================
# Predict loads with uncertainty quantification

# Create surrogates dictionary
surrogates_dict = {OUTPUT_NAME[0]: surrogate}

loads_mean_xr, loads_std_xr = predict_loads_sector_average(
    surrogates_dict, 
    sector_avg, 
    yaw_angles, 
    helix_amplitudes,
    return_std=True
)

# Extract values for both turbines
T1_load_mean = loads_mean_xr.sel(wt=0, name=OUTPUT_NAME[0]).values.squeeze()
T1_load_std = loads_std_xr.sel(wt=0, name=OUTPUT_NAME[0]).values.squeeze()
T2_load_mean = loads_mean_xr.sel(wt=1, name=OUTPUT_NAME[0]).values.squeeze()
T2_load_std = loads_std_xr.sel(wt=1, name=OUTPUT_NAME[0]).values.squeeze()

### 
## Plot figure of the turbine DELs in the same figure
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
time = sector_avg.coords['time'].values
delta_t = time[1] - time[0]

ax[0].plot(time, T1_load_mean, label='Turbine 1 Mean', color='blue')
ax[0].plot(time, T2_load_mean, label='Turbine 2 Mean', color='orange')
ax[0].set_ylabel(f'{OUTPUT_NAME[0]} DELs [kNm]')
ax[0].legend()      

# plot average DELs following miner's rule 
m_coeff = 10
T1_load_avg = (np.mean(T1_load_mean**m_coeff))**(1/m_coeff)
T2_load_avg = (np.mean(T2_load_mean**m_coeff))**(1/m_coeff)

ax[0].axhline(T1_load_avg, color='blue', linestyle='--', label='Turbine 1 Avg')
ax[0].axhline(T2_load_avg, color='orange', linestyle='--', label='Turbine 2 Avg')

# plot yaw misalignment angles
ax[1].plot(time, yaw_angles[0, :], label='Turbine 1 Yaw', color='blue')
ax[1].plot(time, yaw_angles[1, :], label='Turbine 2 Yaw', color='orange')
ax[1].set_ylabel('Yaw Misalignment [deg]')  
plt.show()

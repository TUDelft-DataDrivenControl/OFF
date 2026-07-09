# //////////////////////////////////////////////////////////////////// #
#    ____  ______ ______ 
#   / __ \|  ____|  ____|
#  | |  | | |__  | |__   
#  | |  | |  __| |  __|  
#  | |__| | |    | |     
#   \____/|_|    |_|     
# //////////////////////////////////////////////////////////////////// #

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

# //////////////////////////////////////////////////////////////////// #
# Welcome to the example OFF main file. This showcases how to run a simulation using the OFF framework.
# The settings are defined in the run_example.yaml file, have a look to see what is possible.
# If you experience issues, create a new issue on the GitHub page https://github.com/TUDelft-DataDrivenControl/OFF
# //////////////////////////////////////////////////////////////////// #

import copy
# from copy import deepcopy
import os, logging
logging.basicConfig(level=logging.ERROR)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import off.off as off
import off.off_interface as offi
import time

# import optimizer
from msr import MSR_optimizer
from msr import ObjFuncComponent

# import surrogate model
from gpytorch_gp_model import GPyTorchGPSurrogate, HeteroscedasticGPSurrogate, predict_loads_sector_average

from pathlib import Path
OFF_PATH: Path = Path(off.OFF_PATH)

print("OFF_PATH: ", OFF_PATH)

# This example shows how to run a simulation using the OFF framework in an incremental manner.
def main():
    # ============================================================================
    # Configure and load GPyTorch surrogate model

    OUTPUT_NAME = ['TwrBsMyt']  # Change to 'TwrBsMyt' or other sensors
    USE_HETEROSCEDASTIC = True  # Set to True to use heteroscedastic model
    wPower = 0.2  # Weight for power in the objective function
    m_coeff = 4  # Miner’s rule exponent for DELs (TwrBsMyt = 4, RootMflp = 10)

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

    # ======================================================= #
    # Get continous reference simulation
    # ======================================================= #

    start_time = time.time()
    # # Create an interface object
    oi_ref = offi.OFFInterface()
    
    # Tell the simulation what to run
    #   The run file needs to contain everything, the wake model, the ambient conditions etc.
    # Example case with PyWake
    oi_ref.init_simulation_by_path( OFF_PATH / "02_Examples_and_Cases" / "02_Example_Cases" / "007_two_turbines_wind_dir_change_pywake.yaml")
    
    # Run the simulation
    oi_ref.run_sim()

    print("\n---OFF Simulation took %s seconds ---" % (time.time() - start_time))

    # Get output
    measurements_long = oi_ref.get_measurements()
    control_long      = oi_ref.get_applied_control()

    # oi_ref.store_measurements()
    # oi_ref.store_applied_control()

    # ======================================================= #
    # Run incremental simulation
    # ======================================================= #
    start_time = time.time()

    Delta_t = 600.0 # seconds
    # Create an interface object
    oi_inc = offi.OFFInterface()
    
    # Initialize the simulation with the same settings as the continous simulation
    oi_inc.init_simulation_by_path( OFF_PATH / "02_Examples_and_Cases" / "02_Example_Cases" / "007_two_turbines_wind_dir_change_pywake.yaml")
    
    # Retrieve the simulation end time from the settings of the continous simulation
    sim_start_time = oi_inc.off_sim.settings_sim['time start']
    sim_end_time = oi_inc.off_sim.settings_sim['time end']
    wind_dir_profile = oi_inc.settings_cor["ambient"]["wind_directions"]
    wind_dir_t = oi_inc.settings_cor["ambient"]["wind_directions_t"]

    # Create placeholder for incremental measurements and control
    measurements_inc = pd.DataFrame()
    control_inc      = pd.DataFrame()

    wind_direction  = 225.0   # deg — incoming wind direction
    yaw_rate        = 0.3     # deg/s — physical yaw drive rate
    time_step       = 4

    # Read data settings
    ws_cols  = ['WS_sec_up_PyWake', 'WS_sec_right_PyWake', 'WS_sec_down_PyWake', 'WS_sec_left_PyWake']
    ti_cols  = ['TI_sec_up_PyWake',  'TI_sec_right_PyWake',  'TI_sec_down_PyWake',  'TI_sec_left_PyWake']
    P_cols   = ['Power_PyWake']
    sectors  = ['up', 'right', 'down', 'left']

    for step in np.arange(sim_start_time, sim_end_time, Delta_t):
        # Update the total number of iterations for the incremental simulation (only needed for progress bar)
        oi_inc.off_sim.iterations_total = int(Delta_t / oi_inc.off_sim.settings_sim['time step']) + 1

        if step == sim_start_time:
            # Run the incremental simulation for Delta_t seconds
            oi_inc.increment_sim(Delta_t-oi_inc.off_sim.settings_sim['time step'], start_time=step)

        else:          
            # define objective functions
            def calculate_power_OFF(oi, gamma, current_state, wPower, refPower, refDELs):
                # gamma[i] is the yaw misalignment w.r.t. the average wind direction in the prediction horizon.
                # Absolute target orientation = wind_dir_at_step + gamma[i].
                # delta_t_yaw is based on the actual angular change from the current orientation.
                
                n_wt_ctrl = len(gamma) # number of turbines to control

                oi_cntrl = copy.deepcopy(oi)            # OFF interface object for control
                current_state = oi_cntrl.get_state()    # Get the current state of the simulation
                current_state_ctrl = current_state
                
                # Extract current orientations from the controller's LUT at the current step
                orientation_current = [
                    np.interp(step, current_state.controller.t, current_state.controller.lut[:, i])
                    for i in range(n_wt_ctrl)
                ]

                # Compute average wind direction over [step, step+Delta_t] using exact
                # trapezoid integration of the piecewise-linear profile
                t_window = np.concatenate([[step],
                                           np.array(wind_dir_t)[(np.array(wind_dir_t) > step) & (np.array(wind_dir_t) < step + Delta_t)],
                                           [step + Delta_t]])
                wind_dir_avg = np.trapezoid(np.interp(t_window, wind_dir_t, wind_dir_profile), t_window) / Delta_t
                new_orientations = [wind_dir_avg + float(gamma[i]) for i in range(n_wt_ctrl)]
                delta_t_yaw = max(abs(new_orientations[i] - orientation_current[i]) for i in range(n_wt_ctrl)) / yaw_rate

                # Fill control_data with 4 rows: hold current orientation, then ramp to target
                current_state_ctrl.controller.lut = np.tile(orientation_current, (4, 1))
                for i in range(n_wt_ctrl):
                    current_state_ctrl.controller.lut[2][i] = new_orientations[i]
                    current_state_ctrl.controller.lut[3][i] = new_orientations[i]
                current_state_ctrl.controller.t = [step, step, step + delta_t_yaw, sim_end_time]
                print(current_state_ctrl.controller.lut.shape)

                #  Run the incremental simulation for Delta_t seconds
                oi_cntrl.set_state(current_state_ctrl)
                oi_cntrl.increment_sim(Delta_t-oi_cntrl.off_sim.settings_sim['time step'], start_time=step)
                
                # Extract measusrements and control inputs
                measurements_opt = oi_cntrl.get_measurements()
                control_opt      = oi_cntrl.get_applied_control()

                # Read power and sector-averaged WS and TI from measurements.csv
                n_wt = int(measurements_opt['t_idx'].nunique())
                n_time = int(len(measurements_opt) / n_wt)
            
                sector_data = np.zeros((n_wt, n_time, len(sectors), 2))  # (wt, time, sector, quantity=[WS, TI])
                power_data = np.zeros((n_wt, n_time))                    # (wt, time)
                # Extract data per turbine
                for wt_i in range(n_wt):
                    wt_df = measurements_opt[measurements_opt['t_idx'] == wt_i]
                    sector_data[wt_i, :, :, 0] = wt_df[ws_cols].values
                    sector_data[wt_i, :, :, 1] = wt_df[ti_cols].values
                    power_data[wt_i, :] = wt_df[P_cols].values.squeeze()
                time_coords = measurements_opt[measurements_opt['t_idx'] == 0]['time'].values
                
                # Compute turbine energy farm energy over the prediction horizon
                turb_power_opt = np.sum(power_data * time_step, axis=1)     # cumulative energy for each turbine
                farm_power_opt = np.sum(turb_power_opt) / 1e3 / 3600        # total farm energy in MWh

                print(f"Farm energy for yaw misalignment {gamma}: {farm_power_opt:.1f} kWh")

                sector_avg = xr.DataArray(
                    sector_data,
                    dims=['wt', 'time', 'sector', 'quantity'],
                    coords={'wt': list(range(n_wt)), 'time': time_coords, 'sector': sectors}
                ).expand_dims({'wd': 1, 'ws': 1})

                # Read yaw angles from applied_control
                yaw_angles = np.zeros((n_wt, n_time))
                for wt_i in range(n_wt):
                    yaw_angles[wt_i, :] = control_opt[control_opt['t_idx'] == wt_i]['yaw'].values
                helix_amplitudes = np.zeros((n_wt, n_time))     # zero helix amplitude for load surrogate

                # Create surrogates dictionary
                surrogates_dict = {OUTPUT_NAME[0]: surrogate}

                # Infer turbine loads based on inflow from sector_avg.
                loads_xr = predict_loads_sector_average(
                    surrogates_dict, 
                    sector_avg, 
                    yaw_angles, 
                    helix_amplitudes,
                    return_std=False
                )

                # Average DELs following miner's rule for all turbines
                loads_avg = [
                    (np.mean(loads_xr.sel(wt=wt_i, name=OUTPUT_NAME[0]).values.squeeze() ** m_coeff)) ** (1 / m_coeff)
                    for wt_i in range(n_wt)
                ]
                max_DELs_opt = np.max(loads_avg)

                print(f"Maximum average loading {gamma}: {max_DELs_opt:.1f} kNm")

                wDELs = 1 - wPower
                J_Power = wPower * farm_power_opt / refPower
                J_DELs =  wDELs * max_DELs_opt / refDELs
                J = J_Power - J_DELs    

                print(f"Objective function value {gamma}: {J:.4f} (Power: {J_Power:.4f}, DELs: {J_DELs:.4f})")
                return J
                       
            # Get state of the simulation at the current step
            oi_ref = copy.deepcopy(oi_inc)
            current_state = oi_ref.get_state()
            # get_state() returns a live reference to off_sim; snapshot the wake state
            # NOW before oi_ref.increment_sim() propagates the OPs forward by Delta_t
            state_at_step = copy.deepcopy(current_state)
            # orientation_current = [np.interp(step, current_state.controller.t, current_state.controller.lut[:, 0]),
            #                        np.interp(step, current_state.controller.t, current_state.controller.lut[:, 1])]
            n_wt_ctrl = current_state.controller.lut.shape[1]
            orientation_current = [
                                   np.interp(step, current_state.controller.t, current_state.controller.lut[:, i])
                                   for i in range(n_wt_ctrl)
                                  ]
            
            # Fill control_data with 4 equal rows of the current orientation
            current_state.controller.lut = np.tile(orientation_current, (2, 1))
            current_state.controller.t   = [step, sim_end_time]
            print(f"Current reference lut at step {step}:\n{current_state.controller.lut}")
            print(f"Current reference t at step {step}:\n{current_state.controller.t}")
            
            #  Run the incremental simulation for Delta_t seconds
            oi_ref.set_state(current_state)
            oi_ref.increment_sim(Delta_t-oi_ref.off_sim.settings_sim['time step'], start_time=step)
            print(f"Reference simulation completed for step {step}.")

            # Get the reference power and DELs for the objective function
            measurements_ref = oi_ref.get_measurements()
            control_ref      = oi_ref.get_applied_control()

            # Read power and sector-averaged WS and TI from measurements.csv
            n_wt = int(measurements_ref['t_idx'].nunique())
            n_time = int(len(measurements_ref) / n_wt)
        
            sector_data = np.zeros((n_wt, n_time, len(sectors), 2))  # (wt, time, sector, quantity=[WS, TI])
            power_data = np.zeros((n_wt, n_time))  # (wt, time)
            for wt_i in range(n_wt):
                wt_df = measurements_ref[measurements_ref['t_idx'] == wt_i]
                sector_data[wt_i, :, :, 0] = wt_df[ws_cols].values
                sector_data[wt_i, :, :, 1] = wt_df[ti_cols].values
                power_data[wt_i, :] = wt_df[P_cols].values.squeeze()
            time_coords = measurements_ref[measurements_ref['t_idx'] == 0]['time'].values

            # Compute turbine energy and farm energy over the prediction horizon for the reference case
            turb_power_ref = np.sum(power_data * time_step, axis=1)  # cumulative power for each turbine
            farm_power_ref = np.sum(turb_power_ref) / 1e3 / 3600           # total farm energy in MWh
            print(f"Reference farm power at step {step}: {farm_power_ref:.1f} kWh")

            sector_avg = xr.DataArray(
                sector_data,
                dims=['wt', 'time', 'sector', 'quantity'],
                coords={'wt': list(range(n_wt)), 'time': time_coords, 'sector': sectors}
            ).expand_dims({'wd': 1, 'ws': 1})

            # Read yaw angles from applied_control
            yaw_angles = np.zeros((n_wt, n_time))
            for wt_i in range(n_wt):
                yaw_angles[wt_i, :] = control_ref[control_ref['t_idx'] == wt_i]['yaw'].values
            helix_amplitudes = np.zeros((n_wt, n_time))

            # Infer reference turbine loads based on inflow from sector_avg using the surrogate model
            surrogates_dict = {OUTPUT_NAME[0]: surrogate}
            loads_xr_ref = predict_loads_sector_average(
                surrogates_dict, 
                sector_avg, 
                yaw_angles, 
                helix_amplitudes,
                return_std=False
            )

            # Average DELs following miner's rule for all turbines
            loads_avg_ref = [
                (np.mean(loads_xr_ref.sel(wt=wt_i, name=OUTPUT_NAME[0]).values.squeeze() ** m_coeff)) ** (1 / m_coeff)
                for wt_i in range(n_wt)
            ]
            max_DELs_ref = np.max(loads_avg_ref)

            print(f"Reference maximum average loading at step {step}: {max_DELs_ref:.1f} kNm")
            
            x = np.array([0, 3.83*284, 7.66*284])
            y = np.array([0, 3.21*284, 6.42*284])
            # create objective function object
            f_obj = ObjFuncComponent(obj_func = calculate_power_OFF,
                                    input_keys = ['gamma'],
                                    oi = oi_inc,
                                    wPower = wPower,
                                    current_state = current_state,
                                    refPower = farm_power_ref,
                                    refDELs = max_DELs_ref
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
            yaw_opt = c_opt['gamma']  # yaw misalignment w.r.t. wind direction [deg]

            # Convert optimal misalignment to absolute target orientations using average
            # wind direction over [step, step+Delta_t]
            t_window = np.concatenate([[step],
                                       np.array(wind_dir_t)[(np.array(wind_dir_t) > step) & (np.array(wind_dir_t) < step + Delta_t)],
                                       [step + Delta_t]])
            wind_dir_avg = np.trapezoid(np.interp(t_window, wind_dir_t, wind_dir_profile), t_window) / Delta_t
            n_wt_ctrl = len(yaw_opt)
            new_orientations = [wind_dir_avg + yaw_opt[i] for i in range(n_wt_ctrl)]
            delta_t_yaw = max(abs(new_orientations[i] - orientation_current[i]) for i in range(n_wt_ctrl)) / yaw_rate

            # Fill control_data with 4 rows: hold current orientation, then ramp to target
            # Apply to state_at_step (OPs at 'step') not current_state (OPs at step+Delta_t)
            state_at_step.controller.lut = np.tile(orientation_current, (4, 1))
            for i in range(n_wt_ctrl):
                state_at_step.controller.lut[2][i] = new_orientations[i]
                state_at_step.controller.lut[3][i] = new_orientations[i]
            state_at_step.controller.t   = [step, step, step + delta_t_yaw, sim_end_time]
            print(f"New lut at step {step}:\n{state_at_step.controller.lut}")
            print(f"New t at step {step}:\n{state_at_step.controller.t}")

            oi_inc.set_state(state_at_step)
            
            #  Run the incremental simulation for Delta_t seconds
            oi_inc.increment_sim(Delta_t-oi_inc.off_sim.settings_sim['time step'], start_time=step)
        
        # Get output and concatenate to the previous outputs
        measurements_inc = pd.concat([measurements_inc, oi_inc.get_measurements()])
        control_inc      = pd.concat([control_inc, oi_inc.get_applied_control()])

    print("\n---OFF Incremental Simulation took %s seconds ---" % (time.time() - start_time))

    out_dir = OFF_PATH / "runs" / f"receding_horizon_3turbs_{OUTPUT_NAME[0]}_w{wPower:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    measurements_inc.to_csv(out_dir / "measurements.csv", index=False)
    control_inc.to_csv(out_dir / "applied_control.csv", index=False)
    print(f"Saved measurements and applied_control to {out_dir}")


    # ======================================================= #
    # Plot comparison 
    # ======================================================= #
    wind_dir_min = wind_direction - 30
    wind_dir_max = wind_direction + 30

    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(measurements_long[(measurements_long['t_idx'] == 0)]['time'],
                 measurements_long[(measurements_long['t_idx'] == 0)]['Power_PyWake']/1e6, 
                 label='Continuous Simulation', color='blue')
    axs[0].plot(measurements_inc[(measurements_inc['t_idx'] == 0)]['time'],
                 measurements_inc[(measurements_inc['t_idx'] == 0)]['Power_PyWake']/1e6, 
                 color='orange', linestyle='--', label='Incremental Simulation')
    axs[0].vlines(np.arange(sim_start_time, sim_end_time, Delta_t),
                color='gray', linestyle=':', alpha=0.5,
                ymin=0, ymax=12,
                label='Incremental Simulation Steps')
    axs[0].set_ylim(0, 12)
    axs[0].set_xlim(sim_start_time, sim_end_time)
    axs[0].set_ylabel('Power (MW)')
    axs[0].set_title('Turbine 0 Power Comparison')
    axs[0].legend(loc='lower left')

    axs[1].plot(measurements_long[(measurements_long['t_idx'] == 1)]['time'],
                 measurements_long[(measurements_long['t_idx'] == 1)]['Power_PyWake']/1e6, 
                 label='Continuous Simulation', color='blue')
    axs[1].plot(measurements_inc[(measurements_inc['t_idx'] == 1)]['time'],
                 measurements_inc[(measurements_inc['t_idx'] == 1)]['Power_PyWake']/1e6, 
                 color='orange', linestyle='--', label='Incremental Simulation')
    axs[1].vlines(np.arange(sim_start_time, sim_end_time, Delta_t),
                color='gray', linestyle=':', alpha=0.5,
                ymin=0, ymax=10,
                label='Incremental Simulation Steps')
    axs[1].set_ylim(0, 10)
    axs[1].set_xlim(sim_start_time, sim_end_time)
    axs[1].set_ylabel('Power (MW)')
    axs[1].set_title('Turbine 1 Power Comparison')
    # axs[1].legend()

      # Plot total farm power comparison
    axs[2].plot(measurements_long.groupby('time')['Power_PyWake'].sum()/1e6, 
                 label='Reference Simulation', color='blue')
    axs[2].plot(measurements_inc.groupby('time')['Power_PyWake'].sum()/1e6, 
                 label='Optimized Simulation', color='orange', linestyle='--')
    axs[2].vlines(np.arange(sim_start_time, sim_end_time, Delta_t),
                color='gray', linestyle=':', alpha=0.5,
                ymin=0, ymax=30,
                label='Incremental Simulation Steps')
    axs[2].set_ylabel('Total Farm Power (MW)')
    axs[2].set_ylim(5, 30)
    axs[2].set_title('Farm Power Comparison')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_xlim(sim_start_time, sim_end_time)
    # axs[2].legend()
    fig.savefig(out_dir / f"Optimize_loads_comparison.png", dpi=300)
    plt.tight_layout()

    
    # ======================================================= #
    # Plot comparison 
    # ======================================================= #
    fig2, axs2 = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    axs2[0].plot(measurements_long[(measurements_long['t_idx'] == 0)]['time'],
                 control_long[(control_long['t_idx'] == 0)]['orientation'], 
                 label='Reference Simulation', color='blue')
    axs2[0].plot(measurements_inc[(measurements_inc['t_idx'] == 0)]['time'],
                 control_inc[(control_inc['t_idx'] == 0)]['orientation'], 
                 color='orange', linestyle='--', label='Incremental Simulation')
    axs2[0].vlines(np.arange(sim_start_time, sim_end_time, Delta_t),
                color='gray', linestyle=':', alpha=0.5,
                ymin=wind_dir_min, ymax=wind_dir_max,
                label='Incremental Simulation Steps')
    axs2[0].plot(wind_dir_t, wind_dir_profile, color='green', linestyle='-', label='Wind Direction Profile')
    axs2[0].set_ylim(wind_dir_min, wind_dir_max)
    axs2[0].set_xlim(sim_start_time, sim_end_time)
    axs2[0].set_ylabel('Orientation (deg)')
    axs2[0].set_title('Turbine 0 Orientation Comparison')
    axs2[0].legend(loc='upper left')

    axs2[1].plot(measurements_long[(measurements_long['t_idx'] == 1)]['time'],
                 control_long[(control_long['t_idx'] == 1)]['orientation'], 
                 label='Reference Simulation', color='blue')
    axs2[1].plot(measurements_inc[(measurements_inc['t_idx'] == 1)]['time'],
                 control_inc[(control_inc['t_idx'] == 1)]['orientation'], 
                 color='orange', linestyle='--', label='Incremental Simulation')
    axs2[1].vlines(np.arange(sim_start_time, sim_end_time, Delta_t),
                color='gray', linestyle=':', alpha=0.5,
                ymin=wind_dir_min, ymax=wind_dir_max,
                label='Incremental Simulation Steps')
    axs2[1].plot(wind_dir_t, wind_dir_profile, color='green', linestyle='-', label='Wind Direction Profile')
    axs2[1].set_ylim(wind_dir_min, wind_dir_max)
    axs2[1].set_xlim(sim_start_time, sim_end_time)
    axs2[1].set_ylabel('Orientation (deg)')
    axs2[1].set_title('Turbine 1 Orientation Comparison')
    axs2[1].set_xlabel('Time (s)')
    fig2.savefig(out_dir / f"Optimize_loads_yaw_comparison.png", dpi=300)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

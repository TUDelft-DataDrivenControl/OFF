import numpy as np
from scipy.stats import norm
from numpy import newaxis as na
import pandas as pd
from gpytorch_gp_model import predict_loads_sector_average

# import optimizer
from msr import MSR_optimizer
from msr import ObjFuncComponent
import time

from wind_farm_loads import tool_agnostic as ta
from pathos.multiprocessing import ProcessPool

#%% FUNCTIONS

def compute_WFFC_LUT(x,
                     y,
                     wfm,
                     wd,
                     ws,
                     surrogates_dict,
                     helix_amp_max = 4,
                     wPower = 0.9,
                     n_step = 3,
                     n_values = 5,
                     parallel_execution = False,
                     n_cpu = None):
      
    # define objective functions
    def calculate_power_loads_helix(x,y,wf_model,wd,ws,helix_amp,wPower,refPower,refDELs):
        yaw_angles = np.zeros_like(helix_amp)
        tilt_angles = np.zeros_like(helix_amp)
        sim_res = wf_model(x, y, 
                           ws=ws, 
                           wd=wd, 
                           yaw = yaw_angles, 
                           tilt = tilt_angles, 
                           helix_amp=helix_amp)
        turb_power = sim_res.Power.values
        farm_power = np.sum(turb_power)
        sector_avg = ta.compute_sector_average(sim_res, radius=10, n_azimuth_per_sector=10, look='upwind')
                
        # Predict loads with uncertainty quantification
        loads_mean_xr, loads_std_xr = predict_loads_sector_average(
            surrogates_dict, 
            sector_avg, 
            yaw_angles, 
            helix_amp,
            return_std=True
        )
        
        # Extract values for both turbines
        turb_load_mean = loads_mean_xr.mean(dim='wt').values
        
        wDELs = 1 - wPower
        J_Power = wPower * farm_power/(refPower)
        J_DELs = wDELs * turb_load_mean/(refDELs)
        J = J_Power - J_DELs    
        return J
    
    # define a wrapper function to run in parallel for each flow case
    def wffc_optimizer_wrapper(wd,ws):
        ## Reference values for objective function normalization (from baseline case)
        sim_res_ref = wfm(x, y, 
                        ws=ws, 
                        wd=wd, 
                        yaw = np.zeros_like(x), 
                        tilt = np.zeros_like(x), 
                        helix_amp = np.zeros_like(x))
        turb_power_ref = sim_res_ref.Power.values
        farm_power_ref = np.sum(turb_power_ref)
        sector_avg = ta.compute_sector_average(sim_res_ref, radius=10, n_azimuth_per_sector=10, look='upwind')
        
        # Predict loads with uncertainty quantification
        loads_mean_xr, loads_std_xr = predict_loads_sector_average(
            surrogates_dict, 
            sector_avg, 
            np.zeros_like(x), 
            np.zeros_like(x),
            return_std=True
        )
        
        # Extract values for both turbines
        DELS_ref = loads_mean_xr.mean(dim='wt').values

        # create objective function object
        f_obj = ObjFuncComponent(obj_func = calculate_power_loads_helix,
                                input_keys = ['helix_amp'],
                                x = x,
                                y = y,
                                wf_model = wfm,
                                wd = wd,
                                ws = ws,
                                wPower = wPower,
                                refPower = farm_power_ref,
                                refDELs = DELS_ref)
    
        # create optimizer object
        optimizer_MSR = MSR_optimizer(x = x,
                                    y = y,
                                    wd = wd,
                                    f_obj = f_obj,
                                    n_step = n_step,
                                    exclusivity = True
                                    )
    
        # add strategy (Wake steering - Refine)
        optimizer_MSR.add_strategy(str_name = 'Helix method',
                                var_name = 'helix_amp',
                                opt_method = 'Refine',
                                n_values = n_values,
                                cmin = 0.0,
                                cmax = helix_amp_max)
    
        t = time.time()
        optimizer_MSR.optimize()
        c_opt = optimizer_MSR.c_opt
        helix_amp_opt_values = c_opt['helix_amp']
        
        return helix_amp_opt_values

    
   
    # use wd and ws as timseries
    wd_list = wd.tolist()
    ws_list = ws.tolist()
      
    if parallel_execution:
        with ProcessPool(n_cpu) as pool:
            results = pool.map(wffc_optimizer_wrapper,wd_list,ws_list)
        res_helix_amp_opt = zip(*results)
    else:
        results = map(wffc_optimizer_wrapper,wd_list,ws_list)
        res_helix_amp_opt = zip(*results)

    # reconstruct array
    helix_amp_opt_array = np.array(list(res_helix_amp_opt)).T
    
    return helix_amp_opt_array
    
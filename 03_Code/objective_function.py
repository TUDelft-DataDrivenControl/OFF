from multiprocessing.spawn import freeze_support

import DynamicYawOptimization.optvars_2_yaml as optvars_2_yaml
import os, logging
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.ERROR)

import off.off as off
import off.off_interface as offi
import time
from optimparallel import minimize_parallel
import os


def ObjectiveFunction(opt_vars):

    opt_vars = opt_vars.reshape(2, -1).T
    # file_path = "../02_Examples_and_Cases/02_Example_Cases/dummy_yaml"
    #
    # process_id = str(os.getpid())
    # time_id = str(round(1e7*time.time()))

    # optvars_2_yaml.Optvars2Yaml(opt_vars, file_path, process_id)

    start_time = time.time()
    oi = offi.OFFInterface()
    oi.init_simulation_by_path(f'{off.OFF_PATH}/02_Examples_and_Cases/02_Example_Cases/dummy_yaml.yaml', opt_vars)
    oi.run_sim()
    print("---OFF Simulation took %s seconds ---" % (time.time() - start_time))
    power = oi.measurements["power_OFF"]
    print('cost', sum(power))
    print(opt_vars)

    return -sum(power)


if __name__ == '__main__':
    # t1 = np.array([270, 270, 285])
    # t2 = np.array([270, 270, 270])
    # opt_vars = np.array([t1, t2])
    # ObjectiveFunction(opt_vars)
    freeze_support()

    t1 = np.array([270, 270, 270])
    t2 = np.array([270, 270, 270])
    opt_vars = np.array([t1, t2]).reshape(1, -1).squeeze()

    lb = np.ones(opt_vars.shape) * 295
    ub = np.ones(opt_vars.shape) * 245
    bounds = np.array([ub, lb])

    res = minimize_parallel(ObjectiveFunction, opt_vars, bounds=bounds.T)

    print(res.x)
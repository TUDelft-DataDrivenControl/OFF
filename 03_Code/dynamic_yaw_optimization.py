import numpy as np
import scipy.optimize as opt
import objective_function
from optimparallel import minimize_parallel

t1 = np.array([270, 270, 270])
t2 = np.array([270, 270, 270])
opt_vars = np.array([t1, t2]).reshape(1, -1).squeeze()

lb = np.ones(opt_vars.shape)*295
ub = np.ones(opt_vars.shape)*245
bounds = np.array([ub, lb])

res = minimize_parallel(objective_function.ObjectiveFunction, opt_vars, bounds=bounds.T)

print(res.x)
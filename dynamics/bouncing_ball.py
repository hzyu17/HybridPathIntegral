## 2 dimensional Bouncing ball hitting a plan ground.
# x = [y,z, \dot y, \dot z]

import numpy as np

def dyn_f(t, x):
    g = 9.81
    k = 1.0
    vy_norm = np.sqrt(x[2]*x[2])
    vz_norm = np.sqrt(x[3]*x[3])
    return np.array([x[2], x[3], -k*vy_norm, -g-k*vz_norm])

def event_bouncing(t, x):
    return x[1]

def reset_map(t, x_minus):
    x_plus = np.zeros(4, dtype=np.float64)
    x_plus[0] = x_minus[0]
    x_plus[1] = x_minus[1]
    x_plus[2] = x_minus[2]
    x_plus[3] = -0.9*x_minus[3] 
    return x_plus

# partial derivatives R_x
def par_Rx(t, x):
    
    return 
## 1 dimensional bouncing ball dynamics
# x = [z, \dot z]

import numpy as np

def dyn_f(t, x):
    g = 9.81
    k = 1.0
    vz_norm = np.sqrt(x[1]*x[1])
    return np.array([x[1], -g-k*vz_norm])

def event_bouncing(t, x):
    return x[0]

def reset_map(t, x_minus):
    e = 0.9
    x_plus = np.zeros(2, dtype=np.float64)
    x_plus[0] = x_minus[0]
    x_plus[1] = -e*x_minus[1] 
    return x_plus

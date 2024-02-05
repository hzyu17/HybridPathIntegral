## Bouncing ball hitting a plan ground.

import numpy as np

def dyn_f(t, x):
    g = 9.81
    k = 1.0
    vy_norm = np.sqrt(x[2]*x[2])
    vz_norm = np.sqrt(x[3]*x[3])
    return np.array([x[2], x[3], -k*vy_norm, -g-k*vz_norm])

def event_bouncing(t, x):
    return x[1]

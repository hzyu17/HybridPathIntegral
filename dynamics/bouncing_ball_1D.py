## 1 dimensional bouncing ball dynamics
# x = [z, \dot z]

from saltation_matrix.samtation_matrix import *
import numpy as np

def dyn_f(t, x):
    g = 9.81
    k = 1.0
    vz_norm = np.sqrt(x[1]*x[1])
    return np.array([x[1], -g-k*vz_norm], dtype=np.float64)

def event_bouncing(t, x):
    return x[0]

def reset_map(t, x_minus):
    e1 = 0.8
    e2 = 0.6
    x_plus = np.zeros(2, dtype=np.float64)
    x_plus[0] = e1*x_minus[0]
    x_plus[1] = -e2*x_minus[1] 
    return x_plus

def Rx(t, x):
    e1 = 0.8
    e2 = 0.6
    return np.array([[e1, 0.0],[0.0, -e2]], dtype=np.float64)

def Rt(t, x):
    return np.zeros(2, dtype=np.float64)

def gx(t, x):
    return np.array([1.0, 0.0], dtype=np.float64)
    
def gt(t, x):
    return 0.0
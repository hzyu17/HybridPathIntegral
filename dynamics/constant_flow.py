## The dynamics of constant flow.
# Hongzhe Yu, 02/09/24

import numpy as np
from saltation_matrix.samtation_matrix import *

def dyn_f1(t, x):
    return np.array([1.0, -2.0], dtype=np.float64)

def dyn_f2(t, x):
    return np.array([1.0, 3.0], dtype=np.float64)

def event_cst_flow(t, x):
    return x[0] - 5.0

def reset_map(t, x_minus):
    """
    Identity reset map.
    """
    return x_minus

def Rx(t, x):
    return np.eye(2, dtype=np.float64)

def Rt(t, x):
    return np.zeros(2, dtype=np.float64)

def gx(t, x):
    return np.array([1.0, 0.0], dtype=np.float64)
    
def gt(t, x):
    return 0.0

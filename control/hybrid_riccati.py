# Solve Riccati equation with hybrid events
# Hongzhe Yu, 02/26/2024

import numpy as np
from saltation_matrix.saltation_matrix import *

def dS_Riccati(t, z, *args):
    """
    Args:
        t (scalar): time variable
        z (array): concatenated state and flattened quadratic value matrix [x, S]
        args[0]: dyn (function): system dynamics 
        args[1]: linearization (function): linearization function
        args[2]: Q(t, x): the quadratic state loss matrix
        args[3]: n (scalar): state dimension

    Returns:
        [x(t), S(t)]: solved coupled system
    """
    
    dyn = args[0]
    linearization = args[1]
    Q_tx = args[2]
    n = args[3]
    
    x = z[:n]
    X = z[n:].reshape((n, n))
    
    A, B = linearization(x)

    # state dynamics
    dx_dt = dyn(t, x)
    
    # covariance dynamics
    dX_dt =  -(np.transpose(A) @ X + X @ A - X @ B @ np.transpose(B) @ np.transpose(X) + Q_tx(t, x))
    
    return np.concatenate([dx_dt, dX_dt.flatten()])

    
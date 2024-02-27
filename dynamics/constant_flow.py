## The dynamics of constant flow.
# Hongzhe Yu, 02/09/24

import numpy as np
from saltation_matrix.samtation_matrix import *
import jax
from jax import jacfwd 

def dyn_f1(t, x):
    return np.array([1.0, -2.0], dtype=np.float64)

def dyn_f2(t, x):
    return np.array([1.0, 3.0], dtype=np.float64)

def guard_ctflow(t, x):
    return x[0] - 5.0
guard_jit = jax.jit(guard_ctflow)

def linearization_ctflow(x):
    return np.zeros((2, 2), dtype=np.float64), np.zeros(2)

def resetmap_ctflow(t, x_minus):
    """
    Identity reset map.
    """
    return x_minus
reset_map_jit = jax.jit(resetmap_ctflow)

def Rx_ctflow(t, x):
    return jacfwd(reset_map_jit, argnums=(1))(t, x)

def Rt_ctflow(t, x):
    return jacfwd(reset_map_jit, argnums=(0))(t, x)

def gt_ctflow(t, x):
    return jax.grad(guard_jit, argnums=0)(t, x)

def gx_ctflow(t, x):
    return jax.grad(guard_jit, argnums=1)(t, x)

def mode_jump_ctflow(current_mode):
    if current_mode == 0:
        return 1
    if current_mode == 1:
        return 0

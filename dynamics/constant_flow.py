## The dynamics of constant flow.
# Hongzhe Yu, 02/09/24

import numpy as np
from saltation_matrix.samtation_matrix import *
import jax
from jax import jacfwd, jacrev 

def dyn_f1(t, x):
    return np.array([1.0, -2.0], dtype=np.float64)

def dyn_f2(t, x):
    return np.array([1.0, 3.0], dtype=np.float64)

def guard(t, x):
    return x[0] - 5.0
guard_jit = jax.jit(guard)

def reset_map(t, x_minus):
    """
    Identity reset map.
    """
    return x_minus
reset_map_jit = jax.jit(reset_map)

def Rx(t, x):
    return jacfwd(reset_map_jit, argnums=(1))(t, x)

def Rt(t, x):
    return jacfwd(reset_map_jit, argnums=(0))(t, x)

def gt(t, x):
    return jax.grad(guard_jit, argnums=0)(t, x)

def gx(t, x):
    return jax.grad(guard_jit, argnums=1)(t, x)

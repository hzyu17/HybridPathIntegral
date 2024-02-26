## 1 dimensional bouncing ball dynamics
# x = [z, \dot z]

from saltation_matrix.samtation_matrix import *
import numpy as np
import jax.numpy as jnp

import jax
from jax import jacfwd 

def dyn_bouncing(t, x):
    g = 9.81
    # k = 1.0
    # vz_norm = np.sqrt(x[1]*x[1])
    # return np.array([x[1], -g-k*vz_norm], dtype=np.float64)
    return np.array([x[1], -g], dtype=np.float64)

def guard_bouncing(t, x):
    return x[0]
guard_bouncing_jit = jax.jit(guard_bouncing)

def reset_map_bouncing(t, x_minus):
    e1 = 1.0
    e2 = 0.6
    coeff = np.array([[e1, 0], [0, -e2]])
    x_plus = coeff@x_minus
    
    # x_plus.at[0].set(e1*x_minus[0])
    # x_plus.at[1].set(-e2*x_minus[1]) 
    return x_plus
resetmap_bouncing_jit = jax.jit(reset_map_bouncing)

# def Rx_bouncing(t, x):
#     return jacfwd(resetmap_bouncing_jit, argnums=(1))(t, x)

# def Rt_bouncing(t, x):
#     return jacfwd(resetmap_bouncing_jit, argnums=(0))(t, x)

# def gt_bouncing(t, x):
#     return jax.grad(guard_bouncing_jit, argnums=0)(t, x)

# def gx_bouncing(t, x):
#     return jax.grad(guard_bouncing_jit, argnums=1)(t, x)


def Rx_bouncing(t, x):
    e1 = 1.0
    e2 = 0.6
    return np.array([[e1, 0.0],[0.0, -e2]], dtype=np.float64)

def Rt_bouncing(t, x):
    return np.zeros(2, dtype=np.float64)

def gx_bouncing(t, x):
    return np.array([1.0, 0.0], dtype=np.float64)
    
def gt_bouncing(t, x):
    return 0.0

def linearize_bouncing(x):
    return np.array([[0, 1.0], [0.0, 0.0]], dtype=np.float64)
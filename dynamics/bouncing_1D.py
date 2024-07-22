## 1 dimensional bouncing ball dynamics
# x = [z, \dot z]
import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(root_dir)

from hybrid_ilqr.saltation_matrix import *
import numpy as np

import sympy as sp
from sympy.matrices import Matrix

import jax
from jax import jacfwd 

e1 = 1.0
e2 = 0.6
g = 9.81
m = 1.0

def dyn_bouncing(t, x):
    
    # Define the symbolic variables
    z, z_dot, u = sp.symbols('z z_dot u')
    
    # Defining the dynamics
    dynamics = Matrix([x[1], -g])
    dyn_func = sp.lambdify((z, z_dot), dynamics, 'numpy')
    
    return dyn_func(x[0], x[1]).flatten()


def linearize_bouncing(x):
    """
    Linearization.
    Args:
        x: state

    Returns:
        A, B: linear system matrices.
    """
    
    return np.array([[0, 1.0], [0.0, 0.0]], dtype=np.float64), np.array([0, 1.0/m])

def guard_bouncing_12(t, x):
    return x[0]
guard_bouncing_jit = jax.jit(guard_bouncing_12)

def guard_bouncing_21(t, x):
    return x[1]

def reset_map_bouncing_21(t, x_minus, current_mode):
    if (x_minus[0] > 0) and (current_mode==2):
        x_plus = x_minus
        current_mode = 1
    return x_plus, current_mode

def reset_map_bouncing_12(t, x_minus, current_mode):
    e2 = 0.6
    new_mode = current_mode
    if (x_minus[1] < 0) and (current_mode==1):
        coeff = np.array([[1.0, 0], [0, -e2]])
        x_plus = coeff@x_minus
        new_mode = 2
        
    else: # integration error
        x_plus = x_minus
        
    return x_plus, new_mode

resetmap_bouncing_jit = jax.jit(reset_map_bouncing_12)

# Mode jump: only 1 mode in this case
def mode_jump_bouncing(current_mode):
    mode_map = {0: 0}
    return mode_map[current_mode]

# def Rx_bouncing(t, x):
#     return jacfwd(resetmap_bouncing_jit, argnums=(1))(t, x)

# def Rt_bouncing(t, x):
#     return jacfwd(resetmap_bouncing_jit, argnums=(0))(t, x)

# def gt_bouncing(t, x):
#     return jax.grad(guard_bouncing_jit, argnums=0)(t, x)

# def gx_bouncing(t, x):
#     return jax.grad(guard_bouncing_jit, argnums=1)(t, x)


def Rx_bouncing(t, x):
    return np.array([[e1, 0.0],[0.0, -e2]], dtype=np.float64)

def Rt_bouncing(t, x):
    return np.zeros(2, dtype=np.float64)

def gx_bouncing(t, x):
    return np.array([1.0, 0.0], dtype=np.float64)
    
def gt_bouncing(t, x):
    return 0.0


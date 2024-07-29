## 1 dimensional bouncing ball dynamics
# x = [z, \dot z]
# 2 modes: falling and bouncing, with the same smooth dynamics 

import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(root_dir)

from hybrid_ilqr.saltation_matrix import *
import numpy as np
import jax.numpy as jnp

import sympy as sp
from sympy.matrices import Matrix

import jax
from jax import jacfwd, grad

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

def mode_change_maps(current_mode):
    new_mode = current_mode
    if (current_mode == 0):
        new_mode = 1
    elif (current_mode == 1):
        new_mode = 0
    return new_mode

def linearize_bouncing(x):
    return np.array([[0, 1.0], [0.0, 0.0]], dtype=np.float64), np.array([0, 1.0/m])

def guard_bouncing_12(t, x):
    return x[0]
guard_bouncing_12_jit = jax.jit(guard_bouncing_12)

def guard_bouncing_21(t, x):
    return x[1]

def reset_map_bouncing_21(t, x_minus, current_mode):
    x_plus = x_minus
    next_mode = current_mode
    
    if (x_minus[0] > 0) and (current_mode==1):
        x_plus = x_minus
        next_mode = 0
    return x_plus, next_mode


def reset_map_bouncing_12_jax(t, x_minus, current_mode):
    bouncing_cond = jax.numpy.logical_and(x_minus[1] < 0, current_mode==0)
    def bouncing_true_fun(args):
        x_minus, current_mode = args
        e2 = 0.6
        coeff = jnp.array([[1.0, 0], [0, -e2]])
        x_plus = coeff@x_minus
        new_mode = 1
        return x_plus, new_mode
    
    def bouncing_false_fun(args):
        x_minus, current_mode = args
        return x_minus, current_mode
    
    args = (x_minus, current_mode)
    x_plus, new_mode = jax.lax.cond(bouncing_cond, bouncing_true_fun, bouncing_false_fun, args)
        
    return x_plus, new_mode


def reset_map_bouncing_21_jax(t, x_minus, current_mode):
    takeoff_cond = jax.numpy.logical_and(x_minus[1] > 0, current_mode==1)
    def takeoff_true_fun(args):
        x_plus = x_minus
        new_mode = 0
        return x_plus, new_mode
    
    def takeoff_false_fun(args):
        x_minus, current_mode = args
        return x_minus, current_mode
    
    args = (x_minus, current_mode)
    x_plus, new_mode = jax.lax.cond(takeoff_cond, takeoff_true_fun, takeoff_false_fun, args)
        
    return x_plus, new_mode

    
def reset_map_bouncing_12(t, x_minus, current_mode):
    e2 = 0.6
    new_mode = current_mode
    x_plus = x_minus
    if (x_minus[1] < 0) and (current_mode==0):
        coeff = jnp.array([[1.0, 0], [0, -e2]])
        x_plus = coeff@x_minus
        new_mode = 1
        
    return x_plus, new_mode

resetmap_bouncing_jit = jax.jit(reset_map_bouncing_12)


Rt_bouncing_12 = jax.jit(jacfwd(lambda t, x, current_mode: reset_map_bouncing_12_jax(t, x, current_mode), 0))
Rx_bouncing_12 = jax.jit(jacfwd(lambda t, x, current_mode: reset_map_bouncing_12_jax(t, x, current_mode), 1))

Rt_bouncing_21 = jax.jit(jacfwd(lambda t, x, current_mode: reset_map_bouncing_21_jax(t, x, current_mode), 0))
Rx_bouncing_21 = jax.jit(jacfwd(lambda t, x, current_mode: reset_map_bouncing_21_jax(t, x, current_mode), 1))

gt_bouncing_12 = jax.jit(jacfwd(lambda t, x: guard_bouncing_12(t, x), 0))
gx_bouncing_12 = jax.jit(jacfwd(lambda t, x: guard_bouncing_12(t, x), 1))

gt_bouncing_21 = jax.jit(jacfwd(lambda t, x: guard_bouncing_21(t, x), 0))
gx_bouncing_21 = jax.jit(jacfwd(lambda t, x: guard_bouncing_21(t, x), 1))


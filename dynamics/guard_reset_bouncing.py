## 1 dimensional bouncing ball dynamics
# x = [z, \dot z]
import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(root_dir)

from dynamics.saltation_matrix import *
import numpy as np
import jax.numpy as jnp

import jax
from jax import jacfwd 


m = 1

def linearize_bouncing(x):
    """
    Linearization.
    Args:
        x: state

    Returns:
        A, B: linear system matrices.
    """
    
    return np.array([[0, 1.0], [0.0, 0.0]], dtype=np.float64), np.array([0, 1.0/m])

@jax.jit
def guard_bouncing_12(t, x):
    return x[0]

@jax.jit
def guard_bouncing_21(t, x):
    return x[1]

def reset_map_control_12(t, u_minus):
    return u_minus

def reset_map_control_21(t, u_minus):
    return u_minus

def reset_map_bouncing_21(t, x_minus, current_mode, args_reset):
    x_plus = x_minus
    if (x_minus[1] < 0) and (current_mode==1):
        x_plus = x_minus
        current_mode = 0
    return x_plus, current_mode, args_reset

def reset_map_bouncing_21_jax(t, x_minus, current_mode, args_reset):
    bouncing_cond = jax.numpy.logical_and(x_minus[1] < 0, current_mode==1)
    def bouncing_true_fun(args):
        x_plus = x_minus
        new_mode = 0
        return x_plus, new_mode
    
    def bouncing_false_fun(args):
        x_minus, current_mode = args
        return x_minus, current_mode
    
    args = (x_minus, current_mode)
    x_plus, new_mode = jax.lax.cond(bouncing_cond, bouncing_true_fun, bouncing_false_fun, args)
        
    return x_plus, new_mode, args_reset

@jax.jit
def reset_map_bouncing_12_jax(t, x_minus, current_mode, args_reset):
    # coeff = jnp.array([[1.0, 0.0], [0.0, -0.6]], dtype=jnp.float64)
    x_plus = jnp.array([x_minus[0], -0.6*x_minus[1]], dtype=jnp.float64)
    new_mode = 1
    return x_plus, new_mode, args_reset
        
def reset_map_bouncing_12(t, x_minus, current_mode, args_reset):
    x_plus = np.array([x_minus[0], -0.6*x_minus[1]], dtype=np.float64)
    new_mode = 1
                
    return x_plus, new_mode, args_reset


# def reset_map_bouncing_12_jax(t, x_minus, current_mode, args_reset):
#     bouncing_cond = jax.numpy.logical_and(x_minus[1] < 0, current_mode==0)
#     def bouncing_true_fun(args):
#         x_minus, current_mode = args
#         e2 = 0.6
#         coeff = np.array([[1.0, 0], [0, -e2]])
#         x_plus = coeff@x_minus
#         new_mode = 1
#         return x_plus, new_mode
    
#     def bouncing_false_fun(args):
#         x_minus, current_mode = args
#         return x_minus, current_mode
#     args = (x_minus, current_mode)
#     x_plus, new_mode = jax.lax.cond(bouncing_cond, bouncing_true_fun, bouncing_false_fun, args)
        
#     return x_plus, new_mode, args_reset
    
# def reset_map_bouncing_12(t, x_minus, current_mode, args_reset):
#     e2 = 0.6
#     new_mode = current_mode
#     x_plus = x_minus
#     if (x_minus[1] < 0) and (current_mode==0):
#         coeff = np.array([[1.0, 0], [0, -e2]])
#         x_plus = coeff@x_minus
#         new_mode = 1
        
#     return x_plus, new_mode, args_reset

# resetmap_bouncing_jit = jax.jit(reset_map_bouncing_12)


Rt_bouncing_12 = jax.jit(jacfwd(lambda t, x, current_mode, args: reset_map_bouncing_12_jax(t, x, current_mode, args), 0))
Rx_bouncing_12 = jax.jit(jacfwd(lambda t, x, current_mode, args: reset_map_bouncing_12_jax(t, x, current_mode, args), 1))

Rt_bouncing_21 = jax.jit(jacfwd(lambda t, x, current_mode, args: reset_map_bouncing_21_jax(t, x, current_mode, args), 0))
Rx_bouncing_21 = jax.jit(jacfwd(lambda t, x, current_mode, args: reset_map_bouncing_21_jax(t, x, current_mode, args), 1))

gt_bouncing_12 = jax.jit(jacfwd(lambda t, x: guard_bouncing_12(t, x), 0))
gx_bouncing_12 = jax.jit(jacfwd(lambda t, x: guard_bouncing_12(t, x), 1))

gt_bouncing_21 = jax.jit(jacfwd(lambda t, x: guard_bouncing_21(t, x), 0))
gx_bouncing_21 = jax.jit(jacfwd(lambda t, x: guard_bouncing_21(t, x), 1))

guard_bouncing_12.terminal=True
guard_bouncing_12.direction=-1

guard_bouncing_21.terminal=True
guard_bouncing_21.direction=1
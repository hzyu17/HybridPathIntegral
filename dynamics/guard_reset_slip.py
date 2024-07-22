## 2-dimensional SLIP dynamics
# mode 1: x = [px, vx, pz, vz, theta], u = [theta_dot]
# mode 2: x = [theta, theta_dot, r, r_dot], u = [r_delta, \tau_hip]
# reset maps: identity

import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(root_dir)

from hybrid_ilqr.saltation_matrix import *
import numpy as np
import jax.numpy as jnp

import jax
from jax import grad, jacfwd 


def mode_change_maps(current_mode):
    new_mode = current_mode
    if (current_mode == 0):
        new_mode = 1
    elif (current_mode == 1):
        new_mode = 0
    return new_mode

# =========================================================
# The guard and reset map from flight mode to stance mode
# =========================================================

# xf: [x, x_dot, z, z_dot, theta]
# guard_12 = z - r0*sin(theta) = 0
def guard_slip_12(t, x):
    r0 = 1    
    return  r0*jnp.sin(x[4]) - x[2]
guard_slip_12_jit = jax.jit(guard_slip_12)

# reset map from flight mode to stance mode
def reset_map_slip_12(t, x_event, current_mode):
    # x_event: [x, x_dot, z, z_dot, theta]
    new_mode = current_mode
    x, xdot, z, zdot, theta = x_event
    theta_dot = 0
    x_reset = x_event
    r0 = 1
    args = (x_event, current_mode)
    
    stance_cond = jax.numpy.logical_and(z-r0*jnp.sin(theta) < 0, current_mode==0)
    
    def stance_true_fun(args):
        x_event, current_mode = args
        r = z*jnp.sin(theta)
        r_dot = -xdot*jnp.cos(theta) + zdot*jnp.sin(theta)
        x_reset = jnp.array([theta, theta_dot, r, r_dot])
        return x_reset
    
    def stance_false_fun(args):
        r = r0
        r_dot = -xdot*jnp.cos(theta) + zdot*jnp.sin(theta)
        x_reset = jnp.array([theta, theta_dot, r, r_dot])
        return x_reset
    
    args = (x_event, current_mode)
    x_reset = jax.lax.cond(stance_cond, stance_true_fun, stance_false_fun, args)
    
    return x_reset, 1

resetmap_slip_jit = jax.jit(reset_map_slip_12)

# =========================================================
# The guard and reset map from stance mode to flight mode
# =========================================================
# xs: [theta, theta_dot, r, r_dot]
# guard_21 = r - r0
def guard_slip_21(t, x):
    r0 = 1
    return x[2] - r0

def reset_control_slip_21(t, u_minus):
    return np.zeros(1)

def reset_control_slip_12(t, u_minus):
    return np.zeros(2)
    
# reset map from stance mode to flight mode
def reset_map_slip_21(t, x_event, current_mode):
    r0 = 1
    x_reset = x_event
    new_mode = current_mode
    theta, theta_dot, r, r_dot = x_event
    
    # TODO: add pxs as a constant
    
    # takeoff_cond = jax.numpy.logical_and(r>r0, current_mode==1)
    # def takeoff_true_fun(args):
    #     px_reset = r0*jnp.cos(theta)
    #     vx_reset = r_dot*jnp.cos(theta) - r*theta_dot*jnp.sin(theta)
    #     pz_reset = r0*jnp.sin(theta)
    #     vz_reset = r0*theta_dot*jnp.cos(theta) + r_dot*jnp.sin(theta)
    #     theta_reset = theta
        
    #     x_reset = jnp.array([px_reset, vx_reset, pz_reset, vz_reset, theta_reset])
    #     return x_reset
    
    # def takeoff_false_fun(args):
    #     px_reset = r0*jnp.cos(theta)
    #     vx_reset = r_dot*jnp.cos(theta) - r*theta_dot*jnp.sin(theta)
    #     pz_reset = r0*jnp.sin(theta)
    #     vz_reset = r0*theta_dot*jnp.cos(theta) + r_dot*jnp.sin(theta)
    #     theat_reset = theta
        
    #     x_reset = jnp.array([px_reset, vx_reset, pz_reset, vz_reset, theat_reset])
    #     return x_reset
    
    # args = (x_event, current_mode)
    # x_reset = jax.lax.cond(takeoff_cond, takeoff_true_fun, takeoff_false_fun, args)
    
    px_reset = r0*jnp.cos(theta)
    vx_reset = r_dot*jnp.cos(theta) - r*theta_dot*jnp.sin(theta)
    pz_reset = r0*jnp.sin(theta)
    vz_reset = r0*theta_dot*jnp.cos(theta) + r_dot*jnp.sin(theta)
    theta_reset = theta

    x_reset = jnp.array([px_reset, vx_reset, pz_reset, vz_reset, theta_reset])
        
    return x_reset, 0


# Define derivatives
Rt_slip_12 = jax.jit(jacfwd(lambda t, x, current_mode: reset_map_slip_12(t, x, current_mode), 0))

Rx_slip_12 = jax.jit(jacfwd(lambda t, x, current_mode: reset_map_slip_12(t, x, current_mode), 1))

Rt_slip_21 = jax.jit(jacfwd(lambda t, x, current_mode: reset_map_slip_21(t, x, current_mode), 0))

Rx_slip_21 = jax.jit(jacfwd(lambda t, x, current_mode: reset_map_slip_21(t, x, current_mode), 1))

gt_slip_12 = jax.jit(grad(lambda t, x: guard_slip_12(t, x), 0))
    
gx_slip_12 = jax.jit(grad(lambda t, x: guard_slip_12(t, x), 1))

gx_slip_21 = jax.jit(grad(lambda t, x: guard_slip_21(t, x), 1))
    
gt_slip_21 = jax.jit(grad(lambda t, x: guard_slip_21(t, x), 0))
  
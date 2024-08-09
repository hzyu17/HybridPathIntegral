import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

import jax
import jax.numpy as jnp
from dynamics.dynamics_slip import *
from dynamics.dynamics_discrete import *


def reset_map_slip_21_padding(t, x_event, current_mode, args_reset):
    
    print("Reset map from mode 2 to mode 1, with padding.")
    
    xp = args_reset[0]
    r0 = 1
    theta, theta_dot, r, r_dot = x_event[0], x_event[1], x_event[2], x_event[3]
    
    px_reset = xp + r0*jnp.cos(theta)
    vx_reset = r_dot*jnp.cos(theta) - r*theta_dot*jnp.sin(theta)
    pz_reset = r0*jnp.sin(theta)
    vz_reset = r0*theta_dot*jnp.cos(theta) + r_dot*jnp.sin(theta)
    theta_reset = theta

    x_reset = jnp.array([px_reset, vx_reset, pz_reset, vz_reset, theta_reset])

    return x_reset, 0, args_reset

# @jax.jit
def stochastic_integration_euler_SLIP_padding(mode, x0, u, dt, eps, dW):   
    def mode0_dynamics_true_func_slip_padding(args):
        (x0, u, dW, eps) = args
        # flight mode
        # [x, x_dot, z, z_dot, theta] = x0
        
        B = jnp.array([[0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [1.0, 0.0]], dtype=jnp.float64)
        
        return x0 + jnp.array([x0[1], 0, x0[3], -9.81, u[0]], dtype=jnp.float64) * dt + jnp.sqrt(eps) * B@dW
    
    def mode0_dynamics_false_func_slip_padding(args):
        (x0, u, dW, eps) = args
        # stance mode
        # [theta, theta_dot, r, r_dot] = x0
        
        g = 9.81
        k = 25.0
        m = 0.5
        r0 = 1
        
        theta, theta_dot, r, r_dot = x0[0], x0[1], x0[2], x0[3]
        
        # Defining the stance dynamics of the system
        B = jnp.array([[0.0, 0.0], 
                        [0.0, 0.0], 
                        [0.0, 1/m/r/r], 
                        [k/m, 0.0],
                        [0.0, 0.0]], dtype=jnp.float64)
    
        xt_next = x0 + jnp.array([theta_dot, 
                                -2*theta_dot*r_dot/r-g*jnp.cos(theta)/r, 
                                r_dot + u[1]/m/r/r, 
                                k/m*(r0-r) - g*jnp.sin(theta) + theta_dot*theta_dot*r + k*u[0]/m,
                                0.0], dtype=jnp.float64) * dt + jnp.sqrt(eps) * B@dW
        
        return xt_next
    
    cond_mode0 = (mode == 0)
    args_choose_dynamics = (x0, u, dW, eps)
    xt_next = jax.lax.cond(cond_mode0, 
                           mode0_dynamics_true_func_slip_padding, 
                           mode0_dynamics_false_func_slip_padding, args_choose_dynamics)

    return xt_next


# ===============================================================================================================
#                                        SLIP Guard condition handling 
# ===============================================================================================================
@jax.jit
def guard_condition_slip_padding(xt, xt_next, current_mode):
    # assume time invariant guard for now
    return jnp.logical_and(current_mode==1, jnp.logical_and(guard_slip_21(0.0,xt)<0, guard_slip_21(0.0,xt_next)>=0))

@jax.jit
def guard_true_func_slip_padding(args):
    print("slip_cond: True")
    (xt_current, current_mode, u, t, xt_next, dt_int, dt_shrinkrate, RandN, eps, reset_arg) = args
    
    def while_cond(vars):
        (_, _, _, _, _, _, _, _, _, can_continue) = vars
        return can_continue
    
    def while_loop_body(vars):
        (xt, _, u, t, dt_int, dt_shrinkrate, RandN, eps, cnt_shrink, _) = vars
        
        # Too far from the guard, shrink the step size.
        dt_shrink = dt_int * dt_shrinkrate
        dW_new = jnp.sqrt(dt_shrink)*RandN
        
        xt_shrinked = stochastic_integration_euler_SLIP_padding(current_mode, xt, u, dt_shrink, eps, dW_new)
        
        new_condition = jnp.logical_and(guard_slip_21(t, xt_shrinked) >= 0, cnt_shrink<50)
        cnt_shrink += 1
        
        new_vars = (xt, xt_shrinked, u, t, dt_shrink, dt_shrinkrate, RandN, eps, cnt_shrink, new_condition)
        
        return new_vars
    
    # --------------------------
    #       Shrinking dt
    # --------------------------
    # init_vars = (xt_current, xt_next, u, t, dt_int, dt_shrinkrate, RandN, eps, 0, True)
    # final_vars = jax.lax.while_loop(while_cond, while_loop_body, init_val=init_vars)
    
    # # (_, xt_shrinked, _, _, dt_shrinked, _, RandN, _, _, _) = final_vars
    # t_event = t + dt_shrinked
    # x_event = xt_shrinked
    # xt_reset, next_mode, new_reset_arg = reset_map_slip_21_padding(t_event, xt_shrinked, current_mode, reset_arg)
    # dW_new = jnp.sqrt(dt_shrinked)*RandN
    
    # --------------------------
    #     Direct reset map
    # --------------------------
    t_event = t + dt_int
    x_event = xt_next
    xt_reset, next_mode, new_reset_arg = reset_map_slip_21_padding(t, xt_next, current_mode, reset_arg)
    dW_new = jnp.sqrt(dt_int)*RandN
    
    return t_event, x_event, xt_reset, next_mode, dW_new, new_reset_arg


@jax.jit
def guard_false_func_slip_padding(args):
    (_, current_mode, _, t, xt_next, dt_int, _, RandN, eps, reset_arg) = args
    
    t_next = t + dt_int
    dW = jnp.sqrt(dt_int)*RandN
    
    return t_next, xt_next, xt_next, current_mode, dW, reset_arg

# ===============================================================================================================
#                                       // End of SLIP guard condition handling //
# ===============================================================================================================

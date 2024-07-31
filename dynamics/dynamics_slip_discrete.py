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

# @jax.jit
def stochastic_integration_euler_SLIP(mode, x0, u, dt, eps, dW, padding=True):   
    def mode0_dynamics_true_func_slip(args):
        (x0, u, dW, eps) = args
        # flight mode
        # [x, x_dot, z, z_dot, theta] = x0
        
        if padding:
            B = jnp.array([[0.0, 0.0],
                            [0.0, 0.0],
                            [0.0, 0.0],
                            [0.0, 0.0],
                            [1.0, 0.0]], dtype=jnp.float64)
        else:
            B = jnp.array([[0.0],
                           [0.0],
                           [0.0],
                           [0.0],
                           [1.0]], dtype=jnp.float64)
        
        return x0 + jnp.array([x0[1], 0, x0[3], -9.81, u[0]], dtype=jnp.float64) * dt + jnp.sqrt(eps) * B@dW
    
    def mode0_dynamics_false_func_slip(args):
        (x0, u, dW, eps) = args
        # stance mode
        # [theta, theta_dot, r, r_dot] = x0
        
        g = 9.81
        k = 25.0
        m = 0.5
        r0 = 1
        
        theta, theta_dot, r, r_dot = x0[0], x0[1], x0[2], x0[3]
        
        if padding:
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
            
        else:
            # Defining the stance dynamics of the system
            B = jnp.array([[0.0, 0.0], 
                            [0.0, 0.0], 
                            [0.0, 1/m/r/r], 
                            [k/m, 0.0]], dtype=jnp.float64)
        
            xt_next = x0 + jnp.array([theta_dot, 
                                    -2*theta_dot*r_dot/r-g*jnp.cos(theta)/r, 
                                    r_dot + u[1]/m/r/r, 
                                    k/m*(r0-r) - g*jnp.sin(theta) + theta_dot*theta_dot*r + k*u[0]/m], dtype=jnp.float64) * dt + jnp.sqrt(eps) * B@dW
        
        
        return xt_next
    
    cond_mode0 = (mode == 0)
    args_choose_dynamics = (x0, u, dW, eps)
    xt_next = jax.lax.cond(cond_mode0, mode0_dynamics_true_func_slip, mode0_dynamics_false_func_slip, args_choose_dynamics)

    return xt_next


# ===============================================================================================================
#                                        SLIP Guard condition handling 
# ===============================================================================================================
@jax.jit
def event_condition_slip(xt, xt_next, current_mode):
    # assume time invariant guard for now
    return jnp.logical_and(current_mode==1, jnp.logical_and(guard_slip_21(0.0,xt)<0, guard_slip_21(0.0,xt_next)>=0))

@jax.jit
def event_true_func_slip(args):
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
        
        xt_shrinked = stochastic_integration_euler_SLIP(current_mode, xt, u, dt_shrink, eps, dW_new)
        
        new_condition = jnp.logical_not(jnp.logical_or(guard_slip_21(t, xt_shrinked)<0, cnt_shrink==10))
        cnt_shrink += 1
        
        new_vars = (xt, xt_shrinked, u, t, dt_shrink, dt_shrinkrate, RandN, eps, cnt_shrink, new_condition)
        
        return new_vars
    
    init_vars = (xt_current, xt_next, u, t, dt_int, dt_shrinkrate, RandN, eps, 0, True)
    final_vars = jax.lax.while_loop(while_cond, while_loop_body, init_val=init_vars)
    
    (_, xt_shrinked, _, _, dt_shrinked, _, RandN, _, _, _) = final_vars
    xt_next, next_mode, new_reset_arg = reset_map_slip_21_padding(t, xt_shrinked, current_mode, reset_arg)
    dW_new = jnp.sqrt(dt_shrinked)*RandN
    
    return xt_next, next_mode, dW_new, new_reset_arg


@jax.jit
def event_false_func_slip(args):
    (_, current_mode, _, _, xt_next, dt_int, _, RandN, _, reset_arg) = args
    dW = jnp.sqrt(dt_int)*RandN
    return xt_next, current_mode, dW, reset_arg

# ===============================================================================================================
#                                       // End of SLIP guard condition handling //
# ===============================================================================================================

from functools import partial

hybrid_integration_slip = partial(hybrid_integration_euler, 
                                  stochastic_integration_euler_func = stochastic_integration_euler_SLIP, 
                                  event_condition_func = event_condition_slip, 
                                  event_condition_true_fun = event_true_func_slip, 
                                  event_condition_false_fun = event_false_func_slip)
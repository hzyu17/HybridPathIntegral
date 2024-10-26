import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from dynamics.dynamics_bouncing import *
from dynamics.dynamics_discrete import *


def stoch_integr_bouncing_JAX(mode, x0, u, dt, eps, dW):
    B = jnp.array([[0],[1.0]], dtype=jnp.float64)
    xt_next = x0 + jnp.array([x0[1], u[0]-9.81], dtype=jnp.float64) * dt + jnp.sqrt(eps) * B@dW
    return xt_next


# ===============================================================================================================
#                                        Bouncing condition handling 
# ===============================================================================================================
# --------------------------------------------------
#               From mode 1 to mode 2
# --------------------------------------------------
# @jax.jit
def guard_bouncing_12_JAX(xt, xt_next, current_mode):
    # assume time invariant guard for now
    return jnp.logical_and(current_mode==0, jnp.logical_and(guard_bouncing_12(0.0,xt)>0, guard_bouncing_12(0.0,xt_next)<=0))

# @jax.jit
def guard_true_bouncing_12_JAX(args):
    (xt_current, current_mode, u, t, xt_next, dt_int, dt_shrinkrate, RandN, eps, reset_arg) = args
    
    def while_cond(vars):
        (_, _, _, _, _, _, _, _, _, _, can_continue) = vars
        return can_continue
    
    def while_loop_body(vars):
        (xt_current, xt_shrinked, u, t, dt_int, dt_shrinkrate, RandN, eps, cnt_shrink, reset_arg, _) = vars
        # Too far from the guard, shrink the step size.
        dt_shrinked = dt_int * dt_shrinkrate
        dW_shrinked = jnp.sqrt(dt_shrinked)*RandN
        
        xt_shrinked = stoch_integr_bouncing_JAX(current_mode, xt_current, u, dt_shrinked, eps, dW_shrinked)
        
        new_condition = jnp.logical_and(guard_bouncing_12_JAX(xt_current, xt_shrinked, current_mode), cnt_shrink<20)
        cnt_shrink += 1
        
        new_vars = (xt_current, xt_shrinked, u, t, dt_shrinked, dt_shrinkrate, RandN, eps, cnt_shrink, reset_arg, new_condition)
        
        return new_vars
    
    init_vars = (xt_current, xt_next, u, t, dt_int, dt_shrinkrate, RandN, eps, 0, reset_arg, True)
    
    final_vars = jax.lax.while_loop(while_cond, while_loop_body, init_val=init_vars)
    (xt_current, xt_shrink_final, _, t, dt_final, dt_shrinkrate, RandN, eps, final_cnt, reset_arg, _) = final_vars
    
    t_event = t+dt_final
    xt_reset, next_mode, new_reset_arg = reset_map_bouncing_12_jax(t_event, xt_shrink_final, current_mode, reset_arg)
    
    dW_new = jnp.sqrt(dt_final)*RandN
    
    return t_event, xt_shrink_final, xt_reset, next_mode, dW_new, new_reset_arg


# @jax.jit
def guard_false_bouncing_12_JAX(args):
    (xt_current, current_mode, u, t, xt_next, dt_int, dt_shrinkrate, RandN, eps, reset_arg) = args
    t_event = t+dt_int
    dW = jnp.sqrt(dt_int)*RandN
    return t_event, xt_next, xt_next, current_mode, dW, reset_arg


# --------------------------------------------------
#               From mode 2 to mode 1
# --------------------------------------------------
def guard_bouncing_21_JAX(xt, xt_next, current_mode):
    # assume time invariant guard for now
    return jnp.logical_and(current_mode==1, jnp.logical_and(guard_bouncing_21(0.0,xt)>0, guard_bouncing_21(0.0,xt_next)<=0))

# @jax.jit
def guard_true_bouncing_21_JAX(args):
    (xt_current, current_mode, u, t, xt_next, dt_int, dt_shrinkrate, RandN, eps, reset_arg) = args
    
    def while_cond(vars):
        (_, _, _, _, _, _, _, _, _, _, can_continue) = vars
        return can_continue
    
    def while_loop_body(vars):
        (xt_current, xt_shrinked, u, t, dt_int, dt_shrinkrate, RandN, eps, cnt_shrink, reset_arg, _) = vars
        # Too far from the guard, shrink the step size.
        dt_shrinked = dt_int * dt_shrinkrate
        dW_shrinked = jnp.sqrt(dt_shrinked)*RandN
        
        xt_shrinked = stoch_integr_bouncing_JAX(current_mode, xt_current, u, dt_shrinked, eps, dW_shrinked)
        
        new_condition = jnp.logical_and(guard_bouncing_21_JAX(xt_current, xt_shrinked, current_mode), cnt_shrink<20)
        cnt_shrink += 1
        
        new_vars = (xt_current, xt_shrinked, u, t, dt_shrinked, dt_shrinkrate, RandN, eps, cnt_shrink, reset_arg, new_condition)
        
        return new_vars
    
    init_vars = (xt_current, xt_next, u, t, dt_int, dt_shrinkrate, RandN, eps, 0, reset_arg, True)
    
    final_vars = jax.lax.while_loop(while_cond, while_loop_body, init_val=init_vars)
    (xt_current, xt_shrink_final, _, t, dt_final, dt_shrinkrate, RandN, eps, _, reset_arg, _) = final_vars
    
    t_event = t+dt_final
    xt_reset, next_mode, new_reset_arg = reset_map_bouncing_21_jax(t_event, xt_shrink_final, current_mode, reset_arg)
    
    dW_new = jnp.sqrt(dt_final)*RandN
    
    return t_event, xt_shrink_final, xt_reset, next_mode, dW_new, new_reset_arg


# @jax.jit
def guard_false_bouncing_21_JAX(args):
    (xt_current, current_mode, u, t, xt_next, dt_int, dt_shrinkrate, RandN, eps, reset_arg) = args
    t_event = t+dt_int
    dW = jnp.sqrt(dt_int)*RandN
    return t_event, xt_next, xt_next, current_mode, dW, reset_arg

# ===============================================================================================================
#                                      // End of Bouncing condition handling //
# ===============================================================================================================

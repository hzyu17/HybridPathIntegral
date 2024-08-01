import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from dynamics.dynamics_bouncing import *
from dynamics.dynamics_discrete import *

def stochastic_integration_euler_bouncing(mode, x0, u, dt, eps, dW):
    B = jnp.array([[0],[1.0]], dtype=jnp.float64)
    xt_next = x0 + jnp.array([x0[1], u[0]-9.81], dtype=jnp.float64) * dt + jnp.sqrt(eps) * B@dW
    return xt_next


# ===============================================================================================================
#                                        Bouncing condition handling 
# ===============================================================================================================
@jax.jit
def guard_condition_bouncing_JAX(xt, xt_next, current_mode):
    # assume time invariant guard for now
    return jnp.logical_and(current_mode==0, jnp.logical_and(guard_bouncing_12_jit(0.0,xt)>0, guard_bouncing_12_jit(0.0,xt_next)<=0))

@jax.jit
def guard_true_func_bouncing_JAX(args):
    print("bouncing_cond: True")
    (xt_current, current_mode, u, t, xt_next, dt_int, dt_shr, RandN, eps, reset_arg) = args
    
    def while_cond(vars):
        (_, _, _, _, _, _, _, _, _, _, can_continue) = vars
        return can_continue
    
    def while_loop_body(vars):
        (xt_current, xt_swch, u, t, dt_int, dt_shr, RandN, eps, cnt_shrink, reset_arg, _) = vars
        
        # Too far from the guard, shrink the step size.
        dt_int = dt_int * dt_shr
        dW_new = jnp.sqrt(dt_int)*RandN
        
        xt_swch = stochastic_integration_euler_bouncing(current_mode, xt_current, u, dt_int, eps, dW_new)
        
        new_condition = jnp.logical_or(guard_bouncing_12(t, xt_swch)<=0, cnt_shrink<10)
        cnt_shrink += 1
        
        new_vars = (xt_current, xt_swch, u, t, dt_int, dt_shr, RandN, eps, cnt_shrink, reset_arg, new_condition)
        
        return new_vars
    
    init_condition = True
    init_vars = (xt_current, xt_next, u, t, dt_int, dt_shr, RandN, eps, 0, reset_arg, init_condition)
    final_vars = jax.lax.while_loop(while_cond, while_loop_body, init_val=init_vars)
    
    (xt_current, xt_shrink_final, u_final, t, dt_final, dt_shr, RandN, eps, _, reset_arg, _) = final_vars
    xt_reset, next_mode, reset_arg = reset_map_bouncing_12_jax(t, xt_shrink_final, current_mode, reset_arg)
    
    dW_new = jnp.sqrt(dt_final)*RandN
    
    # dt_remain = dt_int - dt_final
    # dW_remain = jnp.sqrt(dt_remain)*RandN
    # u_reset = reset_map_control_12(t, u_final)
    # xt_final = stochastic_integration_euler_bouncing(next_mode, xt_reset, u_reset, dt_remain, eps, dW_remain)
    
    return xt_reset, next_mode, dW_new, reset_arg


@jax.jit
def guard_false_func_bouncing_JAX(args):
    (_, current_mode, _, _, xt_next, dt_int, _, RandN, _, reset_arg) = args
    dW = jnp.sqrt(dt_int)*RandN
    return xt_next, current_mode, dW, reset_arg

# ===============================================================================================================
#                                      // End of Bouncing condition handling //
# ===============================================================================================================

from functools import partial

hybrid_stochastic_integration_bouncing_JAX = partial(hybrid_stochastic_integration_euler_JAX, 
                                                    stochastic_integration_euler_func = stochastic_integration_euler_bouncing, 
                                                    guard_condition_func = guard_condition_bouncing_JAX, 
                                                    guard_condition_true_fun = guard_true_func_bouncing_JAX, 
                                                    guard_condition_false_fun = guard_false_func_bouncing_JAX)
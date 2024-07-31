import jax
import jax.numpy as jnp

# @jax.jit
def hybrid_integration_euler(xt, current_mode, ut, randN, eps, dt, dt_shrink, t0, reset_arg, 
                            stochastic_integration_euler_func,
                            event_condition_func,
                            event_condition_true_fun,
                            event_condition_false_fun):
    
    dW = jnp.sqrt(dt)*randN
    xt_next = stochastic_integration_euler_func(current_mode, xt, ut, dt, eps, dW)
    
    # ------------
    # change mode
    # ------------
    # next_mode = current_mode
    args_guard = (xt, current_mode, ut, t0, xt_next, dt, dt_shrink, randN, eps, reset_arg)

    guard_hit = event_condition_func(xt, xt_next, current_mode)
    xt_next, next_mode, dW, new_reset_arg = jax.lax.cond(guard_hit, event_condition_true_fun, event_condition_false_fun, args_guard)

    return xt_next, next_mode, dW, new_reset_arg
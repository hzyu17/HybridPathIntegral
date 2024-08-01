import jax
import jax.numpy as jnp
import numpy as np
from dynamics.dynamics import extract_extensions

# @jax.jit
def hybrid_stochastic_integration_euler_JAX(xt, current_mode, ut, 
                                            randN, eps, 
                                            dt, dt_shrink, t0, reset_arg, 
                                            stochastic_integration_euler_func=None,
                                            guard_condition_func=None,
                                            guard_condition_true_fun=None,
                                            guard_condition_false_fun=None):
    
    dW = jnp.sqrt(dt)*randN
    xt_next = stochastic_integration_euler_func(current_mode, xt, ut, dt, eps, dW)
    
    # ------------
    # change mode
    # ------------
    # next_mode = current_mode
    args_guard = (xt, current_mode, ut, t0, xt_next, dt, dt_shrink, randN, eps, reset_arg)

    guard_hit = guard_condition_func(xt, xt_next, current_mode)
    xt_next, next_mode, dW, new_reset_arg = jax.lax.cond(guard_hit, guard_condition_true_fun, guard_condition_false_fun, args_guard)

    return xt_next, next_mode, dW, new_reset_arg



def hybrid_stochastic_integration_euler(xt, current_mode, ut, 
                                        randN, eps, 
                                        dt, dt_shrink, t0, reset_arg, 
                                        stochastic_integration_euler_func=None,
                                        guard_condition_func=None,
                                        guard_condition_true_fun=None,
                                        guard_condition_false_fun=None):
    dW = np.sqrt(dt) * randN
    
    xt_next = stochastic_integration_euler_func(current_mode, xt, ut, dt, eps, dW)
    
    args_guard = (xt, current_mode, ut, t0, xt_next, dt, dt_shrink, randN, eps, reset_arg)
    
    guard_hit = guard_condition_func(xt, xt_next, current_mode)
    
    if guard_hit:
        xt_next, next_mode, dW, new_reset_arg = guard_condition_true_fun(args_guard)
    else:
        xt_next, next_mode, dW, new_reset_arg = guard_condition_false_fun(args_guard)

    return xt_next, next_mode, dW, new_reset_arg



def hybrid_stochastic_feedback_rollout_discrete(init_mode, x0, n_inputs, xt_ref, ref_modes, 
                                                ut, Kt, kt, target_state, Q_T, t0, tf, 
                                                epsilon, GaussianNoise, dt_shrinkingrate, 
                                                reference_extension_helper, init_reset_args,
                                                cond_mode_mismatch_func=None,
                                                reaction_mode_mismatch_func=None,
                                                hybrid_stochastic_integration_func=None):

    (v_event_modechange, v_ref_ext_bwd, v_ref_ext_fwd, 
    v_Kfb_ref_ext_bwd, v_Kfb_ref_ext_fwd, 
    v_kff_ref_ext_bwd, v_kff_ref_ext_fwd, _) = extract_extensions(reference_extension_helper, start_index = 0)
    
    n_timestamps = len(xt_ref)
    
    dt = (tf - t0) / n_timestamps
    dt_int = dt
    
    # returning trajectory    
    xt_trj = [np.array([0.0]) for _ in range(n_timestamps)]
    xt_trj[0] = x0  
    
    mode_trj = np.zeros((n_timestamps), dtype=np.int64) 
    mode_trj[0] = init_mode
    
    # closed-loop controls 
    ut_cl_trj = [np.zeros((n_timestamps, n_inputs[0])), np.zeros((n_timestamps, n_inputs[1]))]
    
    cnt_mismatch = 0
    xt_ref_actual = [np.array([0.0]) for _ in range(n_timestamps)]
    
    # path cost
    Sk = 0
    
    # hybrid event related 
    cnt_event = 0
    reset_args = init_reset_args
    event_args = [init_reset_args[0]]
    
    # -------------- roullout function --------------
    for ii_t in range(n_timestamps-1):   

        t0_i = t0 + ii_t*dt   
        
        current_mode = mode_trj[ii_t]
        xt = xt_trj[ii_t]
        
        ref_current_mode = ref_modes[ii_t]
        reset_args[ii_t] = event_args[cnt_event]
        
        # ======== Handle mode mismatch ========
        K_fb_i = Kt[ii_t]
        k_ff_i = kt[ii_t]
        xref_i = xt_ref[ii_t] 
        
        if cond_mode_mismatch_func(current_mode, ref_current_mode):
            print("mode mismatch")
            xref_i, K_fb_i, k_ff_i, cnt_mismatch = reaction_mode_mismatch_func(ii_t, current_mode, ref_current_mode, 
                                                                                v_ref_ext_fwd[0], v_ref_ext_bwd[0], 
                                                                                v_event_modechange[0],
                                                                                v_Kfb_ref_ext_fwd[0], v_kff_ref_ext_fwd[0],
                                                                                v_Kfb_ref_ext_bwd[0], v_kff_ref_ext_bwd[0],
                                                                                cnt_mismatch)
        
        xt_ref_actual[ii_t] = xref_i
        
        delta_xt_i = xt_trj[ii_t] - xref_i
        current_u = ut[current_mode][ii_t] + K_fb_i@delta_xt_i + k_ff_i
        ut_cl_trj[current_mode][ii_t] = current_u
        
        noise_i = GaussianNoise[current_mode][ii_t]
        dW_i = np.sqrt(dt_int)*noise_i
        
        # ============================== One step integration ==============================        
        xt_next, next_mode, _, new_reset_arg = hybrid_stochastic_integration_func(xt, current_mode, current_u, 
                                                                                    noise_i, epsilon, 
                                                                                    dt, dt_shrinkingrate, t0, reset_args[ii_t])
        
        reset_args[ii_t+1] = new_reset_arg
        
        # ============================== // One step integration // ==============================     
        
        # Collect cost: consider only the terminal state cost for now.
        Sk += current_u.T@current_u/2.0 * dt + np.sqrt(epsilon) * np.dot(current_u.T, dW_i)
        
        # Update trajectories
        xt_trj[ii_t+1] = xt_next
        mode_trj[ii_t+1] = next_mode
    
    xt_ref_actual[-1] = xt_ref[-1]
    
    # Terminal cost
    Sk += (xt-target_state)@Q_T@(xt-target_state) / 2.0
    
    return mode_trj, xt_trj, ut_cl_trj, Sk, xt_ref_actual, reset_args
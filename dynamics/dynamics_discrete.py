import jax
import jax.numpy as jnp
import numpy as np
from dynamics.dynamics import extract_extensions
from dynamics.saltation_matrix import saltation_matrix


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



def guard_true_func_deterministic(args):
    (xt_current, current_mode, u_current, t, xt_next, dt_int, dt_shrinkrate, smooth_dyn, guard, resetmap, reset_arg) = args
    
    def while_loop_body(xt, u, t, dt_int, dt_shrink, cnt_shrink):
        # Too far from the guard, shrink the step size
        dt_shrinked = dt_int * dt_shrink
        
        xt_shrinked = xt + smooth_dyn(t, xt, u) * dt_shrinked
        
        new_condition = True
        if (guard.direction == 1):
            new_condition = ((guard(0.0,xt_shrinked)>0) and (cnt_shrink < 50))
        elif (guard.direction == -1):
            new_condition = ((guard(0.0,xt_shrinked)<=0) and (cnt_shrink < 50))
            
        cnt_shrink += 1
        
        return xt_shrinked, dt_shrinked, cnt_shrink, new_condition
    
    # -------------------
    #   Shrinking step 
    # -------------------
    # cnt_shrink = 0
    # can_continue = True
    # xt_shrinked = xt_next
    
    # # Implementing the loop with Python's while
    # while can_continue:
    #     xt_shrinked, dt_int, cnt_shrink, can_continue = while_loop_body(
    #         xt_current, u_current, t, dt_int, dt_shrinkrate, cnt_shrink
    #     )
    
    # # Execute the reset map after the loop
    # t_event =  t + dt_int
    # x_reset, next_mode, new_reset_arg = resetmap(t, xt_shrinked, current_mode, reset_arg)
    
    # return t_event, xt_shrinked, x_reset, next_mode, new_reset_arg
    
    # -------------------
    #     Bi-section 
    # -------------------
    tol = 1e-8
    t_left = t
    x_left = xt_current
    t_right = t+dt_int
    x_right = xt_next
    
    while (t_right - t_left) / 2.0 > tol:
        t_mid = (t_left + t_right) / 2.0
        t_span = (t_left, t_right)
        dt_new = t_right - t_left
        
        x_mid = x_left + smooth_dyn(t_left, x_left, u_current) * dt_new
        
        if guard(t_mid, x_mid) == 0:
            return (t_mid, x_mid)  # We've found the exact root
        
        elif guard(t_left, x_left) * guard(t_mid, x_mid) < 0:
            t_right = t_mid  # The root is in the left half
            x_right = x_mid
        else:
            t_left = t_mid  # The root is in the right half
            x_left = x_mid
            
    print(f"guard left: {guard(t, xt_current)}, t_right: {guard(t+dt_int, xt_next)}, midpoint: {guard(t_mid, x_mid)}")  # Debug statement
    
    # t_event = (t_left + t_right) / 2.0
    # x_event = (x_left + x_right) / 2.0
    
    t_event =  t_left
    x_event = x_left
    # dt_final = t_event - t_left
    x_reset, next_mode, new_reset_arg = resetmap(t_event, x_event, current_mode, reset_arg)
    
    # # --------------------
    # #   Direct reset map 
    # # --------------------
    # x_reset, next_mode, new_reset_arg = resetmap(t, xt_next, current_mode, reset_arg)
    # t_event =  t + dt_int
    
    return t_event, x_event, x_reset, next_mode, new_reset_arg



def event_detect_onestep_discrete(xt, ut, 
                                  t0, dt, dt_shrink, current_mode, 
                                    smooth_dynamics, 
                                    guards,
                                    gxs, gts,
                                    reset_maps,
                                    Rxs, Rts,
                                    reset_args, 
                                    guard_condition_func=None,
                                    detection=True, backwards=False):
    
    current_dyn = smooth_dynamics[current_mode]
    
    current_guard = guards[current_mode]
    current_resetmap = reset_maps[current_mode]
    
    current_Rx = Rxs[current_mode]
    current_Rt = Rts[current_mode]
    
    current_gx = gxs[current_mode]
    current_gt = gts[current_mode]
    
    current_guard = guards[current_mode]
    current_resetmap = reset_maps[current_mode]
    
    xt_next = None
    x_event = None
    t_event = None
    x_reset = None
    saltation = None
    next_mode = current_mode
    reset_byproduct = (None, )
        
    # smooth dynamics
    if backwards:
        xt_next = xt - current_dyn(t0, xt, ut)*dt
    else:
        xt_next = xt + current_dyn(t0, xt, ut)*dt
    
    # detection
    if detection:
        
        guard_hit = guard_condition_func(xt, xt_next, current_mode)
        
        if guard_hit:
            args_guard = (xt, current_mode, ut, t0, xt_next, dt, dt_shrink, 
                          current_dyn, current_guard, current_resetmap, reset_args)
            t_event, x_event, x_reset, next_mode, reset_byproduct = guard_true_func_deterministic(args_guard)
            
            # ---------- Compute saltation matrix ---------- 
            R_x = current_Rx(t_event, x_event, current_mode, reset_args)[0]
            R_t = current_Rt(t_event, x_event, current_mode, reset_args)[0]
            
            g_x = current_gx(t_event, x_event)
            g_t = current_gt(t_event, x_event)
            
            next_dyn = smooth_dynamics[next_mode]
            
            F_1 = current_dyn(t_event, x_event)
            F_2 = next_dyn(t_event, x_reset) # Important, the F2 is evaluated at the reseted state!
            saltation = saltation_matrix(F_1, F_2, R_t, R_x, g_t, g_x)        
            xt_next = x_reset
        
    mode_mapping = np.array([current_mode, next_mode])
    
    return xt_next, saltation, mode_mapping, t_event, x_event, x_reset, reset_byproduct


def hybrid_stochastic_feedback_rollout_discrete(init_mode, x0, n_inputs, xt_ref, ref_modes, 
                                                ut, Kt, kt, target_state, Q_T, t0, dt, 
                                                epsilon, GaussianNoise, dt_shrinkrate, 
                                                reference_extension_helper, init_reset_args,
                                                cond_mode_mismatch_func=None,
                                                reaction_mode_mismatch_func=None,
                                                hybrid_stochastic_integration_func=None):

    (v_event_modechange, v_ref_ext_bwd, v_ref_ext_fwd, 
    v_Kfb_ref_ext_bwd, v_Kfb_ref_ext_fwd, 
    v_kff_ref_ext_bwd, v_kff_ref_ext_fwd, _) = extract_extensions(reference_extension_helper, start_index = 0)
    
    n_timestamps = len(xt_ref)
    
    # Returning trajectory    
    xt_trj = [np.array([0.0]) for _ in range(n_timestamps)]
    xt_trj[0] = x0  
    
    mode_trj = np.zeros((n_timestamps), dtype=np.int64) 
    mode_trj[0] = init_mode
    
    # Closed-loop controls 
    ut_cl_trj = [np.zeros((n_timestamps, n_inputs[0])), np.zeros((n_timestamps, n_inputs[1]))]
    
    cnt_mismatch = 0
    xt_ref_actual = [np.array([0.0]) for _ in range(n_timestamps)]
    
    # Path cost
    Sk = 0
    
    # Hybrid event related 
    cnt_event = 0
    reset_args = [np.array([0.0]) for _ in range(n_timestamps)]
    event_args = [init_reset_args[0]]
    
    # -------------- roullout function --------------
    for ii_t in range(n_timestamps-1):   
        
        xt = xt_trj[ii_t]
        current_mode = mode_trj[ii_t]
        ref_current_mode = ref_modes[ii_t]
        
        K_fb_i = Kt[ii_t]
        k_ff_i = kt[ii_t]
        xref_i = xt_ref[ii_t] 
        
        reset_args[ii_t] = event_args[cnt_event]
        
        # ======== Handle mode mismatch ========
        if cond_mode_mismatch_func(current_mode, ref_current_mode):
            print("mode mismatch at time: ", ii_t)
            xref_i, K_fb_i, k_ff_i, cnt_mismatch = reaction_mode_mismatch_func(ii_t, current_mode, ref_current_mode, 
                                                                                v_ref_ext_fwd[0], v_ref_ext_bwd[0], 
                                                                                v_event_modechange[0],
                                                                                v_Kfb_ref_ext_fwd[0], v_kff_ref_ext_fwd[0],
                                                                                v_Kfb_ref_ext_bwd[0], v_kff_ref_ext_bwd[0],
                                                                                cnt_mismatch)
        
        xt_ref_actual[ii_t] = xref_i
        delta_xt_i = xt - xref_i
        current_u = ut[current_mode][ii_t] + K_fb_i@delta_xt_i + k_ff_i
        ut_cl_trj[current_mode][ii_t] = current_u
        
        noise_i = GaussianNoise[current_mode][ii_t]
        dW_i = np.sqrt(dt)*noise_i
        
        # ============================== One step integration ==============================        
        xt_next, next_mode, _, new_reset_arg = hybrid_stochastic_integration_func(xt, current_mode, current_u, 
                                                                                    noise_i, epsilon, 
                                                                                    dt, dt_shrinkrate, t0, reset_args[ii_t])
        
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
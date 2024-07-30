import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

# jax import
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
print("Devices:", jax.devices())
print("jax.default_backend()", jax.default_backend())

# ===============================================================================================================
#                                       Mode mismatch condition handling
# ===============================================================================================================

# -----------------------------------
# Mode Mismatch condition handling
# -----------------------------------

def ModeMismatch_true_fun_jax(args):
    
    (current_mode, ref_current_mode, ref_mode_change, ext_trj_fwd, ext_trj_bwd, 
     Kfb_ref_ext_fwd, kff_ref_ext_fwd, Kfb_ref_ext_bwd, kff_ref_ext_bwd, 
     Kfb, kff, xref_i, cnt_indx, cnt_MM)= args
    # TODO: Take the first hybrid event for now. 
    # Need to find the correct corresponding one among all hybrid events.
    
    # # ------------------------------------------------------------ 
    # #                        EArr handling
    # # ------------------------------------------------------------
    
    is_early = cond_EarlyArr_jax(current_mode, ref_current_mode, ref_mode_change)
    
    args_EarlyArrival = (cnt_indx, ext_trj_fwd, ext_trj_bwd, 
                         Kfb_ref_ext_fwd, kff_ref_ext_fwd, 
                         Kfb_ref_ext_bwd, kff_ref_ext_bwd)
    
    xref_i, K_fb_i, k_ff_i = jax.lax.cond(is_early, EarlyArr_true_fun_jax, EarlyArr_false_fun_jax, args_EarlyArrival)
    
    # # ------------------------------------------------------------ 
    # #                     // End of EArr handling
    # # ------------------------------------------------------------
        
    return xref_i, K_fb_i, k_ff_i, cnt_MM+1

def ModeMismatch_false_fun_jax(args):
    (_, _, _, _, _, _, _, _, _, Kfb, kff, xref_i, _, cnt_MM)= args
    
    # Return the original ref and feedback gains
    return xref_i, Kfb, kff, cnt_MM

# ================================================================================================================
#                               Early arrival condition handling function definitions
# ================================================================================================================
    
def cond_EarlyArr_jax(current_mode, ref_current_mode, ref_mode_change):
    # hand picked, need to be automatic
    return jnp.logical_and(current_mode==1, ref_current_mode==0)

def EarlyArr_true_fun_jax(args):
    (cnt_indx, ext_trj_fwd, ext_trj_bwd, Kfb_ref_ext_fwd, kff_ref_ext_fwd, Kfb_ref_ext_bwd, kff_ref_ext_bwd) = args
    return ext_trj_bwd[cnt_indx].flatten(), Kfb_ref_ext_bwd[cnt_indx], kff_ref_ext_bwd[cnt_indx].flatten()

def EarlyArr_false_fun_jax(args):
    (cnt_indx, ext_trj_fwd, ext_trj_bwd, Kfb_ref_ext_fwd, kff_ref_ext_fwd, Kfb_ref_ext_bwd, kff_ref_ext_bwd) = args
    return ext_trj_fwd[cnt_indx].flatten(), Kfb_ref_ext_fwd[cnt_indx], kff_ref_ext_fwd[cnt_indx].flatten()

# ============================ // End of Mode Mismatch condition handling //=======================================


# ===========================================
#    Choose nominal control based on mode 
# ===========================================
@jax.jit
def mode0_condition_jit(current_mode):
    # assume time invariant guard for now
    return current_mode == 0

@jax.jit
def mode0_cond_true_fun_jax(args):
    (xref_0, xref_1, u_0, u_1, K_mode0, k_mode0, K_mode1, k_mode1, randN_0, randN_1) = args
    return xref_0, u_0, K_mode0, k_mode0, randN_0
# mode0_true_fun_jit = jax.jit(mode0_cond_true_fun_jax)

@jax.jit
def mode0_cond_false_fun_jax(args):
    (xref_0, xref_1, u_0, u_1, K_mode0, k_mode0, K_mode1, k_mode1, randN_0, randN_1) = args
    return xref_1, u_1, K_mode1, k_mode1, randN_1
# mode0_false_fun_jit = jax.jit(mode0_cond_false_fun_jax)


def hybrid_integration(xt, current_mode, ut, randN, eps, dt, dt_shrink, t0, reset_arg, 
                       stochastic_integration_euler_func,
                       event_condition_func,
                       event_condition_true_fun,
                       event_condition_false_fun):
    
    dW = jnp.sqrt(dt)*randN
    xt_next = stochastic_integration_euler_func(current_mode, xt, ut, dt, eps, dW)
    
    # ------------
    # change mode
    # ------------
    next_mode = current_mode
    args_guard = (xt, current_mode, ut, t0, xt_next, dt, dt_shrink, randN, eps, reset_arg)
    guard_hit = event_condition_func(xt, xt_next)
    xt, next_mode, dW, new_reset_arg = jax.lax.cond(guard_hit, event_condition_true_fun, event_condition_false_fun, args_guard)

    return xt, next_mode, dW, new_reset_arg
    
"""
One step cost
"""
def cost_i(xt, current_mode, cur_St, xref, uref, K, k, randN, eps, dt, dt_shrink, t0, reset_arg, 
           hybrid_integration_func):
    # -------- compute control input --------
    delta_xt = xt - xref
    ut = uref + jnp.dot(K, delta_xt) + k    
    next_mode = current_mode
    
    # -------- propagate dynamics with hybrid event --------
    xt, next_mode, dW, new_reset_arg = hybrid_integration_func(xt, current_mode, ut, randN, eps, dt, dt_shrink, t0, reset_arg)

    # Collect cost: consider only the terminal state cost for now.
    cur_St += jnp.array([jnp.dot(ut.T, ut)/2.0 * dt + jnp.sqrt(eps) * jnp.dot(ut.T, dW)])
    
    return xt, next_mode, cur_St, new_reset_arg

def feedback_cost_jax(carry, inputs, eps, 
                      dt, dt_shrink, t0, tf, 
                      v_ext_ref_mode_change, 
                      v_ext_trj_fwd, v_ext_trj_bwd,
                      v_Kfb_ref_ext_fwd, v_kff_ref_ext_fwd,
                      v_Kfb_ref_ext_bwd, v_kff_ref_ext_bwd,
                      cost_i_func):
    xt, current_mode, St, cnt_MM, indx = carry
    
    (uref_mode0, uref_mode1, 
    K_mode0, k_mode0, 
    K_mode1, k_mode1, 
    randN_mode0, randN_mode1, 
    xref_mode0, xref_mode1, current_mode_ref, reset_arg) = inputs
    
    # ---------------------------
    # rollout and collect costs 
    # ---------------------------
    # Choose the control in the current mode. Assuming 2-mode system    
    is_mode0 = (current_mode == 0)    
    args_choose_control = (xref_mode0, xref_mode1, uref_mode0, uref_mode1, K_mode0, k_mode0, K_mode1, k_mode1, randN_mode0, randN_mode1)
    xref, u_mode, K_mode, k_mode, randN_mode = jax.lax.cond(is_mode0, mode0_cond_true_fun_jax, mode0_cond_false_fun_jax, args_choose_control)
        
    # -------------------------------
    # Get the trajectory extensions 
    # -------------------------------
    # Consider only 1 mode change for now.
    cnt_event = 0
    ext_ref_mode_change = v_ext_ref_mode_change[0]
    ext_trj_fwd = v_ext_trj_fwd[0]
    ext_trj_bwd = v_ext_trj_bwd[0]
    Kfb_ref_ext_fwd = v_Kfb_ref_ext_fwd[0]
    kff_ref_ext_fwd = v_kff_ref_ext_fwd[0]
    Kfb_ref_ext_bwd = v_Kfb_ref_ext_bwd[0]
    kff_ref_ext_bwd = v_kff_ref_ext_bwd[0]
    
    is_ModeMismatched = (current_mode != current_mode_ref)    
    args_ModeMismatch = (current_mode, 
                         current_mode_ref,
                         ext_ref_mode_change, 
                         ext_trj_fwd, ext_trj_bwd, 
                         Kfb_ref_ext_fwd, kff_ref_ext_fwd, 
                         Kfb_ref_ext_bwd, kff_ref_ext_bwd, 
                         K_mode, k_mode,
                         xref, indx, cnt_MM)
    
    xref, K_mode, k_mode, cnt_MM = jax.lax.cond(is_ModeMismatched, ModeMismatch_true_fun_jax, ModeMismatch_false_fun_jax, args_ModeMismatch)
    
    xt_next, next_mode, St, _ = cost_i_func(xt, current_mode, St, xref, u_mode, 
                                            K_mode, k_mode, randN_mode, eps, dt, dt_shrink, t0, reset_arg)
    indx = indx + 1
    
    return (xt_next, next_mode, St, cnt_MM, indx), (next_mode, xt_next, St, xref) 
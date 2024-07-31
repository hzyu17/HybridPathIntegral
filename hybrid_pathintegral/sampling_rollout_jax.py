import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

# JAX import
import jax
import jax.numpy as jnp
# discrete dynamics import
from dynamics.dynamics_discrete import *

jax.config.update("jax_enable_x64", True)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
print("Devices:", jax.devices())
print("jax.default_backend()", jax.default_backend())


# ------------------------------------- terminal loss ------------------------------------- 
def quadratic_terminal_cost(xT, args):
    x_tar, QT =  args
    return (xT-x_tar)@QT@(xT-x_tar) / 2.0
quadratic_terminal_cost_jit = jax.jit(quadratic_terminal_cost)

# terminal cost function in jax
def terminal_cost_jax(xT, args):
    return jnp.array(quadratic_terminal_cost(xT, args))
terminal_cost_jit = jax.jit(terminal_cost_jax)

# ------------------------------------- / terminal loss ------------------------------------- 


# ===============================================================================================================
#                                       Mode mismatch condition handling
# ===============================================================================================================

# -----------------------------------
# Mode Mismatch condition handling
# -----------------------------------

def ModeMismatch_true_fun_jax(args):
    
    (current_mode, ref_current_mode, 
     ref_mode_change, 
     ext_trj_fwd, ext_trj_bwd, 
     Kfb_ref_ext_fwd, kff_ref_ext_fwd, 
     Kfb_ref_ext_bwd, kff_ref_ext_bwd, 
     Kfb, kff, xref_i, 
     cnt_indx, cnt_MM)= args
        
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
    mode_mismatch = (current_mode != ref_current_mode)
    return jnp.logical_and(mode_mismatch, jnp.logical_and(current_mode==ref_mode_change[1], ref_current_mode==ref_mode_change[0]))

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
# @jax.jit
def mode0_condition_jit(current_mode):
    # assume time invariant guard for now
    return current_mode == 0

# @jax.jit
def mode0_true_fun_jax(args):
    (xref_mode0, _, uref_mode0, _, Kfb_mode0, kff_mode0, _, _, randN_mode0, _) = args
    return xref_mode0, uref_mode0, Kfb_mode0, kff_mode0, randN_mode0


# @jax.jit
def mode0_false_fun_jax(args):
    (_, xref_mode1, _, uref_mode1, _, _, Kfb_mode1, kff_mode1, _, randN_mode1) = args
    return xref_mode1, uref_mode1, Kfb_mode1, kff_mode1, randN_mode1

    
"""
One step cost
"""
# @jax.jit
def cost_i(xt, current_mode, cur_St, 
           xref, uref, Kfb, kff, 
           randN, eps, dt, dt_shrink, 
           t0, reset_arg, 
           hybrid_integration_func):
    
    # -------- compute control input --------
    delta_xt = xt - xref
    ut = uref + jnp.dot(Kfb, delta_xt) + kff    
    # next_mode = current_mode
    
    # -------- propagate dynamics with hybrid event --------
    xt_next, next_mode, dW, new_reset_arg = hybrid_integration_func(xt, current_mode, ut, randN, eps, dt, dt_shrink, t0, reset_arg)

    # Collect cost: consider only the terminal state cost for now.
    cur_St += jnp.array([jnp.dot(ut.T, ut)/2.0 * dt + jnp.sqrt(eps) * jnp.dot(ut.T, dW)])[0]
    
    return xt_next, next_mode, ut, cur_St, new_reset_arg

# @jax.jit
def feedback_cost_jax(carry, inputs, eps, 
                      dt, dt_shrink, t0, 
                      v_ext_ref_mode_change, 
                      v_ext_trj_fwd, v_ext_trj_bwd,
                      v_Kfb_ref_ext_fwd, v_kff_ref_ext_fwd,
                      v_Kfb_ref_ext_bwd, v_kff_ref_ext_bwd,
                      cost_i_func):
    
    xt, current_mode, St, cnt_MM, indx, reset_arg = carry
    
    (uref_mode0, uref_mode1, 
    Kfb_mode0, kff_mode0, 
    kfb_mode1, kff_mode1, 
    randN_mode0, randN_mode1, 
    xref_mode0, xref_mode1, current_mode_ref) = inputs
    
    # ---------------------------
    # rollout and collect costs 
    # ---------------------------
    # Choose the control in the current mode. Assuming 2-mode system    
    cond_mode0 = (current_mode == 0) 
    args_cond_mode0 = (xref_mode0, xref_mode1, 
                       uref_mode0, uref_mode1, 
                       Kfb_mode0, kff_mode0, 
                       kfb_mode1, kff_mode1, 
                       randN_mode0, randN_mode1)
    
    (xref_mode, u_mode, K_mode, k_mode, randN_mode) = jax.lax.cond(cond_mode0, mode0_true_fun_jax, mode0_false_fun_jax, args_cond_mode0)

    # -------------------------------
    # Get the trajectory extensions 
    # -------------------------------
    # Consider only 1 mode change for now.
    # cnt_event = 0
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
                         xref_mode, indx, cnt_MM)
    xref_mode, K_mode, k_mode, cnt_MM_next = jax.lax.cond(is_ModeMismatched, ModeMismatch_true_fun_jax, ModeMismatch_false_fun_jax, args_ModeMismatch)
    
    xt_next, next_mode, ut, St_next, new_reset_arg = cost_i_func(xt, current_mode, St, xref_mode, u_mode, 
                                                                 K_mode, k_mode, randN_mode, 
                                                                 eps, dt, dt_shrink, t0, reset_arg)
    indx_next = indx + 1
    
    return (xt_next, next_mode, St_next, cnt_MM_next, indx_next, new_reset_arg), (current_mode, xt, St, ut, xref_mode, K_mode, k_mode, new_reset_arg) 
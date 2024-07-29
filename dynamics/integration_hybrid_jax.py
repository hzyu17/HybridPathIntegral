import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from dynamics.dynamics_bouncing import *

# numpy
import sympy as sp
from sympy.matrices import Matrix
import numpy as np

# plotting
import matplotlib.pyplot as plt

# jax import
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from functools import partial
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
print("Devices:", jax.devices())
print("jax.default_backend()", jax.default_backend())


# def dyn_bouncing_euler(x, u):
#     return jnp.array([x[1], u[0]-9.81], dtype=jnp.float64)


# def gdWt_bouncing(dWt, eps):
#     B = np.array([[0],[1.0]], dtype=np.float64)
#     return np.sqrt(eps) * B@dWt
    
def stochastic_integration_euler(x0, u, dt, eps, dW):
    B = jnp.array([[0],[1.0]], dtype=jnp.float64)
    xt_next = x0 + jnp.array([x0[1], u[0]-9.81], dtype=jnp.float64) * dt + jnp.sqrt(eps) * B@dW
    return xt_next


def update_u0_pathintegral_jax(u0, PathCosts, GaussianNoise, epsilon, delta_t):
    # ------- numerical processing -------
    PathCosts = PathCosts - jnp.min(PathCosts)
    exp_PathCosts = jnp.exp(-PathCosts/epsilon)
    sum_expPathCosts = jnp.sum(exp_PathCosts)
    
    # ------- Compute the weights ---------
    E_expS_jax = jnp.mean(exp_PathCosts)
    weights = exp_PathCosts / E_expS_jax
    
    # ------- Compute the update to control -------
    
    def Cost_Noise_mult(exp_PathCosts_k, GaussianNoise_k):
        return exp_PathCosts_k*GaussianNoise_k
    
    Cost_Noise_vmap = jax.vmap(Cost_Noise_mult, in_axes=(0,0))
    Cost_Noise_prod = Cost_Noise_vmap(exp_PathCosts, GaussianNoise)
    U_update_jax = jnp.sum(Cost_Noise_prod)
        
    U_update_jax = jnp.sqrt(epsilon/delta_t) * U_update_jax / sum_expPathCosts 
    
    return u0 + U_update_jax, weights

# ------------------------------------- terminal loss ------------------------------------- 
from functools import partial

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


# ===============================================================================================================
#                                        Bouncing condition handling 
# ===============================================================================================================
@jax.jit
def bouncing_event_condition_jit(xt, xt_next):
    # assume time invariant guard for now
    return jnp.logical_and(guard_bouncing_12_jit(0.0,xt)>0, guard_bouncing_12_jit(0.0,xt_next)<=0) 

@jax.jit
def bouncing_true_fun_jit(args):
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
        
        xt_swch = stochastic_integration_euler(xt_current, u, dt_int, eps, dW_new)
        
        new_condition = jnp.logical_not(jnp.logical_or(guard_bouncing_12(t, xt_swch)>0, cnt_shrink==10))
        cnt_shrink += 1
        
        new_vars = (xt_current, xt_swch, u, t, dt_int, dt_shr, RandN, eps, cnt_shrink, reset_arg, new_condition)
        
        return new_vars
    
    init_condition = True
    init_vars = (xt_current, xt_next, u, t, dt_int, dt_shr, RandN, eps, 0, reset_arg, init_condition)
    final_vars = jax.lax.while_loop(while_cond, while_loop_body, init_val=init_vars)
    
    (xt_current, xt_swch, u, t, dt_int, dt_shr, RandN, eps, cnt, reset_arg, can_continue) = final_vars
    xt_next, next_mode, reset_arg = reset_map_bouncing_12_jax(t, xt_swch, current_mode, reset_arg)
    dW_new = jnp.sqrt(dt_int)*RandN
    
    return xt_next, next_mode, dW_new, reset_arg

# bouncing_true_fun_jit = jax.jit(bouncing_cond_true_fun_jax)

@jax.jit
def bouncing_false_fun_jit(args):
    print("Bouncing_cond: False")
    (_, current_mode, _, _, xt_next, dt_int, _, RandN, _, reset_arg) = args
    dW = jnp.sqrt(dt_int)*RandN
    return xt_next, current_mode, dW, reset_arg

# ===============================================================================================================
#                                       // End of Bouncing condition handling //
# ===============================================================================================================

# ===========================================
#    Choose nominal control based on mode 
# ===========================================
def mode0_condition_jax(current_mode):
    # assume time invariant guard for now
    return current_mode == 0
mode0_condition_jit = jax.jit(mode0_condition_jax)

def mode0_cond_true_fun_jax(args):
    print("Mode 0 True")
    (u_0, u_1, randN_0, randN_1) = args
    return u_0, randN_0
mode0_true_fun_jit = jax.jit(mode0_cond_true_fun_jax)

def mode0_cond_false_fun_jax(args):
    (u_0, u_1, randN_0, randN_1) = args
    return u_1, randN_1
mode0_false_fun_jit = jax.jit(mode0_cond_false_fun_jax)
    

# =======================================================
#   // End of choose nominal control based on mode 
# =======================================================

def hybrid_integration(xt, current_mode, ut, randN, eps, dt, dt_shrink, t0, reset_arg):
    dW = jnp.sqrt(dt)*randN
    xt_next = stochastic_integration_euler(xt, ut, dt, eps, dW)
    
    # ------------
    # change mode
    # ------------
    next_mode = current_mode
    args_guard = (xt, current_mode, ut, t0, xt_next, dt, dt_shrink, randN, eps, reset_arg)
    guard_hit = bouncing_event_condition_jit(xt, xt_next)
    xt, next_mode, dW, new_reset_arg = jax.lax.cond(guard_hit, bouncing_true_fun_jit, bouncing_false_fun_jit, args_guard)

    return xt, next_mode, dW, new_reset_arg
    
"""
One step cost
"""
def cost_i(xt, current_mode, cur_St, xref, uref, K, k, randN, eps, dt, dt_shrink, t0, reset_arg):
    # -------- compute control input --------
    delta_xt = xt - xref
    ut = uref + jnp.dot(K, delta_xt) + k    
    next_mode = current_mode
    
    # -------- propagate dynamics with hybrid event --------
    xt, next_mode, dW, new_reset_arg = hybrid_integration(xt, current_mode, ut, randN, eps, dt, dt_shrink, t0, reset_arg)

    # # --------------
    # # change mode
    # # --------------
    # args_guard = (xt, current_mode, ut, t0, xt_next, next_mode, dt, dt_shrink, randN, eps)
    # guard_hit = bouncing_event_condition_jit(xt, xt_next)
    # xt, next_mode, dW = jax.lax.cond(guard_hit, bouncing_true_fun_jit, bouncing_false_fun_jit, args_guard)

    # Collect cost: consider only the terminal state cost for now.
    cur_St += jnp.array([jnp.dot(ut.T, ut)/2.0 * dt + jnp.sqrt(eps) * jnp.dot(ut.T, dW)])
    # cur_MC = jnp.array([current_mode, next_mode])
    
    return xt, next_mode, cur_St, new_reset_arg
    

# collect_cost_jitted = jax.jit(collect_cost)

def feedback_cost_jax(carry, inputs, eps, 
                      dt, dt_shrink, t0, tf, 
                      v_ext_ref_mode_change, 
                      v_ext_trj_fwd, v_ext_trj_bwd,
                      v_Kfb_ref_ext_fwd, v_kff_ref_ext_fwd,
                      v_Kfb_ref_ext_bwd, v_kff_ref_ext_bwd
                      ):
    xt, current_mode, St, cnt_MM, indx = carry
    
    uref_mode0, uref_mode1, K, k, randN_mode0, randN_mode1, xref, current_mode_ref, reset_arg = inputs

    # The next mode of the reference state
    # current_mode_ref = v_ref_modes
    
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
                         K, k,
                         xref, indx, cnt_MM)
    
    xref, K, k, cnt_MM = jax.lax.cond(is_ModeMismatched, ModeMismatch_true_fun_jax, ModeMismatch_false_fun_jax, args_ModeMismatch)
   
    # ---------------------------
    # rollout and collect costs 
    # ---------------------------
    # Choose the control in the current mode. Assuming 2-mode system    
    is_mode0 = (current_mode == 0)    
    args_choose_control = (uref_mode0, uref_mode1, randN_mode0, randN_mode1)
    u_mode, randN_mode = jax.lax.cond(is_mode0, mode0_cond_true_fun_jax, mode0_cond_false_fun_jax, args_choose_control)
    
    xt_next, next_mode, St, _ = cost_i(xt, current_mode, St, xref, u_mode, 
                                        K, k, randN_mode, eps, dt, dt_shrink, t0, reset_arg)
    indx = indx + 1
    
    return (xt_next, next_mode, St, cnt_MM, indx), (xt_next, St, xref) 

    
"""
Synonyms:
MC: mode change
MM: mismatch
shr: shrinking ratio
fb: feedback
ff: feedforward
x_tar: target state
eps: epsilon
"""
def sample_bouncing_jax(n_samples, x0, current_mode, 
                        xref_trj, ref_modes, 
                        uref_mode0_trj, uref_mode1_trj, 
                        K_fb, k_ff, 
                        x_tar, Q_T, 
                        t0, dt, tf, dt_shr, 
                        eps, 
                        noise_mode0,
                        noise_mode1, 
                        v_ext_trj_mode_change, 
                        v_ext_trj_fwd, v_ext_trj_bwd, 
                        v_Kfb_ref_ext_fwd, v_kff_ref_ext_fwd,
                        v_Kfb_ref_ext_bwd, v_kff_ref_ext_bwd,
                        reset_args):
    
    # -----------------------------
    # move the variables onto GPU
    # ----------------------------- 
    xref_trj = jnp.asarray(xref_trj)
    uref_mode0_trj = jnp.asarray(uref_mode0_trj)
    uref_mode1_trj = jnp.asarray(uref_mode1_trj)
    reset_args = jnp.asarray(reset_args)
    ref_modes = jnp.asarray(ref_modes)
    K_fb = jnp.asarray(K_fb)
    k_ff = jnp.asarray(k_ff)
    noise_mode0 = jnp.asarray(noise_mode0)
    noise_mode1 = jnp.asarray(noise_mode1)

    
    # ===========================
    # start jax sampling process
    # ===========================
    x0_jax = jnp.asarray(x0)
    
    # -----------------------------------
    # vectorizing carrys and inputs
    # -----------------------------------
    
    # ----- carrys: (x0, current mode change, mismatch counter, timestep index, Path Cost) ----- 
    v_x0 = jnp.tile(x0_jax, (n_samples, 1))
    v_current_mode = jnp.tile(current_mode, (n_samples, ))
    v_cnt_MM = jnp.zeros((n_samples, ), dtype=jnp.int64)
    v_index = jnp.tile(0, (n_samples, ))
    v_St = jnp.zeros((n_samples, 1), dtype=jnp.float64)
    # --------------- // carrys // --------------- 
    
    # --------------- inputs --------------------
    """ 
    (reference_trj_x, 
    reference_trj_u, 
    feedback gain, 
    feed forward gain, 
    Gaussian randomness, 
    reference mode change sequence)
    """
    # --------------- / inputs --------------------
    v_xref = jnp.tile(xref_trj, (n_samples, 1, 1))
    
    # ------------------------------------------------- 
    # Mode-dependent control, assuming 2-mode system
    # ------------------------------------------------- 
    v_uref_mode0 = jnp.tile(uref_mode0_trj, (n_samples, 1, 1))
    v_uref_mode1 = jnp.tile(uref_mode1_trj, (n_samples, 1, 1))
    v_reset_args = jnp.tile(reset_args, (n_samples, 1, 1))

    v_Kfb = jnp.tile(K_fb, (n_samples, 1, 1, 1))
    v_kff = jnp.tile(k_ff, (n_samples, 1, 1))
    
    v_randN_mode0 = jnp.asarray(noise_mode0)
    v_randN_mode1 = jnp.asarray(noise_mode1)
    v_ref_modes = jnp.tile(ref_modes, (n_samples, 1))
    
    v_initial_carry = (v_x0, v_current_mode, v_St, v_cnt_MM, v_index)
    v_inputs = (v_uref_mode0, v_uref_mode1, 
                v_Kfb, v_kff, 
                v_randN_mode0, v_randN_mode1, 
                v_xref, v_ref_modes, 
                v_reset_args)
    
    # -------------------- // inputs // ------------------------- 
    
    # =================
    # Sampling process
    # =================

    # ================================
    #  Define scan and vmap functions
    # ================================ 
    feedback_cost_scan_fun = partial(feedback_cost_jax, 
                                     eps=eps, dt=dt, 
                                     dt_shrink=dt_shr, 
                                     t0=t0, tf=tf, 
                                     v_ext_ref_mode_change=v_ext_trj_mode_change, 
                                     v_ext_trj_fwd=v_ext_trj_fwd, 
                                     v_ext_trj_bwd=v_ext_trj_bwd,
                                     v_Kfb_ref_ext_fwd=v_Kfb_ref_ext_fwd, 
                                     v_kff_ref_ext_fwd=v_kff_ref_ext_fwd,
                                     v_Kfb_ref_ext_bwd=v_Kfb_ref_ext_bwd, 
                                     v_kff_ref_ext_bwd=v_kff_ref_ext_bwd)
    
    # carry = (v_xt, v_current_mode, v_St, v_cnt_MM, v_index)
    # inputs = (uref_mode0, uref_mode1, Kfb, kff, randN_mode0, randN_mode1, xref, ref_modes, reset_args)
    def feedbackcost_onerow(carrys, inputs):
        initial_carry = (carrys[0], carrys[1], carrys[2], carrys[3], carrys[4])
        _, updated_row = jax.lax.scan(feedback_cost_scan_fun, initial_carry, inputs)
        return updated_row
    
    feedbackcost_vmap = jax.vmap(feedbackcost_onerow, in_axes=(0,0))
    
    v_sample_results = feedbackcost_vmap(v_initial_carry, v_inputs)
    
    
    args_terminal_cost = (x_tar, Q_T)
    terminal_cost_xQrx_vmap = jax.vmap(partial(quadratic_terminal_cost_jit, args=args_terminal_cost), in_axes=0)
    

    # --------------------------
    # results and terminal loss 
    # --------------------------
    Ksamples_jax, PathCosts_jax, actual_ref_jax = v_sample_results
    
    # Move the samples forward by 1 place and add xt to the front, to keep the same with numpy results.
    Ksamples_jax = jnp.concatenate((v_x0.reshape((n_samples, 1, -1)), Ksamples_jax[:,0:-1,:]), axis=1)
    PathCosts_jax = PathCosts_jax[:,-2,1]
    
    # ------------ Terminal cost ------------
    xT_samples = Ksamples_jax[:,-1,:]
    v_S_xT = terminal_cost_xQrx_vmap(xT_samples)
    PathCosts_jax = PathCosts_jax + v_S_xT
    
    return Ksamples_jax, PathCosts_jax, actual_ref_jax
    
    # ============================================== / jax parallel sampling ====================================
    

import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from dynamics.dynamics_slip import *
from hybrid_pathintegral.sampling_rollout_jax import *


def stochastic_integration_euler_SLIP(mode, x0, u, dt, eps, dW):
    def condition_mode0(mode):
        return (mode == 0)
    
    def mode0_dynamics_true_func_slip(args):
        (x0, u, dW, eps) = args
        # flight mode
        # [x, x_dot, z, z_dot, theta] = x0
        B = jnp.array([[0, 0],[0, 0],[0, 0],[0, 0],[1.0, 0.0]], dtype=jnp.float64)
        
        return x0 + jnp.array([x0[1], 0, x0[3], -9.81, u[0]], dtype=jnp.float64) * dt + jnp.sqrt(eps) * B@dW
    
    def mode0_dynamics_false_func_slip(args):
        (x0, u, dW, eps) = args
        # stance mode
        # [theta, theta_dot, r, r_dot] = x0
        
        g = 9.81
        k = 15.0
        m = 0.2
        r0 = 1
        
        theta,theta_dot,r,r_dot = x0[0], x0[1], x0[2], x0[3]
        u1, u2 = u[0], u[1]
        
        # Defining the stance dynamics of the system
        B = jnp.array([[0.0, 0.0], 
                       [0.0, 0.0], 
                       [0.0, 1/m/r/r], 
                       [k/m, 0.0],
                       [0.0, 0.0]], dtype=jnp.float64)
        
        xt_next = x0 + jnp.array([theta_dot, 
                                  -2*theta_dot*r_dot/r-g*jnp.cos(theta)/r, 
                                  r_dot + u2/m/r/r, 
                                  k/m*(r0-r) - g*jnp.sin(theta) + theta_dot*theta_dot*r + k*u1/m,
                                  0.0], dtype=jnp.float64) * dt + jnp.sqrt(eps) * B@dW
        
        
        return xt_next
    
    is_mode0 = (mode == 0)
    args_choose_dynamics = (x0, u, dW, eps)
    xt_next = jax.lax.cond(is_mode0, mode0_dynamics_true_func_slip, mode0_dynamics_false_func_slip, args_choose_dynamics)


    # if mode == 0:
    #     B = jnp.array([[0],[0],[0],[0],[1.0]], dtype=jnp.float64)
    #     xt_next = x0 + jnp.array([x0[1], 0, x0[3], -9.81, u[0]], dtype=jnp.float64) * dt + jnp.sqrt(eps) * B@dW
        
    # elif mode == 1:
    #     # stance mode
    #     # [theta, theta_dot, r, r_dot] = x0
    #     g = 9.81
    #     k = 15.0
    #     m = 0.2
    #     r0 = 1
        
    #     theta,theta_dot,r,r_dot = x0[0], x0[1], x0[2], x0[3]
    #     u1, u2 = u[0], u[1]
        
    #     # Defining the stance dynamics of the system
    #     B = jnp.array([[0.0, 0.0], 
    #                    [0.0, 0.0], 
    #                    [0.0, 1/m/r/r], 
    #                    [k/m, 0.0],
    #                    [0.0, 0.0]], dtype=jnp.float64)
        
    #     xt_next = x0 + jnp.array([theta_dot, 
    #                               -2*theta_dot*r_dot/r-g*jnp.cos(theta)/r, 
    #                               r_dot + u2/m/r/r, 
    #                               k/m*(r0-r) - g*jnp.sin(theta) + theta_dot*theta_dot*r + k*u1/m,
    #                               0.0], dtype=jnp.float64) * dt + jnp.sqrt(eps) * B@dW
        
    return xt_next

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
#                                        SLIP Guard condition handling 
# ===============================================================================================================
@jax.jit
def event_condition_slip(xt, xt_next):
    # assume time invariant guard for now
    return jnp.logical_and(guard_slip_12(0.0,xt)>0, guard_slip_12(0.0,xt_next)<=0) 

@jax.jit
def event_true_func_slip(args):
    print("slip_cond: True")
    (xt_current, current_mode, u, t, xt_next, dt_int, dt_shr, RandN, eps, reset_arg) = args
    
    def while_cond(vars):
        (_, _, _, _, _, _, _, _, _, _, can_continue) = vars
        return can_continue
    
    def while_loop_body(vars):
        (xt_current, xt_swch, u, t, dt_int, dt_shr, RandN, eps, cnt_shrink, reset_arg, _) = vars
        
        # Too far from the guard, shrink the step size.
        dt_int = dt_int * dt_shr
        dW_new = jnp.sqrt(dt_int)*RandN
        
        xt_swch = stochastic_integration_euler_SLIP(current_mode, xt_current, u, dt_int, eps, dW_new)
        
        new_condition = jnp.logical_not(jnp.logical_or(guard_slip_21(t, xt_swch)>0, cnt_shrink==10))
        cnt_shrink += 1
        
        new_vars = (xt_current, xt_swch, u, t, dt_int, dt_shr, RandN, eps, cnt_shrink, reset_arg, new_condition)
        
        return new_vars
    
    init_condition = True
    init_vars = (xt_current, xt_next, u, t, dt_int, dt_shr, RandN, eps, 0, reset_arg, init_condition)
    final_vars = jax.lax.while_loop(while_cond, while_loop_body, init_val=init_vars)
    
    (xt_current, xt_swch, u, t, dt_int, dt_shr, RandN, eps, cnt, reset_arg, can_continue) = final_vars
    xt_next, next_mode, reset_arg = reset_map_slip_21_padding(t, xt_swch, current_mode, reset_arg)
    dW_new = jnp.sqrt(dt_int)*RandN
    
    return xt_next, next_mode, dW_new, reset_arg


@jax.jit
def event_false_func_slip(args):
    (_, current_mode, _, _, xt_next, dt_int, _, RandN, _, reset_arg) = args
    dW = jnp.sqrt(dt_int)*RandN
    return xt_next, current_mode, dW, reset_arg

# ===============================================================================================================
#                                       // End of SLIP guard condition handling //
# ===============================================================================================================

hybrid_integration_slip = partial(hybrid_integration, 
                                      stochastic_integration_euler_func = stochastic_integration_euler_SLIP, 
                                      event_condition_func = event_condition_slip, 
                                      event_condition_true_fun = event_true_func_slip, 
                                      event_condition_false_fun = event_false_func_slip)

cost_i_slip = partial(cost_i, hybrid_integration_func=hybrid_integration_slip)

feedback_cost_slip_jax = partial(feedback_cost_jax, cost_i_func=cost_i_slip)

# =======================================================
#   // End of choose nominal control based on mode 
# =======================================================
    
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
def sample_slip_jax(n_samples, x0, current_mode, 
                        xref_0_trj,
                        xref_1_trj, 
                        ref_modes, 
                        uref_mode0_trj, uref_mode1_trj, 
                        K_fb_0, k_ff_0, 
                        K_fb_1, k_ff_1, 
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
    xref_mode0_trj = jnp.asarray(xref_0_trj)
    xref_mode1_trj = jnp.asarray(xref_1_trj)
    uref_mode0_trj = jnp.asarray(uref_mode0_trj)
    uref_mode1_trj = jnp.asarray(uref_mode1_trj)
    reset_args = jnp.asarray(reset_args)
    ref_modes = jnp.asarray(ref_modes)
    K_fb_0 = jnp.asarray(K_fb_0)
    k_ff_0 = jnp.asarray(k_ff_0)
    K_fb_1 = jnp.asarray(K_fb_1)
    k_ff_1 = jnp.asarray(k_ff_1)
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
    v_xref_mode0 = jnp.tile(xref_mode0_trj, (n_samples, 1, 1))
    v_xref_mode1 = jnp.tile(xref_mode1_trj, (n_samples, 1, 1))
    
    # ------------------------------------------------- 
    # Mode-dependent control, assuming 2-mode system
    # ------------------------------------------------- 
    v_uref_mode0 = jnp.tile(uref_mode0_trj, (n_samples, 1, 1))
    v_uref_mode1 = jnp.tile(uref_mode1_trj, (n_samples, 1, 1))
    v_reset_args = jnp.tile(reset_args, (n_samples, 1, 1))

    v_Kfb_0 = jnp.tile(K_fb_0, (n_samples, 1, 1, 1))
    v_kff_0 = jnp.tile(k_ff_0, (n_samples, 1, 1))
    v_Kfb_1 = jnp.tile(K_fb_1, (n_samples, 1, 1, 1))
    v_kff_1 = jnp.tile(k_ff_1, (n_samples, 1, 1))
    
    v_randN_mode0 = jnp.asarray(noise_mode0)
    v_randN_mode1 = jnp.asarray(noise_mode1)
    v_ref_modes = jnp.tile(ref_modes, (n_samples, 1))
    
    v_initial_carry = (v_x0, v_current_mode, v_St, v_cnt_MM, v_index)
    v_inputs = (v_uref_mode0, v_uref_mode1, 
                v_Kfb_0, v_kff_0,
                v_Kfb_1, v_kff_1, 
                v_randN_mode0, v_randN_mode1, 
                v_xref_mode0, v_xref_mode1,
                v_ref_modes, 
                v_reset_args)
    
    # -------------------- // inputs // ------------------------- 
    
    # =================
    # Sampling process
    # =================

    # ================================
    #  Define scan and vmap functions
    # ================================ 
    feedback_cost_scan_fun = partial(feedback_cost_slip_jax, 
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
    Ksample_modes_jax, Ksamples_jax, PathCosts_jax, actual_ref_jax = v_sample_results
    
    # Move the samples forward by 1 place and add xt to the front, to keep the same with numpy results.
    Ksample_modes_jax = jnp.concatenate((v_current_mode.reshape((n_samples, -1)), Ksample_modes_jax[:,0:-1]), axis=1)
    Ksamples_jax = jnp.concatenate((v_x0.reshape((n_samples, 1, -1)), Ksamples_jax[:,0:-1,:]), axis=1)
    PathCosts_jax = PathCosts_jax[:,-2,1]
    
    # ------------ Terminal cost ------------
    xT_samples = Ksamples_jax[:,-1,:]
    v_S_xT = terminal_cost_xQrx_vmap(xT_samples)
    PathCosts_jax = PathCosts_jax + v_S_xT
    
    return Ksample_modes_jax, Ksamples_jax, PathCosts_jax, actual_ref_jax
    
    # ============================================== / jax parallel sampling ====================================
    

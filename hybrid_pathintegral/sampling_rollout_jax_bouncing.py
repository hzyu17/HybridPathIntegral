from hybrid_pathintegral.sampling_rollout_jax import *
from dynamics.dynamics_bouncing_discrete import *
from functools import partial

hybrid_integration_bouncing = partial(hybrid_integration_euler, 
                                      stochastic_integration_euler_func = stochastic_integration_euler_bouncing, 
                                      event_condition_func = bouncing_event_condition, 
                                      event_condition_true_fun = bouncing_event_true_func, 
                                      event_condition_false_fun = bouncing_event_false_func)

cost_i_bouncing = partial(cost_i, hybrid_integration_func=hybrid_integration_bouncing)

feedback_cost_bouncing_jax = partial(feedback_cost_jax, cost_i_func=cost_i_bouncing)

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
def sample_bouncing_jax(n_samples, x0, current_mode, 
                        xref_0_trj,
                        xref_1_trj, 
                        ref_modes, 
                        uref_mode0_trj, uref_mode1_trj, 
                        K_fb_0, k_ff_0, 
                        K_fb_1, k_ff_1, 
                        x_tar, Q_T, 
                        t0, dt, dt_shr, 
                        eps, 
                        noise_mode0,
                        noise_mode1, 
                        v_ext_trj_mode_change, 
                        v_ext_trj_fwd, v_ext_trj_bwd, 
                        v_Kfb_ref_ext_fwd, v_kff_ref_ext_fwd,
                        v_Kfb_ref_ext_bwd, v_kff_ref_ext_bwd,
                        init_reset_args):
    
    # -----------------------------
    # move the variables onto GPU
    # ----------------------------- 
    xref_mode0_trj = jnp.asarray(xref_0_trj)
    xref_mode1_trj = jnp.asarray(xref_1_trj)
    uref_mode0_trj = jnp.asarray(uref_mode0_trj)
    uref_mode1_trj = jnp.asarray(uref_mode1_trj)
    init_reset_args = jnp.asarray(init_reset_args)
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
    v_init_event_args = jnp.tile(init_reset_args, (n_samples, 1))
    
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

    v_Kfb_0 = jnp.tile(K_fb_0, (n_samples, 1, 1, 1))
    v_kff_0 = jnp.tile(k_ff_0, (n_samples, 1, 1))
    v_Kfb_1 = jnp.tile(K_fb_1, (n_samples, 1, 1, 1))
    v_kff_1 = jnp.tile(k_ff_1, (n_samples, 1, 1))
    
    v_randN_mode0 = jnp.asarray(noise_mode0)
    v_randN_mode1 = jnp.asarray(noise_mode1)
    v_ref_modes = jnp.tile(ref_modes, (n_samples, 1))
    
    v_initial_carry = (v_x0, v_current_mode, v_St, v_cnt_MM, v_index, v_init_event_args)
    v_inputs = (v_uref_mode0, v_uref_mode1, 
                v_Kfb_0, v_kff_0,
                v_Kfb_1, v_kff_1, 
                v_randN_mode0, v_randN_mode1, 
                v_xref_mode0, v_xref_mode1,
                v_ref_modes)
    
    # --------------- / inputs --------------------
    
    
    # ------------------------------------------------- 
    # Mode-dependent control, assuming 2-mode system
    # ------------------------------------------------- 
    # v_uref_mode0 = jnp.tile(uref_mode0_trj, (n_samples, 1, 1))
    # v_uref_mode1 = jnp.tile(uref_mode1_trj, (n_samples, 1, 1))
    # v_reset_args = jnp.tile(reset_args, (n_samples, 1, 1))

    # v_Kfb = jnp.tile(K_fb, (n_samples, 1, 1, 1))
    # v_kff = jnp.tile(k_ff, (n_samples, 1, 1))
    
    # v_randN_mode0 = jnp.asarray(noise_mode0)
    # v_randN_mode1 = jnp.asarray(noise_mode1)
    # v_ref_modes = jnp.tile(ref_modes, (n_samples, 1))
    
    # v_initial_carry = (v_x0, v_current_mode, v_St, v_cnt_MM, v_index)
    # v_inputs = (v_uref_mode0, v_uref_mode1, 
    #             v_Kfb, v_kff, 
    #             v_randN_mode0, v_randN_mode1, 
    #             v_xref, v_ref_modes, 
    #             v_reset_args)
    
    # -------------------- // inputs // ------------------------- 
    
    # =================
    # Sampling process
    # =================

    # ================================
    #  Define scan and vmap functions
    # ================================ 
    feedback_cost_scan_fun = partial(feedback_cost_bouncing_jax, 
                                     eps=eps, dt=dt, 
                                     dt_shrink=dt_shr, 
                                     t0=t0, 
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
        initial_carry = (carrys[0], carrys[1], carrys[2], carrys[3], carrys[4], carrys[5])
        _, updated_row = jax.lax.scan(feedback_cost_scan_fun, initial_carry, inputs)
        return updated_row
    
    feedbackcost_vmap = jax.vmap(feedbackcost_onerow, in_axes=(0,0))
    
    v_sample_results = feedbackcost_vmap(v_initial_carry, v_inputs)
    
    
    args_terminal_cost = (x_tar, Q_T)
    terminal_cost_xQrx_vmap = jax.vmap(partial(quadratic_terminal_cost_jit, args=args_terminal_cost), in_axes=0)
    

    # --------------------------
    # results and terminal loss 
    # --------------------------
    Ksample_modes_jax, Ksamples_jax, PathCosts_jax, Ksamples_ut, actual_ref_jax, Ksamples_Kfb_mode, Ksamples_kff_mode, Ksamples_reset_args = v_sample_results
    
    # Move the samples forward by 1 place and add xt to the front, to keep the same with numpy results.
    # Ksample_modes_jax = jnp.concatenate((v_current_mode.reshape((n_samples, -1)), Ksample_modes_jax[:,0:-1]), axis=1)
    # Ksamples_jax = jnp.concatenate((v_x0.reshape((n_samples, 1, -1)), Ksamples_jax[:,0:-1,:]), axis=1)
    
    # ------------ Terminal cost ------------
    PathCosts_jax = PathCosts_jax[:, -1].flatten()
    xT_samples = Ksamples_jax[:,-1,:]
    v_S_xT = terminal_cost_xQrx_vmap(xT_samples)
    PathCosts_jax = PathCosts_jax + v_S_xT
    
    return Ksample_modes_jax, Ksamples_jax, PathCosts_jax, Ksamples_ut, actual_ref_jax, Ksamples_Kfb_mode, Ksamples_kff_mode, Ksamples_reset_args
    
    # ============================================== / jax parallel sampling ====================================
    

from hybrid_pathintegral.sampling_rollout_jax import *
from hybrid_pathintegral.dynamics_discrete_bouncing_JAX import *
from functools import partial

from functools import partial

hybrid_stochastic_integration_bouncing_JAX = partial(h_stoch_integr_euler_JAX, 
                                                    stochastic_integration_euler_func = stochastic_integration_euler_bouncing_JAX, 
                                                    guard_func = guard_bouncing_12_JAX, 
                                                    guard_true_func = guard_true_bouncing_12_JAX, 
                                                    guard_false_func = guard_false_bouncing_12_JAX)

cost_i_bouncing = partial(cost_i, hybrid_stochastic_integration_func=hybrid_stochastic_integration_bouncing_JAX)
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
                        dt, dt_shr, 
                        eps, 
                        noise_mode0,
                        noise_mode1, 
                        v_ext_trj_mode_change, 
                        v_ext_trj_fwd, v_ext_trj_bwd, 
                        v_Kfb_ref_ext_fwd, v_kff_ref_ext_fwd,
                        v_Kfb_ref_ext_bwd, v_kff_ref_ext_bwd,
                        init_reset_args):
    
    # -----------------------------
    #  move the variables onto GPU
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

    
    # ============================
    #  Start jax sampling process
    # ============================
    x0_jax = jnp.asarray(x0)
    
    # -----------------------------------
    #    Vectorizing carrys and inputs
    # -----------------------------------
    
    # ----- carrys: (t0, x0, current_mode_change, timestep_index, Path_Cost) ----- 
    v_t = jnp.zeros((n_samples, 1), dtype=jnp.float64)
    v_x0 = jnp.tile(x0_jax, (n_samples, 1))
    v_current_mode = jnp.tile(current_mode, (n_samples, ))
    v_index = jnp.tile(0, (n_samples, ))
    v_St = jnp.zeros((n_samples, 1), dtype=jnp.float64)
    v_init_event_args = jnp.tile(init_reset_args, (n_samples, 1))
    v_initial_carry = (v_t, v_x0, v_current_mode, v_St, v_index, v_init_event_args)
    
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
    #  Mode-dependent control, assuming 2-mode system
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
    
    v_inputs = (v_uref_mode0, v_uref_mode1, 
                v_Kfb_0, v_kff_0,
                v_Kfb_1, v_kff_1, 
                v_randN_mode0, v_randN_mode1, 
                v_xref_mode0, v_xref_mode1,
                v_ref_modes)
    
    # --------------- / inputs --------------------

    # ================================
    #  Define scan and vmap functions
    # ================================ 
    feedback_cost_scan_fun = partial(feedback_cost_bouncing_jax, 
                                     eps=eps, dt=dt, 
                                     dt_shrink=dt_shr, 
                                     v_ext_ref_mode_change=v_ext_trj_mode_change, 
                                     v_ext_trj_fwd=v_ext_trj_fwd, 
                                     v_ext_trj_bwd=v_ext_trj_bwd,
                                     v_Kfb_ref_ext_fwd=v_Kfb_ref_ext_fwd, 
                                     v_kff_ref_ext_fwd=v_kff_ref_ext_fwd,
                                     v_Kfb_ref_ext_bwd=v_Kfb_ref_ext_bwd, 
                                     v_kff_ref_ext_bwd=v_kff_ref_ext_bwd)
    
    # carry = (v_xt, v_current_mode, v_St, v_index)
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
    (Ksamples_ts, Ksample_modes_jax, Ksamples_jax, 
     PathCosts_jax, Ksamples_ut, 
     actual_ref_jax, Ksamples_Kfb_mode, Ksamples_kff_mode, Ksamples_reset_args) = v_sample_results
    
    # ------------ Terminal cost ------------
    PathCosts_jax = PathCosts_jax[:, -1].flatten()
    xT_samples = Ksamples_jax[:,-1,:]
    v_S_xT = terminal_cost_xQrx_vmap(xT_samples)
    PathCosts_jax = PathCosts_jax + v_S_xT
    
    return (Ksamples_ts, Ksample_modes_jax, Ksamples_jax, PathCosts_jax, 
            Ksamples_ut, actual_ref_jax, Ksamples_Kfb_mode, Ksamples_kff_mode, Ksamples_reset_args)
    
    # ============================================== / jax parallel sampling ====================================
    

import numpy as np

import os
import sys
file_path = os.path.abspath(__file__)
exp_dir = os.path.dirname(file_path)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
root_dir = os.path.abspath(os.path.join(exp_dir, '..'))
sys.path.append(root_dir)

import gc
import jax.numpy as jnp

# Import iLQR class and reference extension handler
# from hybrid_ilqr.h_ilqr import solve_ilqr, extract_extensions
from hybrid_ilqr.h_ilqr_discrete import solve_ilqr, extract_extensions
# Importing path integral control
from hybrid_pathintegral.hybrid_pathintegral import *
from hybrid_pathintegral.sampling_rollout_jax_slip import *
# Import plotting
import matplotlib.pyplot as plt
# Import experiment parameter class
from experiments.exp_params import *
# Import slip model dynamics
from dynamics.dynamics_discrete_slip import *


import gc
gc.collect()


def run_experiment(i_exp, nt, n_samples, n_states, n_inputs, 
                    init_mode, init_state, target_state, hybrid_ilqr_result, 
                    start_time, end_time, dt, dt_shrink, 
                    Q_k, Q_T, R_k, epsilon, init_reset_args):
    
    gc.collect()

    (time_span,modes,states,inputs,
     k_feedforward,K_feedback,
     current_cost,states_iter,
     ref_modechanges,reference_extension_helper, ref_reset_args) = hybrid_ilqr_result
        
    show_ilqr_reference = False
    if show_ilqr_reference:
        print("Plotting h-iLQR reference state and input trajectories.")
        plot_slip(time_span, modes, states, inputs, init_state, target_state, nt, ref_reset_args)
        plt.show()
    
    RndN_actual = [np.random.randn(nt, n_inputs[0]), np.random.randn(nt, n_inputs[1])]
    
    
    # =================================================================
    #                     Hybrid ilqr for comparison
    # ================================================================= 
    # -------------- result collectors, hybrid ilqr proposal --------------
    xt_trj_ilqr = [np.array([0.0]) for _ in range(nt)]
    ut_trj_ilqr = [np.zeros((nt, n_inputs[0])), np.zeros((nt, n_inputs[1]))]
    
    xt_trj_ilqr[0] = init_state
    (mode_trj_ilqr, xt_trj_ilqr, 
    ut_trj_ilqr, cost_ilqr, _, 
    reset_args_ilqr) = hybrid_stochastic_feedback_rollout_discrete_slip(init_mode, init_state, 
                                                                        n_inputs, 
                                                                        states, modes, inputs, 
                                                                        K_feedback, k_feedforward, 
                                                                        target_state, Q_T,
                                                                        start_time, dt, 
                                                                        epsilon, RndN_actual, 
                                                                        dt_shrink, 
                                                                        reference_extension_helper,
                                                                        init_reset_args)

    show_hilqr_noise_results = False
    if show_hilqr_noise_results:
        time_span = np.arange(start_time, end_time, dt).flatten()
        plot_slip(time_span, mode_trj_ilqr, xt_trj_ilqr, ut_trj_ilqr, init_state, target_state, nt, reset_args_ilqr, step=1)
        plt.show()
    
    # mode-dependent reference states. Assuming 2 modes
    # States with padded dimensions
    max_n_inputs = np.max(n_inputs)
    
    RndN_actual_padding = RndN_actual
    if max_n_inputs > n_inputs[0]:
        RndN_actual_padding = [np.concatenate((RndN_actual[0], np.zeros((nt, max_n_inputs-n_inputs[0]))), axis=1), RndN_actual[1]]
    if max_n_inputs > n_inputs[1]:
        RndN_actual_padding = [RndN_actual[0], np.concatenate((RndN_actual[1], np.zeros((nt, max_n_inputs-n_inputs[1]))), axis=1)]
    
    states_0 = np.zeros((nt, 5))
    states_1 = np.zeros((nt, 5))
    
    inputs_0 = np.zeros((nt, max_n_inputs))
    inputs_1 = np.zeros((nt, max_n_inputs))
    
    K_feedback_0 = np.zeros((nt, max_n_inputs, 5))
    K_feedback_1 = np.zeros((nt, max_n_inputs, 5))
    
    k_feedforward_0 = np.zeros((nt, max_n_inputs))
    k_feedforward_1 = np.zeros((nt, max_n_inputs))
    
    for i in range(nt):
        mode_i = modes[i]
        if mode_i == 0:
            states_0[i, :n_states[0]] = states[i]
            inputs_0[i, :n_inputs[0]] = inputs[0][i]
            K_feedback_0[i, :n_inputs[0], :n_states[0]] = K_feedback[i]
            k_feedforward_0[i, :n_inputs[0]] = k_feedforward[i]
        
        if mode_i == 1:
            states_1[i, :n_states[1]] = states[i]
            equiv_state_0 = convert_state_21_slip(states[i])
            states_0[i, :n_states[0]] = equiv_state_0.flatten()
            inputs_1[i, :n_inputs[1]] = inputs[1][i]
            K_feedback_1[i, :n_inputs[1], :n_states[1]] = K_feedback[i]
            k_feedforward_1[i, :n_inputs[1]] = k_feedforward[i]
    
    k_feedforward_0 = np.array(k_feedforward_0)
    k_feedforward_1 = np.array(k_feedforward_1)
    K_feedback_0 = np.array(K_feedback_0)
    K_feedback_1 = np.array(K_feedback_1)
    ref_reset_args = np.array(ref_reset_args)
    
    print(f"=================== The experiment index: {i_exp} ===================" )
    
    # -------------- result collectors, jax --------------
    modes_pi_jax = np.zeros(nt, dtype=np.int64)
    trj_pi_jax = [np.array([0.0]) for _ in range(nt)]

    u_star_pi_jax = [np.zeros((nt, n_inputs[0])), np.zeros((nt, n_inputs[1]))]
    allPathCosts_jax = np.zeros((nt-1, n_samples))
    
    
    # -------------- initialize the control loop -------------- 
    init_state_pad = np.zeros(5)
    init_state_pad[:n_states[init_mode]] = init_state
    x0_jax = jnp.asarray(init_state_pad)
    xt = x0_jax
    xt_ilqr = init_state_pad
    
    current_mode_actual = init_mode
    next_mode_actual = current_mode_actual
    
    modes_pi_jax[0] = init_mode
    trj_pi_jax[0] = x0_jax
    
    # current_ref_modechange_ilqr = (1, 1)
    cnt_mismatch = 0
    # cnt_mismatch_ilqr = 0
    
    # Assuming 2-mode system
    reset_args_actual = ref_reset_args
    event_args_actual = [ref_reset_args[0]]
    cnt_event_actual = 0
    
    # ---------------------------------
    #  Extract the extended references 
    # ---------------------------------
    (v_mode_change_ref, v_ref_ext_bwd, v_ref_ext_fwd, 
    v_Kfb_ref_ext_bwd, v_Kfb_ref_ext_fwd, 
    v_kff_ref_ext_bwd, v_kff_ref_ext_fwd, v_tevent) = extract_extensions(reference_extension_helper, start_index = 0)
    
    # Add the references to the start of the backward extensions
    n_extensions = len(v_tevent)
    for i_ext in range(n_extensions):
        t_event_i = v_tevent[i_ext]
        v_ref_ext_bwd[i_ext][t_event_i+1:] = states[t_event_i+1:]
    
    show_trj_extensions = False
    if show_trj_extensions:
         
        fig_ext, axes_ext = plt.subplots(3, 1, figsize=(10, 12))
        (ax_ext_1, ax_ext_2, ax_ext_3) = axes_ext.flatten()

        ax_ext_1.grid(True)
        ax_ext_2.grid(True)
        ax_ext_3.grid(True)
        
        ax_ext_1.plot(time_span, v_ref_ext_bwd[0][:, 0], color='r', label='Backward extension')
        ax_ext_1.plot(time_span, v_ref_ext_fwd[0][:, 0], color='b', label='Forward extension')
        ax_ext_1.plot(time_span, states[:, 0], color='k')
        
        ax_ext_2.plot(time_span, v_ref_ext_bwd[0][:, 1], color='r', label='Backward extension')
        ax_ext_2.plot(time_span, v_ref_ext_fwd[0][:, 1], color='b', label='Forward extension')
        ax_ext_2.plot(time_span, states[:, 1], color='k')
        
        ax_ext_3.plot(v_ref_ext_bwd[0][:, 0], v_ref_ext_bwd[0][:, 1], color='r', label='Backward extension')
        ax_ext_3.plot(v_ref_ext_fwd[0][:, 0], v_ref_ext_fwd[0][:, 1], color='b', label='Forward extension')
        ax_ext_3.plot(states[:, 0], states[:, 1], color='k')
        
        ax_ext_1.legend()
        ax_ext_2.legend()
        ax_ext_3.legend()
        fig_ext.tight_layout()
    
        plt.show()

    n_modes = 2
    Ksamples_jax_saving = np.zeros((n_samples, nt, n_modes))
    
    
    
    # ======================================================
    #     Main loop for the hybrid path integral control
    # ======================================================
    for i_t in range(nt-1):
        
        print(f"----------- Time index: {i_t} -----------")
        
        start_time_i = start_time + i_t*dt
        nt_i = nt - i_t
        
        # ------------------------------------------------------------------------
        #   Calculate the slice of the variables for the current future horizon 
        # ------------------------------------------------------------------------
        
        # mode-dependent references
        modes_i = modes[i_t:]
        ref_current_mode = modes[i_t]
        
        states_0_i = states_0[i_t:]
        states_1_i = states_1[i_t:]
        
        inputs_0_i = inputs_0[i_t:]
        inputs_1_i = inputs_1[i_t:]
        
        states_i = [states_0_i, states_1_i]
        inputs_i = [inputs_0_i, inputs_1_i]       
        
        K_feedback_0_i = K_feedback_0[i_t:]
        K_feedback_1_i = K_feedback_1[i_t:]
        
        k_feedforward_0_i = k_feedforward_0[i_t:]
        k_feedforward_1_i = k_feedforward_1[i_t:]
        
        # Combined gains
        K_feedback_i = [K_feedback_0_i, K_feedback_1_i]
        k_feedforward_i = [k_feedforward_0_i, k_feedforward_1_i]
        
        # actual modes and states
        current_mode_actual = modes_pi_jax[i_t]
        xt = trj_pi_jax[i_t]
        
        # Randomness is mode dependent, same as control. Assuming 2-mode system.
        n_modes = 2
        Ksamples_jax_i = np.zeros((n_samples, nt_i, n_modes))
        GaussianNoise_i = [np.random.randn(n_samples, nt_i, max_n_inputs), np.random.randn(n_samples, nt_i, max_n_inputs)]
        init_reset_args_i = np.array(ref_reset_args[i_t])
        
        # --------------------------- 
        # Coupling of the randomness
        # ---------------------------
        GaussianNoise_i[0][int(n_samples/2):, 0] = -GaussianNoise_i[0][:int(n_samples/2), 0]
        GaussianNoise_i[1][int(n_samples/2):, 0] = -GaussianNoise_i[1][:int(n_samples/2), 0]
        
        # ----------------------------------------------------------
        # Extract the extended references for the future horizon
        # ----------------------------------------------------------
        (v_mode_change_ref_i, v_ref_ext_bwd_i, v_ref_ext_fwd_i, 
        v_Kfb_ref_ext_bwd_i, v_Kfb_ref_ext_fwd_i, 
        v_kff_ref_ext_bwd_i, v_kff_ref_ext_fwd_i, v_tevent_i) = extract_extensions(reference_extension_helper, 
                                                                                    start_index = i_t, 
                                                                                    padding=True) 
        
        # Add the references to the start of the backward extensions
        n_ext_i = len(v_tevent_i)
        for i_e in range(n_ext_i):
            mc_i = v_mode_change_ref_i[i_e]
            t_e_i = v_tevent_i[i_e]
            v_ref_ext_bwd_i[i_e][t_e_i+1:] = states_i[mc_i[1]][t_e_i+1:]
        
        # ====================
        # Sampling using jax 
        # ====================
        
        # ---------------------------------------------------------------------------------------
        #                               Sample future trajectories
        # ---------------------------------------------------------------------------------------                    
        
        (Ksamples_ts_i, Kmodes_jax_i, Ksamples_jax_i, PathCosts_jax_i, 
         Ksamples_ut, Ksamples_xref, Ksamples_Kfb_mode, 
         Ksamples_kff_mode, Ksamples_reset_args) = sample_slip_jax(i_exp, n_samples, xt, 
                                                                    current_mode_actual, 
                                                                    states_0_i, states_1_i, 
                                                                    modes_i, 
                                                                    inputs_0_i, inputs_1_i, 
                                                                    K_feedback_0_i, k_feedforward_0_i, 
                                                                    K_feedback_1_i, k_feedforward_1_i, 
                                                                    target_state, Q_T, 
                                                                    dt, dt_shrink, 
                                                                    epsilon, 
                                                                    GaussianNoise_i[0], GaussianNoise_i[1], 
                                                                    v_mode_change_ref_i, 
                                                                    v_ref_ext_fwd_i, v_ref_ext_bwd_i, 
                                                                    v_Kfb_ref_ext_fwd_i, v_kff_ref_ext_fwd_i, 
                                                                    v_Kfb_ref_ext_bwd_i, v_kff_ref_ext_bwd_i, 
                                                                    init_reset_args_i)
        
        # ----------------------------------- 
        
        # save the samples at t=0
        save_samples = False
        if save_samples and (i_t == 0):
            Ksamples_jax_saving = Ksamples_jax_i
        
        # --------------------------------------------------------
        #               Update the control proposal
        # --------------------------------------------------------
        # Compute proposal control, possibly with early arrival
        ubar_actual_i = inputs_i[current_mode_actual][0]
        K_fb_actual = K_feedback_i[current_mode_actual][0]
        k_ff_actual = k_feedforward_i[current_mode_actual][0]
        xref_actual_i = states_i[current_mode_actual][0]
        
        if cond_mode_mismatch_slip(current_mode_actual, ref_current_mode):
            print("--------------- mode mismatch happened ---------------")
            (xref_actual_i, 
             K_fb_actual, 
             k_ff_actual, 
             cnt_mismatch) = reaction_mode_mismatch(i_t, 
                                                    current_mode_actual, ref_current_mode, 
                                                    v_ref_ext_fwd[0], v_ref_ext_bwd[0], 
                                                    v_mode_change_ref[0], 
                                                    v_Kfb_ref_ext_fwd[0], v_kff_ref_ext_fwd[0], 
                                                    v_Kfb_ref_ext_bwd[0], v_kff_ref_ext_bwd[0], 
                                                    cnt_mismatch, 
                                                    cond_early_arrival=cond_early_arrival_slip)
        
        
        GaussianNoises_ustar_jax = GaussianNoise_i[current_mode_actual][:,0,:n_inputs[current_mode_actual]]
        
        # Remove the padding, and recover the actual shapes 
        xt_actual = xt
        if current_mode_actual == 0:
            xt_actual = xt_actual[:n_states[0]]
            xref_actual_i = xref_actual_i[:n_states[0]]
            ubar_actual_i = ubar_actual_i[:n_inputs[0]]
            K_fb_actual = K_fb_actual[:n_inputs[0], :n_states[0]].reshape((n_inputs[0],n_states[0]))
            k_ff_actual = k_ff_actual[:n_inputs[0]]
            # GaussianNoises_ustar_jax = GaussianNoises_ustar_jax[:, :n_inputs[0]]
        
        elif current_mode_actual == 1:
            ubar_actual_i = ubar_actual_i[:n_inputs[1]]
            K_fb_actual = K_fb_actual[:n_inputs[1], :n_states[1]]
            k_ff_actual = k_ff_actual[:n_inputs[1]]
            xt_actual = xt_actual[:n_states[1]]
            xref_actual_i = xref_actual_i[:n_states[1]]
            # GaussianNoises_ustar_jax = GaussianNoises_ustar_jax[:, :n_inputs[1]]
        
        u0_proposal = ubar_actual_i + K_fb_actual@(xt_actual - xref_actual_i) + k_ff_actual      
        Ksamples_delta_t0 = Ksamples_ts_i[:,1] - Ksamples_ts_i[:,0]
        u0_star_jax, weights_jax = update_u0_pathintegral_jax(u0_proposal, PathCosts_jax_i, GaussianNoises_ustar_jax, epsilon, Ksamples_delta_t0, dt)
                
        u_star_pi_jax[current_mode_actual][i_t] = u0_star_jax
        allPathCosts_jax[i_t] = PathCosts_jax_i
        
        print("*** Var weight_jax", jnp.var(weights_jax))
        print("*** lambda_jax", 1.0 / jnp.mean(weights_jax**2))
        
        # ----------------------------------
        #   Visualize sampled trajectories
        # ----------------------------------
        show_samples = False
        if show_samples and (i_t==0):
            print(f"Plotting trajectory samples.")
            nsamples_plot = 10
            
            sorted_cost_indices = [index for index, _ in sorted(enumerate(PathCosts_jax_i), key=lambda x: x[1])]
            cost_tail_index = sorted_cost_indices[-nsamples_plot:]
            time_span_i = np.arange(start_time_i, end_time, dt).flatten()
            
            fig, axes = plot_sample_trajectory_slip(cost_tail_index, nt_i, time_span_i, 
                                                    Kmodes_jax_i, Ksamples_jax_i, Ksamples_ut, Ksamples_reset_args, 
                                                    mode_trj_ilqr, xt_trj_ilqr, ut_trj_ilqr, reset_args_ilqr,
                                                    modes, states, inputs, ref_reset_args,
                                                    init_state, target_state)
                
            plt.tight_layout()
            plt.show()
            
        show_sample_weights = False
        if show_sample_weights:
            plot_step = 1
            print("Plotting the sample weights and path costs.")
            ## -------- show path costs and weights in path integral control -------- 
            _, axes_weights = plt.subplots(2, 1, figsize=(8,6))
            (ax2, ax3) = axes_weights.flatten()
            ax2.grid(True)
            ax3.grid(True)
            
            ax2.bar(range(0,len(PathCosts_jax_i),plot_step), PathCosts_jax_i[::plot_step])
            ax2.set_title("Path Cost distribution")
            ax2.set_xlabel("Sample Number")
            ax2.set_ylabel("Costs")
            
            ax3.bar(range(0,len(weights_jax),plot_step), weights_jax[::plot_step])
            ax3.set_title("Weight distribution")
            ax3.set_xlabel("Sample Number")
            ax3.set_ylabel("Weights")
            
            plt.show()
        # --------------------------------------------- 
        #  Apply optimal control and go to next state  
        # ---------------------------------------------
        reset_args_actual[i_t] = event_args_actual[cnt_event_actual]
        
        actual_noise_i_pad = RndN_actual_padding[current_mode_actual][i_t]
        
        (_, 
         xt_next, 
         next_mode_actual, 
         _, 
         new_reset_arg) = hybrid_stochastic_integration_slip_padding(xt, current_mode_actual,
                                                                    u0_star_jax, actual_noise_i_pad, 
                                                                    epsilon, dt, dt_shrink, start_time_i, 
                                                                    reset_args_actual[i_t])

        next_mode_actual = int(next_mode_actual)
        # current_modechange = np.array([current_mode_actual, next_mode_actual])
        modes_pi_jax[i_t+1] = next_mode_actual
        trj_pi_jax[i_t+1] = xt_next
        
        # Update the hybrid event information under the actual controller
        reset_args_actual[i_t] = event_args_actual[cnt_event_actual]
        if (current_mode_actual!=next_mode_actual):
            print("----------- Mode changed for the actual controlled system ------------")
            event_args_actual.append(new_reset_arg)
            cnt_event_actual += 1

    # -------------
    # Compare cost
    # -------------
    dWs_zeros = [np.zeros((nt, n_inputs[0])), np.zeros((nt, n_inputs[1]))]
    cost_pi = compute_cost(modes_pi_jax, trj_pi_jax, u_star_pi_jax, dWs_zeros, target_state, states, Q_k, R_k, Q_T, epsilon,dt)
    cost_ilqr = compute_cost(mode_trj_ilqr, xt_trj_ilqr, ut_trj_ilqr, dWs_zeros, target_state, states, Q_k, R_k, Q_T, epsilon,dt)
    
    print("cost_pi:", cost_pi)
    print("cost_ilqr:", cost_ilqr)
    
    # -------------
    # Record data
    # -------------
    data_i = DataOneSample(modes_pi_jax, trj_pi_jax, u_star_pi_jax, 
                           mode_trj_ilqr, xt_trj_ilqr, ut_trj_ilqr, 
                           allPathCosts_jax, cost_pi, cost_ilqr, Ksamples_jax_saving)
    
    gc.collect()


    return cost_pi, cost_ilqr, data_i

def main(epsilon, n_samples, dt):
    print(f"The value of time discretization dt is: {dt}")
    print(f"The value of epsilon input is: {epsilon}")
    print(f"The value of number of samples input is: {n_samples}")
    
    n_exp = 20
    
    # === ilqr parameters ===
    # Initialize timings
    
    # ---------------- bouncing example -----------------
    dt_shrink = 0.9
    r0 = 1
    
    n_modes = 2
    
    # mode 1 (flight): x = [px, vx, pz, vz, theta], u = [theta_dot]
    # mode 2 (stance): x = [theta, theta_dot, r, r_dot], u = [r_delta, \tau_hip]
    
    # --------------
    # SLIP Dynamics
    # --------------
    # mode 1 (flight): x = [px, vx, pz, vz, theta], u = [theta_dot]
    # mode 2 (stance): x = [theta, theta_dot, r, r_dot], u = [r_delta, \tau_hip]
    
    # For the slip dynamics, mode 1 has 1 input, and mode 2 has 2 inputs. 
    n_states = [5, 4]
    n_inputs = [3, 2]
    
    # # ----------------------------
    # # Case 1: vertical bouncing
    # # ----------------------------
    # start_time = 0
    # end_time = 1.5
    # init_mode = 0
    # init_state = np.array([0.0, 0.0, 2.0, 0.0, np.pi/2], dtype=np.float64)    # Define the initial state to be the origin with no velocity
    # target_state = np.array([0.0, 0.0, 2.3, 0.0, np.pi/2], dtype=np.float64)  # Swing pendulum upright

    # # Time definitions
    # start_time = 0
    # end_time = 1.5
    # time_span = np.arange(start_time, end_time, dt).flatten()
    # nt = len(time_span)
    
    # # Terminal cost 
    # target_mode = 0
    # Q_T = 0.01*np.eye(n_states[0])
    
    # # Running costs
    # Q_k = [np.zeros((n_states[0],n_states[0])), np.zeros((n_states[1],n_states[1]))] # zero weight to penalties along a strajectory since we are finding a trajectory
    # R_k = [np.eye(n_inputs[0]), np.eye(n_inputs[1])]
    
    # init_reset_args = [np.array([0.0]) for _ in range(nt)]
    # target_reset_args = [np.array([0.0]) for _ in range(nt)]
    
    # ---------------------------
    #  Case 2: Jumping one step
    # ---------------------------
    init_mode = 1
    
    # Time definitions
    start_time = 0
    end_time = 0.5
    
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)
    
    print("nt: ", nt)
    
    # Terminal cost 
    target_mode = 0
    Q_T = 60.0*np.eye(n_states[0])
    # Q_T[4,4] = 500.0
    # Q_T[1,1] = 2000.0
    # Q_T[3,3] = 2000.0
    
    # Running costs
    Q_k = [np.zeros((n_states[0],n_states[0])), np.zeros((n_states[1],n_states[1]))] # zero weight to penalties along a strajectory since we are finding a trajectory
    R_k = [np.eye(n_inputs[0]), np.eye(n_inputs[1])]
    
    init_theta_deg = 100
    init_theta = init_theta_deg / 180 * np.pi
    init_state = np.array([init_theta, -4.0, 0.5*r0, 0.0], dtype=np.float64)
    target_state = np.array([1.1, 2.5, 1.5, 0.0, np.pi/3], dtype=np.float64)  # Swing pendulum upright
    init_reset_args = [np.array([0.0]) for _ in range(nt)]
    target_reset_args = [np.array([0.0]) for _ in range(nt)]
    
    # ---------------- / slip example -----------------
    
    # ================================
    # solve for hybrid ilqr proposal
    # ================================
    exp_params = ExpParams()
    
    initial_guess = [0.0*np.ones((np.shape(time_span)[0],n_inputs[0])), 0.0*np.ones((np.shape(time_span)[0],n_inputs[1]))]
    smooth_dynamics = [symbolic_flight_dynamics_slip, symbolic_stance_dynamics_slip]
    
    exp_params.update_params(n_modes, init_mode, target_mode, n_states, 
                             init_state, target_state, 
                             start_time, end_time, dt, dt_shrink,
                             initial_guess, 
                             epsilon, n_exp, n_samples, 
                             Q_k, R_k, Q_T, smooth_dynamics, 
                             event_detect_discrete_slip, 
                             plot_slip, convert_state_21_slip, 
                             init_reset_args, target_reset_args)
    exp_data = ExpData(exp_params)
    
    print("===================== Solving for h-iLQG proposal controller =====================")
    hybrid_ilqr_result = solve_ilqr(exp_params, detect=True, verbose=False)
    
    (timespan,modes,states,inputs,
     k_feedforward,K_feedback,current_cost,states_iter,
     ref_modechanges,reference_extension_helper, ref_reset_args) = hybrid_ilqr_result
    
    exp_data.add_nominal_data(hybrid_ilqr_result)
    exp_data.add_plotting_function(plot_slip)
    
    # ---------------------
    #  Show h-iLQG results
    # ---------------------
    show_results = True
    if show_results:
        plot_slip(time_span, modes, states, inputs, init_state, target_state, nt, ref_reset_args)

    # ============================================================================================================
    #                                   // End of Solve for hybrid ilqg proposal
    # ============================================================================================================

    # ============================================================================================================
    #               Path Integral Control, do n_exp number of experiments under different randomness
    # ============================================================================================================
    
    cost_pi_exp = np.zeros(n_exp)
    cost_ilqr_exp = np.zeros(n_exp)
    
    # ---------------------------------------
    #     Main Loop over the experiments
    # ---------------------------------------
    for i_exp in range(n_exp):
        result = run_experiment(i_exp, nt, n_samples, n_states, n_inputs, 
                                init_mode, init_state, target_state, hybrid_ilqr_result, 
                                start_time, end_time, dt, dt_shrink, 
                                Q_k, Q_T, R_k, epsilon, init_reset_args)

        cost_pi_exp[i_exp] = result[0]
        cost_ilqr_exp[i_exp] = result[1]
        exp_data.add_data(i_exp, result[2])
        
        
    print("E[cost_pi]: ", np.mean(cost_pi_exp))
    print("E[cost_ilqr_exp]: ", np.mean(cost_ilqr_exp))

    # =========== save data ===========
    from datetime import datetime
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"data_{formatted_datetime}_{script_filename}_{n_exp}experiments_{n_samples}samples_eps_{epsilon}_coupling_dt_{dt}.pickle"
    
    # save_root = '/hddscratch/hyu419/hybrid_pathintegral/exp_200'
    # save_root = '/home/hzyu/git/HybridPathIntegral/experiments'
    save_root = '/ssdscratch/hyu419/hybrid_pathintegral/new_exp'
    save_path = f"{save_root}/data/slip/{filename}"
    exp_data.dump(save_path)
    
    print(" =================== Saved data to: =================== \n", save_path)
    

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="The epsilon parameter.")
    parser.add_argument("--epsilon", type=float, default=0.0008, help="The process noise intensity value, epsilon.")
    parser.add_argument("--nsamples", type=int, default=1000, help="The number of samples used in path integral control.")
    parser.add_argument("--dt", type=int, default=0.0008, help="The time discretization.")
    
    args = parser.parse_args()

    main(args.epsilon, args.nsamples, args.dt)
    
import numpy as np

import os
import sys
file_path = os.path.abspath(__file__)
exp_dir = os.path.dirname(file_path)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
root_dir = os.path.abspath(os.path.join(exp_dir, '..'))
sys.path.append(root_dir)

import jax.numpy as jnp

# Import iLQR class and reference extension handler
from hybrid_ilqr.h_ilqr_discrete import solve_ilqr, extract_extensions
# Importing path integral control
from hybrid_pathintegral.hybrid_pathintegral import *
from hybrid_pathintegral.sampling_rollout_jax_bouncing import *

# Import plotting
import matplotlib.pyplot as plt
# Import experiment parameter class
from experiments.exp_params import *
# Import bouncing ball dynamics
from hybrid_pathintegral.sampling_rollout_jax_bouncing import sample_bouncing_jax
from dynamics.dynamics_bouncing import *
from dynamics.dynamics_discrete_bouncing import *


import gc
gc.collect()


def run_experiment(i_exp, nt, n_samples, n_states, n_inputs, 
                    init_mode, init_state, target_state, hybrid_ilqr_result, 
                    start_time, end_time, dt, dt_shrinkingrate, 
                    Q_k, Q_T, R_k, epsilon, init_reset_args):
    
    (timespan,modes,states,inputs,
     k_feedforward,K_feedback,
     current_cost,states_iter,
     ref_modechanges,reference_extension_helper, ref_reset_args) = hybrid_ilqr_result
    
    RndN_actual = [np.random.randn(nt, 1), np.random.randn(nt, 1)]
    
    # =================================================================
    #                     Hybrid ilqr for comparison
    # ================================================================= 
    print("=================== Hybrid ilqr under uncertainties for comparison ===================")
    # -------------- result collectors, hybrid ilqr proposal --------------
    xt_trj_ilqr = [np.array([0.0]) for _ in range(nt)]
    u_trj_ilqr = [np.zeros((nt, n_inputs[0])), np.zeros((nt, n_inputs[1]))]
    xt_trj_ilqr[0] = init_state
    
    (mode_trj_ilqr, 
     xt_trj_ilqr, 
     u_trj_ilqr, 
     cost_ilqr, _, _) = hybrid_stochastic_feedback_rollout_discrete_bouncing(init_mode, 
                                                                            init_state, 
                                                                            n_inputs, 
                                                                            states, modes, 
                                                                            inputs, K_feedback, k_feedforward, 
                                                                            target_state, Q_T,
                                                                            start_time, dt, 
                                                                            epsilon, RndN_actual, dt_shrinkingrate, 
                                                                            reference_extension_helper,
                                                                            init_reset_args)
    
    show_hilqr_results = False
    if show_hilqr_results:
        time_span = np.arange(start_time, end_time, dt).flatten()
        plot_bouncingball(time_span, mode_trj_ilqr, xt_trj_ilqr, u_trj_ilqr, init_state, target_state, nt, trj_labels='iLQG-stochastic')
        plt.show()
        
    
    # mode-dependent reference states. Assuming 2 modes
    # States with padded dimensions
    states_0 = np.zeros((nt, 2))
    states_1 = np.zeros((nt, 2))
    
    inputs_0 = np.zeros((nt, 1))
    inputs_1 = np.zeros((nt, 1))
    
    K_feedback_0 = np.zeros((nt, 1, 2))
    K_feedback_1 = np.zeros((nt, 1, 2))
    
    k_feedforward_0 = np.zeros((nt, 1))
    k_feedforward_1 = np.zeros((nt, 1))
    
    for i in range(nt):
        mode_i = modes[i]
        if mode_i == 0:
            states_0[i, :n_states[0]] = states[i]
            inputs_0[i, :n_inputs[0]] = inputs[0][i]
            K_feedback_0[i, :n_inputs[0], :n_states[0]] = K_feedback[i]
            k_feedforward_0[i, :n_inputs[0]] = k_feedforward[i]
        
        if mode_i == 1:
            states_1[i, :n_states[1]] = states[i]
            inputs_1[i, :n_inputs[1]] = inputs[1][i]
            K_feedback_1[i, :n_inputs[1], :n_states[1]] = K_feedback[i]
            k_feedforward_1[i, :n_inputs[1]] = k_feedforward[i]
        
    ref_reset_args = np.array(ref_reset_args)
    
    ref_reset_args = np.array(ref_reset_args)
    k_feedforward_0 = np.array(k_feedforward_0)
    k_feedforward_1 = np.array(k_feedforward_1)
    K_feedback_0 = np.array(K_feedback_0)
    K_feedback_1 = np.array(K_feedback_1)
    
    print(f"=================== The experiment index: {i_exp} ===================" )
    
    # -------------- result collectors, jax --------------
    modes_pi_jax = np.zeros(nt, dtype=np.int64)
    trj_pi_jax = [np.array([0.0]) for _ in range(nt)]

    u_star_pi_jax = [np.zeros((nt, n_inputs[0])), np.zeros((nt, n_inputs[1]))]
    allPathCosts_jax = np.zeros((nt-1, n_samples))
    
    # -------------- initialize the control loop -------------- 
    x0_jax = jnp.asarray(init_state)
    xt = x0_jax
    xt_ilqr = init_state
    
    current_mode_actual = init_mode
    next_mode_actual = current_mode_actual
    # current_modechange = np.array([current_mode_actual, next_mode_actual])
    
    modes_pi_jax[0] = init_mode
    trj_pi_jax[0] = x0_jax
    
    # current_ref_modechange_ilqr = (0, 0)
    cnt_mismatch = 0
    
    # Assuming 2-mode system
    reset_args_actual = ref_reset_args
    event_args_actual = [ref_reset_args[0]]
    cnt_event_actual = 0
    
    # ---------------------------------
    #  Extract the extended references 
    # ---------------------------------
    (v_mode_change_ref, v_ref_ext_bwd, v_ref_ext_fwd, 
    v_Kfb_ref_ext_bwd, v_Kfb_ref_ext_fwd, 
    v_kff_ref_ext_bwd, v_kff_ref_ext_fwd, _) = extract_extensions(reference_extension_helper, start_index = 0)
    
    
    show_trj_extensions = False
    if show_trj_extensions:
        time_span = np.arange(start_time, end_time, dt).flatten()
         
        fig_ext, axes_ext = plt.subplots(3, 1, figsize=(10, 12))
        (ax_ext_1, ax_ext_2, ax_ext_3) = axes_ext.flatten()

        ax_ext_1.grid(True)
        ax_ext_2.grid(True)
        ax_ext_3.grid(True)
        
        ax_ext_1.plot(time_span, v_ref_ext_bwd[0][:-1, 0], color='r', label='Backward extension')
        ax_ext_1.plot(time_span, v_ref_ext_fwd[0][:-1, 0], color='b', label='Forward extension')
        ax_ext_1.plot(time_span, states[:, 0], color='k')
        
        ax_ext_2.plot(time_span, v_ref_ext_bwd[0][:-1, 1], color='r', label='Backward extension')
        ax_ext_2.plot(time_span, v_ref_ext_fwd[0][:-1, 1], color='b', label='Forward extension')
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
        
        # ------------------------------------------------------------------------------------
        #         Calculate the slice of the variables for the current future horizon 
        # ------------------------------------------------------------------------------------
        
        # actual modes and states
        current_mode_actual = modes_pi_jax[i_t]
        xt = trj_pi_jax[i_t]
        
        # references
        states_0_i = states_0[i_t:, :]
        states_1_i = states_1[i_t:, :]
        inputs_0_i = inputs_0[i_t:, :]
        inputs_1_i = inputs_1[i_t:, :]
        
        states_i = [states_0_i, states_1_i]
        inputs_i = [inputs_0_i, inputs_1_i]
        
        # ref_modechange_i = ref_modechanges[i_t:]
        modes_i = modes[i_t:]
        
        K_feedback_0_i = K_feedback_0[i_t:,:]
        K_feedback_1_i = K_feedback_1[i_t:,:]
        
        k_feedforward_0_i = k_feedforward_0[i_t:,:]
        k_feedforward_1_i = k_feedforward_1[i_t:,:]
        
        K_feedback_i = [K_feedback_0_i, K_feedback_1_i]
        k_feedforward_i = [k_feedforward_0_i, k_feedforward_1_i]
        
        # states_i = states[i_t:,:]
        # inputs_i = inputs[:, i_t:,:]
        # ref_modechange_i = ref_modechanges[i_t:]
        # modes_i = modes[i_t:]
        # K_feedback_i = K_feedback[i_t:,:]
        # k_feedforward_i = k_feedforward[i_t:,:]
        
        ref_current_mode = ref_modechanges[i_t][0]
        
        # Randomness is mode dependent, same as control. Assuming 2-mode system.
        n_modes = 2
        Ksamples_jax_i = np.zeros((n_samples, nt_i, n_modes))
        GaussianNoise_i = [np.random.randn(n_samples, nt_i, 1), np.random.randn(n_samples, nt_i, 1)]
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
        v_kff_ref_ext_bwd_i, v_kff_ref_ext_fwd_i, _) = extract_extensions(reference_extension_helper, start_index = i_t) 
        
        # ====================
        # Sampling using jax 
        # ====================
        
        # ---------------------------------------------------------------------------------------
        #                               Sample future trajectories
        # ---------------------------------------------------------------------------------------
        
        # timer_start_tic = time.perf_counter()
        (Ksamples_ts_i, Kmodes_jax_i, Ksamples_jax_i, PathCosts_jax_i, 
        Ksamples_ut, Ksamples_xref, Ksamples_Kfb_mode, 
        Ksamples_kff_mode, Ksamples_reset_args) = sample_bouncing_jax(n_samples, xt, 
                                                                        current_mode_actual, 
                                                                        states_0_i, states_1_i, 
                                                                        modes_i, 
                                                                        inputs_0_i, inputs_1_i, 
                                                                        K_feedback_0_i, k_feedforward_0_i, 
                                                                        K_feedback_1_i, k_feedforward_1_i, 
                                                                        target_state, Q_T, 
                                                                        dt, dt_shrinkingrate, 
                                                                        epsilon, 
                                                                        GaussianNoise_i[0], GaussianNoise_i[1], 
                                                                        v_mode_change_ref_i, 
                                                                        v_ref_ext_fwd_i, v_ref_ext_bwd_i, 
                                                                        v_Kfb_ref_ext_fwd_i, v_kff_ref_ext_fwd_i, 
                                                                        v_Kfb_ref_ext_bwd_i, v_kff_ref_ext_bwd_i, 
                                                                        init_reset_args_i)
        
        # --------------------------------------------------------
        #               Update the control proposal
        # --------------------------------------------------------
        # Compute proposal control, possibly with early arrival
        ubar_i = inputs_i[current_mode_actual][0]
        K_fb = K_feedback_i[current_mode_actual][0]
        k_ff = k_feedforward_i[current_mode_actual][0]
        xref_i = states_i[current_mode_actual][0]
        
        if cond_mode_mismatch_bouncing(current_mode_actual, ref_current_mode):
            print("--------------- mode mismatch happened ---------------")
            xref_i, K_fb, k_ff, cnt_mismatch = reaction_mode_mismatch(i_t, 
                                                                      current_mode_actual, ref_current_mode, 
                                                                      v_ref_ext_fwd[0], v_ref_ext_bwd[0], 
                                                                      v_mode_change_ref[0],
                                                                      v_Kfb_ref_ext_fwd[0], v_kff_ref_ext_fwd[0], 
                                                                      v_Kfb_ref_ext_bwd[0], v_kff_ref_ext_bwd[0], 
                                                                      cnt_mismatch,
                                                                      cond_early_arrival=cond_early_arrival_bouncing)
        
        u0_proposal = ubar_i + K_fb@(xt - xref_i) + k_ff
        
        GaussianNoises_ustar_jax = GaussianNoise_i[current_mode_actual][:,0,:]
        Ksamples_delta_t0 = Ksamples_ts_i[:,1] - Ksamples_ts_i[:,0]
        u0_star_jax, weights_jax = update_u0_pathintegral_jax(u0_proposal, PathCosts_jax_i, GaussianNoises_ustar_jax, epsilon, Ksamples_delta_t0, dt)
        
        u_star_pi_jax[current_mode_actual][i_t] = u0_star_jax
        allPathCosts_jax[i_t] = PathCosts_jax_i
        
        print("*** Var weight_jax", jnp.var(weights_jax))
        print("*** lambda_jax", 1.0 / jnp.mean(weights_jax**2))
        
        # -------------------------------
        # Visualize sampled trajectories
        # -------------------------------
        show_samples = False
        if show_samples:
            _, axes_sample = plt.subplots(2, 1)
            (ax1_sample, ax2_sample) = axes_sample.flatten()
            
            ax1_sample.grid(True)
            ax2_sample.grid(True)
            
            # ----------------------
            #  Plot all the samples 
            # ----------------------
            xt_trj_ilqr = np.asarray(xt_trj_ilqr)
            states_ref = np.asarray(states)
                        
            
            for i_s in range(20):
                ax1_sample.plot(Ksamples_jax_i[i_s,:,0], Ksamples_jax_i[i_s,:,1],'b', alpha=0.2)
                ax2_sample.plot(Ksamples_jax_i[i_s,:,0], 'b', alpha=0.2)
                # ax2_sample.plot(Ksamples_ut[i_s,:,0], 'b', alpha=0.2)
                
            ax1_sample.plot(xt_trj_ilqr[:,0], xt_trj_ilqr[:,1],'r', alpha=0.9, label='h-iLQR stochastic control')
            ax1_sample.plot(states_ref[:,0], states_ref[:,1],'k', alpha=0.9, label='Reference')
            ax1_sample.plot(Ksamples_jax_i[i_s,:,0], Ksamples_jax_i[i_s,:,1],'b', alpha=0.2, label='Samples')
            # ax2_sample.plot(Ksamples_ut[i_s,:,0], 'b', alpha=0.2, label='sampled stochastic control')
            ax2_sample.plot(Ksamples_jax_i[i_s,:,0], 'b', alpha=0.2, label='sampled stochastic control')
            
            fig4, ax10 = plt.subplots()
            inputs_ref = np.asarray(inputs)
            u_trj_ilqr = np.asarray(u_trj_ilqr)
            
            # ax2_sample.plot(inputs_ref[0, :, 0], 'k', label='Ref control')
            # ax2_sample.plot(u_trj_ilqr[0, :, 0], 'r', label='h-iLQR stochastic control')
            
            ax2_sample.plot(states_ref[:, 0], 'k', label='Ref control')
            ax2_sample.plot(xt_trj_ilqr[:, 0], 'r', label='h-iLQR stochastic control')
            
            ax1_sample.legend()
            ax2_sample.legend()
            
            plt.show()
            
            ax1_sample.scatter(target_state[0], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
            ax1_sample.scatter(init_state[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')
            ax1_sample.scatter(xt[0], xt[1], color='b', marker='x', s=30.0, linewidths=6, label='Current')
            
        # ----------------------------------- 
        
        # --------------------------------------------- 
        # Apply optimal control and go to next state  
        # ---------------------------------------------
        reset_args_actual[i_t] = event_args_actual[cnt_event_actual]
        actual_noise_i = RndN_actual[current_mode_actual][i_t]
        
        xt_next, next_mode_actual, _, new_reset_arg = hybrid_stochastic_integration_bouncing(xt, current_mode_actual,
                                                                                                u0_star_jax, actual_noise_i, 
                                                                                                epsilon, dt, dt_shrinkingrate, 
                                                                                                start_time_i, reset_args_actual[i_t])

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

    # ---------------
    #  Compare cost
    # ---------------    
    cost_pi = compute_cost_nonoise(modes_pi_jax, trj_pi_jax, u_star_pi_jax, target_state, states, Q_k, R_k, Q_T,dt)
    cost_ilqr = compute_cost_nonoise(mode_trj_ilqr, xt_trj_ilqr, u_trj_ilqr, target_state, states, Q_k, R_k, Q_T,dt)
    
    print("cost_pi:", cost_pi)
    print("cost_ilqr:", cost_ilqr)
    
    # -------------
    #  Record data
    # -------------
    data_i = DataOneSample(modes_pi_jax, trj_pi_jax, u_star_pi_jax, 
                           mode_trj_ilqr, xt_trj_ilqr, u_trj_ilqr, 
                           allPathCosts_jax, cost_pi, cost_ilqr, Ksamples_jax_saving)

    gc.collect()


    return cost_pi, cost_ilqr, data_i

def main(epsilon, n_samples, dt):
    
    print(f"The value of epsilon input is: {epsilon}")
    print(f"The value of number of samples input is: {n_samples}")
    print(f"The value of time discretization dt is: {dt}")
    # === ilqr parameters ===
    # Initialize timings
    
    
    # =========== save data ===========
    # save_root = '/hddscratch/hyu419/hybrid_pathintegral/exp_200'
    save_root = '/home/hzyu/git/HybridPathIntegral/experiments'
    # save_root = '/hddscratch/hyu419/hybrid_pathintegral/new_exp'
    
    save_path = f"{save_root}/data/new_exp/bouncing"
    
    from pathlib import Path
    if not Path(save_path).exists():
        print(f"File {save_path} does not exist.")
        sys.exit(1)  # Exit the program with a non-zero status code
    else:
        print(f"File {save_path} exists.")
        
    n_exp = 1
    
    # Set desired state
    n_modes = 2
    
    # the state and control dimensions, mode-dependent
    n_states = [2, 2]
    n_inputs = [1, 1]
    
    # ---------------- multiple bouncing example -----------------
    dt = 0.0015
    epsilon = 2.0
    dt_shrink = 0.95
    
    start_time = 0
    end_time = 3.0
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)
    
    print(f"Number of time discretizations: {nt}")

    init_state = np.array([5, 1.5])    # Define the initial state to be the origin with no velocity
    target_state = np.array([1.2, 0.0])  # Swing pendulum upright
     
    init_mode = 0
    target_mode = 0
    Q_T = 60*np.eye(n_states[0])
    
    initial_guess = [1.0*np.ones((np.shape(time_span)[0],n_inputs[0])), 1.0*np.ones((np.shape(time_span)[0],n_inputs[1]))]
    
    # ------------------ one bounce --------------------
    # # dt = 0.005
    # dt_shrink = 0.9
    # start_time = 0
    # end_time = 2.0
    # time_span = np.arange(start_time, end_time, dt).flatten()
    # nt = len(time_span)

    # init_state = np.array([5, 1.5], dtype=np.float64)    # Define the initial state to be the origin with no velocity
    # target_state = np.array([2.5, 0], dtype=np.float64)  # Swing pendulum upright
    
    # init_mode = 0
    # initial_guess = [0.5*np.ones((np.shape(time_span)[0],n_inputs[0])), 0.5*np.ones((np.shape(time_span)[0],n_inputs[1]))]

    # ---------------- / bouncing example -----------------

    # ===== OR =====
    # dt = 5e-5
    # # ------------- verification with no contact ------------- 
    # start_time = 0
    # end_time = 1.0
    # time_span = np.arange(start_time, end_time, dt).flatten()
    # nt = len(time_span)

    # init_state = np.array([5, 1.5])    # Define the initial state to be the origin with no velocity
    # target_state = np.array([1.0, 0.0])
    
    # target_mode = 0
    # Q_T = 200*np.eye(n_states[0], dtype=np.float64)
    # Q_T[0,0] = 2000.0

    # # ------------- /verification with no contact ------------- 

    # ---------------------------- 
    # Define weighting matrices
    # ----------------------------
    Q_k = [np.zeros((n_states[0],n_states[0]), dtype=np.float64), np.zeros((n_states[1],n_states[1]), dtype=np.float64)] # zero weight to penalties along a strajectory since we are finding a trajectory
    R_k = [np.eye(n_inputs[0], dtype=np.float64), np.eye(n_inputs[1], dtype=np.float64)]
    
    init_reset_args = [np.array([0.0]) for _ in range(nt)]
    target_reset_args = [np.array([0.0]) for _ in range(nt)]
    
    # ============================================================================================================
    #                                       Solve for hybrid ilqg proposal
    # ============================================================================================================
    exp_params = ExpParams()
    
    flow_dynamics = [symbolic_dynamics_bouncing, symbolic_dynamics_bouncing]
    
    exp_params.update_params(n_modes, init_mode, target_mode, n_states, init_state, target_state, 
                             start_time, end_time, dt, dt_shrink, initial_guess, 
                             epsilon, n_exp, n_samples, 
                             Q_k, R_k, Q_T, flow_dynamics, 
                             event_detect_bouncing_discrete, 
                             plot_bouncingball, 
                             convert_state_21_bouncing, 
                             init_reset_args, target_reset_args)
    exp_data = ExpData(exp_params)
    
    print("===================== Solving for h-iLQG proposal controller =====================")
    hybrid_ilqr_result = solve_ilqr(exp_params, detect=True, verbose=False)
    
    (timespan,modes,states,inputs,
     k_feedforward,K_feedback,
     current_cost,states_iter,
     ref_modechanges,reference_extension_helper, ref_reset_args) = hybrid_ilqr_result
    
    
    exp_data.add_nominal_data(hybrid_ilqr_result)
    exp_data.add_plotting_function(plot_bouncingball)
    
    # ---------------------
    #  Show h-iLQG results
    # ---------------------
    show_results = False
    if show_results:
        plot_bouncingball(time_span, modes, states, inputs, init_state, target_state, nt)
    
    
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
    
    save_file = f"{save_path}/{filename}"
    exp_data.dump(save_file)
    
    print(" =================== Saved data to: =================== \n", save_file)
    
    
    # ---------------------------------------------
    # plot the path integral controlled trajectory
    # --------------------------------------------- 
    show_results = False
    if show_results:
        fig4, axes = plt.subplots(1, 2)
        (ax4, ax5) = axes.flatten()
        fig4, axes, fig5, ax6 =  plot_bouncingball_nexp(n_exp, exp_data, time_span, init_state, 
                                                            target_state, args=None)
        
        fig4.tight_layout()
        fig5.tight_layout()
        
        plt.show()


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="The epsilon parameter.")
    parser.add_argument("--epsilon", type=float, default=2.0, help="The process noise intensity value, epsilon.")
    parser.add_argument("--nsamples", type=int, default=500, help="The number of samples used in path integral control.")
    parser.add_argument("--dt", type=int, default=0.0025, help="The time discretization.")
    
    args = parser.parse_args()

    main(args.epsilon, args.nsamples, args.dt)
    
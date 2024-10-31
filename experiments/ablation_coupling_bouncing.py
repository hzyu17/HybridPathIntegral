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
from dynamics.dynamics_bouncing import *
from dynamics.dynamics_discrete_bouncing import *
# Importing path integral control
from hybrid_pathintegral.hybrid_pathintegral import *
from hybrid_pathintegral.sampling_rollout_jax_bouncing import *
# Import experiment parameter class
from experiments.exp_params import *

from hybrid_pathintegral.sampling_rollout_jax_bouncing import sample_bouncing_jax

import copy


import gc
gc.collect()

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
def run_experiment(i_exp, nt, n_samples, n_states, n_inputs, 
                    init_mode, init_state, target_state, hybrid_ilqr_result, 
                    start_time, end_time, dt, dt_shrink, 
                    Q_k, Q_T, R_k, epsilon, init_reset_args):
    
    gc.collect()

    (time_span,modes,states,inputs,
     k_feedforward,K_feedback,
     current_cost,states_iter,
     ref_modechanges,
     ref_ext_helper, ref_reset_args) = hybrid_ilqr_result
        
    RndN_actual = [np.random.randn(nt, n_inputs[0]), np.random.randn(nt, n_inputs[1])]
    
    
    # =================================================================
    #                     Hybrid ilqr rollout
    # ================================================================= 
    # -------------- result collectors, hybrid ilqr proposal --------------
    xt_trj_ilqr = [np.array([0.0]) for _ in range(nt)]
    ut_trj_ilqr = [np.zeros((nt, n_inputs[0])), np.zeros((nt, n_inputs[1]))]
    
    xt_trj_ilqr[0] = init_state
    
    inputs_zero = [np.zeros_like(inputs[0]), np.zeros_like(inputs[1])]
    states_zero = [np.zeros_like(states[i]) for i in range(len(states))]
    K_feedback_zero = [np.zeros_like(K_feedback[i]) for i in range(len(K_feedback))]
    k_feedforward_zero = [np.zeros_like(k_feedforward[i]) for i in range(len(K_feedback))]
    
    reference_extension_helper_zero = ref_ext_helper
    mc = reference_extension_helper_zero[0]["Mode Change"]
    reference_extension_helper_zero[0]["Feedback gains"][mc[0]] = np.zeros_like(reference_extension_helper_zero[0]["Feedback gains"][mc[0]])
    reference_extension_helper_zero[0]["Feedback gains"][mc[1]] = np.zeros_like(reference_extension_helper_zero[0]["Feedback gains"][mc[1]])
    reference_extension_helper_zero[0]["Feedforward gains"][mc[0]] = np.zeros_like(reference_extension_helper_zero[0]["Feedforward gains"][mc[0]])
    reference_extension_helper_zero[0]["Feedforward gains"][mc[1]] = np.zeros_like(reference_extension_helper_zero[0]["Feedforward gains"][mc[1]])
    
    (mode_trj_ilqr, 
     xt_trj_ilqr, 
     u_trj_ilqr, 
     cost_ilqr, _, _) = h_stoch_fb_rollout_bouncing(init_mode, 
                                                    init_state, n_inputs, 
                                                    states_zero, modes, 
                                                    inputs_zero, K_feedback_zero, 
                                                    k_feedforward_zero, 
                                                    target_state, Q_T,
                                                    start_time, dt, 
                                                    epsilon, RndN_actual, dt_shrink, 
                                                    reference_extension_helper_zero,
                                                    init_reset_args)
    
    print("------------------- finished rollout under h-ilqr ---------------------")

    show_hilqr_results = False
    if show_hilqr_results:
        time_span = np.arange(start_time, end_time, dt).flatten()
        plot_bouncingball(time_span, mode_trj_ilqr, xt_trj_ilqr, u_trj_ilqr, init_state, target_state, nt, trj_labels='iLQG-stochastic')
        plt.show()
    
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
    allPathCosts_jax_coupled = np.zeros((nt-1, n_samples))
    
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
    (v_mode_change_ref, v_ext_bwd, v_ext_fwd, 
    v_Kfb_ext_bwd, v_Kfb_ext_fwd, 
    v_kff_ext_bwd, v_kff_ext_fwd, _) = extract_extensions(ref_ext_helper, start_index = 0)
    
    
    show_trj_extensions = False
    if show_trj_extensions:
        time_span = np.arange(start_time, end_time, dt).flatten()
         
        fig_ext, axes_ext = plt.subplots(3, 1, figsize=(10, 12))
        (ax_ext_1, ax_ext_2, ax_ext_3) = axes_ext.flatten()

        ax_ext_1.grid(True)
        ax_ext_2.grid(True)
        ax_ext_3.grid(True)
        
        ax_ext_1.plot(time_span, v_ext_bwd[0][:-1, 0], color='r', label='Backward extension')
        ax_ext_1.plot(time_span, v_ext_fwd[0][:-1, 0], color='b', label='Forward extension')
        ax_ext_1.plot(time_span, states[:, 0], color='k')
        
        ax_ext_2.plot(time_span, v_ext_bwd[0][:-1, 1], color='r', label='Backward extension')
        ax_ext_2.plot(time_span, v_ext_fwd[0][:-1, 1], color='b', label='Forward extension')
        ax_ext_2.plot(time_span, states[:, 1], color='k')
        
        ax_ext_3.plot(v_ext_bwd[0][:, 0], v_ext_bwd[0][:, 1], color='r', label='Backward extension')
        ax_ext_3.plot(v_ext_fwd[0][:, 0], v_ext_fwd[0][:, 1], color='b', label='Forward extension')
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
        
        # actual modes and states
        current_mode_actual = modes_pi_jax[i_t]
        xt = trj_pi_jax[i_t]
        
        # =============================== Create samples under zero-control input ==========================        
        
        states_0_i_zero = np.zeros_like(states_0[i_t:, :])
        states_1_i_zero = np.zeros_like(states_1[i_t:, :])
        inputs_0_i_zero = np.zeros_like(inputs_0[i_t:, :])
        inputs_1_i_zero = np.zeros_like(inputs_1[i_t:, :])
        
        states_i = [states_0_i_zero, states_1_i_zero]
        inputs_i = [inputs_0_i_zero, inputs_1_i_zero]
        
        K_feedback_0_i_zero = np.zeros_like(K_feedback_0[i_t:,:])
        k_feedforward_0_i_zero = np.zeros_like(k_feedforward_0[i_t:,:])
        K_feedback_1_i_zero = np.zeros_like(K_feedback_1[i_t:,:])
        k_feedforward_1_i_zero = np.zeros_like(k_feedforward_1[i_t:,:])
        
        # ref_modechange_i = ref_modechanges[i_t:]
        modes_i = modes[i_t:]
        
        ref_current_mode = ref_modechanges[i_t][0]
        
        # Randomness is mode dependent, same as control. Assuming 2-mode system.
        n_modes = 2
        Ksamples_jax_i = np.zeros((n_samples, nt_i, n_modes))
        GaussianNoise_i = [np.random.randn(n_samples, nt_i, 1), np.random.randn(n_samples, nt_i, 1)]
        init_reset_args_i = np.array(ref_reset_args[i_t])
        
        # ----------------------------------------------------------
        #   Extract the extended references for the future horizon
        # ----------------------------------------------------------
        (v_mode_change_ref_i, v_ref_ext_bwd_i, v_ref_ext_fwd_i, 
        v_Kfb_ref_ext_bwd_i, v_Kfb_ref_ext_fwd_i, 
        v_kff_ref_ext_bwd_i, v_kff_ref_ext_fwd_i, v_tevent_i) = extract_extensions(ref_ext_helper, 
                                                                                    start_index = i_t, 
                                                                                    padding=True) 
        
        # ====================
        # Sampling using jax 
        # ====================
        
        # ---------------------------------------------------------------------------------------
        #                               Sample future trajectories
        # ---------------------------------------------------------------------------------------                    
        
        # --------------------------- 
        # Coupling of the randomness
        # ---------------------------
        # GaussianNoise_i_coupled = [np.random.randn(n_samples, nt_i, max_n_inputs), np.random.randn(n_samples, nt_i, max_n_inputs)]
        GaussianNoise_i_coupled = copy.deepcopy(GaussianNoise_i)
        GaussianNoise_i_coupled[0][int(n_samples/2):, :] = GaussianNoise_i_coupled[0][:int(n_samples/2), :]
        GaussianNoise_i_coupled[1][int(n_samples/2):, :] = GaussianNoise_i_coupled[1][:int(n_samples/2), :]
        GaussianNoise_i_coupled[0][int(n_samples/2):, 0] = -GaussianNoise_i_coupled[0][:int(n_samples/2), 0]
        GaussianNoise_i_coupled[1][int(n_samples/2):, 0] = -GaussianNoise_i_coupled[1][:int(n_samples/2), 0]
        
        print("--------------- Start Sampling ---------------")
        target_state_t = np.tile(target_state, (nt_i,1))
        Qk_t = np.tile(Q_k[0], (nt_i,1,1))
        (Ksamples_ts_i_coupled, Kmodes_jax_i_coupled, 
         Ksamples_jax_i_coupled, PathCosts_jax_i_coupled, 
         Ksamples_ut, Ksamples_xref, Ksamples_Kfb_mode, 
         Ksamples_kff_mode, Ksamples_reset_args) = sample_bouncing_jax(n_samples, xt, 
                                                                        current_mode_actual, 
                                                                        states_0_i_zero, states_1_i_zero, 
                                                                        modes_i, 
                                                                        inputs_0_i_zero, inputs_1_i_zero, 
                                                                        K_feedback_0_i_zero, k_feedforward_0_i_zero, 
                                                                        K_feedback_1_i_zero, k_feedforward_1_i_zero, 
                                                                        target_state_t, Q_T, Qk_t,
                                                                        dt, dt_shrink, 
                                                                        epsilon, 
                                                                        GaussianNoise_i_coupled[0], GaussianNoise_i_coupled[1], 
                                                                        v_mode_change_ref_i, 
                                                                        v_ref_ext_fwd_i, v_ref_ext_bwd_i, 
                                                                        v_Kfb_ref_ext_fwd_i, v_kff_ref_ext_fwd_i, 
                                                                        v_Kfb_ref_ext_bwd_i, v_kff_ref_ext_bwd_i, 
                                                                        init_reset_args_i)
         
        print("--------------- End Sampling ---------------")
         
        # ----------------------------------- samples using un-coupled randomnesses ---------------------------------------
        (Ksamples_ts_i, Kmodes_jax_i, Ksamples_jax_i, PathCosts_jax_i, 
         Ksamples_ut, Ksamples_xref, Ksamples_Kfb_mode, 
         Ksamples_kff_mode, Ksamples_reset_args) = sample_bouncing_jax(n_samples, xt, 
                                                                        current_mode_actual, 
                                                                        states_0_i_zero, states_1_i_zero, 
                                                                        modes_i, 
                                                                        inputs_0_i_zero, inputs_1_i_zero, 
                                                                        K_feedback_0_i_zero, k_feedforward_0_i_zero, 
                                                                        K_feedback_1_i_zero, k_feedforward_1_i_zero, 
                                                                        target_state_t, Q_T, Qk_t,
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
            Ksamples_jax_saving = Ksamples_jax_i_coupled
        
        # --------------------------------------------------------
        #               Update the control proposal
        # --------------------------------------------------------
        # Compute proposal control, possibly with early arrival
        ubar_i = inputs_i[current_mode_actual][0]
        u0_proposal = np.zeros_like(ubar_i)
        
        # ------------------------------------------- compute the weights using coupling -------------------------------------------    
        GaussianNoises_ustar_jax_coupled = GaussianNoise_i_coupled[current_mode_actual][:,0,:n_inputs[current_mode_actual]]
        Ksamples_delta_t0_coupled = Ksamples_ts_i_coupled[:,1] - Ksamples_ts_i_coupled[:,0]
        u0_star_jax, weights_jax_coupled = update_u0_pathintegral_jax(u0_proposal, PathCosts_jax_i_coupled, GaussianNoises_ustar_jax_coupled, epsilon, Ksamples_delta_t0_coupled, dt)
        
        # ------------------------------------------- compute the weights without coupling ------------------------------------------- 
        GaussianNoises_ustar_jax = GaussianNoise_i[current_mode_actual][:,0,:n_inputs[current_mode_actual]]
        Ksamples_delta_t0 = Ksamples_ts_i[:,1] - Ksamples_ts_i[:,0]
        u0_star_jax, weights_jax = update_u0_pathintegral_jax(u0_proposal, PathCosts_jax_i, GaussianNoises_ustar_jax, epsilon, Ksamples_delta_t0, dt)
        
        u_star_pi_jax[current_mode_actual][i_t] = u0_star_jax
        allPathCosts_jax_coupled[i_t] = PathCosts_jax_i_coupled
        
        # un-coupled path costs
        allPathCosts_jax[i_t] = PathCosts_jax_i
        
        print("=== Var weight_jax: ", jnp.var(weights_jax))
        print("*** Var weight_jax coupled: ", jnp.var(weights_jax_coupled))
        print("Improved Var: ", (jnp.var(weights_jax)-jnp.var(weights_jax_coupled))/jnp.var(weights_jax))
        
        lambda_weights_jax = 1.0 / jnp.mean(weights_jax**2)
        lambda_weights_jax_coupled = 1.0 / jnp.mean(weights_jax_coupled**2)
        print("=== lambda_jax: ", lambda_weights_jax)
        print("*** lambda_jax coupled: ", lambda_weights_jax_coupled)
        print("Improved Lambda: ", (lambda_weights_jax_coupled-lambda_weights_jax)/lambda_weights_jax_coupled)
        
        # ----------------------------------
        #   Visualize sampled trajectories
        # ----------------------------------
        show_samples = False
        if show_samples and (i_t==0):
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
            ax2_sample.plot(Ksamples_jax_i[i_s,:,0], 'b', alpha=0.2, label='sampled stochastic control')
            
            fig4, ax10 = plt.subplots()
            inputs_ref = np.asarray(inputs)
            u_trj_ilqr = np.asarray(u_trj_ilqr)
            
            ax2_sample.plot(states_ref[:, 0], 'k', label='Ref control')
            ax2_sample.plot(xt_trj_ilqr[:, 0], 'r', label='h-iLQR stochastic control')
            
            ax1_sample.legend()
            ax2_sample.legend()
            
            plt.show()
            
            ax1_sample.scatter(target_state[0], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
            ax1_sample.scatter(init_state[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')
            ax1_sample.scatter(xt[0], xt[1], color='b', marker='x', s=30.0, linewidths=6, label='Current')
            
            
        show_sample_weights = False
        if show_sample_weights:
            plot_step = 1
            print("Plotting the sample weights and path costs.")
            ## -------- show path costs and weights in path integral control -------- 
            _, axes_weights = plt.subplots(2, 1, figsize=(8,6))
            (ax2, ax3) = axes_weights.flatten()
            ax2.grid(True)
            ax3.grid(True)
            
            ax2.bar(range(0,len(PathCosts_jax_i_coupled),plot_step), PathCosts_jax_i_coupled[::plot_step])
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
        actual_noise_i = RndN_actual[current_mode_actual][i_t]
        
        xt_next, next_mode_actual, _, new_reset_arg = h_stoch_integr_bouncing(xt, current_mode_actual,
                                                                            u0_star_jax, actual_noise_i, 
                                                                            epsilon, dt, dt_shrink, 
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

    # -------------
    # Compare cost
    # -------------
    dWs_zeros = [np.zeros((nt, n_inputs[0])), np.zeros((nt, n_inputs[1]))]
    cost_pi = compute_cost(modes_pi_jax, trj_pi_jax, u_star_pi_jax, dWs_zeros, target_state, states, Q_k, R_k, Q_T, epsilon,dt)
    cost_ilqr = compute_cost(mode_trj_ilqr, xt_trj_ilqr, ut_trj_ilqr, dWs_zeros, target_state, states, Q_k, R_k, Q_T, epsilon,dt)
    
    print("cost_pi:", cost_pi)
    print("cost_ilqr:", cost_ilqr)
    
    # -------------
    #  Record data
    # -------------
    data_i = DataOneSample(modes_pi_jax, trj_pi_jax, u_star_pi_jax, 
                           mode_trj_ilqr, xt_trj_ilqr, u_trj_ilqr, 
                           allPathCosts_jax_coupled, cost_pi, cost_ilqr, 
                           Ksamples_jax_saving, allPathCosts_uncoupled=allPathCosts_jax)

    gc.collect()

    return cost_pi, cost_ilqr, data_i

def main(epsilon, n_samples, dt):
    
    gc.collect()
    
    print(f"The value of time discretization dt is: {dt}")
    print(f"The value of epsilon input is: {epsilon}")
    print(f"The value of number of samples input is: {n_samples}")

    n_exp = 3
    
    # save_root = '/hddscratch/hyu419/hybrid_pathintegral/exp_200'
    # save_root = '/ssdscratch/hyu419/hybrid_pathintegral/new_exp/data/slip'
    save_file_path = '/home/hzyu/git/HybridPathIntegral/experiments/data/new_exp/slip'
    
    if os.path.exists(save_file_path):
        print("The directory exists.")
    else:
        print("The directory does not exist.")
        sys.exit(1)
    
    # ---------------- bouncing example -----------------
    dt_shrink = 0.9
    n_modes = 2
    
    n_states = [2, 2]
    n_inputs = [1, 1]
    
    # Time definitions
    start_time = 0
    end_time = 2.0
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)
    
    print("nt: ", nt)
    
    # Terminal cost 
    # Q_T = 60.0*np.eye(n_states[0])
    
    # Running costs
    # Q_k = [np.zeros((n_states[0],n_states[0])), np.zeros((n_states[1],n_states[1]))] # zero weight to penalties along a strajectory since we are finding a trajectory
    Q_k = [0.5*np.eye(n_states[0]), 0.5*np.eye(n_states[0])]
    R_k = [np.eye(n_inputs[0]), np.eye(n_inputs[1])]

    # ---------------------------- Set the terminal cost ----------------------------
    target_mode = 0
    Q_T = 200*np.eye(n_states[0])
    Q_T[0,0] = 2000.0
    
    init_mode = 0
    target_mode = 0
    
    init_state = np.array([5, 1.5])    # Define the initial state to be the origin with no velocity
    target_state = np.array([2.5, 0])  # Swing pendulum upright
    
    init_reset_args = [np.array([0.0]) for _ in range(nt)]
    target_reset_args = [np.array([0.0]) for _ in range(nt)]
    
    # ---------------- / slip example -----------------
    
    # ================================
    # solve for hybrid ilqr proposal
    # ================================
    exp_params = ExpParams()
    
    initial_guess = [0.5*np.ones((np.shape(time_span)[0],n_inputs[0])), 0.5*np.ones((np.shape(time_span)[0],n_inputs[1]))]
    
    flow_dynamics = [sym_dyn_bouncing, sym_dyn_bouncing]
    
    exp_params.update_params(n_modes, init_mode, target_mode, n_states, init_state, target_state, 
                             start_time, end_time, dt, dt_shrink,
                             initial_guess, 
                             epsilon, n_exp, n_samples, 
                             Q_k, R_k, Q_T, flow_dynamics, 
                             event_detect_bouncing_discrete, 
                             plot_bouncingball, 
                             convert_state_21_bouncing, 
                             init_reset_args, target_reset_args)
    exp_data = ExpData(exp_params)
    
    hybrid_ilqr_result = solve_ilqr(exp_params, detect=True, verbose=False)
    
    exp_data.add_nominal_data(hybrid_ilqr_result)
    exp_data.add_plotting_function(plot_bouncingball)
    
    # ============================================================================================================
    #                                   // End of Solve for hybrid ilqg proposal
    # ============================================================================================================

    # ============================================================================================================
    #               Path Integral Control, do n_exp number of experiments under different randomness
    # ============================================================================================================
    
    cost_pi_exp = np.zeros(n_exp)
    cost_ilqr_exp = np.zeros(n_exp)
    
    # ==================== 
    # Run ith experiment 
    # ====================
    import multiprocessing as mp 
    mp.set_start_method('spawn', force=True)
        
    # Pool of workers
    num_process = max(1, min(mp.cpu_count()-10, 3))
    with mp.Pool(processes=num_process) as pool:
        # Prepare the arguments for each experiment
        args = [(i, nt, n_samples, n_states, n_inputs, 
                 init_mode, init_state, target_state, hybrid_ilqr_result, 
                 start_time, end_time, dt, dt_shrink, 
                 Q_k, Q_T, R_k, epsilon, init_reset_args) for i in range(n_exp)]
        
        # Map each experiment to the pool
        results = pool.starmap(run_experiment, args)
    
    cost_pi_exp = np.zeros(n_exp)
    cost_ilqr_exp = np.zeros(n_exp)
    
    for i_exp, result in enumerate(results):
        print("Experiment result:", result)
        cost_pi_exp[i_exp] = result[0]
        cost_ilqr_exp[i_exp] = result[1]
        exp_data.add_data(i_exp, result[2])
    
    # # ---------------------------------------
    # #     Main Loop over the experiments
    # # ---------------------------------------
    # for i_exp in range(n_exp):
    #     result = run_experiment(i_exp, nt, n_samples, n_states, n_inputs, 
    #                             init_mode, init_state, target_state, hybrid_ilqr_result, 
    #                             start_time, end_time, dt, dt_shrink, 
    #                             Q_k, Q_T, R_k, epsilon, init_reset_args)

    #     cost_pi_exp[i_exp] = result[0]
    #     cost_ilqr_exp[i_exp] = result[1]
    #     exp_data.add_data(i_exp, result[2])
        
        
    print("E[cost_pi]: ", np.mean(cost_pi_exp))
    print("E[cost_ilqr_exp]: ", np.mean(cost_ilqr_exp))

    # =========== save data ===========
    from datetime import datetime
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"data_{formatted_datetime}_{script_filename}_{n_exp}exp_{n_samples}samples_eps_{epsilon}_coupling_ablation.pickle"
    filename = save_file_path+"/"+filename
    print("=================== Saving data to: =================== \n", filename)
    exp_data.dump(filename)
    

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="The epsilon parameter.")
    
    parser.add_argument("--epsilon", type=float, default=2.0, help="The process noise intensity value, epsilon.")
    parser.add_argument("--nsamples", type=int, default=500, help="The number of samples used in path integral control.")
    parser.add_argument("--dt", type=int, default=0.01, help="The time discretization.")
    
    args = parser.parse_args()

    main(args.epsilon, args.nsamples, args.dt)
    
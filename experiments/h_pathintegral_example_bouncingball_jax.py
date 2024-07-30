import numpy as np

import os
import sys
file_path = os.path.abspath(__file__)
exp_dir = os.path.dirname(file_path)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
root_dir = os.path.abspath(os.path.join(exp_dir, '..'))
sys.path.append(root_dir)

import time
import jax.numpy as jnp

# Import iLQR class and reference extension handler
from hybrid_ilqr.h_ilqr import solve_ilqr, extract_extensions
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

# Set environment variable to control the GPU memory fraction used by JAX
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"


import gc
gc.collect()

# ====================================== Path Integral Control ====================================== 
def compute_cost(modes,states,inputs,randN,target_state,trj_ref, Qk, Rk, QT, epsilon, dt):
    
    n_timestamps = len(states)
    
    # Initialize cost
    total_cost = 0.0
    # current_cost = 0.0
    for ii in range(n_timestamps-1):
        mode_i = modes[ii]
        current_u = inputs[mode_i][ii]
        randN_i = randN[mode_i][ii]
        Rk_i = Rk[mode_i]
        
        total_cost += current_u.T@Rk_i@current_u/2.0*dt + np.sqrt(epsilon*dt) * np.dot(current_u.T, randN_i)
        # total_cost = total_cost+current_cost
        
    # Compute terminal cost
    terminal_difference = (target_state-states[-1]).flatten()
    terminal_cost = terminal_difference.T@QT@terminal_difference/2.0

    total_cost = total_cost+terminal_cost

    return total_cost

def process_compute_costs(sample_i, inputs, dWs, target_state, ref_states, index, Q_k, R_k, Q_T, epsilon, dt):
    costs_i = compute_cost(sample_i, inputs, dWs, target_state, ref_states, Q_k, R_k, Q_T, epsilon, dt)
    return costs_i, index


def run_experiment(i_exp, nt, n_samples, n_states, n_inputs, 
                    init_mode, init_state, target_state, hybrid_ilqr_result, 
                    start_time, end_time, dt, dt_shrink, 
                    Q_k, Q_T, R_k, epsilon, init_reset_args):
    
    (modes,states,inputs,
     k_feedforward,K_feedback,
     current_cost,states_iter,
     ref_modechanges,reference_extension_helper, ref_reset_args) = hybrid_ilqr_result
    
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
    
    # states = np.array(states)
    # inputs = np.array(inputs)
    
    # k_feedforward = np.array(k_feedforward)
    # K_feedback = np.array(K_feedback)
    
    print(f"=================== The experiment index: {i_exp} ===================" )
    
    # -------------- result collectors, jax --------------
    modes_pi_jax = np.zeros(nt, dtype=np.int64)
    trj_pi_jax = [np.array([0.0]) for _ in range(nt)]

    u_star_pi_jax = [np.zeros((nt, n_inputs[0])), np.zeros((nt, n_inputs[1]))]
    allPathCosts_jax = np.zeros((nt-1, n_samples))
    
    # -------------- result collectors, hybrid ilqr proposal --------------
    trj_ilqr = [np.array([0.0]) for _ in range(nt)]
    u_trj_ilqr = [np.zeros((nt, n_inputs[0])), np.zeros((nt, n_inputs[1]))]
    
    # -------------- initialize the control loop -------------- 
    x0_jax = jnp.asarray(init_state)
    xt = x0_jax
    xt_ilqr = init_state
    
    current_mode_actual = init_mode
    next_mode_actual = current_mode_actual
    # current_modechange = np.array([current_mode_actual, next_mode_actual])
    
    dt_shrinkingrate = 0.7
    
    modes_pi_jax[0] = init_mode
    trj_pi_jax[0] = x0_jax
    trj_ilqr[0] = init_state
    
    # current_ref_modechange_ilqr = (1, 1)
    cnt_mismatch = 0
    # cnt_mismatch_ilqr = 0
    
    # Assuming 2-mode system
    RndN_actual = [np.random.randn(nt, 1), np.random.randn(nt, 1)]
    reset_args_actual = ref_reset_args
    event_args_actual = [ref_reset_args[0]]
    cnt_event_actual = 0
    
    # ---------------------------------
    # Extract the extended references 
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
    # Main loop for the hybrid path integral control
    # ======================================================
    for i_t in range(nt-1):
        
        print(f"----------- Time index: {i_t} -----------")
        
        start_time_i = start_time + i_t*dt
        nt_i = nt - i_t
        
        # ------------------------------------------------------------------------------------
        # Calculate the slice of the variables for the current future horizon 
        # ------------------------------------------------------------------------------------
        
        # actual modes and states
        current_mode_actual = modes_pi_jax[i_t]
        xt = trj_pi_jax[i_t]
        
        # references
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
        reset_args_i = ref_reset_args[i_t:]
        
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
        Kmodes_jax_i, Ksamples_jax_i, PathCosts_jax_i, _ = sample_bouncing_jax(n_samples, xt, 
                                                                                current_mode_actual, 
                                                                                states_0_i, states_1_i, 
                                                                                modes_i, 
                                                                                inputs_0_i, inputs_1_i, 
                                                                                K_feedback_0_i, k_feedforward_0_i, 
                                                                                K_feedback_1_i, k_feedforward_1_i, 
                                                                                target_state, Q_T, 
                                                                                start_time_i, dt, dt_shrinkingrate, 
                                                                                epsilon, 
                                                                                GaussianNoise_i[0], GaussianNoise_i[1], 
                                                                                v_mode_change_ref_i, 
                                                                                v_ref_ext_fwd_i, v_ref_ext_bwd_i, 
                                                                                v_Kfb_ref_ext_fwd_i, v_kff_ref_ext_fwd_i, 
                                                                                v_Kfb_ref_ext_bwd_i, v_kff_ref_ext_bwd_i, 
                                                                                reset_args_i)
        
        # save the samples at t=0
        # if (i_t == 0):
        #     Ksamples_jax_saving = Ksamples_jax_i
        
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
            xref_i, K_fb, k_ff, cnt_mismatch = reaction_mode_mismatch(cond_early_arrival_bouncing, i_t, 
                                                                      current_mode_actual, ref_current_mode, 
                                                                      v_ref_ext_fwd[0], v_ref_ext_bwd[0], 
                                                                      v_Kfb_ref_ext_fwd[0], v_kff_ref_ext_fwd[0], 
                                                                      v_Kfb_ref_ext_bwd[0], v_kff_ref_ext_bwd[0], cnt_mismatch)
        
        u0_proposal = ubar_i + K_fb@(xt - xref_i) + k_ff
        
        GaussianNoises_ustar_jax = GaussianNoise_i[current_mode_actual][:,0,:]
        u0_star_jax, weights_jax = update_u0_pathintegral_jax(u0_proposal, PathCosts_jax_i, GaussianNoises_ustar_jax, epsilon, dt)
        
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
            # Plot all the samples 
            # ----------------------
            for i_s in range(n_samples):
                ax1_sample.plot(Ksamples_jax_i[i_s,:,0], Ksamples_jax_i[i_s,:,1],'b', alpha=0.2)
            
            ax1_sample.scatter(target_state[0], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
            ax1_sample.scatter(init_state[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')
            ax1_sample.scatter(xt[0], xt[1], color='b', marker='x', s=30.0, linewidths=6, label='Current')
            
            # --------------------------------------------------
            # Plot the sample corresponding to the highest cost
            # --------------------------------------------------
            highcost_index = [index for index, _ in sorted(enumerate(PathCosts_jax_i), key=lambda x: x[1])][-1]
            
            print(f"Ploting the high cost sample. Index: {highcost_index}")
            
            ax2_sample.plot(Ksamples_jax_i[highcost_index,:,0], Ksamples_jax_i[highcost_index,:,1],'b', alpha=0.2)
            ax2_sample.scatter(target_state[0], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
            ax2_sample.scatter(init_state[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')
            ax2_sample.scatter(xt[0], xt[1], color='b', marker='x', s=30.0, linewidths=6, label='Current')
            
            ## -------- show path costs and weights in path integral control -------- 
            _, axes_weights = plt.subplots(2, 1, figsize=(8,6))
            (ax2, ax3) = axes_weights.flatten()
            ax2.grid(True)
            ax3.grid(True)
            
            ax2.bar(range(len(PathCosts_jax_i)), PathCosts_jax_i)
            ax2.set_title("Path Cost distribution")
            ax2.set_xlabel("Sample Number")
            ax2.set_ylabel("Costs")
            
            ax3.bar(range(len(weights_jax)), weights_jax)
            ax3.set_title("Weight distribution")
            ax3.set_xlabel("Sample Number")
            ax3.set_ylabel("Weights")
            
            plt.show()
        
        # ----------------------------------- 
        
        # --------------------------------------------- 
        # Apply optimal control and go to next state  
        # ---------------------------------------------
        reset_args_actual[i_t] = event_args_actual[cnt_event_actual]
        actual_noise_i = RndN_actual[current_mode_actual][i_t]
        
        xt_next, next_mode_actual, _, new_reset_arg = hybrid_integration_bouncing(xt, current_mode_actual,
                                                                                    u0_star_jax, actual_noise_i, 
                                                                                    epsilon, dt, dt_shrink, start_time_i, reset_args_actual[i_t])

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
            
    # =================================================================
    #                   Hybrid ilqr for comparison
    # =================================================================            
    mode_trj_ilqr, trj_ilqr, u_trj_ilqr, cost_ilqr, _ = stochastic_feedback_rollout_bouncing(init_mode, init_state, n_inputs,
                                                                                            states, ref_modechanges, 
                                                                                            inputs, K_feedback, k_feedforward, 
                                                                                            target_state, Q_T,
                                                                                            start_time, end_time, epsilon, 
                                                                                            RndN_actual, dt_shrinkingrate, 
                                                                                            reference_extension_helper,
                                                                                            init_reset_args)
    
    show_hilqr_results = False
    if show_hilqr_results:
        time_span = np.arange(start_time, end_time, dt).flatten()
        plot_bouncingball(time_span, mode_trj_ilqr, trj_ilqr, u_trj_ilqr, init_state, target_state, nt, trj_labels='iLQG-stochastic')


    # -------------
    # Compare cost
    # -------------
    dWs_zeros = [np.zeros((nt, n_inputs[0])), np.zeros((nt, n_inputs[1]))]
    cost_pi = compute_cost(modes_pi_jax, trj_pi_jax, u_star_pi_jax, dWs_zeros, target_state, states, Q_k, R_k, Q_T, epsilon,dt)
    cost_ilqr = compute_cost(mode_trj_ilqr, trj_ilqr, u_trj_ilqr, dWs_zeros, target_state, states, Q_k, R_k, Q_T, epsilon,dt)
    
    print("cost_pi:", cost_pi)
    print("cost_ilqr:", cost_ilqr)
    
    # -------------
    # Record data
    # -------------
    data_i = DataOneSample(modes_pi_jax, trj_pi_jax, u_star_pi_jax, 
                           mode_trj_ilqr, trj_ilqr, u_trj_ilqr, 
                           allPathCosts_jax, cost_pi, cost_ilqr, Ksamples_jax_saving)
    
    gc.collect()


    return cost_pi, cost_ilqr, data_i

def main(epsilon, n_samples, dt):
    
    print(f"The value of epsilon input is: {epsilon}")
    print(f"The value of number of samples input is: {n_samples}")
    print(f"The value of time discretization dt is: {dt}")
    # === ilqr parameters ===
    # Initialize timings
    
   # ---------------- bouncing example -----------------
    # dt = 0.005
    dt_shrink = 0.7
    
    start_time = 0
    end_time = 2.0
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)

    init_state = np.array([5, 1.5])    # Define the initial state to be the origin with no velocity
    target_state = np.array([2.5, 0])  # Swing pendulum upright
    
    init_mode = 0

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

    # # ------------- /verification with no contact ------------- 

    # Set desired state
    n_modes = 2
    
    # the state and control dimensions, mode-dependent
    n_states = [2, 2]
    n_inputs = [1, 1]

    # ---------------------------- 
    # Define weighting matrices
    # ----------------------------
    Q_k = [np.zeros((n_states[0],n_states[0])), np.zeros((n_states[1],n_states[1]))] # zero weight to penalties along a strajectory since we are finding a trajectory
    R_k = [np.eye(n_inputs[0]), np.eye(n_inputs[1])]

    # ---------------------------- Set the terminal cost ----------------------------
    target_mode = 0
    Q_T = 200*np.eye(n_states[0])
    Q_T[0,0] = 2000.0

    n_exp = 1
    
    init_reset_args = [np.array([0.0]) for _ in range(nt)]
    target_reset_args = [np.array([0.0]) for _ in range(nt)]
    
    # ============================================================================================================
    #                                       Solve for hybrid ilqg proposal
    # ============================================================================================================
    exp_params = ExpParams()
    
    initial_guess = [0.5*np.ones((np.shape(time_span)[0],n_inputs[0])), 0.5*np.ones((np.shape(time_span)[0],n_inputs[1]))]
    
    flow_dynamics = [symbolic_dynamics_bouncing, symbolic_dynamics_bouncing]
    
    exp_params.update_params(n_modes, init_mode, target_mode, n_states, init_state, target_state, 
                             start_time, end_time, dt, initial_guess, 
                             epsilon, n_exp, n_samples, 
                             Q_k, R_k, Q_T, flow_dynamics, 
                             event_detect_bouncing, plot_bouncingball, convert_state_21_bouncing, 
                             init_reset_args, target_reset_args)
    exp_data = ExpData(exp_params)
    
    print("===================== Solving for h-iLQG proposal controller =====================")
    hybrid_ilqr_result = solve_ilqr(exp_params, detect=True, verbose=False)
    
    (modes,states,inputs,
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
    
    # save_root = '/hddscratch/hyu419/hybrid_pathintegral/exp_200'
    save_root = '/home/hzyu/git/HybridPathIntegral/experiments'
    save_path = f"{save_root}/data/bouncing/{filename}"
    exp_data.dump(save_path)
    
    print(f" =================== Saved data to: =================== \n {{save_path}}")
    
    
    # --------------------------------------------
    # plot the path integral controlled trajectory
    # -------------------------------------------- 
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
    parser.add_argument("--epsilon", type=float, default=5, help="The process noise intensity value, epsilon.")
    parser.add_argument("--nsamples", type=int, default=5000, help="The number of samples used in path integral control.")
    parser.add_argument("--dt", type=int, default=0.005, help="The time discretization.")
    
    args = parser.parse_args()

    main(args.epsilon, args.nsamples, args.dt)
    
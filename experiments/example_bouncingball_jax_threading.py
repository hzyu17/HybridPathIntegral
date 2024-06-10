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

# Import pendulum dynamics
from dynamics.integration_hybrid_jax import roullout_bouncing_jax, hybrid_integration, update_u0_pathintegral_jax
# from dynamics.integration_hybrid_jax import *
# Import iLQR class
from hybrid_ilqr.hybrid_ilqr import solve_ilqr
# Import Riccati class
from hybrid_ilqr.hybrid_riccati import *
# Importing path integral control
from hybrid_pathintegral.hybrid_pathintegral import *
# Import plotting
import matplotlib.pyplot as plt
# Import experiment parameter class
from experiments.exp_params import *

# for paralle sampling on cpu
from joblib import Parallel, delayed

# Set environment variable to control the GPU memory fraction used by JAX
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"


import gc
gc.collect()

# ====================================== Path Integral Control ====================================== 
def compute_cost(states,inputs,randN,target_state,trj_ref, Qk, Rk, QT, epsilon, dt):
    
    n_timestamps = states.shape[0]
    
    # Initialize cost
    total_cost = 0.0
    current_cost = 0.0
    for ii in range(n_timestamps-1):
        current_u = inputs[ii]
        
        current_cost = current_u.T@Rk@current_u/2.0*dt + np.sqrt(epsilon*dt) * np.dot(current_u.T, randN[ii])
        total_cost = total_cost+current_cost
        
    # Compute terminal cost
    terminal_difference = (target_state-states[-1]).flatten()
    terminal_cost = terminal_difference.T@QT@terminal_difference/2.0

    total_cost = total_cost+terminal_cost

    return total_cost

def process_compute_costs(sample_i, inputs, dWs, target_state, ref_states, index, Q_k, R_k, Q_T, epsilon, dt):
    costs_i = compute_cost(sample_i, inputs, dWs, target_state, ref_states, Q_k, R_k, Q_T, epsilon, dt)
    return costs_i, index


def run_experiment(i_exp, nt, n_samples, n_states, n_inputs, 
                   init_state, target_state, hybrid_ilqr_result, 
                   start_time, end_time, dt, dt_shrink, 
                   Q_k, Q_T, R_k, epsilon):
    print("===================== experiment: ", i_exp, " =====================")
    (ref_states,ref_inputs,k_feedforward,K_feedback,_,_,modechanges,mode_exttrjs_maps) = hybrid_ilqr_result
    
    # -------------- result collectors, jax --------------
    trj_pi_jax = np.zeros((nt, n_states))
    u_star_pi_jax = np.zeros((nt, n_inputs))
    allPathCosts_jax = np.zeros((nt-1, n_samples))
    
    # -------------- result collectors, hybrid ilqr proposal --------------
    trj_ilqr = np.zeros((nt, n_states))
    u_trj_ilqr = jnp.zeros((nt, n_inputs))
    
    # -------------- initialize the control loop -------------- 
    x0_jax = jnp.asarray(init_state)
    xt = x0_jax
    current_modechange = jnp.array([1, 1])
    current_mode = current_modechange[0]
    next_mode = current_modechange[1]
    dt_shrinkingrate = 0.7
    
    trj_pi_jax[0] = x0_jax
    trj_ilqr[0] = init_state
    
    # current_modechange_ilqr = (1, 1)
    cnt_mismatch = 0
    
    RndN_actual = np.random.randn(nt, n_inputs)
    
    # ======================================================
    # main loop for the hybrid path integral control
    # ======================================================
    for i_t in range(nt-1):
        print("----------- time index: ", i_t)
        
        start_time_i = start_time + i_t*dt
        nt_i = nt - i_t
        
        # ------------------------------------------------------------------------------------
        # Calculate the slice of the variables for the current future horizon 
        # ------------------------------------------------------------------------------------
        states_i = ref_states[i_t:,:]
        inputs_i = ref_inputs[i_t:,:]
        modechange_i = modechanges[i_t:]
        K_feedback_i = K_feedback[i_t:,:]
        k_feedforward_i = k_feedforward[i_t:,:]
        GaussianNoise_i = np.random.randn(n_samples, nt_i, n_inputs)
        
        # # --------------------------- 
        # # Coupling of the randomness
        # # ---------------------------
        GaussianNoise_i[int(n_samples/2):, 0] = -GaussianNoise_i[:int(n_samples/2), 0]
        
        cur_ref_modechange = modechanges[i_t]
        ref_next_mode = cur_ref_modechange[1]        
        
        # ====================
        # Sampling using jax 
        # ====================
        # --------------------------------------------------------
        # Compute proposal control, possibly with early arrival
        # --------------------------------------------------------
        xref_i = states_i[0]
        
        if (next_mode != ref_next_mode):    
            if mode_exttrjs_maps is not None: # has extensions
                # Take the first hybrid event for now. Needs to find the correct corresponding one among all hybrid events.
                mode_change_i, mode_exttrjs_i = mode_exttrjs_maps[0]
                extended_trj = mode_exttrjs_i[next_mode]
                
            xref_i = extended_trj[i_t]
            cnt_mismatch += 1
        
        u0_proposal = inputs_i[0] + K_feedback_i[0]@(xt - xref_i) + k_feedforward_i[0]
        
        # ---------------------------------------------------
        #            Extract the extended references 
        # ---------------------------------------------------
        num_events = len(mode_exttrjs_maps)
        v_mode_change = []
        v_ext_trj_fwd = []
        v_ext_trj_bwd = []
        
        for i_event in range(num_events):
            # find out the mode changes
            MC_i = mode_exttrjs_maps[i_event][0]
            cur_mode_i = MC_i[0]
            next_mode_i = MC_i[1]
            
            v_mode_change.append((cur_mode_i, next_mode_i))
            
            # add the forward and backward extensions to the collection
            MC_EXTTRJ_MAP = mode_exttrjs_maps[i_event][1]
            v_ext_trj_fwd.append(MC_EXTTRJ_MAP[cur_mode_i][i_t:])
            v_ext_trj_bwd.append(MC_EXTTRJ_MAP[next_mode_i][i_t:])
        
        # ---------------------------------------------------------------------------------------
        #                                       Sampling
        # ---------------------------------------------------------------------------------------
        # timer_start_tic = time.perf_counter()
        Ksamples_jax, PathCosts_jax, actual_ref_jax = roullout_bouncing_jax(n_samples, xt, current_modechange, 
                                                                            states_i, modechange_i, 
                                                                            inputs_i, K_feedback_i, k_feedforward_i, 
                                                                            target_state, Q_T, 
                                                                            start_time_i, dt, end_time, dt_shrinkingrate, 
                                                                            epsilon, GaussianNoise_i, 
                                                                            v_mode_change, v_ext_trj_fwd, v_ext_trj_bwd)
        
        # ------------------------------
        # update the control proposal
        # ------------------------------
        GaussianNoises_ustar_jax = GaussianNoise_i[:,0,:]
        u0_star_jax, weights_jax = update_u0_pathintegral_jax(u0_proposal, PathCosts_jax, 
                                                                GaussianNoises_ustar_jax, epsilon, dt)
        u_star_pi_jax[i_t] = u0_star_jax
        allPathCosts_jax[i_t] = PathCosts_jax
        
        # print("*** Var weight_jax", jnp.var(weights_jax))
        # print("*** lambda_jax", 1.0 / jnp.mean(weights_jax**2))
        
        # --------------------------------------------- 
        # Apply optimal control and go to next state  
        # ---------------------------------------------
        actual_noise_i = RndN_actual[i_t]
        # t_span = (start_time_i, start_time_i+dt)
        
        xt, next_mode, _ = hybrid_integration(xt, current_mode, next_mode, 
                                                u0_star_jax, actual_noise_i, epsilon, dt, dt_shrink, start_time_i)
        # xt, next_mode = hybrid_stochastic_integration(xt, u0_star_jax, current_mode, t_span, epsilon, actual_noise_i, dt, dt_shrinkingrate)
        next_mode = int(next_mode)
        trj_pi_jax[i_t+1] = xt
        
        current_modechange = np.array([current_mode, next_mode])
        current_mode = next_mode

        gc.collect()

    # // endfor i_t in range(nt-1):
    
    # ------------------------------
    # hybrid ilqr for comparison
    # ------------------------------
    # ---------------------------------------------------
    #            Extract the extended references 
    # ---------------------------------------------------
    num_events = len(mode_exttrjs_maps)
    v_mode_change = []
    v_ext_trj_fwd = []
    v_ext_trj_bwd = []
    
    for i_event in range(num_events):
        # find out the mode changes
        MC_i = mode_exttrjs_maps[i_event][0]
        cur_mode_i = MC_i[0]
        next_mode_i = MC_i[1]
        
        v_mode_change.append((cur_mode_i, next_mode_i))
        
        # add the forward and backward extensions to the collection
        MC_EXTTRJ_MAP = mode_exttrjs_maps[i_event][1]
        v_ext_trj_fwd.append(MC_EXTTRJ_MAP[cur_mode_i])
        v_ext_trj_bwd.append(MC_EXTTRJ_MAP[next_mode_i])
    
    trj_ilqr, u_trj_ilqr, cost_ilqr, _ = rollout_bouncing_feedback(init_state, np.array([1, 1]), ref_states, modechanges, 
                                                                    ref_inputs, K_feedback, k_feedforward, target_state, Q_T,
                                                                    start_time, end_time, epsilon, 
                                                                    RndN_actual, dt_shrinkingrate, v_ext_trj_fwd, v_ext_trj_bwd)
    
    # ----------------
    # Compare cost
    # ----------------
    dWs_zeros = np.zeros((nt, n_inputs))
    cost_pi = compute_cost(trj_pi_jax, u_star_pi_jax, dWs_zeros, target_state, ref_states, Q_k, R_k, Q_T, epsilon,dt)
    cost_ilqr = compute_cost(trj_ilqr, u_trj_ilqr, dWs_zeros, target_state, ref_states, Q_k, R_k, Q_T, epsilon,dt)
    
    # -------------
    # Record data
    # -------------
    data_i = DataOneSample(trj_pi_jax, u_star_pi_jax, trj_ilqr, u_trj_ilqr, allPathCosts_jax, cost_pi, cost_ilqr)
    
    gc.collect()

    return cost_pi, cost_ilqr, data_i

def main(epsilon, n_samples):
    print(f"The value of epsilon input is: {epsilon}")
    print(f"The value of number of samples input is: {n_samples}")
    # === ilqr parameters ===
    # Initialize timings
    
    # ---------------- bouncing example -----------------
    dt = 0.01
    dt_shrink = 0.7
    start_time = 0
    end_time = 2.0
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)
    
    init_state = np.array([5, 1.5])    # Define the initial state to be the origin with no velocity
    target_state = np.array([3.5, 0])  # Swing pendulum upright
    
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
    n_states = 2
    n_inputs = 1
    
    # ---------------------------- 
    # Define weighting matrices
    # ----------------------------
    Q_k = np.zeros((n_states,n_states)) # zero weight to penalties along a strajectory since we are finding a trajectory
    # R_k = 0.01*np.eye(n_inputs)
    R_k = np.eye(n_inputs)

    # ---------------------------- Set the terminal cost ----------------------------
    # Q_T = 1000*np.eye(n_states)
    Q_T = 200*np.eye(n_states)
    Q_T[0,0] = 2000.0
    
    # -------------------------- 
    # path integral parameters 
    # --------------------------
    n_exp = 100
    
    # ----------------------------------------------------
    # Do N experiments and compare the expected costs 
    # ----------------------------------------------------
    cost_pi_exp = np.zeros(n_exp)
    cost_ilqr_exp = np.zeros(n_exp)

    # ====================================
    # solve for hybrid ilqr proposal
    # ====================================
    exp_params = ExpParams()
    initial_guess = 0.5*np.ones((np.shape(time_span)[0],n_inputs))
    exp_params.update_params(init_state, target_state, start_time, end_time, dt, initial_guess, 
                             epsilon, n_exp, n_samples, Q_k, R_k, Q_T, symbolic_dynamics_bouncing,detect_bouncing)
    exp_data = ExpData(exp_params)
    hybrid_ilqr_result = solve_ilqr(exp_params, detect=True)
    
    (states,inputs,k_feedforward,K_feedback,current_cost,states_iter,modechanges,mode_exttrjs_maps) = hybrid_ilqr_result
    
    exp_data.add_nominal_data((states,inputs,k_feedforward,K_feedback,current_cost,states_iter))

    # =============================================================================================
    # Do sample experiments for n_exp number of experiments, under different randomness
    # =============================================================================================          
    
    import multiprocessing as mp
    # ==================== 
    # Run ith experiment 
    # ==================== 
    mp.set_start_method('spawn', force=True)
        
    # Pool of workers
    num_process = max(1, min(mp.cpu_count()-10, 5))
    with mp.Pool(processes=num_process) as pool:
        # Prepare the arguments for each experiment
        args = [(i, nt, n_samples, n_states, n_inputs, 
                 init_state, target_state, hybrid_ilqr_result, 
                 start_time, end_time, dt, dt_shrink, 
                 Q_k, Q_T, R_k, epsilon) for i in range(n_exp)]
        
        # Map each experiment to the pool
        results = pool.starmap(run_experiment, args)
    
    for i_exp, result in enumerate(results):
        print("Experiment result:", result)
        cost_pi_exp[i_exp] = result[0]
        cost_ilqr_exp[i_exp] = result[1]
        exp_data.add_data(i_exp, result[2])
    
    print(exp_data._data.keys())
    
    print("E[cost_pi]: ", np.mean(cost_pi_exp))
    print("E[cost_ilqr_exp]: ", np.mean(cost_ilqr_exp))

    # =========== save data ===========
    from datetime import datetime
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    # filename = f"data_{formatted_datetime}_{script_filename}_{n_samples}samples_eps_{epsilon}_coupling.pickle"
    filename = f"data_{n_samples}samples_eps_{epsilon}_coupling.pickle"
    save_root = '/hddscratch/hyu419/hybrid_pathintegral/exp_200'
    file_path = f"{save_root}/data/bouncing/{filename}"
    print("Saving data to: ", file_path)
    exp_data.dump(file_path)

    show_results = False
    if show_results:
        # =============== plotting ===============
        fig1, axes = plt.subplots(1, 2)
        (ax1, ax2) = axes.flatten()
        ax1.grid(True)
        ax2.grid(True)

        # # ----------- plot the path integral controlled trajectory -----------
        for i_exp in range(n_exp):
            trj_ilqr = exp_data.get_data(i_exp).x_trj_ilqr()
            trj_pi_jax = exp_data.get_data(i_exp).x_trj_pi()
            
            ax1.plot(time_span, trj_ilqr[:, 0], 'b', alpha=0.2)
            ax2.plot(time_span, trj_ilqr[:, 1], 'b', alpha=0.2)
            
            ax1.plot(time_span, trj_pi_jax[:, 0], 'r', alpha=0.8)
            ax2.plot(time_span, trj_pi_jax[:, 1], 'r', alpha=0.8)
            
        ax1.plot(time_span, trj_ilqr[:, 0], 'b', alpha=0.2, label='iLQR')
        ax2.plot(time_span, trj_ilqr[:, 1], 'b', alpha=0.2, label='iLQR')

        ax1.plot(time_span, trj_pi_jax[:, 0], 'r', alpha=0.8, label='Path Integral')
        ax2.plot(time_span, trj_pi_jax[:, 1], 'r', alpha=0.8, label='Path Integral')

        # ----------- Plot the start and goal states -----------
        ax1.scatter(time_span[-1], target_state[0], color='g', marker='x', s=50.0, linewidths=6, label='Target')
        ax1.scatter(time_span[0], init_state[0], color='r', marker='x', s=50.0, linewidths=6, label='Start')

        ax2.scatter(time_span[-1], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
        ax2.scatter(time_span[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')

        # ----------- Plot the reference -----------
        ax1.plot(time_span, states[:,0],'k',label='iLQR-deterministic')
        ax2.plot(time_span, states[:,1],'k',label='iLQR-deterministic')

        ax1.set_xlabel(r"Time")
        ax1.set_ylabel(r"$z$")
        ax1.set_title("Bouncing Ball Vertical Position")

        ax2.set_xlabel(r"Time")
        ax2.set_ylabel(r"$\dot z$")
        ax2.set_title("Bouncing Ball Vertical Velocity")

        ax1.legend()
        ax2.legend()

        # =========== Plot the z-\dot_z figure ===========
        fig2, ax5 = plt.subplots()
        ax5.grid(True)

        # ----------- Plot the last iteration of iLQR controller ----------
        for i_exp in range(n_exp):
            trj_ilqr = exp_data.get_data(i_exp).x_trj_ilqr()
            trj_pi_jax = exp_data.get_data(i_exp).x_trj_pi()
            ax5.plot(trj_ilqr[:, 0], trj_ilqr[:, 1], 'b', alpha=0.2)
            ax5.plot(trj_pi_jax[:, 0], trj_pi_jax[:, 1], 'r', alpha=0.8)

        ax5.plot(trj_ilqr[:, 0], trj_ilqr[:, 1], 'b', alpha=0.2, label='iLQR')
        ax5.plot(trj_pi_jax[:, 0], trj_pi_jax[:, 1], 'r', alpha=0.8, label='Path Integral')
        ax5.plot(states[:,0], states[:,1],'k',label='iLQR-deterministic')

        # ----------- Plot the start and goal states -----------
        ax5.scatter(target_state[0], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
        ax5.scatter(init_state[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')

        ax5.legend()

        # plot control inputs
        fig3, ax7 = plt.subplots(1, 1)
        ax7.grid(True)
        ax7.plot(time_span, inputs[:,0],'k',label='Final iteration ilqr')
        ax7.plot(time_span, u_star_pi_jax[:,0],'r',label='Path integral controller')
        ax7.set_xlabel(r"Timestep")
        ax7.set_ylabel(r"$u$")
        ax7.set_title("Bouncing Ball Final Control Input")

        ax7.legend()

        # plot PathCosts
        fig3, ax8 = plt.subplots()
        ax8.grid(True)
        ax8.bar(range(n_exp), cost_ilqr_exp, width = 2, color='navy', alpha=0.1, label='Cost iLQR')
        ax8.bar(range(n_exp), cost_pi_exp, width = 2, color='red', alpha=0.5, label='Cost PathIntegralControl')

        ax8.set_xlabel(r"Experiment ID")
        ax8.set_ylabel(r"$Costs$")
        ax8.legend()

        plt.show()


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="The epsilon parameter.")
    parser.add_argument("--epsilon", type=float, default=2, help="The process noise intensity value, epsilon.")
    parser.add_argument("--nsamples", type=int, default=5000, help="The number of samples used in path integral control.")

    args = parser.parse_args()

    main(args.epsilon, args.nsamples)
    
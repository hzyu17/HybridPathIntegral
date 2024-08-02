import numpy as np

import os
import sys
file_path = os.path.abspath(__file__)
exp_dir = os.path.dirname(file_path)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
root_dir = os.path.abspath(os.path.join(exp_dir, '..'))
sys.path.append(root_dir)

from dynamics.dynamics_bouncing import *
# Import iLQR class
from hybrid_ilqr.h_ilqr_discrete import solve_ilqr
from dynamics.dynamics_bouncing import *
from dynamics.dynamics_discrete_bouncing import *
# Importing path integral control
from hybrid_pathintegral.hybrid_pathintegral import *
# Import plotting
import matplotlib.pyplot as plt
# Import experiment parameter class
from experiments.exp_params import *
from experiments.h_pathintegral_example_bouncingball_jax import run_experiment

# Set environment variable to control the GPU memory fraction used by JAX
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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


def main(epsilon, n_samples, dt):
    
    print(f"The value of epsilon input is: {epsilon}")
    print(f"The value of number of samples input is: {n_samples}")
    print(f"The value of time discretization dt is: {dt}")
    # === ilqr parameters ===
    # Initialize timings
    
    # ---------------- bouncing example -----------------
    # dt = 0.01
    dt_shrink = 0.99
    
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

    n_exp = 10
    
    init_reset_args = [np.array([0.0]) for _ in range(nt)]
    target_reset_args = [np.array([0.0]) for _ in range(nt)]
    
    # ============================================================================================================
    #                                       Solve for hybrid ilqg proposal
    # ============================================================================================================
    exp_params = ExpParams()
    
    initial_guess = [0.5*np.ones((np.shape(time_span)[0],n_inputs[0])), 0.5*np.ones((np.shape(time_span)[0],n_inputs[1]))]
    
    flow_dynamics = [symbolic_dynamics_bouncing, symbolic_dynamics_bouncing]
    
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
    
    print(exp_data._data.keys())
    
    print("E[cost_pi]: ", np.mean(cost_pi_exp))
    print("E[cost_ilqr_exp]: ", np.mean(cost_ilqr_exp))

    # =========== save data ===========
    from datetime import datetime
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"data_{formatted_datetime}_{script_filename}_{n_samples}samples_eps_{epsilon}_coupling.pickle"
    # filename = f"data_{n_samples}samples_eps_{epsilon}_coupling.pickle"
    # save_root = '/hddscratch/hyu419/hybrid_pathintegral/exp_200'
    save_root = '/hddscratch/hyu419/hybrid_pathintegral/new_exp'
    
    # save_root = '/home/hzyu/git/HybridPathIntegral/experiments'
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
    
    parser.add_argument("--epsilon", type=float, default=2.0, help="The process noise intensity value, epsilon.")
    parser.add_argument("--nsamples", type=int, default=5000, help="The number of samples used in path integral control.")
    parser.add_argument("--dt", type=int, default=0.002, help="The time discretization.")
    
    args = parser.parse_args()

    main(args.epsilon, args.nsamples, args.dt)
    
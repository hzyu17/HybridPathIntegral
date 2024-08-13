import numpy as np

import os
import sys
file_path = os.path.abspath(__file__)
exp_dir = os.path.dirname(file_path)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
root_dir = os.path.abspath(os.path.join(exp_dir, '..'))
sys.path.append(root_dir)

# Import iLQR class
from hybrid_ilqr.h_ilqr_discrete import solve_ilqr
from dynamics.dynamics_slip import *
from dynamics.dynamics_discrete_slip import *
# Importing path integral control
from hybrid_pathintegral.hybrid_pathintegral import *
# Import experiment parameter class
from experiments.exp_params import *
from experiments.h_PI_slip_jax import run_experiment

from jax import pmap

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import gc
gc.collect()

def main(epsilon, n_samples, dt):
    
    print(f"The value of epsilon input is: {epsilon}")
    print(f"The value of number of samples input is: {n_samples}")
    print(f"The value of time discretization dt is: {dt}")
    # === ilqr parameters ===
    # Initialize timings
    
    n_exp = 10
    
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
    
    # ----------------------------
    # Case 1: vertical bouncing
    # ----------------------------
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
    
    # --------------------------
    # Case 2: Running one step
    # --------------------------
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
    
    (time_span,modes,states,inputs,
     k_feedforward,K_feedback,
     current_cost,states_iter,
     ref_modechanges,reference_extension_helper, ref_reset_args) = hybrid_ilqr_result
    
    exp_data.add_nominal_data(hybrid_ilqr_result)
    exp_data.add_plotting_function(plot_slip)
    
    # ---------------------
    #  Show h-iLQG results
    # ---------------------
    show_results = False
    if show_results:
        plot_slip(time_span, modes, states, inputs, init_state, target_state, nt)
    
    # =============================================================================================
    # Do sample experiments for n_exp number of experiments, under different randomness
    # =============================================================================================          
    
    import multiprocessing as mp
    # ==================== 
    # Run ith experiment 
    # ==================== 
    mp.set_start_method('spawn', force=True)

    # Pool of workers
    num_process = max(1, min(mp.cpu_count()-10, 1))
    with mp.Pool(processes=num_process) as pool:
        # Prepare the arguments for each experiment
        args = [(i_exp, nt, n_samples, n_states, n_inputs, 
                 init_mode, init_state, target_state, hybrid_ilqr_result, 
                 start_time, end_time, dt, dt_shrink, 
                 Q_k, Q_T, R_k, epsilon, init_reset_args) for i_exp in range(n_exp)]
        
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
    filename = f"data_{formatted_datetime}_{script_filename}_{n_exp}exp_{n_samples}samples_eps_{epsilon}_coupling.pickle"
    filename = save_file_path+"/"+filename
    print("Saving data to: ", filename)
    exp_data.dump(filename)


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="The epsilon parameter.")
    
    parser.add_argument("--epsilon", type=float, default=0.001, help="The process noise intensity value, epsilon.")
    parser.add_argument("--nsamples", type=int, default=1000, help="The number of samples used in path integral control.")
    parser.add_argument("--dt", type=int, default=0.0008, help="The time discretization.")
    
    args = parser.parse_args()

    main(args.epsilon, args.nsamples, args.dt)
    

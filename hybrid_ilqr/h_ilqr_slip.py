import numpy as np
import os
import sys
file_path = os.path.abspath(__file__)
exp_dir = os.path.dirname(file_path)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
root_dir = os.path.abspath(os.path.join(exp_dir, '..'))
sys.path.append(root_dir)


# Import iLQR class
from hybrid_ilqr.h_ilqr import solve_ilqr, extract_extensions
# Import SLIP dynamics
from dynamics.dynamics_slip import *
# Importing path integral control
from hybrid_pathintegral.hybrid_pathintegral import *
# Import plotting
import matplotlib.pyplot as plt
# Import experiment parameter class
from experiments.exp_params import *


if __name__ == '__main__':
    # ------------- 
    # SLIP example 
    # -------------
    dt = 0.005
    epsilon = 2.0
    dt_shrink = 0.7
    
    start_time = 0
    end_time = 1.5
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)

    # ==============
    # SLIP dynamics
    # ==============
    # mode 1 (flight): x = [px, vx, pz, vz, theta], u = [theta_dot]
    # mode 2 (stance): x = [theta, theta_dot, r, r_dot], u = [r_delta, \tau_hip]
    
    init_state = np.array([0.0, 0.0, 2.0, 0.0, np.pi/2], dtype=np.float64)    # Define the initial state to be the origin with no velocity
    target_state = np.array([0.0, 0.0, 2.2, 0.0, np.pi/2], dtype=np.float64)  # Swing pendulum upright

    n_modes = 2
    init_mode = 0
    
    # ---------------- / slip example -----------------
    
    # For the slip dynamics, mode 1 has 1 input, and mode 2 has 2 inputs. 
    n_states = [5, 4]
    n_inputs = [1, 2]

    # ---------------------------
    # Define weighting matrices
    # ---------------------------
    Q_k = [np.zeros((n_states[0],n_states[0])), np.zeros((n_states[1],n_states[1]))] # zero weight to penalties along a strajectory since we are finding a trajectory
    R_k = [np.eye(n_inputs[0]), np.eye(n_inputs[1])]
    
    # ----------------------- 
    # Set the terminal cost 
    # -----------------------
    target_mode = 0
    Q_T = 0.01*np.eye(n_states[0])

    n_exp = 1
    n_samples = 10

    # ================================
    # solve for hybrid ilqr proposal
    # ================================
    exp_params = ExpParams()
    
    initial_guess = [0.0*np.ones((np.shape(time_span)[0],n_inputs[0])), np.zeros((np.shape(time_span)[0],n_inputs[1]))]
    symbolic_dynamics = [symbolic_flight_dynamics_slip, symbolic_stance_dynamics_slip]
    
    exp_params.update_params(n_modes, init_mode, target_mode, n_states, init_state, target_state, 
                             start_time, end_time, dt, initial_guess, 
                             epsilon, n_exp, n_samples, Q_k, R_k, Q_T, symbolic_dynamics, detect_slip, plot_slip, convert_state_21_slip)
    exp_data = ExpData(exp_params)
    hybrid_ilqr_result = solve_ilqr(exp_params, detect=True)
    
    (modes,states,inputs,k_feedforward,K_feedback,current_cost,states_iter,modechanges,mode_exttrjs_maps) = hybrid_ilqr_result
    
    exp_data.add_nominal_data((states,inputs,k_feedforward,K_feedback,current_cost,states_iter))


    show_results = True
    if show_results:
        plot_slip(time_span, modes, states, inputs, init_state, target_state, nt)

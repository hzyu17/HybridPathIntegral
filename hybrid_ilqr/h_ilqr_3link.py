import numpy as np
import os
import sys

file_path = os.path.abspath(__file__)
exp_dir = os.path.dirname(file_path)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
root_dir = os.path.abspath(os.path.join(exp_dir, '..'))
sys.path.append(root_dir)


# Import iLQR class
from hybrid_ilqr.h_ilqr_jax import *
# Import 3 link walker dynamics
from dynamics.walking_3link import *
# Import experiment parameter class
from experiments.exp_params import *


if __name__ == '__main__':
    # ---------------- 3link walking example -----------------
    dt = 0.005
    epsilon = 2.0
    dt_shrink = 0.95
    
    start_time = 0
    end_time = 2.0
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)

    # generate initial state
    omega_1 = 1.55
    init_state = sigma_three_link(omega_1, a)
    init_state = resetmap_3link(init_state).T

    target_state = init_state  # A limit cycle hopes to go back to the initial state
    
    init_mode = 0

    # Set desired state
    n_modes = 1
    
    # the state and control dimensions, mode-dependent
    n_states = [6]
    n_inputs = [2]

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
    n_samples = 10
    
    init_reset_args = [np.array([0.0]) for _ in range(nt)]
    target_reset_args = [np.array([0.0]) for _ in range(nt)]
    
    # ====================================
    #   Solve for hybrid ilqr proposal
    # ====================================
    
    initial_guess = [0.5*np.ones((np.shape(time_span)[0],n_inputs[0])), 0.5*np.ones((np.shape(time_span)[0],n_inputs[1]))]
    
    smooth_flow = [fxgu_3link_jax]
    
    target_hipvel = 2.0
    runningcost_arg = target_hipvel
    terminalcost_arg = init_state
    
    niters = 20

    hilqr_obj = hybrid_ilqr_jax(n_states, init_state, target_state, initial_guess, 
                                dt, start_time, end_time, 
                                niters, 
                                detect=True, 
                                detect_func=detet_3link, smooth_dynamics=smooth_flow, 
                                running_cost=hipmoving_cost, cost_args=runningcost_arg,
                                terminal_cost=statedeviation_norm_cost, terminal_cost_args=init_state, 
                                verbose=True
                                )
    
    hybrid_ilqr_result = solve_ilqr(exp_params, detect=True)
    
    (modes,states,inputs,
     k_feedforward,K_feedback,
     current_cost,states_iter,
     ref_modechanges,ref_ext_helper, ref_reset_args) = hybrid_ilqr_result
    
    exp_data.add_nominal_data(hybrid_ilqr_result)


    show_results = True
    if show_results:
        plot_bouncingball(time_span, modes, states, inputs, init_state, target_state, nt, color='k')
## Abalation study on the effect of the trajectory extension.
import numpy as np

import os
import sys
file_path = os.path.abspath(__file__)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

# Import pendulum dynamics
from dynamics.symbolic_bouncing_1D import *
# Import iLQR class
from hybrid_ilqr import solve_ilqr
# Importing path integral control
from hybrid_pathintegral.hybrid_pathintegral import *
# Import plotting
import matplotlib.pyplot as plt
# Import experiment parameter class
from exp_params import *

# for paralle sampling on cpu
from joblib import Parallel, delayed


# ====================================== Path Integral Control ====================================== 
def compute_cost(states,inputs,randN,target_state,trj_ref, Qk, Rk, QT, epsilon, dt):
    
    n_timestamps = states.shape[0]
    
    # Initialize cost
    total_cost = 0.0
    current_cost_xref = 0.0
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

def process_sampling_feedback(sample_i, init_state, current_modechange, xt_ref, ref_modechanges, 
                              ut, K_feedback, k_feedforward, 
                              target_state, R_k, Q_T, 
                              start_time, end_time, 
                              epsilon, RandN, 
                              mode_exttrjs_maps, index):
    # print("Sampling trajectory: ", index)
    sample_i, ut_cl_i, Su_i = rollout_bouncing_stochastic_feedback(init_state, current_modechange, xt_ref, ref_modechanges,
                                                                    ut, K_feedback, k_feedforward, target_state, R_k, Q_T,
                                                                    start_time, end_time, epsilon, RandN[index], mode_exttrjs_maps)
    return sample_i, ut_cl_i, Su_i, index

# Compute path costs function
# @njit(numba.types.Tuple((float64, int32))(
#     float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:,:], int32, float64[:,:], float64[:,:], float64[:,:], float64, float64))
def process_compute_costs(sample_i, inputs, dWs, target_state, ref_states, index, Q_k, R_k, Q_T, epsilon, dt):
    # print("Computing costs: ", index)
    costs_i = compute_cost(sample_i, inputs, dWs, target_state, ref_states, Q_k, R_k, Q_T, epsilon, dt)
    return costs_i, index


if __name__ == '__main__':
    # === ilqr parameters ===
    # Initialize timings
    dt = 0.002
    dt_pathintegral = dt
    
    # ---------------- bouncing example -----------------
    start_time = 0
    end_time = 2.0
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)
    
    init_state = np.array([5, 1.5])    # Define the initial state to be the origin with no velocity
    target_state = np.array([2.5, 0])  # Swing pendulum upright
    
    # Set desired state
    n_states = 2
    n_inputs = 1
    
    # Define weighting matrices
    Q_k = np.zeros((n_states,n_states)) # zero weight to penalties along a strajectory since we are finding a trajectory
    # R_k = 0.01*np.eye(n_inputs)
    R_k = np.eye(n_inputs)

    # Set the terminal cost
    Q_T = 200*np.eye(n_states)
    Q_T[0,0] = 2000.0
    
    # === path integral parameters ===
    epsilon = 2.0
    n_samples = 100
    n_exp = 1
    
    # Horizon
    nt_ode_solve = 1000 # number of points used to solve the ode
    
    # === solve for ilqr ===
    exp_params = ExpParams()
    initial_guess = 0.5*np.ones((np.shape(time_span)[0],n_inputs))
    exp_params.update_params(init_state, target_state, start_time, end_time, dt, initial_guess, 
                             epsilon, n_exp, n_samples, Q_k, R_k, Q_T, symbolic_dynamics_bouncing,detect_bouncing)
    exp_data = ExpData(exp_params)
    (states,inputs,k_feedforward,K_feedback,current_cost,states_iter,modechanges,mode_exttrjs_maps) = solve_ilqr(exp_params, detect=True)
    
    step_one_samples = np.zeros((n_samples, n_states))
    
    trj_pi = np.zeros((nt, n_states))
    trj_ilqr = np.zeros((nt, n_states))

    trj_pi[0] = init_state
    trj_ilqr[0] = init_state
    u_star_pi = np.zeros((nt, n_inputs))
    u_trj_ilqr = np.zeros((nt, n_inputs))
    
    allPathCosts = np.zeros((nt-1, n_samples))
    
    xt = init_state
    xt_ilqr = init_state
    current_modechange = (1, 1)
    current_mode = current_modechange[0]
    next_mode = current_modechange[1]
    
    # current_modechange_ilqr = (1, 1)
    cnt_mismatch = 0
    cnt_mismatch_ilqr = 0
    
    RndN_actual = np.random.randn(nt, n_inputs)
    
    recompute_porposal = False
    
    i_t = 0
        
    print("----------- time index: ", i_t)
    
    start_time_i = start_time + i_t*dt

    time_span_i = np.arange(start_time_i, end_time, dt).flatten()
    nt_i = nt - i_t
    
    states_i = states[i_t:,:]
    inputs_i = inputs[i_t:,:]
    modechange_i = modechanges[i_t:]
    K_feedback_i = K_feedback[i_t:,:]
    k_feedforward_i = k_feedforward[i_t:,:]
    ref_next_mode = modechanges[i_t][1]        
    
    xref_i = states_i[0]
    
    # ---------- Consider the mode mismatch ----------
    if (next_mode != ref_next_mode):    
        print("mode mismatch true trajectory")
        print("true state mode change: ", current_modechange)
        print("reference mode change: ", modechange_i)
        if mode_exttrjs_maps is not None: # has extensions
            # Take the first hybrid event for now. Needs to find the correct corresponding one among all hybrid events.
            mode_change_i, mode_exttrjs_i = mode_exttrjs_maps[0]
            extended_trj = mode_exttrjs_i[next_mode]
            
            # First time early arrival: find and reverse the ref
            if (next_mode==2) and (ref_next_mode==1) and (cnt_mismatch==0): 
                len_ref = 0
                i_ext = 0
                while True: # Find the correct length of the extension
                    if (modechange_i[i_ext][1] == next_mode):
                        len_ref = i_ext
                        break
                    i_ext += 1
                extended_trj = extended_trj[0:len_ref]
                extended_trj = extended_trj[::-1]
                
            xref_i = extended_trj[cnt_mismatch]
        cnt_mismatch += 1
    # ---------- / Consider the mode mismatch ----------
    
    u0_proposal = inputs_i[0] + K_feedback_i[0]@(xt - xref_i) + k_feedforward_i[0]
    
    
    # ---------- Not consider the mode mismatch ----------
    xref_i_compare = states_i[0]
    u0_proposal_compare = inputs_i[0] + K_feedback_i[0]@(xt - xref_i_compare) + k_feedforward_i[0]
    # ---------- / Not consider the mode mismatch ----------
    
    
    # sampling stochastic rollouts
    sampled_trjs = np.zeros((n_samples, nt_i, n_states))
    sampled_controls = np.zeros((n_samples, nt_i, n_inputs))
    PathCosts = np.zeros(n_samples)  
    
    # --- comparison ---
    sampled_trjs_compare = np.zeros((n_samples, nt_i, n_states))
    sampled_controls_compare = np.zeros((n_samples, nt_i, n_inputs))
    PathCosts_compare = np.zeros(n_samples)  
    
    GaussianNoise_i = np.random.randn(n_samples, nt_i, n_inputs)
    
    for i_sample in prange(n_samples):
        noise_i = GaussianNoise_i[i_sample]
        sample_i, ut_cl_i, Su_i = rollout_bouncing_stochastic_feedback(xt, current_modechange, states_i, modechange_i, 
                                                                        inputs_i, K_feedback_i, k_feedforward_i, target_state, R_k, Q_T,
                                                                        start_time_i, end_time, epsilon, noise_i, mode_exttrjs_maps)
        sampled_trjs[i_sample] = sample_i
        sampled_controls[i_sample] = ut_cl_i
        PathCosts[i_sample] = Su_i
        
    # --- comparison ---
    for i_sample in prange(n_samples):
        noise_i = GaussianNoise_i[i_sample]
        sample_i, ut_cl_i, Su_i = rollout_bouncing_stochastic_feedback(xt, current_modechange, states_i, modechange_i, 
                                                                        inputs_i, K_feedback_i, k_feedforward_i, target_state, R_k, Q_T,
                                                                        start_time_i, end_time, epsilon, noise_i, mode_exttrjs_maps=None)
        sampled_trjs_compare[i_sample] = sample_i
        sampled_controls_compare[i_sample] = ut_cl_i
        PathCosts_compare[i_sample] = Su_i
        
    PathCosts = PathCosts - np.min(PathCosts)
    PathCosts_eps = PathCosts / epsilon
    
    expS = np.exp(-PathCosts_eps)

    # ------- Compute the expected value ---------
    E_expS = np.mean(expS)
    
    # ------- Compute weights -------
    weights = expS / E_expS
    
    print("*** Var weight", np.var(weights))
    print("*** lambda", 1.0 / np.mean(weights**2))
    
    # --- comparison ---
    PathCosts_compare = PathCosts_compare - np.min(PathCosts_compare)
    PathCosts_eps_compare = PathCosts_compare / epsilon
    
    expS_compare = np.exp(-PathCosts_eps_compare)

    # ------- Compute the expected value ---------
    E_expS_compare = np.mean(expS_compare)
    
    # ------- Compute weights -------
    weights_compare = expS_compare / E_expS_compare
    
    print("*** Var weight (with mismatches)", np.var(weights_compare))
    print("*** lambda (with mismatches)", 1.0 / np.mean(weights_compare**2))
    
    # ------------------------ Visualize sampled trajectories ------------------------ 
    show_samples = True
    if show_samples:
    
        fig3, axes = plt.subplots(1, 2, figsize=(9, 6))
        ax5, ax6 = axes.flatten()
        ax5.grid(True)
        ax6.grid(True)
        
        # --- comparison --- 
        for i_s in range(n_samples):
            ax5.plot(sampled_trjs_compare[i_s,:,0], sampled_trjs_compare[i_s,:,1],'b', alpha=0.2)
        ax5.plot(sampled_trjs_compare[-1,:,0], sampled_trjs_compare[-1,:,1],'b', alpha=0.2, label='Rollouts')
        ax5.scatter(target_state[0], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
        ax5.scatter(init_state[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')
        ax5.legend()
        ax5.set_title("Rollouts with Mode Mismatchs", fontsize=18)
        fig3.tight_layout()
        
        for i_s in range(n_samples):
            ax6.plot(sampled_trjs[i_s,:,0], sampled_trjs[i_s,:,1],'b', alpha=0.2)
        ax6.plot(sampled_trjs[-1,:,0], sampled_trjs[-1,:,1],'b', alpha=0.2, label='Rollouts')
        ax6.scatter(target_state[0], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
        ax6.scatter(init_state[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')
        ax6.legend()
        ax6.set_title("Mode Mismatch Corrected", fontsize=18)
        fig3.tight_layout()
        
        plt.show()
        
        fig3.savefig(root_dir+'/hybrid_pathintegral/comparison_mismatch.pdf', format='pdf', dpi=2000)
        
        ## ============= show path costs and weights in path integral control ============= 
        
        fig6, ax11 = plt.subplots(figsize=(8,6))
        ax11.grid(True)
        ax11.bar(range(len(PathCosts)), PathCosts)
        ax11.set_title("Path Cost distribution")
        ax11.set_xlabel("Sample Number")
        ax11.set_ylabel("Costs")
        
        fig7, ax12 = plt.subplots(figsize=(8,6))
        ax12.grid(True)
        ax12.bar(range(len(weights)), weights)
        ax12.set_title("Weight distribution")
        ax12.set_xlabel("Sample Number")
        ax12.set_ylabel("Weights")
        plt.show()
    
    ### ------------------------ Visualize sampled trajectories / ------------------------ 
            
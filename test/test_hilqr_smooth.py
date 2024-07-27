# test the hybrid ilqr in the smooth case for a linear system. 
# The result should be the optimal solution
import numpy as np

import os
import sys
file_path = os.path.abspath(__file__)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

# Import pendulum dynamics
from dynamics.dynamics_bouncing import *
# Import iLQR class
from hybrid_ilqr.h_ilqr import solve_ilqr
# Importing path integral control
from hybrid_pathintegral.hybrid_pathintegral import *
# Import experiment parameter class
from experiments.exp_params import *

# for paralle sampling on cpu
from joblib import Parallel, delayed

import unittest

class TestHybirdPathIntegral(unittest.TestCase):
    def test_smooth_case(self):
        # === ilqr parameters ===
        # Initialize timings
        dt = 5e-5
        
        # ------------- verification with no contact ------------- 
        start_time = 0
        end_time = 1.0
        time_span = np.arange(start_time, end_time, dt).flatten()
        nt = len(time_span)
        
        init_state = np.array([5, 1.5])    # Define the initial state to be the origin with no velocity
        target_state = np.array([1.0, 0.0])
        
        # ------------- /verification with no contact ------------- 
        # Set desired state
        n_states = 2
        n_inputs = 1
        
        # Define weighting matrices
        Q_k = np.zeros((n_states,n_states)) # zero weight to penalties along a strajectory since we are finding a trajectory
        R_k = np.eye(n_inputs)

        # Set the terminal cost
        Q_T = 200*np.eye(n_states)
        Q_T[0,0] = 2000.0
        
        # === path integral parameters ===
        epsilon = 2.0
        n_samples = 50
        
        # === solve for ilqr ===
        exp_params = ExpParams()
        initial_guess = 0.5*np.ones((np.shape(time_span)[0],n_inputs))
        exp_params.update_params(init_state, target_state, start_time, end_time, dt, initial_guess, 
                                epsilon, 1, n_samples, Q_k, R_k, Q_T, symbolic_dynamics_bouncing,detect_bouncing)
        (states,inputs,k_feedforward,K_feedback,current_cost,states_iter,modechanges,mode_exttrjs_maps) = solve_ilqr(exp_params, detect=True)
        
        trj_pi = np.zeros((nt, n_states))

        trj_pi[0] = init_state
        u_star_pi = np.zeros((nt, n_inputs))
        
        allPathCosts = np.zeros((nt-1, n_samples))
        
        xt = init_state
        current_modechange = (1, 1)
        current_mode = current_modechange[0]
        next_mode = current_modechange[1]
        
        cnt_mismatch = 0
        
        # only test for the first time step
        for i_t in range(1):
            
            start_time_i = start_time + i_t*dt

            nt_i = nt - i_t
            
            # else:
            states_i = states[i_t:,:]
            inputs_i = inputs[i_t:,:]
            modechange_i = modechanges[i_t:]
            K_feedback_i = K_feedback[i_t:,:]
            k_feedforward_i = k_feedforward[i_t:,:]
            ref_next_mode = modechanges[i_t][1]            
            xref_i = states_i[0]
            
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
            
            u0_proposal = inputs_i[0] + K_feedback_i[0]@(xt - xref_i) + k_feedforward_i[0]
            
            # sampling stochastic rollouts
            sampled_trjs = np.zeros((n_samples, nt_i, n_states))
            sampled_controls = np.zeros((n_samples, nt_i, n_inputs))
            PathCosts = np.zeros(n_samples)  
            
            GaussianNoise_i = np.random.randn(n_samples, nt_i, n_inputs)                
            
            # -- cpu parallel ---
            dt_shrinkingrate = 0.7
            samples_index = Parallel(n_jobs=-1)(delayed(process_sampling_feedback)(sampled_trjs[i,:,:], xt, current_modechange, 
                                                                                   states_i, modechange_i, 
                                                                                   inputs_i, K_feedback_i, k_feedforward_i, 
                                                                                   target_state, R_k, Q_T,
                                                                                   start_time_i, end_time, epsilon, GaussianNoise_i, dt_shrinkingrate, mode_exttrjs_maps, i) for i in range(n_samples))

            for sample_i, sample_input_i, Su_i, _, index in samples_index:
                sampled_trjs[index] = sample_i
                sampled_controls[index] = sample_input_i
                PathCosts[index] = Su_i
                
            # update the control proposal using path integral 
            u0_star = update_u0_pathintegral(u0_proposal, PathCosts, epsilon, dt)
            u_star_pi[i_t] = u0_star
            allPathCosts[i_t] = PathCosts
            
            PathCosts = PathCosts - np.min(PathCosts)
            PathCosts_eps = PathCosts / epsilon
            
            expS = np.exp(-PathCosts_eps)

            # ------- Compute the expected value ---------
            E_expS = np.mean(expS)
            
            # ------- Compute weights -------
            weights = expS / E_expS
            
            lambda_effectiveness = 1.0 / np.mean(weights**2)
            variance_weight = np.var(weights)
            # print("*** Var weight", variance_weight)
            # print("*** lambda", lambda_effectiveness)
            
        self.assertAlmostEqual(variance_weight, 0.0, delta=1e-1)
        self.assertAlmostEqual(lambda_effectiveness, 1.0, delta=1e-1)
        
        
if __name__ == '__main__':
    unittest.main()
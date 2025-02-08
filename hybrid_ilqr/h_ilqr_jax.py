# Allows for general nonlinear cost functions and 
# jacobian computations using jax automatic differentiations 
# Only considering 1 same smooth flow (stance mode dynamics) in all the modes.

# For walking robot, we assume 2 modes: divided by the swing foot height and velocity sign.

import jax
from jax import grad, jacfwd, hessian
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from dynamics.trajectory_extension import *
from walking_3link import *

from dynamics.saltation_matrix import compute_saltation

class hybrid_ilqr_jax:
    def __init__(self, 
                 nstates, ninputs,
                 init_state, 
                 target_state,
                 initial_guess,
                 timespan, 
                 n_iterations, 
                 is_detect, 
                 detect_func,
                 smooth_dynamics,
                 running_cost,
                 cost_args,
                 terminal_cost,
                 terminal_cost_args):
        
        self._init_mode = 1

        self._nx = nstates
        self._nu = ninputs

        self._initstate = init_state
        self._tarstate = target_state
        self._inputs = initial_guess
        self._initial_guess = initial_guess
        self._verbose = is_detect
        
        # time definitions
        self._timespan = timespan
        self._dt = timespan[1:] - timespan[:-1]
        self._starttime = timespan[0]
        self._endtime = timespan[-1]
        
        # self._timespan = np.arange(start_time, end_time, dt).flatten()
        self._nt = np.shape(self._timespan)[0]
        
        self._states = [np.zeros(self._nx[0]) for _ in range(self._nt)]
        
        # feedback and feedforward
        self._k_ff = [np.zeros((self._nu[0])) for _ in range(self._nt)]
        self._K_fb = [np.zeros((self._nu[0], self._nx[0])) for _ in range(self._nt)]
        
        # hybrid events and mode changes
        self._saltations = [np.array([None]) for i in range(self._nt)]
        self._modechanges = [np.array([0, 0]) for _ in range(self._nt)]
        self._modes = [0 for _ in range(self._nt)]
        
        # feedback and feedforward for the trajectory extensions 
        # shapes: feedback gains: [[K_fwd_extension, K_bwd_extension]_jump1, [K_fwd_extension, K_bwd_extension]_jump2, ...]
        # shapes: feedforwad gains: [[k_fwd_extension, k_bwd_extension]_jump1, [k_fwd_extension, k_bwd_extension]_jump2, ...]
        self._k_ff_ext = [np.zeros((self._nu[0])), np.zeros((self._nu[1]))]
        self._K_fb_ext = [np.zeros((self._nu[0], self._nx[0])), np.zeros((self._nu[1], self._nx[1]))]
        
        # ------------------------------------------------------------------------------------------------------------------ 
        #   Map that maps the index at hybrid event to the event informations (t_event, x_event, x_reset, mode_change). 
        # ------------------------------------------------------------------------------------------------------------------ 
        self._event_info = {}
        self._refext_helper = []
        
        # Dynamics
        self._smooth_dyn = smooth_dynamics
        self._detectfunc = partial(detect_func, detect=is_detect)

        # Jacobians of the smooth dynamics
        self._A = jax.jit(jacfwd(lambda x, u, dt: smooth_dynamics(x, u, dt), argnums=0))
        self._B = jax.jit(jacfwd(lambda x, u, dt: smooth_dynamics(x, u, dt), argnums=1))

        # Running costs
        self._running_cost = running_cost
        self._cost_args = cost_args
        
        # Gradients of the cost functions
        self._cost_x = jax.jit(grad(lambda x, u: running_cost(x, u, cost_args), 0))
        self._cost_u = jax.jit(grad(lambda x, u: running_cost(x, u, cost_args), 1))
        self._cost_xx = jax.jit(hessian(lambda x, u: running_cost(x, u, cost_args), 0))
        self._cost_uu = jax.jit(hessian(lambda x, u: running_cost(x, u, cost_args), 1))

        # Terminal costs
        self._terminal_cost = terminal_cost
        self._terminal_cost_args = terminal_cost_args

        # Gradients of the terminal costs
        self._terminal_cost_x = grad(lambda x: terminal_cost(x, terminal_cost_args), 0)
        self._terminal_cost_xx = jacfwd(lambda x: self._terminal_cost_x(x), 0)
        
        # Max iterations
        self._niters = n_iterations
        
    def desamble_control(self, modes, mode_inputs):
        nt = len(modes)
        inputs = np.zeros((nt, self._nu[0]))
        for i in range(nt):
            mode = modes[i]
            inputs[i,:] = mode_inputs[mode][i,:].flatten()

        return inputs
    
    def rollout(self):        
        
        fig = plt.figure()
        plt.subplot(1, 1, 1)
        plt.plot(self._timespan, self._inputs[0][:,0], label=r'$u_0 mode 0$')
        plt.legend(loc="best", fontsize=10)
        plt.title('Initial Control GUess')
        plt.xlabel('Time (sec)')
        plt.grid()
        plt.show()
        
        (timespan,modes,states,inputs,
         saltations,mode_changes,event_info) = self.forward_pass(self._timespan,
                                                                    self._modes,
                                                                    self._states,
                                                                    self._inputs,
                                                                    self._refext_helper,
                                                                    use_feedback=False, 
                                                                    learning_rate=1, 
                                                                    check_modemismatch=False)
        
        
        
        fig1 = plt.figure(figsize=(16, 9))
        inputs_whole = self.desamble_control(modes, inputs)
        plot_3link_states(timespan, states, inputs_whole)
        
        plt.subplot(3, 4, 9)
        plt.plot(timespan, modes, label=r'$mode$')
        plt.legend(loc="best", fontsize=10)
        plt.title('Mode')
        plt.xlabel('Time (sec)')
        plt.grid()
        
        plt.subplot(3, 4, 10)
        plt.plot(timespan, self._initial_guess[0][:,0], label=r'$u_0 mode 0$')
        plt.plot(timespan, self._initial_guess[0][:,1], label=r'$u_1 mode 1$')
        plt.legend(loc="best", fontsize=10)
        plt.title('Initial Control GUess')
        plt.xlabel('Time (sec)')
        plt.grid()
        
        # fig, ax = plt.subplots()
        anim(timespan, states, 1/30, speed=1, fig=fig1, loop=5)
        
        plt.show()

        # fig, ax = plt.subplots()
        # states_arr = np.array(states)
        
        # ax.plot(states_arr[:,0], label='x_0')
        # ax.plot(states_arr[:,1], label='x_1')
        # ax.plot(states_arr[:,2], label='x_2')
        
        # ax.legend()
        
        plt.show()
        
        return (timespan, modes, states, inputs, saltations, mode_changes, event_info)

    def compute_cost(self,modes,states,inputs,timespan):
        # Initialize cost
        total_cost = 0.0
        dt = timespan[1:] - timespan[:-1]
        nt = len(timespan)
        
        for ii in range(0,nt-1):
            mode_i = modes[ii]
            x_i = states[ii] # Not being used currently
            u_i = inputs[mode_i][ii].flatten()

            total_cost = total_cost+self._running_cost(x_i, u_i)*dt[ii]
            
        # Compute terminal cost
        total_cost = total_cost + self._terminal_cost(states[-1], self._tarstate)

        return total_cost

    def backward_pass(self, timespan, modes, states, inputs, saltations, event_info):
        
        nt = timespan.shape[0]
        dt = timespan[1:] - timespan[:-1]
        
        V_xx = self._terminal_cost_xx(self._states[-1])
        V_x = self._terminal_cost_x(self._states[-1])

        k_trj = [np.zeros((self._nu[0])) for _ in range(nt)]
        K_trj = [np.zeros((self._nu[0], self._nx[0])) for _ in range(nt)]
        
        k_ff_trj_ext = []
        K_fb_trj_ext = []
        
        # Initialize cost reduction
        expected_cost_reduction = 0
        expected_cost_reduction_grad = 0
        expected_cost_reduction_hess = 0

        
        # for loop backwards in time
        for idx in reversed(range(0, nt-1)):
            # Grab the current variables in the trajectory
            mode_i = modes[idx]
            x_i = states[idx]
            u_i = inputs[mode_i][idx]
            saltation_i = saltations[idx]
            dt_i = dt[idx]
            
            # R_k_updated
            # Define the expansion coefficients and the loss gradients
            l_xx = self._cost_xx(x_i, u_i) # For now zeros, can add in a target to track later on
            l_uu = self._cost_uu(x_i, u_i)

            l_x = self._cost_x(x_i, u_i)@np.zeros(self._nx[0]).flatten() # For now zeros, can add in a target to track later on
            l_u = self._cost_u(x_i, u_i)@u_i.flatten()

            # Get the jacobian of the discretized dynamics
            A_k = self._A(x_i, u_i, dt_i)
            B_k = self._B(x_i, u_i, dt_i)
            
            if saltation_i is None:
                Q_x = l_x*dt_i + A_k.T@V_x
                Q_u = l_u*dt_i + B_k.T@V_x
                Q_ux = B_k.T@V_xx@A_k
                Q_uu = l_uu*dt_i + B_k.T@V_xx@B_k
                Q_xx = l_xx*dt_i + A_k.T@V_xx@A_k
                
                # Compute gains           
                k = -np.linalg.solve(Q_uu, Q_u)
                K = -np.linalg.solve(Q_uu, Q_ux).reshape((self._nu[0], self._nx[0]))

            else:
                # print("Found contact dynamics! Computing the gains with saltation matrix.")
                Q_x = l_x*dt_i + A_k.T @ saltation_i.T @ V_x
                Q_u = l_u*dt_i + B_k.T @ saltation_i.T @ V_x
                Q_ux = B_k.T @ saltation_i.T @ V_xx @ saltation_i @ A_k
                Q_uu = l_uu*dt_i + B_k.T @ saltation_i.T @ V_xx @ saltation_i @ B_k
                Q_xx = l_xx*dt_i + A_k.T @ saltation_i.T @ V_xx @ saltation_i @ A_k    
                
                # Compute gains           
                k = -np.linalg.solve(Q_uu, Q_u)
                K = -np.linalg.solve(Q_uu, Q_ux).reshape((self._nu[0], self._nx[0]))

                # Compute the (gains for the forward extension, gains for the backward extension): use the gain at the immediate next state (reseted)
                k_ff_trj_ext.append((k, k_trj[idx+1]))
                K_fb_trj_ext.append((K, K_trj[idx+1]))
                
                # update the hybrid dynamics information
                previous_event_info = list(event_info[idx])
                previous_event_info[4] = [K, K_trj[idx+1]]
                previous_event_info[5] = [k, k_trj[idx+1]]
                event_info[idx] = tuple(previous_event_info)

            # Store gains
            k_trj[idx] = k
            K_trj[idx] = K

            # Update the expected reduction
            current_cost_reduction_grad = -Q_u.T@k
            current_cost_reduction_hess = 0.5 * k.T @ (Q_uu) @ (k)
            current_cost_reduction = current_cost_reduction_grad + current_cost_reduction_hess

            expected_cost_reduction_grad +=  current_cost_reduction_grad
            expected_cost_reduction_hess +=  current_cost_reduction_hess
            expected_cost_reduction += + current_cost_reduction

            # Update hessian and gradient for value function (If we arent using regularization we can simplify this computation)
            # V_x = Q_x +K.T@Q_uu@k + K.T@Q_u + Q_ux.T@k
            # V_xx = (Q_xx+Q_ux.T@K+K.T@Q_ux+K.T@Q_uu@K)
            V_x = Q_x - K.T@Q_uu@k
            V_xx = Q_xx - K.T@Q_uu@K

        # Store expected cost reductions
        self.expt_cost_redu_grad_ = expected_cost_reduction_grad
        self.expt_cost_redu_hess_ = expected_cost_reduction_hess
        
        
        # Store the gain for the backward extensions
        K_fb_trj_ext.reverse()
        k_ff_trj_ext.reverse()
        
        return (k_trj,K_trj,k_ff_trj_ext,K_fb_trj_ext,expected_cost_reduction,event_info)

    def forward_pass(self, 
                     timespan,
                     modes,
                     states,
                     inputs,
                     ref_ext_helper,
                     use_feedback=True, 
                     learning_rate=1, 
                     check_modemismatch=True):
        
        # temporary variables in the current forward pass
        dt = timespan[1:] - timespan[:-1]
        nt = len(timespan)
        
        saltations = [None for i in range(nt)]
        mode_changes = np.tile(np.array([0, 0]), (nt, 1))
        
        if self._verbose:
            if (not use_feedback):
                print("---------- Initial rollout ----------")
            else:
                print(f"---------- Forward pass. Learning rate: {learning_rate} ----------")
        
        # Lists to collect the current forward pass trajectories (Dimensions might vary so use list)
        if states is None:
            states = [np.array([0.0]) for _ in range(nt)]
        if inputs is None:
            inputs = [np.zeros((nt, self._nu[0])), np.zeros((nt, self._nu[0]))]
        if modes is None:
            modes = [0 for _ in range(nt)]

        # Set the first state to be the initial
        current_state = self._initstate
        current_mode = self._init_mode
        
        modes[0] = current_mode
        states[0] = current_state
        mode_changes[0] = np.array([current_mode, current_mode])
        
        # Extend reference trj, if a hybrid event is hit.
        hybrid_index = set()
        event_info = {} # The dictionary that stores all the information of the jump dynamics and states.
        
           
        # -------------------------------
        # Current rollout hybrid events 
        # -------------------------------
        cnt_event = 0
        hybrid_index_ref = 0
        
        fig, ax = plt.subplots()
        
        ax.plot(inputs[0][:, 0], label='mode 0 u_1')
        ax.plot(inputs[1][:, 0], label='mode 1 u_1')
        
        ax.legend()
        
        # plt.show()
        
        if use_feedback:
            K_fb_ext = []
            k_ff_ext = []
            
            # Reference hybrid events and extensions from the last iteration
            (v_modechg_ref, v_ext_bwd, v_ext_fwd, 
            v_Kfb_ext_bwd, v_Kfb_ext_fwd, 
            v_kff_ext_bwd, v_kff_ext_fwd, v_tevents_ref) = extract_extensions(ref_ext_helper)

            # Construct local extension feedback and feedforward gains
            
            for i_ext in range(len(v_modechg_ref)):
                K_fb_ext.append((v_Kfb_ext_fwd[i_ext], v_Kfb_ext_bwd[i_ext]))
                k_ff_ext.append((v_kff_ext_fwd[i_ext], v_kff_ext_bwd[i_ext]))
                
                # if self._verbose:
                #     print(f"Reference trajectory bouncing event numbers: {len(v_ext_bwd)}")
                #     for i_bounce in range(len(v_ext_bwd)):
                #         print(f"bounce {i_bounce}: From mode {v_modechg_ref[i_bounce][0]} to mode {v_modechg_ref[i_bounce][1]} at time {v_tevents_ref[i_bounce]}")
                #     print("------------------------------------------------")
                        
        else:
            K_fb_ext = self._K_fb_ext
            k_ff_ext = self._k_ff_ext
        
        
        for ii in range(nt-1):
            
            x_i = states[ii]
            mode_i = modes[ii]
            
            # ------------------- 
            # Get the references 
            # ------------------- 
            u_i = inputs[mode_i][ii]
            
            # ====================================
            #  If it is not the first time rollout
            # ====================================
            if use_feedback:
                           
                ref_state = self._states[ii]
                mode_i_ref = self._modes[ii]
                
                # ----------------------------------------------- 
                # Get the current (feedback, feedforward) gains 
                # ----------------------------------------------- 
                current_feedforward = learning_rate * self._k_ff[ii]
                current_feedback = self._K_fb[ii]
                
                # ---------------
                #  Mode Mismatch
                # --------------- 
                if (mode_i != mode_i_ref) and (check_modemismatch):
                    
                    trj_extension = []
                    fb_ext_trj = []
                    ff_ext_trj = []
                    
                    hybrid_index_ref = np.argmin(abs(np.array(v_tevents_ref)-ii)) # find the nearest hybrid event in the reference
                    ref_modechange_hybrid = v_modechg_ref[hybrid_index_ref]
                    
                    # ----------------------------------- Early Arrival ----------------------------------- 
                    if ((mode_i == ref_modechange_hybrid[1]) and (mode_i_ref==ref_modechange_hybrid[0])):
                        # if self._verbose:
                        #     # print(f"early arrival, Time: {ii}. Current mode: {mode_i}, Reference mode: {mode_i_ref}")
                        #     print(f"Reference mode change from mode {ref_modechange_hybrid[0]} to mode {ref_modechange_hybrid[1]} at time {v_tevents_ref[hybrid_index_ref]}")
                        
                        trj_extension = v_ext_bwd[hybrid_index_ref]
                        
                        ff_ext_trj = v_kff_ext_bwd[hybrid_index_ref]
                        fb_ext_trj = v_Kfb_ext_bwd[hybrid_index_ref]
                        
                    # ----------------------------------- Late Arrival ----------------------------------- 
                    elif ((mode_i == ref_modechange_hybrid[0]) and (mode_i_ref==ref_modechange_hybrid[1])):
                        # if self._verbose:
                        #     # print(f"late arrival, Time: {ii}. Current mode: {mode_i}, Reference mode: {mode_i_ref}")
                        #     print(f"Reference mode change from mode {ref_modechange_hybrid[0]} to mode {ref_modechange_hybrid[1]} at time {v_tevents_ref[hybrid_index_ref]}")
                        
                        trj_extension = v_ext_fwd[hybrid_index_ref]
                        
                        ff_ext_trj = v_kff_ext_fwd[hybrid_index_ref]
                        fb_ext_trj = v_Kfb_ext_fwd[hybrid_index_ref]
                    
                    # Modify the reference to the extension
                    ref_state = trj_extension[ii]
                    current_feedback = fb_ext_trj[ii]
                    current_feedforward = learning_rate * ff_ext_trj[ii]
                    
                    # if self._verbose:
                    #     print("current_nominal_input: ", u_i)
                
                current_feedback_input = current_feedback@(x_i-ref_state)
                u_i = u_i + current_feedback_input + current_feedforward
            
            # =================
            # Simulate forward
            # =================
            t_ii = timespan[ii]
            dt_ii = dt[ii]
            t_ii_plus = t_ii + dt_ii
            
            (next_state, saltation, 
             mode_change, t_event, x_event, 
             x_reset, reset_byproduct) = self._detectfunc(x_i, u_i, t_ii, t_ii_plus, mode_i, reset_args=None)

            # ------------------------------
            # Update the hybrid information
            # ------------------------------
            if saltation is not None:
                timespan = np.concatenate(
                    (np.concatenate(
                        (timespan[:ii+1], np.array([t_event]))), timespan[ii+1:]))
                nt += 1
                
                modes = np.concatenate(
                    (np.concatenate(
                        (modes[:ii+1], np.array([mode_change[1]]))), modes[ii+1:]))
                
                states = np.concatenate(
                    (np.concatenate(
                        (states[:ii+1], np.array([x_reset]))), states[ii+1:]))
                
                next_state = x_reset
                
                inputs[mode_change[1]] = np.concatenate(
                    (np.concatenate(
                        (inputs[mode_change[1]][:ii+1], u_i.reshape(1,-1))), inputs[mode_change[1]][ii+1:]))

                inputs[mode_change[0]] = np.concatenate(
                    (np.concatenate(
                        (inputs[mode_change[0]][:ii+1], u_i.reshape(1,-1))), inputs[mode_change[0]][ii+1:]))
                
                inputs[mode_change[0]][ii] = u_i.flatten()

                hybrid_index.add(ii+1)
                
                saltations = saltations[:ii+1] + [saltation] + saltations[ii+1:]
                event_info[ii+1] = (t_event, x_event, x_reset, mode_change, K_fb_ext[hybrid_index_ref], k_ff_ext[hybrid_index_ref])
            else:
                states[ii+1] = next_state
                inputs[mode_i][ii] = u_i.flatten()
                
            # Only consider the transition from mode 0 to mode 1 for now
            if (mode_change[0]!=mode_change[1]):
                if self._verbose:
                    print(f"At Time {ii}, the system has a mode change from mode {mode_change[0]} to mode {mode_change[1]}")
                # event_args.append(reset_byproduct)
                # event_args = reset_byproduct
                cnt_event += 1
                
            # ---------------------
            #  Move forward in time
            # ---------------------            
            mode_changes[ii+1] = mode_change
            modes[ii+1] = mode_change[1]
        
        if self._verbose:
            print(f"--------------------- Total number of contacts: {cnt_event} ---------------------" )
        
        return (timespan, modes,states,inputs,saltations,mode_changes,event_info)
    
    
    def solve(self):
        # ------ collect the iteration data ------
        states_iter = []
        
        # ------------------------------------
        #  First rollout using initial guess
        # ------------------------------------
        print("--------------------- Starting initial rollout ---------------------")
        [timespan,modes,states,inputs,saltations,modechanges,event_info] = self.rollout()
        
            
        # compute reference extensions
        print("------ Computing the reference trajectory extensions ------")
        self._refext_helper = compute_trejactory_extension(event_info, 
                                                            timespan,
                                                            self._nx, self._nu,
                                                            self._initstate, self._tarstate, 
                                                            self._detectfunc)
        
        # Store the rollout as default values
        self._timespan = timespan
        self._dt = timespan[1:] - timespan[:-1]
        self._states = states
        self._inputs = inputs
        self._saltations = saltations
        self._modes = modes
        self._modechanges = modechanges
        self._event_info = event_info    

        # ------------ Plot first rollout ------------ 
        show_rollout = False
        if show_rollout:
            plt.figure(figsize=(6, 6))
            plt.subplot(1, 1, 1)
            ut = []
            for ii in range(len(timespan)):
                ut.append(inputs[modes[ii]][:, 0])
            ut_arr = np.array(ut)
            plt.plot(timespan, ut_arr[:, 0], label=r'$u_1$')
            plt.plot(timespan, ut_arr[:, 1], label=r'$u_2$')
            plt.legend(loc="best", fontsize=10)
            plt.title('Control Input Torque')
            plt.xlabel('Time (sec)')
            plt.grid()
            plt.show()

        print("===================== Finished initial rollout =====================")
        
        # ----------------------------------------------------
        #               Compute the initial cost 
        # ----------------------------------------------------
        current_cost = self.compute_cost(modes,states,inputs,timespan)
        
        # =============
        #   Main Loop
        # =============
        
        learning_speed = 0.9 # This can be modified, 0.95 is very slow
        low_learning_rate = 0.01 # if learning rate drops to this value stop the optimization
        low_expected_reduction = 1e-4 # Determines optimality
        armijo_threshold = 0.1 # Determines if current line search solve is good (this is typically labeled as "c")
        
        for i_iter in range(0,self._niters):
            print('========== Starting Iteration: ',i_iter,', Current cost: ',current_cost, ' ==========')
            print("-------- Backward Pass --------")
                
            # --------------------------------------------------------
            # Compute the backwards pass and update the control gains
            # --------------------------------------------------------
            (k_feedforward,K_feedback,
             k_ff_trj_ext,K_fb_trj_ext,
             expected_reduction,
             updated_event_info) = self.backward_pass(self._timespan, 
                                                      self._modes, self._states, 
                                                      self._inputs, 
                                                      self._saltations, 
                                                      self._event_info)    
            
            # Store updated variables
            self._k_ff = k_feedforward
            self._K_fb = K_feedback
            self.expt_cost_redu_ = expected_reduction
            self._K_fb_ext = K_fb_trj_ext
            self._k_ff_ext = k_ff_trj_ext
            self._event_info = updated_event_info
            
            print('-------- Expected cost reduction: ',expected_reduction, ' --------')
            
            if(abs(expected_reduction)<low_expected_reduction):
                print(" -------- Stopping optimization, Optimal trajectory found --------")
                break
            learning_rate = 1
            armijo_flag = 0
            
            # ---------------------------------------------
            # Forward pass under the updated control gains
            # ---------------------------------------------
            (new_timespan,new_modes,new_states,new_inputs,
             new_saltations,mode_changes,new_event_info)=self.forward_pass(self._timespan, 
                                                                           self._modes, 
                                                                           self._states, 
                                                                           self._inputs,
                                                                           self._refext_helper, 
                                                                           use_feedback=True, 
                                                                           learning_rate=learning_rate,
                                                                           check_modemismatch=True)
            
            # --------------------------------------------------------------
            #  Compute the new trajectory extensions and the gains for them
            # --------------------------------------------------------------
            print("------ Computing the new trajectory extensions ------")
            self._refext_helper = compute_trejactory_extension(new_event_info, 
                                                               new_timespan, 
                                                               self._nx, self._nu,
                                                               self._initstate, 
                                                               self._tarstate, 
                                                               self._detectfunc)
            
            # ---------------------------------------------------------
            #   Compute new costs and check the optimality conditions
            # ---------------------------------------------------------
            new_cost = self.compute_cost(new_modes,new_states, new_inputs, new_timespan)
            
            # armijo condition
            print("---------- Backtracking line search process ----------")
            while(learning_rate > 0.05 and armijo_flag == 0):
                # Decrease learning rate and continue line search
                learning_rate = learning_speed*learning_rate
                
                # Forward pass: line search 
                (new_timespan,new_modes,new_states,new_inputs,
                 new_saltations,mode_changes,new_event_info)=self.forward_pass(self._timespan, 
                                                                                self._modes, 
                                                                                self._states, 
                                                                                self._inputs, 
                                                                                self._refext_helper,
                                                                                use_feedback=True, 
                                                                                learning_rate=learning_rate,
                                                                                check_modemismatch=True)
                
                show_forwardpass = False
                if show_forwardpass:
                    pass
                
                new_cost = self.compute_cost(new_modes, new_states, new_inputs, new_timespan)
                
                print("new_cost: ", new_cost)
                
                # Calculate armijo condition
                cost_difference = (current_cost - new_cost)
                
                expected_cost_redu = learning_rate*self.expt_cost_redu_grad_ + learning_rate*learning_rate*self.expt_cost_redu_hess_
                armijo_flag = cost_difference/expected_cost_redu > armijo_threshold
                
                if(armijo_flag == 1):
                    print(" -------- Armijo condition met --------")
                    # ------------------------------------------------------
                    # Accept the new trajectory if armijo condition is met
                    # ------------------------------------------------------
                    current_cost = new_cost
                    self._timespan = new_timespan
                    self._states = new_states
                    self._inputs = new_inputs
                    self._saltations = new_saltations
                    self._modechanges = mode_changes
                    self._modes = new_modes
                    self._event_info = new_event_info
                    self._refext_helper = self.compute_trejactory_extension(new_event_info)
                    states_iter.append(new_states)
                    
            if(learning_rate<low_learning_rate):
                # If learning rate is low, then stop optimization
                print(" -------- Stopping optimization, low learning rate --------")
                
                current_cost = new_cost
                self._timespan = new_timespan
                self._states = new_states
                self._inputs = new_inputs
                self._saltations = new_saltations
                self._modechanges = mode_changes
                self._modes = new_modes
                self._event_info = new_event_info
                
                # Update the hybrid event maps
                self._refext_helper = self.compute_trejactory_extension(new_event_info)
                
                states_iter.append(new_states)
                    
                break
          
        
        print(" -------- Stopping optimization, reached max iteration --------")
          
        # Return the current trajectory
        timespan = self._timespan
        modes = self._modes
        states = self._states
        inputs = self._inputs
        modechanges = self._modechanges
        event_info = self._event_info
        show_results = False
        if show_results:
            pass

        ref_ext_helper = self.compute_trejactory_extension(event_info)

        return (timespan,modes,states,inputs,
                k_feedforward,K_feedback,
                current_cost,states_iter,
                modechanges,ref_ext_helper)


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
    init_state, _, _ = resetmap_3link_12(start_time, init_state)

    # Target is to go back to the initial state
    target_state = init_state  
    
    init_mode = 0

    # Set desired state
    n_modes = 1
    
    # the state and control dimensions, mode-dependent
    n_states = [6, 6]
    n_inputs = [2, 2]

    # ---------------------------- 
    #   Define weighting matrices
    # ----------------------------

    # Q_k = np.zeros((n_states[0],n_states[0]))
    # Q_k = [np.zeros((n_states[0],n_states[0])), np.zeros((n_states[1],n_states[1]))] # zero weight to penalties along a strajectory since we are finding a trajectory
    # R_k = [np.eye(n_inputs[0]), np.eye(n_inputs[1])]

    # # ---------------------------- Set the terminal cost ----------------------------
    # target_mode = 0
    # Q_T = 200*np.eye(n_states[0])
    # Q_T[0,0] = 2000.0

    n_exp = 1
    n_samples = 10
    n_iters = 20
    is_detect = True

    # ====================================
    #    Solve for hybrid ilqr proposal
    # ====================================

    tout, xout, uout, t_events, x_events, saltations = solve_limcycle_3link()

    initial_guess = [uout, uout] 

    target_hip_velocity = 2.0
    h_ilqr_solver = hybrid_ilqr_jax(n_states, n_inputs,
                                    init_state, target_state, 
                                    initial_guess, dt, 
                                    start_time, end_time, 
                                    n_iters, is_detect, 
                                    onestep_detect_3link, dyn_control_3link_discrete_jax, 
                                    hip_moving_cost, target_hip_velocity,
                                    statedeviation_norm_cost, target_state)
    
    h_ilqr_results = h_ilqr_solver.solve()

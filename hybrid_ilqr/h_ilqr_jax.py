# Allows for general nonlinear cost functions and 
# jacobian computations using jax automatic differentiations 

# Only considering 1 mode for now.

import jax
from jax import grad, jacfwd, hessian
import numpy as np
import matplotlib.pyplot as plt

from dynamics.trajectory_extension import *
# Import 3 link walker dynamics
from dynamics.walking_3link import *

class hybrid_ilqr_jax:
    def __init__(self, nmodes, nstates, 
                 init_state,target_state,initial_guess,
                 dt,start_time,end_time, 
                 contact_detect,smooth_dynamics,
                 running_cost,cost_args,
                 terminal_cost,terminal_cost_args,
                 n_iterations,detect, verbose=True):
        
        self.nmodes_ = nmodes
        self._n_states = nstates
        self._init_state = init_state
        self._target_state = target_state
        self._inputs = initial_guess
        self._n_inputs = np.shape(initial_guess)[1]
        self._verbose = verbose
        
        # time definitions
        self.dt_ = dt
        self.start_time_ = start_time
        self.end_time_ = end_time
        self.time_span_ = np.arange(start_time, end_time, dt).flatten()
        self._nt = np.shape(self.time_span_)[0]
        
        # feedback and feedforward
        self._k_ff = [np.zeros((self._n_inputs[0])) for _ in range(self._nt)]
        self._K_fb = [np.zeros((self._n_inputs[0], self._n_states[0])) for _ in range(self._nt)]
        
        # hybrid events and mode changes
        self._saltations = [np.array([None]) for i in range(self._nt)]
        self._modechanges = [np.array([0, 0]) for _ in range(self._nt)]
        self._modes = [0 for _ in range(self._nt)]
        
        # feedback and feedforward for the trajectory extensions 
        # shapes: feedback gains: [[K_fwd_extension, K_bwd_extension]_jump1, [K_fwd_extension, K_bwd_extension]_jump2, ...]
        # shapes: feedforwad gains: [[k_fwd_extension, k_bwd_extension]_jump1, [k_fwd_extension, k_bwd_extension]_jump2, ...]
        self._k_ff_ext = [[None], [None]]
        self._K_fb_ext = [[None], [None]]
        
        # ------------------------------------------------------------------------------------------------------------------ 
        # Map that maps the index at hybrid event to the event informations (t_event, x_event, x_reset, mode_change). 
        # ------------------------------------------------------------------------------------------------------------------ 
        self._event_info = {}
        self._refext_helper = []
        
        # Dynamics
        self.smooth_dyn_ = smooth_dynamics
        self.detection_func_ = contact_detect

        # Jacobians of the smooth dynamics
        self._A = jax.jit(grad(lambda x, u: smooth_dynamics(x, u), 0))
        self._B = jax.jit(grad(lambda x, u: smooth_dynamics(x, u), 1))

        # Running costs
        self._running_cost = running_cost
        self._cost_args = cost_args

        # Gradients of the cost functions
        self._cost_x = jax.jit(grad(lambda x, u, args: running_cost(x, u, args), 0))
        self._cost_u = jax.jit(grad(lambda x, u, args: running_cost(x, u, args), 1))
        self._cost_xx = jax.jit(hessian(lambda x, u, args: running_cost(x, u, args), 0))
        self._cost_uu = jax.jit(hessian(lambda x, u, args: running_cost(x, u, args), 1))

        # Terminal costs
        self._terminal_cost = terminal_cost
        self._terminal_cost_args = terminal_cost_args

        # Gradients of the terminal costs
        self._terminal_cost_x = grad(lambda x, args: terminal_cost(x, args), 0)
        self._terminal_cost_xx = jacfwd(lambda x: self._terminal_cost_x(x), 0)
        
        # Max iterations
        self._niters = n_iterations
        
        # Flag detection
        self.detect_ = detect

    def rollout(self):        

        (modes,states,inputs,
         saltations,mode_changes,
         hybrid_event_info) = self.forward_pass(use_feedback=False,
                                                           learning_rate=1,
                                                           check_modemismatch=False)

        # Store the trajectory(states, inputs)
        self._states = states
        self._inputs = inputs
        self._saltations = saltations
        self._modes = modes
        self._event_info = hybrid_event_info
        
        return modes, states, inputs, saltations, mode_changes

    def compute_cost(self,states,inputs,dt):
        # Initialize cost
        total_cost = 0.0
        for ii in range(0,self._nt-1):
            current_x = states[ii] # Not being used currently
            current_u = inputs[ii].flatten()

            total_cost = total_cost+self._running_cost(current_x, current_u)*dt
            
        # Compute terminal cost
        total_cost = total_cost + self._terminal_cost(states[-1])

        # terminal_state = states[-1]
        # terminal_difference = (self._target_state - terminal_state).flatten()
        # terminal_cost = 0.5*terminal_difference.T@self.Q_T_@terminal_difference
        # total_cost = total_cost+terminal_cost
        return total_cost

    def backwards_pass(self):
        end_difference = (self._states[-1] - self._target_state).flatten()

        V_xx = self._terminal_cost_xx(end_difference)
        V_x = self._terminal_cost_x(end_difference)

        k_trj = [np.zeros((self._n_inputs[0])) for _ in range(self._nt)]
        K_trj = [np.zeros((self._n_inputs[0], self._n_states[0])) for _ in range(self._nt)]
        
        k_feedforward_trj_extension = []
        K_feedback_trj_extension = []
        
        # Initialize cost reduction
        expected_cost_reduction = 0
        expected_cost_reduction_grad = 0
        expected_cost_reduction_hess = 0

        # for loop backwards in time
        for idx in reversed(range(0, self._nt-1)):
            # Grab the current variables in the trajectory
            current_mode = self._modes[idx]
            current_x = self._states[idx]
            current_u = self._inputs[idx]
            saltation = self._saltations[idx]

            # R_k_updated
            # Define the expansion coefficients and the loss gradients
            l_xx = self._cost_xx # For now zeros, can add in a target to track later on
            l_uu = self._cost_uu

            l_x = self._cost_x@np.zeros(self._n_states).flatten() # For now zeros, can add in a target to track later on
            l_u = self._cost_u@(current_u).flatten()

            # Get the jacobian of the discretized dynamics
            A_k = self._A(current_x, current_u, self.dt_)
            B_k = self._B(current_x, current_u, self.dt_)
            
            if saltation is None:
                Q_x = l_x*self.dt_ + A_k.T@V_x
                Q_u = l_u*self.dt_+ B_k.T@V_x
                Q_ux = B_k.T@V_xx@A_k
                Q_uu = l_uu*self.dt_ + B_k.T@V_xx@B_k
                Q_xx = l_xx*self.dt_ + A_k.T@V_xx@A_k
                
                # Compute gains           
                k = -np.linalg.solve(Q_uu, Q_u)
                K = -np.linalg.solve(Q_uu, Q_ux).reshape((self._n_inputs, self._n_states))

            else:
                # print("Found contact dynamics! Computing the gains with saltation matrix.")
                Q_x = l_x*self.dt_ + A_k.T @ saltation.T @ V_x
                Q_u = l_u*self.dt_ + B_k.T @ saltation.T @ V_x
                Q_ux = B_k.T @ saltation.T @ V_xx @ saltation @ A_k
                Q_uu = l_uu*self.dt_ + B_k.T @ saltation.T @ V_xx @ saltation @ B_k
                Q_xx = l_xx*self.dt_ + A_k.T @ saltation.T @ V_xx @ saltation @ A_k    
                
                # Compute gains           
                k = -np.linalg.solve(Q_uu, Q_u)
                K = -np.linalg.solve(Q_uu, Q_ux).reshape((self._n_inputs, self._n_states))

                # Compute the (gains for the forward extension, gains for the backward extension): use the gain at the immediate next state (reseted)
                k_feedforward_trj_extension.append((k, k_trj[idx+1]))
                K_feedback_trj_extension.append((K, K_trj[idx+1]))
                
                # update the hybrid dynamics information
                previous_hybrid_event_info = list(self._event_info[idx])
                previous_hybrid_event_info[4] = [K, K_trj[idx+1]]
                previous_hybrid_event_info[5] = [k, k_trj[idx+1]]
                self._event_info[idx] = tuple(previous_hybrid_event_info)

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
        self.expt_cost_redu_ = expected_cost_reduction

        # Store gain schedule
        self._k_ff = k_trj
        self._K_fb = K_trj
        
        # Store the gain for the backward extensions
        K_feedback_trj_extension.reverse()
        k_feedforward_trj_extension.reverse()
        self._K_fb_ext = K_feedback_trj_extension
        self._k_ff_ext = k_feedforward_trj_extension
        
        return (k_trj,K_trj,expected_cost_reduction)

    def forward_pass(self, use_feedback=True, 
                     learning_rate=1,
                     check_modemismatch=True):
        
        if self._verbose:
            if (not use_feedback):
                print("---------- Initial rollout ----------")
            else:
                print(f"---------- Forward pass. Learning rate: {learning_rate} ----------")
        
        # Lists to collect the current forward pass trajectories (Dimensions might vary so use list)
        states = [np.array([0.0]) for _ in range(self._nt)]
        inputs = [np.zeros((self._nt, self._n_inputs[0])), np.zeros((self._nt, self._n_inputs[1]))]
        modes = [0 for _ in range(self._nt)]
        saltations = [None for i in range(self._nt)]
        mode_changess = np.tile(np.array([0, 0]), (self._nt, 1))
        
        # Set the first state to be the initial
        current_state = self._init_state
        current_mode = self._init_mode
        
        modes[0] = self._init_mode
        states[0] = current_state
        mode_changess[0] = np.array([current_mode, current_mode])
        
        # Extend reference trj, if a hybrid event is hit.
        hybrid_index = set()
        hybrid_event_info = {} # The dictionary that stores all the information of the jump dynamics and states.
        
        if use_feedback:
            # Reference hybrid events and extensions from the last iteration
            (v_modechg_ref, v_ext_bwd, v_ext_fwd, 
             v_Kfb_ext_bwd, v_Kfb_ext_fwd, 
             v_kff_ext_bwd, v_kff_ext_fwd, v_tevents_ref) = extract_extensions(self._refext_helper)

            if self._verbose:
                print(f"Reference trajectory bouncing event numbers: {len(v_ext_bwd)}")
                for i_bounce in range(len(v_ext_bwd)):
                    print(f"bounce {i_bounce}: From mode {v_modechg_ref[i_bounce][0]} to mode {v_modechg_ref[i_bounce][1]} at time {v_tevents_ref[i_bounce]}")
                print("------------------------------------------------")
            
           
        # -------------------------------
        # Current rollout hybrid events 
        # -------------------------------
        cnt_event = 0
        hybrid_index_ref = 0
        
        for ii in range(self._nt-1):
            
            current_state = states[ii]
            current_mode = modes[ii]
            
            # ------------------- 
            # Get the references 
            # ------------------- 
            current_input = self._inputs[ii]
            
            # ====================================
            #  If it is not the first time rollout
            # ====================================
            if use_feedback:
            
                ref_state = self._states[ii]
                current_mode_ref = self._modes[ii]
                
                # ----------------------------------------------- 
                # Get the current (feedback, feedforward) gains 
                # ----------------------------------------------- 
                current_feedforward = learning_rate * self._k_ff[ii]
                current_feedback = self._K_fb[ii]
                
                # ---------------
                # Mode Mismatch
                # --------------- 
                if (current_mode != current_mode_ref) and (check_modemismatch):
                    
                    trj_extension = []
                    fb_ext_trj = []
                    ff_ext_trj = []
                    
                    hybrid_index_ref = np.argmin(abs(np.array(v_tevents_ref)-ii)) # find the nearest hybrid event in the reference
                    ref_modechange_hybrid = v_modechg_ref[hybrid_index_ref]
                    
                    # ----------------------------------- Early Arrival ----------------------------------- 
                    if ((current_mode == ref_modechange_hybrid[1]) and (current_mode_ref==ref_modechange_hybrid[0])):
                        if self._verbose:
                            print(f"early arrival, Time: {ii}. Current mode: {current_mode}, Reference mode: {current_mode_ref}")
                            print(f"Reference mode change from mode {ref_modechange_hybrid[0]} to mode {ref_modechange_hybrid[1]} at time {v_tevents_ref[hybrid_index_ref]}")
                        
                        trj_extension = v_ext_bwd[hybrid_index_ref]
                        
                        ff_ext_trj = v_kff_ext_bwd[hybrid_index_ref]
                        fb_ext_trj = v_Kfb_ext_bwd[hybrid_index_ref]
                        
                    # ----------------------------------- Late Arrival ----------------------------------- 
                    elif ((current_mode == ref_modechange_hybrid[0]) and (current_mode_ref==ref_modechange_hybrid[1])):
                        if self._verbose:
                            print(f"late arrival, Time: {ii}. Current mode: {current_mode}, Reference mode: {current_mode_ref}")
                            print(f"Reference mode change from mode {ref_modechange_hybrid[0]} to mode {ref_modechange_hybrid[1]} at time {v_tevents_ref[hybrid_index_ref]}")
                        
                        trj_extension = v_ext_fwd[hybrid_index_ref]
                        
                        ff_ext_trj = v_kff_ext_fwd[hybrid_index_ref]
                        fb_ext_trj = v_Kfb_ext_fwd[hybrid_index_ref]
                    
                    # Modify the reference to the extension
                    ref_state = trj_extension[ii]
                    current_feedback = fb_ext_trj[ii]
                    current_feedforward = learning_rate * ff_ext_trj[ii]
                    
                    if self._verbose:
                        print("current_nominal_input: ", current_input)
                
                current_feedback_input = current_feedback@(current_state-ref_state)
                current_input = current_input + current_feedback_input + current_feedforward
            
            # =================
            # Simulate forward
            # =================
            t_ii = self.time_span_[ii]
            
            (next_state, saltation, mode_change, 
             t_event, x_event, x_reset, reset_byproduct) = self.detection_func_(current_state, current_input, 
                                                                                t_ii, t_ii+self.dt_, 
                                                                                current_mode, self.detect_)
            
            # ------------------------------
            # Update the hybrid information
            # ------------------------------
            if saltation is not None:
                hybrid_index.add(ii)
                saltations[ii] = saltation
                hybrid_event_info[ii] = (t_event, x_event, x_reset, mode_change, self._K_fb_ext[hybrid_index_ref], self._k_ff_ext[hybrid_index_ref])
            
            # Only consider the transition from mode 0 to mode 1 for now
            if (mode_change[0]!=mode_change[1]):
                if self._verbose:
                    print(f"At Time {ii}, the system has a mode change from mode {mode_change[0]} to mode {mode_change[1]}")
                # event_args.append(reset_byproduct)
                event_args = reset_byproduct
                cnt_event += 1
                
            # ---------------------
            # Move forward in time
            # ---------------------
            states[ii+1] = next_state
            inputs[ii] = current_input.flatten()
            mode_changess[ii+1] = mode_change
            modes[ii+1] = mode_change[1]
        
        if self._verbose:
            print(f"--------------------- Total number of contacts: {cnt_event} ---------------------" )
        
        return (modes,states,inputs,saltations,mode_changess,hybrid_event_info)
    
    
    def compute_trejactory_extension(self, hybrid_event_info):
        
        # NO hybrid events
        if len(hybrid_event_info.keys()) == 0:
            return []
        
        # hybrid_event_info:
        sorted_hybrid_index = sorted(hybrid_event_info.keys())
        ref_ext_helper = []
        
        i_events = []
        t_events = []
        x_events = []
        x_resets = []
        K_fb_fwd_extensions = []
        k_ff_fwd_extensions = []
        K_fb_bwd_extensions = []
        k_ff_bwd_extensions = []
        
        mode_changes = []
        
        i_events.append(0)
        t_events.append(self.start_time_)
        x_events.append(self._init_state)
        x_resets.append(self._init_state)
        K_fb_fwd_extensions.append(np.zeros(1))
        K_fb_bwd_extensions.append(np.zeros(1))
        k_ff_fwd_extensions.append(np.zeros(1))
        k_ff_bwd_extensions.append(np.zeros(1))
        mode_changes.append(np.array([0, 0]))
        
        # hybrid_event_info[i_key] = (t_event, x_event, x_reset, mode_change, K_feedback_extensions, K_feedforward_extensions)
        for i_key in sorted_hybrid_index:
            i_events.append(i_key)
            t_events.append(hybrid_event_info[i_key][0])
            x_events.append(hybrid_event_info[i_key][1])
            x_resets.append(hybrid_event_info[i_key][2])
            mode_changes.append(hybrid_event_info[i_key][3])
            
            K_fb_fwd_extensions.append(hybrid_event_info[i_key][4][0])
            K_fb_bwd_extensions.append(hybrid_event_info[i_key][4][1])
            
            k_ff_fwd_extensions.append(hybrid_event_info[i_key][5][0])
            k_ff_bwd_extensions.append(hybrid_event_info[i_key][5][1])
        
        i_events.append(self._nt)
        t_events.append(self.end_time_)
        x_events.append(self._target_state)
        x_resets.append(self._target_state)
        K_fb_fwd_extensions.append(np.zeros(1))
        K_fb_bwd_extensions.append(np.zeros(1))
        k_ff_fwd_extensions.append(np.zeros(1))
        k_ff_bwd_extensions.append(np.zeros(1))
        
        if hybrid_event_info.keys():
            mode_changes.append(hybrid_event_info[sorted_hybrid_index[-1]][3])
        
        # Forward and backward trajectory extensions and (feedback, feedforward) gains for the two extensions
        for ii, tevent_i in enumerate(t_events[1:-1], start=1):
            i_event_i = i_events[ii]
            x_event_i = x_events[ii]
            x_reset_i = x_resets[ii]
            current_mode_i = mode_changes[ii][0]
            next_mode_i = mode_changes[ii][1]
            
            K_feedback_fwd_extension_i = K_fb_fwd_extensions[ii]
            k_feedforward_fwd_extension_i = k_ff_fwd_extensions[ii]
            
            K_feedback_bwd_extension_i = K_fb_bwd_extensions[ii]
            k_feedforward_bwd_extension_i = k_ff_bwd_extensions[ii]
            
            # --------------------------------------
            # Choose a time span for the extensions
            # --------------------------------------
            t_ext_fwd_i = self.end_time_
            
            # [0, t_event] for padding
            time_span_ext_fwd_padding = np.arange(0, tevent_i, self.dt_)
            # [t_event, t_trj_ext_fwd]
            timespan_ext_fwd = np.arange(tevent_i, t_ext_fwd_i, self.dt_)
            # [t_trj_ext_bwd: t_event]
            timespan_ext_bwd = np.arange(0, tevent_i, self.dt_)[::-1]
            
            # time span lengths
            nt_ext_fwd = len(timespan_ext_fwd)
            nt_ext_bwd = len(timespan_ext_bwd)
            nt_ext_padding_fwd = len(time_span_ext_fwd_padding)
            nt_ext_padding_bwd = nt_ext_fwd
            
            xtrj_ext_padding_fwd_i = np.zeros((nt_ext_padding_fwd, self._n_states[current_mode_i]))
            xtrj_ext_fwd_i = np.zeros((nt_ext_fwd, self._n_states[current_mode_i]))
            xtrj_ext_fwd_i[0] = x_event_i
            
            xtrj_ext_bwd_i = np.zeros((nt_ext_bwd, self._n_states[next_mode_i]))
            xtrj_ext_padding_bwd_i = np.zeros((nt_ext_padding_bwd, self._n_states[next_mode_i]))    
            
            # Use the same gain for the whole extensions, and padding it to the whole time span
            K_feedback_ext_fwd_i = np.tile(K_feedback_fwd_extension_i, (self._nt, 1, 1))
            k_feedforward_ext_fwd_i = np.tile(k_feedforward_fwd_extension_i, (self._nt, 1))
            
            K_feedback_ext_bwd_i = np.tile(K_feedback_bwd_extension_i, (self._nt, 1, 1))
            k_feedforward_ext_bwd_i = np.tile(k_feedforward_bwd_extension_i, (self._nt, 1))
            
            # ----------------------------------
            # simulate the forward extension
            # ----------------------------------
            current_state = x_event_i
            for jj in range(nt_ext_fwd-1):
                t_jj = timespan_ext_fwd[jj]
                
                # Using zero control, modify if needed.
                current_input = np.zeros(self._n_inputs[current_mode_i])
                
                next_state, _, _, _, _, _, _ = self.detection_func_(current_state, current_input, 
                                                                    t_jj, t_jj+self.dt_, 
                                                                    current_mode_i, detection=False)
                
                # Store states and inputs
                xtrj_ext_fwd_i[jj+1] = next_state

                # Update the current state
                current_state = next_state
            
            xtrj_ext_fwd_i = np.vstack((xtrj_ext_padding_fwd_i, xtrj_ext_fwd_i))
            
            # ----------------------------------
            #   Simulate the backward extension
            # ----------------------------------
            xtrj_ext_bwd_i[0] = x_reset_i
            current_state = x_reset_i
            for jj in range(nt_ext_bwd-1):
                t_jj = timespan_ext_bwd[jj]
                
                # modify if needed
                current_input = np.zeros(self._n_inputs[next_mode_i])
                
                next_state, _, _, _, _, _, _ = self.detection_func_(current_state, current_input, t_jj, t_jj-self.dt_, next_mode_i, detection=False)
                
                # Store states and inputs
                xtrj_ext_bwd_i[jj+1] = next_state

                # Update the current state
                current_state = next_state
            
            # -------------------------------
            # reverse the backward extension 
            # -------------------------------   
            xtrj_ext_bwd_i = xtrj_ext_bwd_i[::-1]
            
            # --------------------- padding ---------------------
            xtrj_ext_bwd_i = np.vstack((xtrj_ext_bwd_i, xtrj_ext_padding_bwd_i))
            
            # ------------------------ collect the trajectory extensions ------------------------
            ref_ext_helper.append((np.array([current_mode_i, next_mode_i]), 
                                      {current_mode_i:xtrj_ext_fwd_i, next_mode_i:xtrj_ext_bwd_i}, 
                                      {current_mode_i:K_feedback_ext_fwd_i, next_mode_i:K_feedback_ext_bwd_i}, 
                                      {current_mode_i:k_feedforward_ext_fwd_i, next_mode_i:k_feedforward_ext_bwd_i}, 
                                      i_event_i))
            
        return ref_ext_helper
    
    
    def solve(self):
        # ------ collect the iteration data ------
        states_iter = []
        
        # ------------------------------------
        # First rollout using initial guess
        # ------------------------------------
        [modes,states,inputs,saltations,modechanges] = self.rollout()
        

        print("===================== Finished initial rollout =====================")
        
        show_rollout = False
        r0 = 1
        if show_rollout:
            pass
            
        # ----------------------------------------------------
        # Compute the current cost of the initial trajectory
        # ----------------------------------------------------
        current_cost = self.compute_cost(states,inputs,self.dt_)
        
        learning_speed = 0.9 # This can be modified, 0.95 is very slow
        low_learning_rate = 0.01 # if learning rate drops to this value stop the optimization
        low_expected_reduction = 1e-4 # Determines optimality
        armijo_threshold = 0.1 # Determines if current line search solve is good (this is typically labeled as "c")
        
        # =============
        #   Main Loop
        # =============
        for ii in range(0,self._niters):
            print('========== Starting Iteration: ',ii,', Current cost: ',current_cost, ' ==========')
            print("-------- Backward Pass --------")
                
            # --------------------------------------------------------
            # Compute the backwards pass and update the control gains
            # --------------------------------------------------------
            (k_feedforward,K_feedback,expected_reduction) = self.backwards_pass()    
            
            # --------------------------------------------------------------
            # Compute the new trajectory extensions and the gains for them
            # --------------------------------------------------------------
            self._refext_helper = self.compute_trejactory_extension(self._event_info)
            
            print('-------- Expected cost reduction: ',expected_reduction, ' --------')
            
            if(abs(expected_reduction)<low_expected_reduction):
                print(" -------- Stopping optimization, Optimal trajectory found --------")
                break
            learning_rate = 1
            armijo_flag = 0
            
            # ---------------------------------------------
            # Forward pass under the updated control gains
            # ---------------------------------------------
            (new_modes,new_states,new_inputs,
             new_saltations,mode_changes,new_hybrid_event_info)=self.forward_pass(learning_rate)
            
            # ---------------------------------------------------------
            # Compute new costs and check the optimality conditions
            # ---------------------------------------------------------
            new_cost = self.compute_cost(new_states, new_inputs, self.dt_)
            
            # Execute linesearch until the armijo condition is met (for
            # now just check if the cost decreased) TODO add real
            # armijo condition
            while(learning_rate > 0.05 and armijo_flag == 0):
                # Decrease learning rate and continue line search
                learning_rate = learning_speed*learning_rate
                
                # Forward pass: line search 
                (new_modes,new_states,new_inputs,
                 new_saltations,mode_changes,new_hybrid_event_info)=self.forward_pass(learning_rate)
                
                show_forwardpass = False
                if show_forwardpass:
                    pass
            
                new_cost = self.compute_cost(new_states, new_inputs, self.dt_)
                
                print("new_cost: ", new_cost)
                
                # Calculate armijo condition
                cost_difference = (current_cost - new_cost)
                
                expected_cost_redu = learning_rate*self.expt_cost_redu_grad_ + learning_rate*learning_rate*self.expt_cost_redu_hess_
                armijo_flag = cost_difference/expected_cost_redu > armijo_threshold
                
                if(armijo_flag == 1):
                    # ------------------------------------------------------
                    # Accept the new trajectory if armijo condition is met
                    # ------------------------------------------------------
                    current_cost = new_cost
                    self._states = new_states
                    self._inputs = new_inputs
                    self._saltations = new_saltations
                    self._modechanges = mode_changes
                    self._modes = new_modes
                    self._event_info = new_hybrid_event_info
                    self._refext_helper = self.compute_trejactory_extension(new_hybrid_event_info)
                    states_iter.append(new_states)
                    
            if(learning_rate<low_learning_rate):
                # If learning rate is low, then stop optimization
                
                print(" -------- Stopping optimization, low learning rate --------")
                
                current_cost = new_cost
                self._states = new_states
                self._inputs = new_inputs
                self._saltations = new_saltations
                self._modechanges = mode_changes
                self._modes = new_modes
                self._event_info = new_hybrid_event_info
                
                # Update the hybrid event maps
                self._refext_helper = self.compute_trejactory_extension(new_hybrid_event_info)
                
                states_iter.append(new_states)
                    
                break
          
        
        print(" -------- Stopping optimization, reached max iteration --------")
          
        # Return the current trajectory
        modes = self._modes
        states = self._states
        inputs = self._inputs
        modechanges = self._modechanges
        hybrid_event_info = self._event_info
        show_results = False
        if show_results:
            pass

        ref_ext_helper = self.compute_trejactory_extension(hybrid_event_info)

        return (modes,states,inputs,
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
    
    #   Solve for hybrid ilqr proposal
    # ====================================
    #     
    initial_guess = [0.5*np.ones((np.shape(time_span)[0],n_inputs[0])), 0.5*np.ones((np.shape(time_span)[0],n_inputs[1]))]        

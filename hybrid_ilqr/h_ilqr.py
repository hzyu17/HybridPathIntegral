import numpy as np
import matplotlib.pyplot as plt
from dynamics.dynamics_slip import *

# Helper function to handle reference trajectory extensions
def extract_extensions(mode_exttrjs_maps):
    # ---------------------------------------------------
    #            Extract the extended references 
    # ---------------------------------------------------
    num_events = len(mode_exttrjs_maps)
    
    if num_events == 0: # reference has no hybrid events
        return None, None, None
    
    v_mode_change = []
    v_ext_trj_fwd = []
    v_ext_trj_bwd = []
    v_Kfb_ext_trj_fwd = []
    v_kff_ext_trj_fwd = []
    v_Kfb_ext_trj_bwd = []
    v_kff_ext_trj_bwd = []
    v_tevents = []
    
    for i_event in range(num_events):
        # find out the mode changes
        MC_i = mode_exttrjs_maps[i_event][0]
        cur_mode_i = MC_i[0]
        next_mode_i = MC_i[1]
        
        v_mode_change.append((cur_mode_i, next_mode_i))
        
        # Add the forward and backward extensions to the collection
        MC_EXTTRJ_MAP = mode_exttrjs_maps[i_event][1]
        v_ext_trj_fwd.append(MC_EXTTRJ_MAP[cur_mode_i])
        v_ext_trj_bwd.append(MC_EXTTRJ_MAP[next_mode_i])
        
        # Add the feedback gain for forward and backward extensions to the collection
        MC_FEEDBACK_EXTTRJ_MAP = mode_exttrjs_maps[i_event][2]
        v_Kfb_ext_trj_fwd.append(MC_FEEDBACK_EXTTRJ_MAP[cur_mode_i])
        v_Kfb_ext_trj_bwd.append(MC_FEEDBACK_EXTTRJ_MAP[next_mode_i])
        
        # Add the feedforward gain for backward and backward extensions to the collection
        MC_FEEDFWD_EXTTRJ_MAP = mode_exttrjs_maps[i_event][3]
        v_kff_ext_trj_fwd.append(MC_FEEDFWD_EXTTRJ_MAP[cur_mode_i])
        v_kff_ext_trj_bwd.append(MC_FEEDFWD_EXTTRJ_MAP[next_mode_i])
        
        tevent_i = mode_exttrjs_maps[i_event][4]
        v_tevents.append(tevent_i)
        
    return v_mode_change, v_ext_trj_bwd, v_ext_trj_fwd, v_Kfb_ext_trj_bwd, v_Kfb_ext_trj_fwd, v_kff_ext_trj_bwd, v_kff_ext_trj_fwd, v_tevents


class hybrid_ilqr:
    def __init__(self, nmodes, init_mode, target_mode, nstates,
                 init_state,target_state,initial_guess,
                 dt,start_time,end_time,
                 contact_detect,smooth_dynamics,
                 Q_k,R_k,Q_T,parameters,n_iterations,
                 detect,plot_func,state_convert_func, 
                 init_reset_args, target_reset_args):
        
        self.nmodes_ = nmodes
        self.n_states_ = nstates
        self._init_mode = init_mode
        self._target_mode = target_mode
        self._init_state = init_state
        self._target_state = target_state
        self.mode_inputs_ = initial_guess
        self.inputs_ = initial_guess
        self.n_inputs_ = [np.shape(initial_guess[i_mode])[1] for i_mode in range(nmodes)]
        self.plot_states_func_ = plot_func
        self.state_convert_func_ = state_convert_func
        
        # time definitions
        self.dt_ = dt
        self.start_time_ = start_time
        self.end_time_ = end_time
        self.time_span_ = np.arange(start_time, end_time, dt).flatten()
        self.n_timesteps_ = np.shape(self.time_span_)[0]
        
        # feedback and feedforward
        self.k_feedforward_ = [np.zeros((self.n_inputs_[0])) for _ in range(self.n_timesteps_)]
        self.K_feedback_ = [np.zeros((self.n_inputs_[0], self.n_states_[0])) for _ in range(self.n_timesteps_)]
        
        # hybrid events and mode changes
        self.saltations_ = [np.array([None]) for i in range(self.n_timesteps_)]
        self.modechanges_ = [np.array([0, 0]) for _ in range(self.n_timesteps_)]
        self.modes_ = [0 for _ in range(self.n_timesteps_)]
        self._reset_args = [0 for _ in range(self.n_timesteps_)]
        self._reset_args[0] = init_reset_args
        self._target_reset_args = target_reset_args
        self._init_reset_args = init_reset_args
        
        # feedback and feedforward for the trajectory extensions 
        # shapes: feedback gains: [[K_fwd_extension, K_bwd_extension]_jump1, [K_fwd_extension, K_bwd_extension]_jump2, ...]
        # shapes: feedforwad gains: [[k_fwd_extension, k_bwd_extension]_jump1, [k_fwd_extension, k_bwd_extension]_jump2, ...]
        self.K_feedforward_extensions_ = [[None], [None]]
        self.K_feedback_extensions_ = [[None], [None]]
        
        # ------------------------------------------------------------------------------------------------------------------ 
        # Map that maps the index at hybrid event to the event informations (t_event, x_event, x_reset, mode_change). 
        # ------------------------------------------------------------------------------------------------------------------ 
        self.txmode_map_ = {}
        self.mode_exttrjs_maps_ = []
        
        # Dynamics
        self.smooth_dyn_ = smooth_dynamics
        self.detection_func_ = contact_detect
        
        self.f_, self.A_, self.B_ = [None for _ in range(self.nmodes_)],[None for _ in range(self.nmodes_)],[None for _ in range(self.nmodes_)]
        for ii in range(self.nmodes_):
            self.f_[ii], self.A_[ii], self.B_[ii] = smooth_dynamics[ii]()
        
        self.detection_func_ = contact_detect
        
        # Weighting
        self.Q_k_ = Q_k
        self.R_k_ = R_k
        self.Q_T_ = Q_T
        self.parameters_ = parameters

        # Max iterations
        self.n_iterations_ = n_iterations
        
        # Flag detection
        self.detect_ = detect

    def rollout(self):        
        # Collection of the rollout results
        states = [np.array([0.0]) for _ in range(self.n_timesteps_)]
        inputs = [np.zeros((self.n_timesteps_, self.n_inputs_[0])), np.zeros((self.n_timesteps_, self.n_inputs_[1]))]
        modes = [0 for _ in range(self.n_timesteps_)]
        mode_changes = [np.array([0, 0]) for _ in range(self.n_timesteps_)]
        saltations = [None for i in range(self.n_timesteps_)]
        reset_args = [(np.array([0.0]), ) for i in range(self.n_timesteps_)]
        reset_args[0] = self._init_reset_args
        
        # Initial states
        current_state = self._init_state
        current_mode = self._init_mode
        states[0] = current_state
        modes[0] = current_mode
        mode_changes[0] = np.array([current_mode, current_mode])
        
        # extend reference trj, if a hybrid event is hit.
        hybrid_index = set()
        txmode_map = {}
        
        cnt_event = 0
        event_args = [self._init_reset_args]
        
        for ii in range(self.n_timesteps_-1):
            
            current_input = self.mode_inputs_[current_mode][ii]
            t_ii = self.time_span_[ii]
            
            next_state, saltation, mode_change, t_event, x_event, x_reset, reset_byproduct = self.detection_func_(current_state, current_input, t_ii, t_ii+self.dt_, 
                                                                                                                  current_mode, event_args[cnt_event], self.detect_)
            
            saltations[ii] = saltation
            reset_args[ii] = event_args[cnt_event]
            next_state = next_state.flatten()
            
            # If a contact happened: compute the extension
            if saltation is not None:
                print("A contact happened in the forward pass.")
                # Collect the tevent and xevent for the hybrid events
                hybrid_index.add(ii)
                txmode_map[ii] = (t_event, x_event, x_reset, mode_change, self.K_feedback_extensions_, self.K_feedforward_extensions_)
            
            if (mode_change[0]!=mode_change[1]):
                cnt_event += 1
                event_args.append(reset_byproduct)
                    
            # Store states and inputs
            states[ii + 1] = next_state
            inputs[current_mode][ii] = current_input # in case we have a control law, we store the input used
            
            mode_changes[ii+1] = mode_change
            next_mode = mode_change[1]
            modes[ii+1] = next_mode
            
            # Update the current state
            current_state = next_state
            current_mode = next_mode

        # Store the trajectory(states, inputs)
        self.states_ = states
        self.inputs_ = inputs
        self.saltations_ = saltations
        self.modechages_ = mode_changes
        self.modes_ = modes
        self.txmode_map_ = txmode_map
        self._reset_args = reset_args
        self.mode_exttrjs_maps_ = self.compute_trejactory_extension(txmode_map)
        
        return states, inputs, saltations, mode_changes, modes

    def compute_cost(self,modes,states,inputs,dt):
        # Initialize cost
        total_cost = 0.0
        for ii in range(0,self.n_timesteps_-1):
            current_mode = modes[ii]
            current_x = states[ii] # Not being used currently
            current_u = inputs[current_mode][ii].flatten()

            current_cost = 0.5*current_u.T@self.R_k_[current_mode]@current_u # Right now only considering cost in input
            total_cost = total_cost+current_cost*dt
            
        # Compute terminal cost
        terminal_state = states[-1]
        if modes[-1] != self._target_mode:
            terminal_state = self.state_convert_func_(states[-1]).flatten()
        terminal_difference = (self._target_state - terminal_state).flatten()
        terminal_cost = 0.5*terminal_difference.T@self.Q_T_@terminal_difference
        total_cost = total_cost+terminal_cost
        return total_cost

    def backwards_pass(self):
        V_xx = self.Q_T_
        
        end_difference = (self.states_[-1] - self._target_state).flatten()
        V_x = self.Q_T_@end_difference

        # # Initialize storage variables
        # k_trj = np.zeros((self.n_timesteps_,self.n_inputs_))
        # K_trj = np.zeros((self.n_timesteps_,self.n_inputs_,self.n_states_))
        
        k_trj = [np.zeros((self.n_inputs_[0])) for _ in range(self.n_timesteps_)]
        K_trj = [np.zeros((self.n_inputs_[0], self.n_states_[0])) for _ in range(self.n_timesteps_)]
        
        k_feedforward_trj_extension = []
        K_feedback_trj_extension = []
        
        # Initialize cost reduction
        expected_cost_reduction = 0
        expected_cost_reduction_grad = 0
        expected_cost_reduction_hess = 0

        # for loop backwards in time
        for idx in reversed(range(0, self.n_timesteps_-1)):
            # Grab the current variables in the trajectory
            current_mode = self.modes_[idx]
            current_x = self.states_[idx]
            current_u = self.inputs_[current_mode][idx]
            saltation = self.saltations_[idx]

            # R_k_updated
            # Define the expansion coefficients and the loss gradients
            l_xx = self.Q_k_[current_mode] # For now zeros, can add in a target to track later on
            l_uu = self.R_k_[current_mode]

            l_x = self.Q_k_[current_mode]@np.zeros(self.n_states_[current_mode]).flatten() # For now zeros, can add in a target to track later on
            # print("current_mode: ", current_mode)
            # print("current_u: ", current_u)
            # print("self.R_k_[current_mode]: ", self.R_k_[current_mode])
            
            l_u = self.R_k_[current_mode]@(current_u).flatten()

            # Get the jacobian of the discretized dynamics
            A_k = self.A_[current_mode](current_x, current_u, self.dt_)
            B_k = self.B_[current_mode](current_x, current_u, self.dt_)
            
            if saltation is None:
                Q_x = l_x*self.dt_ + A_k.T@V_x
                Q_u = l_u*self.dt_+ B_k.T@V_x
                Q_ux = B_k.T@V_xx@A_k
                Q_uu = l_uu*self.dt_ + B_k.T@V_xx@B_k
                Q_xx = l_xx*self.dt_ + A_k.T@V_xx@A_k
                
                # Compute gains           
                k = -np.linalg.solve(Q_uu, Q_u)
                K = -np.linalg.solve(Q_uu, Q_ux)

            else:
                # print("Found contact dynamics! Computing the gains with saltation matrix.")
                Q_x = l_x*self.dt_ + A_k.T @ saltation.T @ V_x
                Q_u = l_u*self.dt_ + B_k.T @ saltation.T @ V_x
                Q_ux = B_k.T @ saltation.T @ V_xx @ saltation @ A_k
                Q_uu = l_uu*self.dt_ + B_k.T @ saltation.T @ V_xx @ saltation @ B_k
                Q_xx = l_xx*self.dt_ + A_k.T @ saltation.T @ V_xx @ saltation @ A_k    
                
                # Compute gains           
                k = -np.linalg.solve(Q_uu, Q_u)
                K = -np.linalg.solve(Q_uu, Q_ux)

                # Compute the (gains for the forward extension, gains for the backward extension): use the gain at the immediate next state (reseted)
                k_feedforward_trj_extension.append((k, k_trj[idx+1]))
                K_feedback_trj_extension.append((K, K_trj[idx+1]))
                
                # update the hybrid dynamics information
                l_txmode_map_ = list(self.txmode_map_[idx])
                l_txmode_map_[4] = [K, K_trj[idx+1]]
                l_txmode_map_[5] = [k, k_trj[idx+1]]
                self.txmode_map_[idx] = tuple(l_txmode_map_)

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
        self.expected_cost_reduction_grad_ = expected_cost_reduction_grad
        self.expected_cost_reduction_hess_ = expected_cost_reduction_hess
        self.expected_cost_reduction_ = expected_cost_reduction

        # Store gain schedule
        self.k_feedforward_ = k_trj
        self.K_feedback_ = K_trj
        
        # store the gain for the backward extensions
        K_feedback_trj_extension.reverse()
        k_feedforward_trj_extension.reverse()
        self.K_feedback_extensions_ = K_feedback_trj_extension
        self.K_feedforward_extensions_ = k_feedforward_trj_extension
        
        return (k_trj,K_trj,expected_cost_reduction)

    def forward_pass(self, learning_rate):
        
        print(f"---------- forward pass learning rate: {learning_rate} ----------")
        
        # Lists to collect the current forward pass trajectories (Dimensions might vary so use list)
        states = [np.array([0.0]) for _ in range(self.n_timesteps_)]
        inputs = [np.zeros((self.n_timesteps_, self.n_inputs_[0])), np.zeros((self.n_timesteps_, self.n_inputs_[1]))]
        modes = [0 for _ in range(self.n_timesteps_)]
        saltations = [None for i in range(self.n_timesteps_)]
        mode_changess = np.tile(np.array([0, 0]), (self.n_timesteps_, 1))
        reset_args = [(None, ) for i in range(self.n_timesteps_)]
        reset_args[0] = self._init_reset_args
        
        # Set the first state to be the initial
        current_state = self._init_state
        current_mode = self._init_mode
        
        modes[0] = self._init_mode
        states[0] = current_state
        mode_changess[0] = np.array([current_mode, current_mode])
        
        # Extend reference trj, if a hybrid event is hit.
        hybrid_index = set()
        txmode_map = {} # The dictionary that stores all the information of the jump dynamics and states.
        
        # Reference hybrid events and extensions from the last iteration
        (v_mode_change_ref, v_ext_trj_bwd_ref, v_ext_trj_fwd_ref, 
         v_Kfb_ext_trj_bwd_ref, v_Kfb_ext_trj_fwd_ref, 
         v_kff_ext_trj_bwd_ref, v_kff_ext_trj_fwd_ref, v_tevents_ref) = extract_extensions(self.mode_exttrjs_maps_)
        
        print(f"Reference trajectory bouncing event numbers: {len(v_ext_trj_bwd_ref)}")
        for i_bounce in range(len(v_ext_trj_bwd_ref)):
            print(f"bounce {i_bounce}: From mode {v_mode_change_ref[i_bounce][0]} to mode {v_mode_change_ref[i_bounce][1]} at time {v_tevents_ref[i_bounce]}")
        print("------------------------------------------------")
        cnt_event = 0
        hybrid_index_ref = 0
        event_args = [self._init_reset_args]
        
        for ii in range(self.n_timesteps_-1):
            
            current_state = states[ii]
            current_mode = modes[ii]
            reset_args[ii] = event_args[cnt_event]
            
            # ------------------- 
            # Get the references 
            # ------------------- 
            ref_state = self.states_[ii]
            current_nominal_input = self.inputs_[current_mode][ii]
            current_mode_ref = self.modes_[ii]
            
            # ----------------------------------------------- 
            # Get the current (feedback, feedforward) gains 
            # ----------------------------------------------- 
            current_feedforward = learning_rate * self.k_feedforward_[ii]
            current_feedback = self.K_feedback_[ii]
            
            # ---------------
            # Mode Mismatch
            # --------------- 
            if (current_mode != current_mode_ref):
                
                trj_extension = []
                fb_ext_trj = []
                ff_ext_trj = []
                
                hybrid_index_ref = np.argmin(abs(np.array(v_tevents_ref)-ii)) # find the nearest hybrid event in the reference
                ref_modechange_hybrid = v_mode_change_ref[hybrid_index_ref]
                
                # ----------------------------------- Early Arrival ----------------------------------- 
                if ((current_mode == ref_modechange_hybrid[1]) and (current_mode_ref==ref_modechange_hybrid[0])):
                    
                    print(f"early arrival, Time: {ii}. Current mode: {current_mode}, Reference mode: {current_mode_ref}")
                    print(f"Reference mode change from mode {ref_modechange_hybrid[0]} to mode {ref_modechange_hybrid[1]} at time {v_tevents_ref[hybrid_index_ref]}")
                    
                    trj_extension = v_ext_trj_bwd_ref[hybrid_index_ref]
                    
                    ff_ext_trj = v_kff_ext_trj_bwd_ref[hybrid_index_ref]
                    fb_ext_trj = v_Kfb_ext_trj_bwd_ref[hybrid_index_ref]
                    
                # ----------------------------------- Late Arrival ----------------------------------- 
                elif ((current_mode == ref_modechange_hybrid[0]) and (current_mode_ref==ref_modechange_hybrid[1])):
                    
                    print(f"late arrival, Time: {ii}. Current mode: {current_mode}, Reference mode: {current_mode_ref}")
                    print(f"Reference mode change from mode {ref_modechange_hybrid[0]} to mode {ref_modechange_hybrid[1]} at time {v_tevents_ref[hybrid_index_ref]}")
                    
                    trj_extension = v_ext_trj_fwd_ref[hybrid_index_ref]
                    
                    ff_ext_trj = v_kff_ext_trj_fwd_ref[hybrid_index_ref]
                    fb_ext_trj = v_Kfb_ext_trj_fwd_ref[hybrid_index_ref]
                
                # Modify the reference to the extension
                ref_state = trj_extension[ii]
                current_feedback = fb_ext_trj[ii]
                current_feedforward = learning_rate * ff_ext_trj[ii]
                
                print("current_nominal_input: ", current_nominal_input)
            
            current_feedback_input = current_feedback@(current_state-ref_state)
            current_input = current_nominal_input + current_feedback_input + current_feedforward
            
            # simulate forward
            t_ii = self.time_span_[ii]
            
            (next_state, saltation, mode_change, 
             t_event, x_event, x_reset, reset_byproduct) = self.detection_func_(current_state, current_input, t_ii, t_ii+self.dt_, 
                                                                                current_mode, reset_args[ii], self.detect_)
            
            # ------------------------------
            # Update the hybrid information
            # ------------------------------
            if saltation is not None:
                hybrid_index.add(ii)
                saltations[ii] = saltation
                txmode_map[ii] = (t_event, x_event, x_reset, mode_change, self.K_feedback_extensions_[hybrid_index_ref], self.K_feedforward_extensions_[hybrid_index_ref])
                
            # ---------------------
            # Move forward in time
            # ---------------------
            states[ii+1] = next_state
            inputs[current_mode][ii] = current_input.flatten()
            mode_changess[ii+1] = mode_change
            modes[ii+1] = mode_change[1]
            
            # Only consider the transition from mode 0 to mode 1 for now
            if (mode_change[0]!=mode_change[1]):
                print(f"At Time {ii}, the system has a mode change from mode {mode_change[0]} to mode {mode_change[1]}")
                event_args.append(reset_byproduct)
                cnt_event += 1
                
        print(f"--------------------- Total number of contacts: {cnt_event} ---------------------" )
        
        return (modes,states,inputs,current_feedback,current_feedforward,saltations,mode_changess,txmode_map,reset_args)
    
    
    def compute_trejactory_extension(self, txmode_map):
        
        # NO hybrid events
        if len(txmode_map.keys()) == 0:
            return []
        
        # txmode_map:
        sorted_hybrid_index = sorted(txmode_map.keys())
        mode_exttrjs_maps = []
        
        i_events = []
        t_events = []
        x_events = []
        x_resets = []
        K_feedback_fwd_extensions = []
        k_feedforward_fwd_extensions = []
        K_feedback_bwd_extensions = []
        k_feedforward_bwd_extensions = []
        
        mode_changes = []
        
        i_events.append(0)
        t_events.append(self.start_time_)
        x_events.append(self._init_state)
        x_resets.append(self._init_state)
        K_feedback_fwd_extensions.append(np.zeros(1))
        K_feedback_bwd_extensions.append(np.zeros(1))
        k_feedforward_fwd_extensions.append(np.zeros(1))
        k_feedforward_bwd_extensions.append(np.zeros(1))
        mode_changes.append(np.array([0, 0]))
        
        # txmode_map[i_key] = (t_event, x_event, x_reset, mode_change, K_feedback_extensions, K_feedforward_extensions)
        for i_key in sorted_hybrid_index:
            i_events.append(i_key)
            t_events.append(txmode_map[i_key][0])
            x_events.append(txmode_map[i_key][1])
            x_resets.append(txmode_map[i_key][2])
            mode_changes.append(txmode_map[i_key][3])
            
            K_feedback_fwd_extensions.append(txmode_map[i_key][4][0])
            K_feedback_bwd_extensions.append(txmode_map[i_key][4][1])
            
            k_feedforward_fwd_extensions.append(txmode_map[i_key][5][0])
            k_feedforward_bwd_extensions.append(txmode_map[i_key][5][1])
        
        i_events.append(self.n_timesteps_)
        t_events.append(self.end_time_)
        x_events.append(self._target_state)
        x_resets.append(self._target_state)
        K_feedback_fwd_extensions.append(np.zeros(1))
        K_feedback_bwd_extensions.append(np.zeros(1))
        k_feedforward_fwd_extensions.append(np.zeros(1))
        k_feedforward_bwd_extensions.append(np.zeros(1))
            
        if txmode_map.keys():
            mode_changes.append(txmode_map[sorted_hybrid_index[-1]][3])
        
        # Forward and backward trajectory extensions and (feedback, feedforward) gains for the two extensions
        for ii, tevent_i in enumerate(t_events[1:-1], start=1):
            i_event_i = i_events[ii]
            x_event_i = x_events[ii]
            x_reset_i = x_resets[ii]
            current_mode_i = mode_changes[ii][0]
            next_mode_i = mode_changes[ii][1]
            
            K_feedback_fwd_extension_i = K_feedback_fwd_extensions[ii]
            k_feedforward_fwd_extension_i = k_feedforward_fwd_extensions[ii]
            
            K_feedback_bwd_extension_i = K_feedback_bwd_extensions[ii]
            k_feedforward_bwd_extension_i = k_feedforward_bwd_extensions[ii]
            
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
            
            xtrj_ext_padding_fwd_i = np.zeros((nt_ext_padding_fwd, self.n_states_[current_mode_i]))
            xtrj_ext_fwd_i = np.zeros((nt_ext_fwd, self.n_states_[current_mode_i]))
            xtrj_ext_fwd_i[0] = x_event_i
            
            xtrj_ext_bwd_i = np.zeros((nt_ext_bwd, self.n_states_[next_mode_i]))
            xtrj_ext_padding_bwd_i = np.zeros((nt_ext_padding_bwd, self.n_states_[next_mode_i]))    
            
            # Use the same gain for the whole extensions, and padding it to the whole time span
            K_feedback_ext_fwd_i = np.tile(K_feedback_fwd_extension_i, (self.n_timesteps_, 1))
            k_feedforward_ext_fwd_i = np.tile(k_feedforward_fwd_extension_i, (self.n_timesteps_, 1))
            
            K_feedback_ext_bwd_i = np.tile(K_feedback_bwd_extension_i, (self.n_timesteps_, 1))
            k_feedforward_ext_bwd_i = np.tile(k_feedforward_bwd_extension_i, (self.n_timesteps_, 1))
            
            # ----------------------------------
            # simulate the forward extension
            # ----------------------------------
            current_state = x_event_i
            for jj in range(nt_ext_fwd-1):
                t_jj = timespan_ext_fwd[jj]
                
                # Using zero control, modify if needed.
                current_input = np.zeros(self.n_inputs_[current_mode_i])
                
                next_state, _, _, _, _, _, _ = self.detection_func_(current_state, current_input, t_jj, t_jj+self.dt_, current_mode_i, self._reset_args[jj], detection=False)
                
                # Store states and inputs
                xtrj_ext_fwd_i[jj+1] = next_state

                # Update the current state
                current_state = next_state
            
            xtrj_ext_fwd_i = np.vstack((xtrj_ext_padding_fwd_i, xtrj_ext_fwd_i))
            
            # ----------------------------------
            # simulate the backward extension
            # ----------------------------------
            xtrj_ext_bwd_i[0] = x_reset_i
            current_state = x_reset_i
            for jj in range(nt_ext_bwd-1):
                t_jj = timespan_ext_bwd[jj]
                
                # modify if needed
                current_input = np.zeros(self.n_inputs_[next_mode_i])
                
                next_state, _, _, _, _, _, _ = self.detection_func_(current_state, current_input, t_jj, t_jj-self.dt_, next_mode_i, self._reset_args[jj], detection=False)
                
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
            mode_exttrjs_maps.append((np.array([current_mode_i, next_mode_i]), 
                                      {current_mode_i:xtrj_ext_fwd_i, next_mode_i:xtrj_ext_bwd_i}, 
                                      {current_mode_i:K_feedback_ext_fwd_i, next_mode_i:K_feedback_ext_bwd_i}, 
                                      {current_mode_i:k_feedforward_ext_fwd_i, next_mode_i:k_feedforward_ext_bwd_i}, 
                                      i_event_i))
            
        return mode_exttrjs_maps
    
    
    def solve(self):
        # ------ collect the iteration data ------
        states_iter = []
        # Compute the rollout to get the initial trajectory with the
        # ------------------------------------
        # First rollout using initial guess
        # ------------------------------------
        [states,inputs,saltations,modechanges,modes] = self.rollout()
        
        print("-------------------- Finished first rollout --------------------")
        
        show_rollout = True
        r0 = 1
        if show_rollout:
            self.plot_states_func_(self.time_span_, modes, states, inputs, 
                                    self._init_state, self._target_state, self.n_timesteps_, self._reset_args)
            
            
            fig, ax = plt.subplots()
            ax.grid(True)
            for ii in range(self.n_timesteps_):
                if modes[ii] == 0:
                    plot_slip_flight_animate(states[ii].flatten(), r0, ax)
                elif modes[ii] == 1:
                    plot_slip_stance_animate(states[ii].flatten(), self._reset_args[ii][0], ax)
            
            # Plot start and goal 
        
            if self._init_mode == 0:
                plot_slip_flight_animate(self._init_state, r0, ax, 'r-')
            elif self._init_mode == 1:
                plot_slip_stance_animate(self._init_state, self._reset_args[0], ax, 'r-')
                
                
            if self._target_mode == 0:
                plot_slip_flight_animate(self._target_state, r0, ax, 'g-')
            elif self._target_mode == 1:
                plot_slip_stance_animate(self._target_state, self._target_reset_args, ax, 'g-')
            
            plt.show()
        
        # ----------------------------------------------------
        # Compute the current cost of the initial trajectory
        # ----------------------------------------------------
        current_cost = self.compute_cost(modes,states,inputs,self.dt_)
        
        learning_speed = 0.9 # This can be modified, 0.95 is very slow
        low_learning_rate = 0.01 # if learning rate drops to this value stop the optimization
        low_expected_reduction = 1e-4 # Determines optimality
        armijo_threshold = 0.1 # Determines if current line search solve is good (this is typically labeled as "c")
        
        # ===========
        #  Main Loop
        # ===========
        for ii in range(0,self.n_iterations_):
            print('========== Starting Iteration: ',ii,', Current cost: ',current_cost, ' ==========')
            # --------------------------------------------------------
            # Compute the backwards pass and update the control gains
            # --------------------------------------------------------
            print("-------- Backward Pass --------")
            (k_feedforward,K_feedback,expected_reduction) = self.backwards_pass()    
            
            # --------------------------------------------------------------
            # Compute the new trajectory extensions and the gains for them
            # --------------------------------------------------------------
            self.mode_exttrjs_maps_ = self.compute_trejactory_extension(self.txmode_map_)
            
            print('-------- Expected cost reduction: ',expected_reduction, ' --------')
            
            if(abs(expected_reduction)<low_expected_reduction):
                # If the expected reduction is low, then end the optimization
                print(" -------- Stopping optimization, Optimal trajectory found --------")
                break
            learning_rate = 1
            armijo_flag = 0
            
            # ---------------------------------------------
            # Forward pass under the updated control gains
            # ---------------------------------------------
            (new_modes,new_states,new_inputs,
             new_feedback,new_feedforward,
             new_saltations,mode_changes,new_txmode_map,new_stance_xpos)=self.forward_pass(learning_rate)
            
            # ---------------------------------------------------------
            # Compute new costs and check the optimality conditions
            # ---------------------------------------------------------
            new_cost = self.compute_cost(new_modes, new_states, new_inputs, self.dt_)
            
            # Execute linesearch until the armijo condition is met (for
            # now just check if the cost decreased) TODO add real
            # armijo condition
            while(learning_rate > 0.05 and armijo_flag == 0):
                # Decrease learning rate and continue line search
                learning_rate = learning_speed*learning_rate
                
                # Forward pass: line search 
                (new_modes,new_states,new_inputs,
                 new_feedback, new_feedforward,
                 new_saltations,mode_changes,new_txmode_map,new_reset_args)=self.forward_pass(learning_rate)
                
                
                show_forwardpass = False
                if show_forwardpass:
                    self.plot_states_func_(self.time_span_, new_modes, new_states, new_inputs, 
                                           self._init_state, self._target_state, self.n_timesteps_, new_reset_args)

                    fig, ax = plt.subplots()
                    ax.grid(True)
                    for ii in range(self.n_timesteps_-1):
                        if new_modes[ii] == 0:
                            plot_slip_flight_animate(new_states[ii].flatten(), r0, ax)
                        elif new_modes[ii] == 1:
                            plot_slip_stance_animate(new_states[ii].flatten(), new_reset_args[ii][0], ax)
                    
                    # Plot start and goal 
                
                    if self._init_mode == 0:
                        plot_slip_flight_animate(self._init_state, r0, ax, 'r-')
                    elif self._init_mode == 1:
                        plot_slip_stance_animate(self._init_state, self._reset_args[0], ax, 'r-')
                        
                        
                    if self._target_mode == 0:
                        plot_slip_flight_animate(self._target_state, r0, ax, 'g-')
                    elif self._target_mode == 1:
                        plot_slip_stance_animate(self._target_state, self._target_reset_args, ax, 'g-')
                    
                    plt.show()
            
                new_cost = self.compute_cost(new_modes, new_states, new_inputs, self.dt_)

                print("new_cost: ", new_cost)
                
                # Calculate armijo condition
                cost_difference = (current_cost - new_cost)
                
                expected_cost_redu = learning_rate*self.expected_cost_reduction_grad_ + learning_rate*learning_rate*self.expected_cost_reduction_hess_
                armijo_flag = cost_difference/expected_cost_redu > armijo_threshold
                
                if(armijo_flag == 1):
                    # ------------------------------------------------------
                    # Accept the new trajectory if armijo condition is met
                    # ------------------------------------------------------
                    current_cost = new_cost
                    self.states_ = new_states
                    self.inputs_ = new_inputs
                    self.saltations_ = new_saltations
                    self.modechanges_ = mode_changes
                    self.modes_ = new_modes
                    self.txmode_map_ = new_txmode_map
                    self.mode_exttrjs_maps_ = self.compute_trejactory_extension(new_txmode_map)
                    self._reset_args = new_reset_args
                    states_iter.append(new_states)
                    
            if(learning_rate<low_learning_rate):
                # If learning rate is low, then stop optimization
                print(" -------- Stopping optimization, low learning rate --------")
                current_cost = new_cost
                self.states_ = new_states
                self.inputs_ = new_inputs
                self.saltations_ = new_saltations
                self.modechanges_ = mode_changes
                self.modes_ = new_modes
                self.txmode_map_ = new_txmode_map
                self._reset_args = new_reset_args
                
                # Update the hybrid event maps
                self.mode_exttrjs_maps_ = self.compute_trejactory_extension(new_txmode_map)
                
                states_iter.append(new_states)
                    
                break
            
        # Return the current trajectory
        modes = self.modes_
        states = self.states_
        inputs = self.inputs_
        modechanges = self.modechanges_
        txmode_map = self.txmode_map_
        reset_args = self._reset_args
        show_results = False
        if show_results:
            self.plot_states_func_(self.time_span_, self.modes_, self.states_, self.inputs_, 
                                    self._init_state, self._target_state, self.n_timesteps_, self._reset_args)
            
        mode_exttrjs_maps = self.compute_trejactory_extension(txmode_map)

        return modes,states,inputs,k_feedforward,K_feedback,current_cost,states_iter,modechanges,mode_exttrjs_maps, reset_args


def solve_ilqr(params, detect=True):
    # Import dynamics
    smooth_dynamis = params.symbolic_dynamics()

    detect_integration = params.detection_func()
    
    plotting_function = params.plotting_function()
    
    state_convert_function = params.state_convert_function()

    # Initialize timings
    dt = params._dt
    
    start_time = params._start_time
    end_time = params._end_time

    # Set desired state
    init_state = params._init_state  # Define the initial state to be the origin with no velocity
    target_state = params._target_state  # Swing pendulum upright

    # Initial guess of zeros, but you can change it to any guess
    initial_guess = params._initial_guess
    
    # Initial resetmap arguments, for instance the stance positions for the SLIP
    init_reset_args = params._init_reset_args
    
    target_reset_args = params._target_reset_args

    # Define weighting matrices
    Q_k = params._Q_k # zero weight to penalties along a strajectory since we are finding a trajectory
    R_k = params._R_k

    # Set the terminal cost
    Q_T = params._Q_T

    # Set the physical parameters of the system
    mass = 1
    gravity = 9.8
    parameters = np.array([mass,gravity])

    # Specify max number of iterations
    n_iterations = 10
    
    init_mode = params.current_mode()
    target_mode = params._target_mode
    
    nmodes = params.nmodes()
    nstates = params._nstates

    ilqr_ = hybrid_ilqr(nmodes,init_mode,target_mode,nstates,init_state,target_state,initial_guess,
                        dt,start_time,end_time,detect_integration,smooth_dynamis,
                        Q_k,R_k,Q_T,parameters,n_iterations,
                        detect,plotting_function,state_convert_function, 
                        init_reset_args, target_reset_args)
    
    (modes,states,inputs,k_feedforward,K_feedback,
     current_cost,states_iter,
     modechanges,mode_exttrjs_maps,stance_xpos) = ilqr_.solve()
        
    return (modes,states,inputs,k_feedforward,K_feedback,current_cost,states_iter,modechanges,mode_exttrjs_maps,stance_xpos)
        

# Helper functions to handle reference trajectory extensions




def extract_extensions(ref_ext_helper, start_index=0, padding=False):
    # ---------------------------------------------------
    #           Extract the extended references 
    # ---------------------------------------------------
    num_events = len(ref_ext_helper)
    
    if num_events == 0: # reference has no hybrid events
        return None, None, None, None, None, None, None, None
    
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
        MC_i = ref_ext_helper[i_event]["Mode Change"]
        Ext_Trjs_i = ref_ext_helper[i_event]["Trajectory Extensions"]
        Ext_Kfb_i = ref_ext_helper[i_event]["Feedback gains"]
        Ext_kff_i = ref_ext_helper[i_event]["Feedforward gains"]
        tevent_i = ref_ext_helper[i_event]["event index"]
        
        cur_mode_i = MC_i[0]
        next_mode_i = MC_i[1]
        
        v_mode_change.append((cur_mode_i, next_mode_i))
        v_tevents.append(tevent_i)
        
        if padding: # padding to the larger dimension of the two modes.
            n_states = np.array([Ext_Trjs_i[cur_mode_i].shape[1], Ext_Trjs_i[next_mode_i].shape[1]])
            n_inputs = np.array([Ext_Kfb_i[cur_mode_i].shape[1], Ext_Kfb_i[next_mode_i].shape[1]])
            
            max_nstate = np.max(n_states)
            max_ninput = np.max(n_inputs)
            
            nt_length = Ext_Trjs_i[cur_mode_i].shape[0] - start_index 
            
            ext_trj_fwd = np.zeros((nt_length, max_nstate))
            ext_trj_bwd = np.zeros((nt_length, max_nstate))
            ext_trj_fwd[:, :n_states[0]] = Ext_Trjs_i[cur_mode_i][start_index:]
            ext_trj_bwd[:, :n_states[1]] = Ext_Trjs_i[next_mode_i][start_index:]
            
            v_ext_trj_fwd.append(ext_trj_fwd)
            v_ext_trj_bwd.append(ext_trj_bwd)
            
            Kfb_ext_trj_fwd = np.zeros((nt_length, max_ninput, max_nstate))
            Kfb_ext_trj_bwd = np.zeros((nt_length, max_ninput, max_nstate))
            Kfb_ext_trj_fwd[:, :n_inputs[0], :n_states[0]] = Ext_Kfb_i[cur_mode_i][start_index:]
            Kfb_ext_trj_bwd[:, :n_inputs[1], :n_states[1]] = Ext_Kfb_i[next_mode_i][start_index:]
            
            v_Kfb_ext_trj_fwd.append(Kfb_ext_trj_fwd)
            v_Kfb_ext_trj_bwd.append(Kfb_ext_trj_bwd)
            
            kff_ext_trj_fwd = np.zeros((nt_length, max_ninput))
            kff_ext_trj_bwd = np.zeros((nt_length, max_ninput))
            kff_ext_trj_fwd[:, :n_inputs[0]] = Ext_kff_i[cur_mode_i][start_index:]
            kff_ext_trj_bwd[:, :n_inputs[1]] = Ext_kff_i[next_mode_i][start_index:]
            
            v_kff_ext_trj_fwd.append(kff_ext_trj_fwd)
            v_kff_ext_trj_bwd.append(kff_ext_trj_bwd)
            
        else:
            # Add the forward and backward extensions to the collection        
            v_ext_trj_fwd.append(Ext_Trjs_i[cur_mode_i][start_index:])
            v_ext_trj_bwd.append(Ext_Trjs_i[next_mode_i][start_index:])
            
            # Add the feedback gain for forward and backward extensions to the collection
            v_Kfb_ext_trj_fwd.append(Ext_Kfb_i[cur_mode_i][start_index:])
            v_Kfb_ext_trj_bwd.append(Ext_Kfb_i[next_mode_i][start_index:])
            
            # Add the feedforward gain for backward and backward extensions to the collection
            v_kff_ext_trj_fwd.append(Ext_kff_i[cur_mode_i][start_index:])
            v_kff_ext_trj_bwd.append(Ext_kff_i[next_mode_i][start_index:])
        
        
    return (v_mode_change, v_ext_trj_bwd, v_ext_trj_fwd, 
            v_Kfb_ext_trj_bwd, v_Kfb_ext_trj_fwd, v_kff_ext_trj_bwd, v_kff_ext_trj_fwd, v_tevents)
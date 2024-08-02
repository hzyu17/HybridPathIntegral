import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from dynamics.guard_reset_bouncing import *

# numpy and scipy
import scipy
import numpy as np


# Helper function to handle reference trajectory extensions
def extract_extensions(reference_extension_helper, start_index=0, padding=False):
    # ---------------------------------------------------
    #           Extract the extended references 
    # ---------------------------------------------------
    num_events = len(reference_extension_helper)
    
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
        MC_i = reference_extension_helper[i_event][0]
        MC_EXTTRJ_MAP = reference_extension_helper[i_event][1]
        MC_FEEDBACK_EXTTRJ_MAP = reference_extension_helper[i_event][2]
        MC_FEEDFWD_EXTTRJ_MAP = reference_extension_helper[i_event][3]
        tevent_i = reference_extension_helper[i_event][4]
        
        cur_mode_i = MC_i[0]
        next_mode_i = MC_i[1]
        
        v_mode_change.append((cur_mode_i, next_mode_i))
        v_tevents.append(tevent_i)
        
        if padding: # padding to the larger dimension of the two modes.
            n_states = np.array([MC_EXTTRJ_MAP[cur_mode_i].shape[1], MC_EXTTRJ_MAP[next_mode_i].shape[1]])
            n_inputs = np.array([MC_FEEDBACK_EXTTRJ_MAP[cur_mode_i].shape[1], MC_FEEDBACK_EXTTRJ_MAP[next_mode_i].shape[1]])
            
            max_nstate = np.max(n_states)
            max_ninput = np.max(n_inputs)
            
            nt_length = MC_EXTTRJ_MAP[cur_mode_i].shape[0] - start_index - 1
            
            ext_trj_fwd = np.zeros((nt_length, max_nstate))
            ext_trj_bwd = np.zeros((nt_length, max_nstate))
            ext_trj_fwd[:, :n_states[0]] = MC_EXTTRJ_MAP[cur_mode_i][start_index:-1]
            ext_trj_bwd[:, :n_states[1]] = MC_EXTTRJ_MAP[next_mode_i][start_index:-1]
            
            v_ext_trj_fwd.append(ext_trj_fwd)
            v_ext_trj_bwd.append(ext_trj_bwd)
            
            Kfb_ext_trj_fwd = np.zeros((nt_length, max_ninput, max_nstate))
            Kfb_ext_trj_bwd = np.zeros((nt_length, max_ninput, max_nstate))
            Kfb_ext_trj_fwd[:, :n_inputs[0], :n_states[0]] = MC_FEEDBACK_EXTTRJ_MAP[cur_mode_i][start_index:]
            Kfb_ext_trj_bwd[:, :n_inputs[1], :n_states[1]] = MC_FEEDBACK_EXTTRJ_MAP[next_mode_i][start_index:]
            
            v_Kfb_ext_trj_fwd.append(Kfb_ext_trj_fwd)
            v_Kfb_ext_trj_bwd.append(Kfb_ext_trj_bwd)
            
            kff_ext_trj_fwd = np.zeros((nt_length, max_ninput))
            kff_ext_trj_bwd = np.zeros((nt_length, max_ninput))
            kff_ext_trj_fwd[:, :n_inputs[0]] = MC_FEEDFWD_EXTTRJ_MAP[cur_mode_i][start_index:]
            kff_ext_trj_bwd[:, :n_inputs[1]] = MC_FEEDFWD_EXTTRJ_MAP[next_mode_i][start_index:]
            
            v_kff_ext_trj_fwd.append(kff_ext_trj_fwd)
            v_kff_ext_trj_bwd.append(kff_ext_trj_bwd)
            
        else:
            # Add the forward and backward extensions to the collection        
            v_ext_trj_fwd.append(MC_EXTTRJ_MAP[cur_mode_i][start_index:-1])
            v_ext_trj_bwd.append(MC_EXTTRJ_MAP[next_mode_i][start_index:-1])
            
            # Add the feedback gain for forward and backward extensions to the collection
            v_Kfb_ext_trj_fwd.append(MC_FEEDBACK_EXTTRJ_MAP[cur_mode_i][start_index:])
            v_Kfb_ext_trj_bwd.append(MC_FEEDBACK_EXTTRJ_MAP[next_mode_i][start_index:])
            
            # Add the feedforward gain for backward and backward extensions to the collection
            v_kff_ext_trj_fwd.append(MC_FEEDFWD_EXTTRJ_MAP[cur_mode_i][start_index:])
            v_kff_ext_trj_bwd.append(MC_FEEDFWD_EXTTRJ_MAP[next_mode_i][start_index:])
        
        
    return (v_mode_change, v_ext_trj_bwd, v_ext_trj_fwd, 
            v_Kfb_ext_trj_bwd, v_Kfb_ext_trj_fwd, v_kff_ext_trj_bwd, v_kff_ext_trj_fwd, v_tevents)


# ------------------------------------- 
# Function that handles mode mismatch 
# -------------------------------------    
def reaction_mode_mismatch(current_index, 
                            current_mode, ref_current_mode, 
                            ext_trj_fwd, ext_trj_bwd, 
                            ref_ext_modechange,
                            Kfb_ref_ext_fwd, kff_ref_ext_fwd,
                            Kfb_ref_ext_bwd, kff_ref_ext_bwd,
                            cnt_mismatch, cond_early_arrival=None):
    # Take the first hybrid event for now. Needs to find the correct corresponding one among all hybrid events.
    
    if (cond_early_arrival(current_mode, ref_current_mode, ref_ext_modechange)):
        extended_trj = ext_trj_bwd
        K_fb = Kfb_ref_ext_bwd
        k_ff = kff_ref_ext_bwd
    else:
        extended_trj = ext_trj_fwd
        K_fb = Kfb_ref_ext_fwd
        k_ff = kff_ref_ext_fwd
    
    xref_i = extended_trj[current_index]
    K_fb_i = K_fb[current_index]
    k_ff_i = k_ff[current_index]
    
    cnt_mismatch += 1
    
    return xref_i, K_fb_i, k_ff_i, cnt_mismatch


def stochastic_integration(x0, u, t_span, epsilon, dW, dyn_f, dyn_gdWt):
    """ Rollout function assuming constant control input during the time span.
    Returns:
        array: stochastic integrated state at tf.
    """
    args = (u, )
    # ============= ode solver =============
    
    # solution = scipy.integrate.solve_ivp(fun=lambda t, y: dyn_bouncing(t, y, *args), 
    #                                     t_span=t_span, y0=x0, method='RK45', 
    #                                     t_eval=t_eval, dense_output=True)
    
    # t0, tf = t_span[0], t_span[-1]
    
    # # Solve for the continuous trajectory before the contact 
    # t_sol = np.linspace(t0, tf, nt).flatten()
    
    # # The solved trajecoty, in shape (nx+nx*nx, nt)
    # f_disc = solution.sol(t_sol) 
    # x_next_det = f_disc[:, -1]
    
    # xt_next = x_next_det + np.sqrt(epsilon)*dW
    
    # ============= method 2: forward Euler =============
    t0, tf = t_span[0], t_span[-1]
    dt = tf - t0
    xt_next = x0 + dyn_f(t0, x0, *args)*dt + dyn_gdWt(x0, dW, epsilon)
    
    return xt_next

def event_condition(xt, xt_next, guard):
    # assume time invariant guard for now
    return (guard(0.0,xt)>0) and (guard(0.0,xt_next)<=0) 

def event_reactive_fun(args):
    (xt_current, current_mode, u, 
     t, t_next, xt_next, dt_int, dt_shrinkingrate, 
     RandN, epsilon, 
     smooth_integration_fun, guard_fun, reset_map_fun, reset_args) = args
    
    xt_swch = xt_next
    current_guard = guard_fun[current_mode]
    current_resetmap = reset_map_fun[current_mode]
    
    # Sandwich rule to find finer grind 
    cnt = 0
    while (True):
        cnt += 1
        
        # Too far from the guard, shrink the step size.
        dt_int = dt_int * dt_shrinkingrate
        dW_new = np.sqrt(dt_int)*RandN
        
        # ---- solver for the deterministic part
        t_span = (t, t+dt_int)
        
        xt_swch = smooth_integration_fun(current_mode, xt_current, u, t_span, epsilon, dW_new)

        reset_byproduct = reset_args
        
        # /---- solver for the deterministic part
        if (not event_condition(xt_current, xt_swch, current_guard)) or (cnt==10): # Until the guard condition is no longer met.
            # The reset map is called
            xt_next, next_mode, reset_byproduct = current_resetmap(t, xt_swch, current_mode, reset_args)
            dW = dW_new
            break
        
    return xt_next, next_mode, dW, reset_byproduct


def event_detect_onestep(x0, u, t0, tf, current_mode, 
                         smooth_dynamics, 
                         guards,
                         gxs,
                         gts,
                         reset_maps,
                         reset_controls,
                         Rxs,
                         Rts,
                         reset_args, detection=True, backwards=False):
    """Integrate controlled dynamics in a short period of time with hybrid event detection.

    Args:
        x0 (array): starting state
        u (array): control input
        t0 (scalar): start time
        tf (scalar): end time
        current_mode (int): the current mode
        smooth_dynamics(map): mode-dependent smooth dynamics 
        guards: mode-dependent guard functions
        gxs: mode-dependent \par g(t,x) / \par x
        gts: mode-dependent \par g(t,x) / \par t
        reset_maps: mode-dependent reset_map functions R(t,x)
        reset_controls: mode-dependent reset_control_functions
        Rxs: mode-dependent \par R(t,x) / \par x
        Rts: mode-dependent \par R(t,x) / \par t
        detection (bool, optional): With detection flag. Defaults to True.
        backwards (bool, optional): Integrate backwards flag. Defaults to False.

    Returns:
        tuple: Containing the next state and contact information if a hybrid event happens.
    """
    # Define the dynamics using the integration
    nt = 100
    
    current_dyn = smooth_dynamics[current_mode]
    
    current_guard = guards[current_mode]
    current_resetmap = reset_maps[current_mode]
    current_resetcontrl = reset_controls[current_mode]
    
    current_Rx = Rxs[current_mode]
    current_Rt = Rts[current_mode]
    
    current_gx = gxs[current_mode]
    current_gt = gts[current_mode]
    
    current_guard = guards[current_mode]
    current_resetmap = reset_maps[current_mode]
    
    args = (u, )
    if backwards:
        # integrate backwards
        t_span = (t0, tf)
    else:
        t_span = (t0, tf)
        t_eval = np.linspace(t0, tf, nt)
        dyn_fun=lambda t, y: current_dyn(t, y, *args)
    
    x_next = None
    x_event = None
    t_event = None
    x_reset = None
    saltation = None
    next_mode = current_mode
    reset_byproduct = (None, )
    
    if detection:
        solution = scipy.integrate.solve_ivp(fun=dyn_fun, 
                                            t_span=t_span, y0=x0, method='RK45', 
                                            t_eval=t_eval, dense_output=True, 
                                            events=current_guard, vectorized=False)
    
        # Hit guard
        if len(solution.t_events[0]) > 0:
            t_event = solution.t_events[0][0]
            x_event = solution.y_events[0][0]
            x_reset, next_mode, reset_byproduct = current_resetmap(t_event, x_event, current_mode, reset_args)
            u_reset = current_resetcontrl(t_event, u)
            x0 = x_reset
            
            # ---------- Compute saltation matrix ---------- 
            R_x = current_Rx(t_event, x_event, current_mode, reset_args)[0]
            R_t = current_Rt(t_event, x_event, current_mode, reset_args)[0]
            
            g_x = current_gx(t_event, x_event)
            g_t = current_gt(t_event, x_event)
            
            next_dyn = smooth_dynamics[next_mode]
            
            F_1 = current_dyn(t_event, x_event)
            F_2 = next_dyn(t_event, x_reset) # Important, the F2 is evaluated at the reseted state!
            saltation = saltation_matrix(F_1, F_2, R_t, R_x, g_t, g_x)
            
            t0 = t_event
                        
            x_next = x_reset.flatten()
        
        # Had no contact
        else:
            x0 = None
            
            t = np.linspace(t0, tf, nt).flatten()
            
            # The solved trajecoty, in shape (nx+nx*nx, nt)
            f_disc = solution.sol(t) 
            
            x_next = f_disc[:, -1]
            
    else: # Do not detect contact 
        solution = scipy.integrate.solve_ivp(fun=dyn_fun, 
                                            t_span=t_span, y0=x0, method='RK45', 
                                            t_eval=t_eval, dense_output=True)
        
        # Solve for the continuous trajectory before the contact 
        t = np.linspace(t0, tf, nt).flatten()
        
        # The solved trajecoty, in shape (nx+nx*nx, nt)
        f_disc = solution.sol(t) 
        
        x_next = f_disc[:, -1]
    
    mode_mapping = np.array([current_mode, next_mode])
    
    return x_next, saltation, mode_mapping, t_event, x_event, x_reset, reset_byproduct
# Simulation for 2 dimensional state space dynamics
# Hongzhe Yu, 02/27/2024

import numpy as np
import scipy
from simulation.propagate_covariance import *

def simulation_2d_saltation(z0, vz0, t0, tf, nt, 
                  Modes, guard, reset_map, reset_mode, 
                  Rx, Rt, gx, gt, linearization, 
                  Sig0, guard_direction=1, n_events=2):
    """Solving the ODE with hybrid events.

    Args:
        z0 (numpy array): Initial height
        vz0 (numpy array): Initial velocity in z direction
        t0 (float): Initial time
        tf (float): Terminal time
        Modes (list(functions)): The collection of all the mode functions.
        n_events (int): Number of events before stopping, 
                        if set to None, then the simulation will continue until tf.
        return_saltation (bool): If return the saltation matrices.

    Returns:
        t_ttl: The total time span.
        x_trj: The whole trajectory including the hybrid events.
        tevents: Time of the hybrid event happening.
        xevents: States at the hybrid event.
        xresets: States after the reset map, for each hybrid event.
        saltations: Collection of saltation matrices at the hybrid events.
    """
    x_trj = []
    tevents = []
    xevents = []
    xresets = []
    saltations = []
    current_mode = 0
    num_events = 0
    x0 = np.array([z0, vz0], dtype=np.float64)    

    guard.terminal=True
    guard.direction=guard_direction

    tf_final = tf

    while (num_events < n_events):   
        dyn_1 = Modes[current_mode]
        dyn_2 = Modes[reset_mode(current_mode)]
        
        t_span = (t0, tf)
        t_eval = np.linspace(t0, tf, 1000)
        
        # Compute saltation matrix and solve for covariance matrix
        initial_conditions = np.concatenate([x0, Sig0.flatten()])  # Combine initial conditions
        
        # ---------- Solve ODE for mean and covariance jointly, with hybrid event detection ---------- 
        args = (dyn_1, linearization, 2)    
        solution = scipy.integrate.solve_ivp(fun=lambda t, y: dxdX_solve_ivp(t, y, *args), 
                                            t_span=t_span, 
                                            y0=initial_conditions, method='RK45', 
                                            t_eval=t_eval, 
                                            dense_output=True, 
                                            events=guard)
        # Has contact
        if len(solution.t_events[0]) > 0:
            t_event = solution.t_events[0][0]
            xX_event = solution.y_events[0][0]
            x_event = xX_event[0:2]
            x_reset = reset_map(t_event, x_event)
            current_mode = reset_mode(current_mode)
            x0 = x_reset
                                    
            # ---------- Compute saltation matrix ---------- 
            R_x = Rx(t_event, x_event)
            R_t = Rt(t_event, x_event)
            g_x = gx(t_event, x_event)
            g_t = gt(t_event, x_event)
            F_1 = dyn_1(t_event, x_event)
            F_2 = dyn_2(t_event, x_reset) # Important, the F2 is evaluated at the reseted state!
            saltation = compute_saltation(F_1, F_2, R_t, R_x, g_t, g_x)
            saltations.append(saltation)
            
            X_event = xX_event[2:6].reshape([2, 2])
            Sig0 = saltation @ X_event @ saltation.transpose()           
            
        else: # Has no contact
            t_event = tf
            x_event = np.array([np.inf, np.inf])
            x_reset = np.array([np.inf, np.inf])
            x0 = None
        
        # Solve for the continuous trajectory before the contact 
        t = np.linspace(t0, t_event, nt).flatten()
        
        # The solved trajecoty, in shape (nx+nx*nx, nt)
        x_trj_i = solution.sol(t) 
        
        # ----------- Add the piece-wise trajectory
        if len(x_trj) == 0:
            x_trj = x_trj_i
            tevents = np.array([t_event])
            xevents = x_event
            xresets = x_reset
            t_ttl = t
        else:
            x_trj = np.concatenate([x_trj, x_trj_i], axis=1)
            tevents = np.concatenate([tevents, np.array([t_event])], axis=0)
            xevents = np.concatenate([xevents, x_event], axis=0)
            xresets = np.concatenate([xresets, x_reset], axis=0)
            t_ttl = np.concatenate([t_ttl, t]).flatten()
        
        if x0 is None:
            x0 = x_trj_i[0:2, -1]
            
        t0 = t_event
        tf = min(t0 + 5.0, tf_final)
                
        num_events += 1

    return t_ttl, x_trj, tevents, xevents, xresets, saltations


def simulation_2d(z0, vz0, t0, tf, nt, 
                  Modes, guard, reset_map, reset_mode,
                  n_events=2, guard_direction=1):
    """Solving the ODE with hybrid events.

    Args:
        z0 (numpy array): Initial height
        vz0 (numpy array): Initial velocity in z direction
        t0 (float): Initial time
        tf (float): Terminal time
        Modes (list(functions)): The collection of all the mode functions.
        n_events (int): Number of events before stopping, 
                        if set to None, then the simulation will continue until tf.
        return_saltation (bool): If return the saltation matrices.

    Returns:
        t_ttl: The total time span.
        x_trj: The whole trajectory including the hybrid events.
        tevents: Time of the hybrid event happening.
        xevents: States at the hybrid event.
        xresets: States after the reset map, for each hybrid event.
    """
    x_trj = []
    tevents = []
    xevents = []
    xresets = []
    current_mode = 0
    num_events = 0
    x0 = np.array([z0, vz0], dtype=np.float64)    

    guard.terminal=True
    guard.direction=guard_direction

    tf_final = tf

    while (num_events < n_events):   
        dyn_1 = Modes[current_mode]
        
        t_span = (t0, tf)
        t_eval = np.linspace(t0, tf, 1000)
        
        # ---------- Solve ODE with hybrid event detection ---------- 
        solution = scipy.integrate.solve_ivp(dyn_1, t_span, x0, method='RK45', 
                                            t_eval=t_eval, dense_output=True, 
                                            events=guard, vectorized=False, args=None)
        if len(solution.t_events[0]) > 0:
            t_event = solution.t_events[0][0]
            x_event = solution.y_events[0][0]
            current_mode = reset_mode(current_mode)
            x_reset = reset_map(t_event, x_event, current_mode)
            x0 = x_reset
        else:
            t_event = tf
            x_event = np.array([np.inf, np.inf])
            x_reset = np.array([np.inf, np.inf])
            x0 = None

        # Solve for the continuous trajectory before the contact 
        t = np.linspace(t0, t_event, nt).flatten()
        
        # The solved trajecoty, in shape (nx+nx*nx, nt)
        x_trj_i = solution.sol(t) 
        
        if x0 is None:
            x0 = x_trj_i[:, -1]
        
        # ----------- Add the piece-wise trajectory
        if len(x_trj) == 0:
            x_trj = x_trj_i
            tevents = np.array([t_event])
            xevents = x_event
            xresets = x_reset
            t_ttl = t
        else:
            x_trj = np.concatenate([x_trj, x_trj_i], axis=1)
            tevents = np.concatenate([tevents, np.array([t_event])], axis=0)
            xevents = np.concatenate([xevents, x_event], axis=0)
            xresets = np.concatenate([xresets, x_reset], axis=0)
            t_ttl = np.concatenate([t_ttl, t]).flatten()
        
        print("t_event")
        print(t_event)
        
        print("tf_final")
        print(tf_final)
        
        t0 = t_event
        tf = min(t0 + 5.0, tf_final)
                
        num_events += 1

    return t_ttl, x_trj, tevents, xevents, xresets

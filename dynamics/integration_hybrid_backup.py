import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)
from dynamics.bouncing_guard_reset import *

# numpy import 
import scipy
import sympy as sp
from sympy.matrices import Matrix
import numpy as np

# jax import
import jax
import jax.numpy as jnp
from functools import partial
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
print("Devices:", jax.devices())

# plotting
import matplotlib.pyplot as plt


def dyn_bouncing(t, x, *args):
    """
    Args:
        t (_type_): time variable
        x (_type_): state
        args[0]: control input
    """
   
    if len(args) == 0:
        u = np.array([0.0])
    else:
        u = args[0]
    return np.array([x[1], u[0]-g])


def gdWt_bouncing(dWt, eps):
    B = np.array([[0],[1.0]], dtype=np.float64)
    return np.sqrt(eps) * B@dWt
    
    
def symbolic_dynamics_bouncing_continuoustime():
    g = 9.81
    z,z_dot,u,dt = sp.symbols('z z_dot u dt')

    # Define the states and inputs
    inputs = Matrix([u])
    states = Matrix([z, z_dot])
    # Defining the dynamics of the system
    f_contin = Matrix([z_dot, u-g])
    
    A_contin = f_contin.jacobian(states)
    B_contin = f_contin.jacobian(inputs)

    A_contin_func = sp.lambdify((states,inputs),A_contin)
    B_contin_func = sp.lambdify((states,inputs),B_contin)
    
    return (f_contin,A_contin_func,B_contin_func)


def symbolic_dynamics_bouncing():
    g = 9.81
    z,z_dot,u,dt = sp.symbols('z z_dot u dt')

    # Define the states and inputs
    inputs = Matrix([u])
    states = Matrix([z, z_dot])
    # Defining the dynamics of the system
    f = Matrix([z_dot, u-g])

    # Discretize the dynamics usp.sing euler integration
    f_disc = states+f*dt
    
    # Take the jacobian with respect to states and inputs
    A_disc = f_disc.jacobian(states)
    B_disc = f_disc.jacobian(inputs)

    f_disc_func = sp.lambdify((states,inputs,dt),f_disc)
    A_disc_func = sp.lambdify((states,inputs,dt),A_disc)
    B_disc_func = sp.lambdify((states,inputs,dt),B_disc)
    return (f_disc_func,A_disc_func,B_disc_func)
    

def stochastic_integration_bouncing(x0, u, current_mode, t_span, t_eval, epsilon, RandN, dt, nt):
    dW = np.sqrt(dt)*RandN
    xt_next = stochastic_integration(x0, u, t_span, t_eval, epsilon, dW, nt)
    t_next = t_eval[-1]
    
    # Sandwich rule for contact detection
    dt_shrinkingrate = 0.7
    dt_int = dt
    t, t_plus = t_span[0], t_span[1]
    
    next_mode = current_mode
    # Guard condition: direction is -1     
    if (guard_bouncing_12(t, x0)>0) and (guard_bouncing_12(t_plus, xt_next)<=0): # Hit the guard function.
        xt_swch = xt_next
        
        # Sandwich rule to find finer grind 
        cnt = 0
        while (True):
            cnt += 1
            xt_last = xt_swch
            
            # Too far from the guard, shrink the step size.
            dt_int = dt_int * dt_shrinkingrate
            dW = np.sqrt(dt_int)*RandN
            
            # ---- solver for the deterministic part
            t_span = (t, t_next)
            t_eval = np.linspace(t, t_next, nt)
            
            xt_swch = stochastic_integration(x0, u, t_span, t_eval, epsilon, dW, nt)

            # /---- solver for the deterministic part
            if (guard_bouncing_12(t, xt_swch)>0) or (cnt==10): # Until the guard condition is no longer met.
                # The reset map is called
                xt_next, next_mode = reset_map_bouncing_12(t, xt_last, current_mode)
                dt_int = dt
                break
    return xt_next, next_mode

def stochastic_integration(x0, u, t_span, t_eval, epsilon, dW, nt):
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
    xt_next = x0 + dyn_bouncing(t0, x0, *args)*dt + gdWt_bouncing(dW, epsilon)
    
    return xt_next


def rollout_bouncing_stochastic_feedback(x0, cur_mode_change, xt_ref, ref_modechanges, 
                                         ut, Kt, kt, target_state, R_k, Q_T, t0, tf, 
                                         epsilon, GaussianNoise, mode_exttrjs_maps=None):

    n_timestamps = xt_ref.shape[0]
    _, nu, nx = Kt.shape
    
    dt = (tf - t0) / n_timestamps
    
    # Integration of the stochastic system
    xt = x0
    
    # returning trajectory
    xt_trj = np.zeros((n_timestamps, nx), dtype=np.float64)
    xt_trj[0] = xt
    
    # guard_thres = 1e-4
    dt_shrinkingrate = 1e-4
    dt_int = dt
    
    ut_cl = np.zeros((n_timestamps, nu))
    
    guard_bouncing_12.terminal=True
    guard_bouncing_12.direction=-1
    
    guard_bouncing_21.terminal=True
    guard_bouncing_21.direction=1
    
    guards = {1:guard_bouncing_12, 2: guard_bouncing_21}
    reset_maps = {1:reset_map_bouncing_12, 2:reset_map_bouncing_21}
    
    current_mode = cur_mode_change[0]
    # only consider the 1->2 reset for now (bouncing)
    current_guard = guards[1]
    current_resetmap = reset_maps[1]
    
    mismatched_states = None
    mismatched_refs = None
    modified_refs = None
    cnt_mismatch = 0
    actual_xtref = np.zeros((n_timestamps, nx), dtype=np.float64)
    
    cnt_events = 0 # count the number of hybrid events happened during the execusion.
    mode_change = cur_mode_change
    
    # path cost
    Sk = 0
    for i in range(n_timestamps-1):    
        
        xref_i = xt_ref[i]
        
        t0_i = t0 + i*dt  
        
        # # Riccati
        # u = Kt[i]@xt + kt[i]
        
        # i-lqr
        # ======== Check mode mismatch ======== 
        current_mode = mode_change[0]
        next_mode = mode_change[1]
        ref_next_mode = ref_modechanges[i][1]
        if (next_mode != ref_next_mode) and len(mode_exttrjs_maps) > 0:
            # Take the first hybrid event for now. Needs to find the correct corresponding one among all hybrid events.
            mode_change_i, mode_exttrjs_i = mode_exttrjs_maps[0]
            extended_trj = mode_exttrjs_i[next_mode]
            
            # First time early arrival: find and reverse the ref
            if (next_mode==2) and (ref_next_mode==1) and (cnt_mismatch==0): 
                len_ref = 0
                i_ext = 0
                while True: # Find the correct length of the extension
                    if (ref_modechanges[i+i_ext][1] == next_mode):
                        len_ref = i_ext
                        break
                    i_ext += 1
                extended_trj = extended_trj[0:len_ref]
                extended_trj = extended_trj[::-1]
                
            xref_i = extended_trj[cnt_mismatch]
            
            if modified_refs is None:
                modified_refs = xref_i.reshape((1, -1))
            else:
                modified_refs = np.vstack((modified_refs, xref_i.reshape((1, -1))))
                
            # Collect data
            if mismatched_states is None:
                mismatched_states = xt_trj[i].reshape((1, -1))
                mismatched_refs = xt_ref[i].reshape((1, -1))
            else:
                mismatched_states = np.vstack((mismatched_states, xt_trj[i].reshape((1, -1))))
                mismatched_refs = np.vstack((mismatched_refs, xt_ref[i].reshape((1, -1))))
            
            cnt_mismatch += 1
            
            # print("mode mismatch")
            # print("current_mode:", current_mode)
            # print("ref mode change:", ref_modechanges[i])
            
        actual_xtref[i] = xref_i
        
        delta_xt = xt_trj[i] - xref_i
        u = ut[i] + Kt[i]@delta_xt + kt[i]
        ut_cl[i] = u
        
        dW_i = np.sqrt(dt_int)*GaussianNoise[i]
        
        # One step integration        
        # ---- solver for the deterministic part
        t_plus = t0_i + dt_int
        t_span = (t0_i, t_plus)
        t_eval = np.linspace(t0_i, t_plus, n_timestamps)
        
        xt_next = stochastic_integration(xt, u, t_span, t_eval, epsilon, dW_i, n_timestamps).flatten()
        
        # Hit the guard function.  
        if (current_guard(t0_i, xt)>0) and (current_guard(t_plus, xt_next)<=0): 
            print("rollout mode change")
            cnt_events += 1
            xt_swch = xt_next
            
            # Sandwich rule to find finer grind 
            cnt = 0
            while (True):
                cnt += 1
                xt_last = xt_swch
                
                # Too far from the guard, shrink the step size.
                dt_int = dt_int * dt_shrinkingrate
                dW_new = np.sqrt(dt_int)*GaussianNoise[i]
                
                # ---- solver for the deterministic part
                t_span_new = (t0_i, t0_i+dt_int)
                t_eval_new = np.linspace(t0_i, t0_i+dt_int, n_timestamps)
                xt_swch = stochastic_integration(xt, u, t_span_new, t_eval_new, epsilon, dW_new, n_timestamps)
                # /---- solver for the deterministic part
                
                # Until the guard condition is no longer met.
                if (current_guard(t0_i, xt_swch)>0) or (cnt==10): 
                    # The reset map is called on the last integration for which the guard is not met.
                    # print("xt ", xt)
                    # print("xt_last ", xt_last)
                    xt_next, next_mode = current_resetmap(t0_i, xt_last, current_mode)
                    dt_int = dt
                    dW_i = dW_new
                    break
        
        # Collect cost: consider only the terminal state cost for now.
        Sk += u.T@R_k@u/2.0 * dt + np.sqrt(epsilon) * np.dot(u.T, dW_i)
        
        mode_change = (current_mode, next_mode)
        xt_trj[i+1] = xt_next
        xt = xt_next
    
    actual_xtref[-1] = xt_ref[-1]
    
    # Terminal cost
    Sk += (xt-target_state)@Q_T@(xt-target_state) / 2.0
    
    show_mismatch = False
    if show_mismatch:
        # ======== Show mode mismatch ======== 
        fig2, axes = plt.subplots(1,2, figsize=(9, 6))
        ax5, ax6 = axes.flatten()
        ax5.grid(True)
        ax6.grid(True)
        
        ax5.plot(xt_trj[:,0], xt_trj[:,1],color='b',linewidth=1.5,label='Rollout')
        ax5.plot(xt_ref[:,0], xt_ref[:,1],color='k',linewidth=2.5,label='Original Ref.')
        # ax5.plot(actual_xtref[:,0], actual_xtref[:,1],color='r',linewidth=1.5,linestyle='--', label='Modified Reference')
        
        ax5.plot(mismatched_states[:,0], mismatched_states[:,1],linewidth=2.5,color='g',label='Mismatched States')
        ax5.plot(mismatched_refs[:,0], mismatched_refs[:,1],linewidth=2.5,color='cyan',label='Mismatched Ref.')
        ax5.plot(modified_refs[:,0], modified_refs[:,1],linewidth=2.5,color='r',label='Extended Ref.')
        
        ax5.set_xlabel(r"z", fontsize=14)
        ax5.set_ylabel(r"$\dot z$", fontsize=14)
        ax5.legend(loc='upper right')
        plt.tight_layout()
        
        ax6.plot(xt_trj[:,0], xt_trj[:,1],color='b',linewidth=1.5,label='Rollout')
        ax6.plot(xt_ref[:,0], xt_ref[:,1],color='k',linewidth=2.5,label='Original Ref.')
        ax6.plot(actual_xtref[:,0], actual_xtref[:,1],color='r',linewidth=1.5,linestyle='--',label='Modified Ref.')
        ax6.set_xlabel(r"z", fontsize=14)
        ax6.set_ylabel(r"$\dot z$", fontsize=14)
        ax6.legend(loc='upper right')
        plt.tight_layout()
        
        plt.show()
    
    return xt_trj, ut_cl, Sk


def rollout_bouncing_stochastic(x0, ut, t0, tf, epsilon, GaussianNoise):

    nt, nu = ut.shape
    nx = len(x0)
    
    dt = (tf - t0) / (nt-1.0)
    # Define the time span and discretizations
    t_eval = np.linspace(t0, tf, nt)
    
    # Integration of the stochastic system
    xt = x0
    
    # returning trajectory
    xt_trj = np.zeros((nt, nx), dtype=np.float64)
    xt_trj[0] = xt
    
    # guard_thres = 1e-4
    dt_shrinkingrate = 0.3
    dt_int = dt
    
    for i in range(nt-1):
        u = ut[i]
        u_next = ut[i+1]
        args = (u, )
        
        # One step integration
        t = t_eval[i]
        t_next = t_eval[i+1]
        
        dW = np.sqrt(dt_int)*GaussianNoise[i]
        # ---- solver for the deterministic part
        t_plus = t + dt_int
        t_span = (t, t_plus)
        t_eval = np.linspace(t, t_plus, nt)
        
        xt_next = stochastic_integration(xt, u, t_span, t_eval, epsilon, dW, nt)
        # /---- solver for the deterministic part
        
        # Guard condition: direction is -1     
        if (guard_bouncing(t, xt)>0) and (guard_bouncing(t_plus, xt_next)<=0): # Hit the guard function.
            xt_swch = xt_next
            
            # Sandwich rule to find finer grind 
            cnt = 0
            while (True):
                cnt += 1
                xt_last = xt_swch
                
                # Too far from the guard, shrink the step size.
                dt_int = dt_int * dt_shrinkingrate
                dW = np.sqrt(dt_int)*GaussianNoise[i]
                
                # ---- solver for the deterministic part
                t_span = (t, t_next)
                t_eval = np.linspace(t, t_next, nt)
                
                xt_swch = stochastic_integration(xt, u, t_span, t_eval, epsilon, dW, nt)

                # /---- solver for the deterministic part
                
                if (guard_bouncing(t, xt_swch)>0) or (cnt==10): # Until the guard condition is no longer met.
                    # The reset map is called on the last integration for which the guard is not met.
                    # print("xt ", xt)
                    # print("xt_last ", xt_last)
                    xt_next = reset_map_bouncing(t, xt_last)
                    dt_int = dt
                    break
        xt = xt_next
        xt_trj[i+1] = xt
    
    return xt_trj
    
    
def detect_bouncing(x0, u, t0, tf, current_mode, detection=True, backwards=False):
    """Integrate controlled dynamics in a short period of time with hybrid event detection.

    Args:
        x0 (array): starting state
        u (array): control input
        t0 (scalar): start time
        tf (scalar): end time
        current_mode (int): the current mode
        detection (bool, optional): With detection flag. Defaults to True.
        backwards (bool, optional): Integrate backwards flag. Defaults to False.

    Returns:
        tuple: Containing the next state and contact information if a hybrid event happens.
    """
    # Define the dynamics using the integration
    nt = 100
    
    guard_bouncing_12.terminal=True
    guard_bouncing_12.direction=-1
    
    guard_bouncing_21.terminal=True
    guard_bouncing_21.direction=1
    
    guards = {1:guard_bouncing_12, 2: guard_bouncing_21}
    reset_maps = {1:reset_map_bouncing_12, 2:reset_map_bouncing_21}
    current_guard = guards[current_mode]
    current_resetmap = reset_maps[current_mode]
    
    args = (u, )
    if backwards:
        # integrate backwards
        t_span = (t0, tf)
    else:
        t_span = (t0, tf)
        t_eval = np.linspace(t0, tf, nt)
        dyn_fun=lambda t, y: dyn_bouncing(t, y, *args)
    
    x_event = None
    t_event = None
    x_reset = None
    saltation = None
    next_mode = current_mode
    
    if detection:
        solution = scipy.integrate.solve_ivp(fun=dyn_fun, 
                                            t_span=t_span, y0=x0, method='RK45', 
                                            t_eval=t_eval, dense_output=True, 
                                            events=current_guard, vectorized=False)
    
        # Hit guard
        if len(solution.t_events[0]) > 0:
            t_event = solution.t_events[0][0]
            x_event = solution.y_events[0][0]
            x_reset, next_mode = current_resetmap(t_event, x_event, current_mode)
            x0 = x_reset
            
            # ---------- Compute saltation matrix ---------- 
            R_x = Rx_bouncing(t_event, x_event)
            R_t = Rt_bouncing(t_event, x_event)
            g_x = gx_bouncing(t_event, x_event)
            g_t = gt_bouncing(t_event, x_event)
            F_1 = dyn_bouncing(t_event, x_event)
            F_2 = dyn_bouncing(t_event, x_reset) # Important, the F2 is evaluated at the reseted state!
            saltation = saltation_matrix(F_1, F_2, R_t, R_x, g_t, g_x)
            
            t0 = t_event
            
            # ---------- Regardless of contact, integrate until t=tf ----------
            t_span = (t0, tf)
            t_eval = np.linspace(t0, tf, nt)
            
            solution = scipy.integrate.solve_ivp(fun=dyn_fun, 
                                                t_span=t_span, y0=x0, method='RK45', 
                                                t_eval=t_eval, dense_output=True)
                                
        # Had no contact
        else:
            x0 = None
            
    else: # Do not detect contact 
        solution = scipy.integrate.solve_ivp(fun=dyn_fun, 
                                            t_span=t_span, y0=x0, method='RK45', 
                                            t_eval=t_eval, dense_output=True)
        
    # Solve for the continuous trajectory before the contact 
    t = np.linspace(t0, tf, nt).flatten()
    
    # The solved trajecoty, in shape (nx+nx*nx, nt)
    f_disc = solution.sol(t) 
    
    x_next = f_disc[:, -1]
    
    mode_mapping = (current_mode, next_mode)
    
    return x_next, saltation, mode_mapping, t_event, x_event, x_reset


if __name__ == '__main__':
   
    x0 = np.array([5.0, 0.0])
    u = np.array([0.0])
    
    t0 = 0.0
    dt = 5.0
    
    x_next, saltation = detect_bouncing(x0, u, t0, dt)
    
    print(x_next.shape)
    print(saltation)
    
    # stochastic rollouts
    nt = 1000
    nu = 1
    ut = np.zeros((nt, nu), dtype=np.float64)
    epsilon = 0.1
    t0 = 0.0
    tf = 3.0
    t_eval = np.linspace(t0, tf, nt)
    x0 = np.array([5.0, 1.0])
    xt_trj = rollout_bouncing_stochastic(x0, ut, t0, tf, epsilon)
    
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.scatter(t_eval, xt_trj[:, 0], color='k', s=0.8)
    plt.show()
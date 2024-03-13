import scipy
import sympy as sp
from sympy.matrices import Matrix
import numpy as np

import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from dynamics.bouncing_ball_1D import *

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


def symbolic_dynamics_bouncing_stochastic():
    g = 9.81
    z,z_dot,u,dt,dW,epsilon = sp.symbols('z z_dot u dt dW epsilon')

    # Define the states and inputs
    inputs = Matrix([u])
    states = Matrix([z, z_dot])
    
    # Defining the dynamics of the system
    f = Matrix([z_dot, u-g])

    # Discretize the dynamics usp.sing euler integration
    f_disc = states+f*dt + sp.sqrt(epsilon)*dW
    
    # Take the jacobian with respect to states and inputs
    A_disc = f_disc.jacobian(states)
    B_disc = f_disc.jacobian(inputs)

    f_disc_func = sp.lambdify((states,inputs,dt,epsilon,dW),f_disc)
    A_disc_func = sp.lambdify((states,inputs,dt),A_disc)
    B_disc_func = sp.lambdify((states,inputs,dt),B_disc)
    return (f_disc_func,A_disc_func,B_disc_func)


def stochastic_integration(x0, u, t_span, t_eval, epsilon, dW, nt):
    args = (u, )
    
    solution = scipy.integrate.solve_ivp(fun=lambda t, y: dyn_bouncing(t, y, *args), 
                                        t_span=t_span, y0=x0, method='RK45', 
                                        t_eval=t_eval, dense_output=True)
    
    t0, tf = t_span[0], t_span[-1]
    
    # Solve for the continuous trajectory before the contact 
    t_sol = np.linspace(t0, tf, nt).flatten()
    
    # The solved trajecoty, in shape (nx+nx*nx, nt)
    f_disc = solution.sol(t_sol) 
    x_next_det = f_disc[:, -1]
    
    xt_next = x_next_det + np.sqrt(epsilon)*dW
    # /---- solver for the deterministic part
    
    return xt_next

def rollout_bouncing_stochastic_feedback(x0, Kt, kt, t0, tf, epsilon):
    nt, nu, nx = Kt.shape
    
    dt = (tf - t0) / (nt-1.0)
    # Define the time span and discretizations
    t_eval = np.linspace(t0, tf, nt)
    
    # Integration of the stochastic system
    xt = x0
    
    # returning trajectory
    xt_trj = np.zeros((nt, nx), dtype=np.float64)
    xt_trj[0] = xt
    
    # guard_thres = 1e-4
    dt_shrinkingrate = 0.5
    dt_int = dt
    
    # Generate the randomness
    GaussianNoise = np.random.randn(nt, nu)
    for i in range(nt-1):
        u = Kt[i]@xt + kt[i]
        
        args = (u, )
        
        # One step integration
        t = t_eval[i]
        
        dW = np.sqrt(dt_int)*GaussianNoise[i]
        
        # ---- solver for the deterministic part
        t_plus = t + dt_int
        t_span = (t, t_plus)
        t_eval = np.linspace(t, t_plus, nt)
        
        xt_next = stochastic_integration(xt, u, t_span, t_eval, epsilon, dW, nt)
        
        # solution = scipy.integrate.solve_ivp(fun=lambda t, y: dyn_bouncing(t, y, *args), 
        #                                     t_span=t_span, y0=xt, method='RK45', 
        #                                     t_eval=t_eval, dense_output=True)
        # # Solve for the continuous trajectory before the contact 
        # t_sol = np.linspace(t, t_plus, nt).flatten()
        
        # # The solved trajecoty, in shape (nx+nx*nx, nt)
        # f_disc = solution.sol(t_sol) 
        # x_next_det = f_disc[:, -1]
        # xt_next = x_next_det + np.sqrt(epsilon)*dW
        
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
                t_plus = t + dt_int
                t_span = (t, t_plus)
                t_eval = np.linspace(t, t_plus, nt)
                
                xt_swch = stochastic_integration(xt, u, t_span, t_eval, epsilon, dW, nt)
                
                # solution = scipy.integrate.solve_ivp(fun=lambda t, y: dyn_bouncing(t, y, *args), 
                #                                     t_span=t_span, y0=xt, method='RK45', 
                #                                     t_eval=t_eval, dense_output=True)
                # # Solve for the continuous trajectory before the contact 
                # t_sol = np.linspace(t, t_plus, nt).flatten()
                
                # # The solved trajecoty, in shape (nx+nx*nx, nt)
                # f_disc = solution.sol(t_sol) 
                # x_next_det = f_disc[:, -1]
                # xt_swch = x_next_det + np.sqrt(epsilon)*dW
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
                
                # solution = scipy.integrate.solve_ivp(fun=lambda t, y: dyn_bouncing(t, y, *args), 
                #                                     t_span=t_span, y0=xt, method='RK45', 
                #                                     t_eval=t_eval, dense_output=True)
                # # Solve for the continuous trajectory before the contact 
                # t_sol = np.linspace(t, t_next, nt).flatten()
                
                # # The solved trajecoty, in shape (nx+nx*nx, nt)
                # f_disc = solution.sol(t_sol) 
                # x_next_det = f_disc[:, -1]
                
                # xt_swch = x_next_det + np.sqrt(epsilon)*dW
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
    
    
def detect_bouncing(x0, u, t0, tf):
    # Define the dynamics using the integration
    nt = 100
        
    guard_bouncing.terminal=True
    guard_bouncing.direction=-1
    
    args = (u, )
     
    t_span = (t0, tf)
    t_eval = np.linspace(t0, tf, nt)
    
    solution = scipy.integrate.solve_ivp(fun=lambda t, y: dyn_bouncing(t, y, *args), 
                                            t_span=t_span, y0=x0, method='RK45', 
                                            t_eval=t_eval, dense_output=True, 
                                            events=guard_bouncing, vectorized=False)

    # Had contact
    if len(solution.t_events[0]) > 0:
        t_event = solution.t_events[0][0]
        x_event = solution.y_events[0][0]
        x_reset = reset_map_bouncing(t_event, x_event)
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
        
        solution = scipy.integrate.solve_ivp(fun=lambda t, y: dyn_bouncing(t, y, *args), 
                                                t_span=t_span, y0=x0, method='RK45', 
                                                t_eval=t_eval, dense_output=True)
                    
    # Had no contact
    else:
        x_event = None
        x_reset = None
        x0 = None
        saltation = None
        
    # Solve for the continuous trajectory before the contact 
    t = np.linspace(t0, tf, nt).flatten()
    
    # The solved trajecoty, in shape (nx+nx*nx, nt)
    f_disc = solution.sol(t) 
    
    x_next = f_disc[:, -1]
    
    return x_next, saltation


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
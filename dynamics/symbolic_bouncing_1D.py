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
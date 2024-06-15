# Integration of mean and covariance simultanuously for a bouncing ball dynamics

import scipy
import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(root_dir)

from dynamics.bouncing_guard_reset import *
from simulation.propagate_covariance import *


if __name__ == '__main__':
    x0 = np.array([10.0, 0.0], dtype=np.float64)
    Sig0 = np.eye(2, dtype=np.float64).flatten()
    initial_conditions = np.concatenate([x0, Sig0])  # Combine initial conditions

    guard_bouncing.terminal=True
    guard_bouncing.direction=-1
    
    t0 = 0
    tf = 10.0
    
    t_span = (t0, tf)
    t_eval = np.linspace(t0, tf, 500)  # Time points

    # Solve the coupled system of ODEs using odeint
    # tuple: (linearization_functtion, state_dim)
    
    args = (dyn_bouncing, linearize_bouncing, 2)    
    solution = scipy.integrate.solve_ivp(fun=lambda t, y: dxdX_solve_ivp(t, y, *args), 
                                        t_span=t_span, 
                                        y0=initial_conditions, method='RK45', 
                                        t_eval=t_eval, 
                                        dense_output=True, 
                                        events=guard_bouncing)
    
    t_event = solution.t_events[0][0]
    x_X_event = solution.y_events[0][0]
    x_event = x_X_event[:2]
    X_event = x_X_event[2:].reshape((-1, 2, 2))
        
    # Solve the trajectory before the first contact event
    nt = 300
    t = np.linspace(t0, t_event, nt).flatten()
    x_trj_i = solution.sol(t)      
    
    solution_x = x_trj_i[0:2, :]
    solution_X = x_trj_i[2:6, :].reshape([2,2,nt])
    
    # Plot results
    fig, ax = plt.subplots(1, 1)
    
    # plot mean trajectory
    for i in range(nt):
        ax.scatter(solution_x[0, i], solution_x[1, i], s=1.2, c='b', alpha=0.5)
    
    # plot covariance trajecotry
    for i in range(0, nt, 5):
        ellipse_boundary, ax = plot_2d_ellipsoid_boundary(solution_x[:,i], solution_X[:,:,i], ax, 'r')
    
    ax.grid(True)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\dot z$')
    plt.show()
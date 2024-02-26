# Integration of mean and covariance simultanuously for a bouncing ball dynamics

import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(root_dir)

from dynamics.bouncing_ball_1D import *
from tools.propagate_covariance import *


if __name__ == '__main__':
    x0 = np.array([10.0, 0.0], dtype=np.float64)
    Sig0 = np.eye(2, dtype=np.float64).flatten()
    
    initial_conditions = np.concatenate([x0, Sig0])  # Combine initial conditions

    t = np.linspace(0, 1, 500)  # Time points

    # Solve the coupled system of ODEs using odeint
    # tuple: (linearization_functtion, state_dim)
    args = (dyn_f, linearize, 2)
    solution = odeint(dxdX_odeint, t=t, y0=initial_conditions, args=args)

    # Extract the solution for x and X
    solution_x = solution[:, :2]
    solution_X = solution[:, 2:].reshape((-1, 2, 2))
    
    # Plot results
    fig, ax = plt.subplots(1, 1)
    
    # plot mean trajectory
    for i in range(solution.shape[0]):
        ax.scatter(solution_x[i, 0], solution_x[i, 1], s=1.2, c='b', alpha=0.5)
    
    # plot covariance trajecotry
    for i in range(0, solution_X.shape[0], 5):
        ellipse_boundary, ax = plot_2d_ellipsoid_boundary(solution_x[i], solution_X[i], ax, 'r')
    
    ax.grid(True)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\dot z$')
    plt.show()
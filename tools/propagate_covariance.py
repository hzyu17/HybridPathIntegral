# Propagate the covariance matrix of a ltv system
# Hongzhe Yu, 02/20

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from tools.common_imports import *
from dynamics.constant_flow import *
from tools.plot_ellipsoid import *

def dxdX_odeint(z, t, *args):
    """
    Args:
        z (array): concatenated state and flattened covariance matrix [x, X]
        t (scalar): time variable
        args[0]: dyn (function): system dynamics 
        args[1]: linearization (function): linearization function
        args[2]: n (scalar): state dimension

    Returns:
        [x(t), X(t)]: solved coupled system
    """
    
    dyn = args[0]
    linearization = args[1]
    n = args[2]
    
    x = z[:n]
    X = z[n:].reshape((n, n))
    
    A = linearization(x)

    # state dynamics
    dx_dt = dyn(t, x)
    
    # covariance dynamics
    dX_dt = A @ X + X @ np.transpose(A)

    return np.concatenate([dx_dt, dX_dt.flatten()])


def dxdX_solve_ivp(t, z, *args):
    """
    Args:
        t (scalar): time variable
        z (array): concatenated state and flattened covariance matrix [x, X]
        args[0]: dyn (function): system dynamics 
        args[1]: linearization (function): linearization function
        args[2]: n (scalar): state dimension

    Returns:
        [x(t), X(t)]: solved coupled system
    """
    
    dyn = args[0]
    linearization = args[1]
    n = args[2]
    
    x = z[:n]
    X = z[n:].reshape((n, n))
    
    A = linearization(x)

    # state dynamics
    dx_dt = dyn(t, x)
    
    # covariance dynamics
    dX_dt = A @ X + X @ np.transpose(A)

    return np.concatenate([dx_dt, dX_dt.flatten()])


# constant flow example
if __name__ == '__main__':
    x0 = np.array([0.0, 0.0], dtype=np.float64)
    Sig0 = np.eye(2, dtype=np.float64).flatten()
    
    initial_conditions = np.concatenate([x0, Sig0])  # Combine initial conditions

    t = np.linspace(0, 1, 100)  # Time points

    # Solve the coupled system of ODEs using odeint
    # tuple: (linearization_functtion, state_dim)
    args = (dyn_f1, linearization, 2)
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
    
    plt.show()
    
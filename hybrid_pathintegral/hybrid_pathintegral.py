import numpy as np

# Hybrid path integral control
def update_control_pathintegral(ut, PathCosts, epsilon, dt=0.0001):
    nt, nu = ut.shape
    n_samples = len(PathCosts)
    
    GaussianNoises = np.random.randn(n_samples, nt, nu)
    dWs = GaussianNoises * np.sqrt(dt*epsilon)
    
    # ------- numerical processing -------
    minS = np.min(PathCosts)
    PathCosts = PathCosts - minS
    exp_PathCosts = np.exp(-PathCosts/epsilon)
    sum_expPathCosts = np.sum(exp_PathCosts)
    
    # ------- Compute the update to control -------
    U_update = np.zeros((nt, nu), dtype=np.float64)
    for k in range(n_samples):
        U_update += exp_PathCosts[k]*dWs[k]
    U_update = np.sqrt(epsilon/dt) * U_update / sum_expPathCosts 
    
    return ut + U_update
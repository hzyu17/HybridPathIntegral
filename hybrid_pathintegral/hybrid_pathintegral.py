import numpy as np

# Hybrid path integral control
def update_control_pathintegral(ut, PathCosts, epsilon, dt):
    nt, nu = ut.shape
    n_samples = len(PathCosts)
    
    GaussianNoises = np.random.randn(n_samples, nt, nu)
    dWs = GaussianNoises * np.sqrt(dt*epsilon)
    
    # ------- numerical processing -------
    PathCosts = PathCosts - np.min(PathCosts)
    exp_PathCosts = np.exp(-PathCosts/epsilon)
    sum_expPathCosts = np.sum(exp_PathCosts)
    
    # ------- Compute the update to control -------
    U_update = np.zeros((nt, nu), dtype=np.float64)
    for k in range(n_samples):
        U_update += exp_PathCosts[k]*GaussianNoises[k]
    U_update = np.sqrt(epsilon/dt) * U_update / sum_expPathCosts 
    
    return ut + U_update
import numpy as np
import jax.numpy as jnp
import jax


# ====================================== Path Integral Control ====================================== 
def compute_cost(modes,states,inputs,randN,target_state,trj_ref, Qk, Rk, QT, epsilon, dt):
    
    n_timestamps = len(states)
    
    # Initialize cost
    total_cost = 0.0

    for ii in range(n_timestamps-1):
        mode_i = modes[ii]
        current_u = inputs[mode_i][ii]
        randN_i = randN[mode_i][ii]
        Rk_i = Rk[mode_i]
        
        total_cost += current_u.T@Rk_i@current_u/2.0*dt + np.sqrt(epsilon*dt) * np.dot(current_u.T, randN_i)
        # total_cost = total_cost+current_cost
        
    # Compute terminal cost
    terminal_difference = (target_state-states[-1]).flatten()
    terminal_cost = terminal_difference.T@QT@terminal_difference/2.0

    total_cost = total_cost+terminal_cost

    return total_cost


def compute_cost_nonoise(modes,states,inputs,target_state,trj_ref, Qk, Rk, QT, dt):
    
    n_timestamps = len(states)
    
    # Initialize cost
    total_cost = 0.0

    for ii in range(n_timestamps-1):
        mode_i = modes[ii]
        current_u = inputs[mode_i][ii]
        total_cost += current_u.T@Rk[mode_i]@current_u/2.0*dt 
        
    # Compute terminal cost
    terminal_difference = (target_state-states[-1]).flatten()
    terminal_cost = terminal_difference.T@QT@terminal_difference/2.0

    total_cost = total_cost+terminal_cost

    return total_cost

def update_u0_pathintegral(u0, PathCosts, GaussianNoise, epsilon, dt):
    nu = len(u0)
    n_samples = len(PathCosts)
    
    # ------- numerical processing -------
    PathCosts = PathCosts - np.min(PathCosts)
    exp_PathCosts = np.exp(-PathCosts/epsilon)
    sum_expPathCosts = np.sum(exp_PathCosts)
    
    # ------- Compute the update to control -------
    U_update = np.zeros(nu)
    for k in range(n_samples):
        U_update += exp_PathCosts[k]*GaussianNoise[k]
    U_update = np.sqrt(epsilon/dt) * U_update / sum_expPathCosts 
    
    return u0 + U_update


def update_u0_pathintegral_jax(u0, PathCosts, GaussianNoise, epsilon, Ksamples_delta_t, delta_t):
    # ------- numerical processing -------
    PathCosts = PathCosts - jnp.min(PathCosts)
    exp_PathCosts = jnp.exp(-PathCosts/epsilon)
    sum_expPathCosts = jnp.sum(exp_PathCosts)
    
    # ------- Compute the weights ---------
    E_expS_jax = jnp.mean(exp_PathCosts)
    weights = exp_PathCosts / E_expS_jax
    
    # ------- Compute the update to control -------
    
    def Cost_Noise_mult(exp_PathCosts_k, GaussianNoise_k, delta_t_k):
        return exp_PathCosts_k*GaussianNoise_k*jnp.sqrt(delta_t_k)
    
    Cost_Noise_vmap = jax.vmap(Cost_Noise_mult, in_axes=(0,0,0))
    Cost_Noise_prod = Cost_Noise_vmap(exp_PathCosts, GaussianNoise, Ksamples_delta_t)
    
    U_update_jax = jnp.sqrt(epsilon) * jnp.sum(Cost_Noise_prod) / sum_expPathCosts / delta_t
            
    return u0 + U_update_jax, weights


def compute_weights(PathCosts, epsilon):
    import matplotlib.pyplot as plt
    plot_weights = False
    if plot_weights:
        fig6, ax11 = plt.subplots(figsize=(8,6))
        ax11.grid(True)
        ax11.bar(range(len(PathCosts)), PathCosts)
        ax11.set_title("Path Cost distribution")
        ax11.set_xlabel("Sample Number")
        ax11.set_ylabel("Costs")
    
    # ------- minus minimum value for numerical stability --------   
    PathCosts = PathCosts - np.min(PathCosts)
    PathCosts_eps = PathCosts / epsilon
    expS = np.exp(-PathCosts_eps)

    # ------- Compute the expected value ---------
    E_expS = np.mean(expS)
    
    # ------- Compute weights -------
    weights = expS / E_expS
    
    if plot_weights:
        fig7, ax12 = plt.subplots(figsize=(8,6))
        ax12.grid(True)
        ax12.bar(range(len(weights)), weights)
        ax12.set_title("Weight distribution")
        ax12.set_xlabel("Sample Number")
        ax12.set_ylabel("Costs")
        plt.show()
    
    return weights


def variance_usefulportion(pathcosts, epsilon):
    """Compute variance and useful pertentage of the samples.
    """   
    weights = compute_weights(pathcosts, epsilon)
    
    variance = np.var(weights)

    # ------- Fraction of effective samples -------
    lbda = 1.0 / np.mean(weights**2)
    
    return variance, lbda


def compute_var_lbd_nexp(n_exp, nt, exp_data):
    variances, lbdas = np.zeros((n_exp, nt-1)), np.zeros((n_exp, nt-1))
    epsilon = exp_params = exp_data.get_params()._epsilon
    for i in range(n_exp):
        allPathCosts = exp_data.get_data(i).allPathCosts()
        for j in range(nt -1):
            variances[i, j], lbdas[i, j] = variance_usefulportion(allPathCosts[j], epsilon)
    
    # Calculating the mean and standard deviation along the repetitions
    avg_variances = np.mean(variances, axis=0)
    std_variances = np.std(variances, axis=0)
    
    avg_lbdas = np.mean(lbdas, axis=0)
    std_lbdas = np.std(lbdas, axis=0)
    
    return avg_variances, std_variances, avg_lbdas, std_lbdas
from example_bouncingball import *
from matplotlib.patches import Circle


if __name__ == '__main__':
    
    exp_params = ExpParams()
    exp_data = ExpData(exp_params)
    n_exp = 20

    # ====================================
    # Ablation study for number of samples
    # ====================================
    v_nsamples = [10, 100, 150, 200, 250, 300, 510, 1010, 1510, 2010, 2510, 3010, 3510, 4010]
    eps = 2.0
    v_costs_pi = np.zeros((len(v_nsamples), n_exp))
    v_costs_hilqr = np.zeros((len(v_nsamples), n_exp))
    for i_nsample, n_samples in enumerate(v_nsamples):
        filename = root_dir+f"/data/bouncing/data_{n_samples}samples_eps_{eps}.pickle"
    
        print("loading data: ", filename)
        exp_data.load(filename)
    
        exp_params = exp_data.get_params()
    
        n_exp = exp_params._n_exp
        n_samples = exp_params._n_samples
        time_span = np.arange(exp_params._start_time, exp_params._end_time, exp_params._dt).flatten()
        nt = len(time_span)
        epsilon = exp_params._epsilon
        init_state = exp_params._init_state
        target_state = exp_params._target_state
    
        # # compute the costs and variances
        cost_pi_exp = np.zeros(n_exp)
        cost_ilqr_exp = np.zeros(n_exp)
        variances, lbdas = np.zeros((n_exp, nt-1)), np.zeros((n_exp, nt-1))
        
        for i in range(n_exp):
            trj_ilqr = exp_data.get_data(i).x_trj_ilqr()
            trj_pi = exp_data.get_data(i).x_trj_pi()
            u_star_pi = exp_data.get_data(i).u_trj_pi()
            u_trj_ilqr = exp_data.get_data(i).u_trj_ilqr()
            
            allPathCosts = exp_data.get_data(i).allPathCosts()
            
            for j in range(nt -1):
                variances[i, j], lbdas[i, j] = variance_usefulportion(allPathCosts[j], epsilon)
            
            cost_pi = exp_data.get_data(i).cost_pi()
            cost_ilqr = exp_data.get_data(i).cost_ilqr()
            
            cost_pi_exp[i] = cost_pi
            cost_ilqr_exp[i] = cost_ilqr
        
        v_costs_pi[i_nsample] = cost_pi_exp
        v_costs_hilqr[i_nsample] = cost_ilqr_exp
        
        print("E[cost_pi]: ", np.mean(cost_pi_exp))
        print("E[cost_ilqr_exp]: ", np.mean(cost_ilqr_exp))
        print("improved: ", (np.mean(cost_ilqr_exp) - np.mean(cost_pi_exp)) / np.mean(cost_ilqr_exp))
        
    # Plot the costs versus the number of samples
    fig, ax = plt.subplots()
    
    avg_cost_pi = np.mean(v_costs_pi, axis=1)
    std_cost_pi = np.std(v_costs_pi, axis=1)
    
    avg_cost_hilqr = np.mean(v_costs_hilqr, axis=1)
    std_cost_hilqr = np.std(v_costs_hilqr, axis=1)
    
    # Plotting        
    ax.grid(True)
    
    # Mean as a solid line
    ax.plot(v_nsamples, avg_cost_pi, 'r-', label='avg cost path integral')
    ax.plot(v_nsamples, avg_cost_hilqr, 'b-', label='avg cost h-ilqr')
    
    # Shaded area for variability (e.g., ±1 standard deviation)
    ax.fill_between(v_nsamples, avg_cost_pi - std_cost_pi, avg_cost_pi + std_cost_pi, color='red', alpha=0.1, label='±1 Std. cost path integral')
    ax.fill_between(v_nsamples, avg_cost_hilqr - std_cost_hilqr, avg_cost_hilqr + std_cost_hilqr, color='blue', alpha=0.1, label='±1 Std. cost h-ilqr')

    ax.set_xlabel(r'Number of samples')
    ax.set_ylabel(r'Cost')
    ax.legend()
    
    plt.show()
    
    
    
    # ==========================
    # Ablation study for epsilon
    # ==========================
    n_exp = 20
    v_eps = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0]
    n_samples = 10000
    v_costs_pi = np.zeros((len(v_eps), n_exp))
    v_costs_hilqr = np.zeros((len(v_eps), n_exp))
    for i_eps, epsilon in enumerate(v_eps):
        filename = root_dir+f"/data/bouncing/data_{n_samples}samples_eps_{epsilon}.pickle"
    
        print("loading data: ", filename)
        exp_data.load(filename)
    
        exp_params = exp_data.get_params()
    
        n_exp = exp_params._n_exp
        time_span = np.arange(exp_params._start_time, exp_params._end_time, exp_params._dt).flatten()
        nt = len(time_span)
        init_state = exp_params._init_state
        target_state = exp_params._target_state
    
        # # compute the costs and variances
        cost_pi_exp = np.zeros(n_exp)
        cost_ilqr_exp = np.zeros(n_exp)
        variances, lbdas = np.zeros((n_exp, nt-1)), np.zeros((n_exp, nt-1))
        
        for i in range(n_exp):
            trj_ilqr = exp_data.get_data(i).x_trj_ilqr()
            trj_pi = exp_data.get_data(i).x_trj_pi()
            u_star_pi = exp_data.get_data(i).u_trj_pi()
            u_trj_ilqr = exp_data.get_data(i).u_trj_ilqr()
            
            allPathCosts = exp_data.get_data(i).allPathCosts()
            
            for j in range(nt -1):
                variances[i, j], lbdas[i, j] = variance_usefulportion(allPathCosts[j], epsilon)
            
            cost_pi = exp_data.get_data(i).cost_pi()
            cost_ilqr = exp_data.get_data(i).cost_ilqr()
            
            cost_pi_exp[i] = cost_pi
            cost_ilqr_exp[i] = cost_ilqr
        
        v_costs_pi[i_eps] = cost_pi_exp
        v_costs_hilqr[i_eps] = cost_ilqr_exp
        
        print("E[cost_pi]: ", np.mean(cost_pi_exp))
        print("E[cost_ilqr_exp]: ", np.mean(cost_ilqr_exp))
        print("improved: ", (np.mean(cost_ilqr_exp) - np.mean(cost_pi_exp)) / np.mean(cost_ilqr_exp))
        
    # Plot the costs versus the number of samples
    fig1, ax1 = plt.subplots()
    
    avg_cost_pi = np.mean(v_costs_pi, axis=1)
    std_cost_pi = np.std(v_costs_pi, axis=1)
    
    avg_cost_hilqr = np.mean(v_costs_hilqr, axis=1)
    std_cost_hilqr = np.std(v_costs_hilqr, axis=1)
    
    # Plotting        
    ax1.grid(True)
    
    # Mean as a solid line
    ax1.plot(v_eps, avg_cost_pi, 'r-', label='avg cost path integral')
    ax1.plot(v_eps, avg_cost_hilqr, 'b-', label='avg cost h-ilqr')
    
    # Shaded area for variability (e.g., ±1 standard deviation)
    ax1.fill_between(v_eps, avg_cost_pi - std_cost_pi, avg_cost_pi + std_cost_pi, color='red', alpha=0.1, label='±1 Std. cost path integral')
    ax1.fill_between(v_eps, avg_cost_hilqr - std_cost_hilqr, avg_cost_hilqr + std_cost_hilqr, color='blue', alpha=0.1, label='±1 Std. cost h-ilqr')

    ax1.set_xlabel(r'Noise Intensity $\epsilon$')
    ax1.set_ylabel(r'Cost')
    ax1.legend()
    
    plt.show()
from example_bouncingball import *
from matplotlib.patches import Circle
import matplotlib.cbook as cbook 

if __name__ == '__main__':
    
    # ---------------------------------------
    # Setting font properties using fontdict
    # ---------------------------------------
    from matplotlib.font_manager import FontProperties
    font_props = FontProperties(family='serif', size=10, weight='normal')
    
    exp_params = ExpParams()
    exp_data = ExpData(exp_params)
    n_exp = 20

    # # ====================================
    # # Ablation study for number of samples
    # # ====================================
    # v_nsamples = [20, 30, 40, 50, 100, 150, 200, 300, 350, 400, 450,
    #               500, 550, 600, 700, 800, 900, 1000, 1200, 1500, 
    #               2000, 5000]
    
    # # v_nsamples = [20, 30, 40, 150, 250, 300, 400, 450,
    # #               550, 600]
    # eps = 2.0
    # v_costs_pi = np.zeros((len(v_nsamples), n_exp))
    # v_costs_hilqr = np.zeros((len(v_nsamples), n_exp))
    # for i_nsample, n_samples in enumerate(v_nsamples):
    #     filename = root_dir+f"/data/bouncing/data_{n_samples}samples_eps_{eps}.pickle"
    
    #     print("loading data: ", filename)
    #     exp_data.load(filename)
    #     print("loading data: ", filename)
    #     exp_data.load(filename)
    
    #     exp_params = exp_data.get_params()
    #     exp_params = exp_data.get_params()
    
    #     n_exp = exp_params._n_exp
    #     n_samples = exp_params._n_samples
    #     time_span = np.arange(exp_params._start_time, exp_params._end_time, exp_params._dt).flatten()
    #     nt = len(time_span)
    #     epsilon = exp_params._epsilon
    #     init_state = exp_params._init_state
    #     target_state = exp_params._target_state
    #     n_exp = exp_params._n_exp
    #     n_samples = exp_params._n_samples
    #     time_span = np.arange(exp_params._start_time, exp_params._end_time, exp_params._dt).flatten()
    #     nt = len(time_span)
    #     epsilon = exp_params._epsilon
    #     init_state = exp_params._init_state
    #     target_state = exp_params._target_state
    
    #     # # compute the costs and variances
    #     cost_pi_exp = np.zeros(n_exp)
    #     cost_ilqr_exp = np.zeros(n_exp)
    #     variances, lbdas = np.zeros((n_exp, nt-1)), np.zeros((n_exp, nt-1))
    #     # # compute the costs and variances
    #     cost_pi_exp = np.zeros(n_exp)
    #     cost_ilqr_exp = np.zeros(n_exp)
    #     variances, lbdas = np.zeros((n_exp, nt-1)), np.zeros((n_exp, nt-1))
        
    #     for i in range(n_exp):
    #         trj_ilqr = exp_data.get_data(i).x_trj_ilqr()
    #         trj_pi = exp_data.get_data(i).x_trj_pi()
    #         u_star_pi = exp_data.get_data(i).u_trj_pi()
    #         u_trj_ilqr = exp_data.get_data(i).u_trj_ilqr()
    #     for i in range(n_exp):
    #         trj_ilqr = exp_data.get_data(i).x_trj_ilqr()
    #         trj_pi = exp_data.get_data(i).x_trj_pi()
    #         u_star_pi = exp_data.get_data(i).u_trj_pi()
    #         u_trj_ilqr = exp_data.get_data(i).u_trj_ilqr()
            
    #         allPathCosts = exp_data.get_data(i).allPathCosts()
    #         allPathCosts = exp_data.get_data(i).allPathCosts()
            
    #         for j in range(nt -1):
    #             variances[i, j], lbdas[i, j] = variance_usefulportion(allPathCosts[j], epsilon)
    #         for j in range(nt -1):
    #             variances[i, j], lbdas[i, j] = variance_usefulportion(allPathCosts[j], epsilon)
            
    #         cost_pi = exp_data.get_data(i).cost_pi()
    #         cost_ilqr = exp_data.get_data(i).cost_ilqr()
    #         cost_pi = exp_data.get_data(i).cost_pi()
    #         cost_ilqr = exp_data.get_data(i).cost_ilqr()
            
    #         cost_pi_exp[i] = cost_pi
    #         cost_ilqr_exp[i] = cost_ilqr
    #         cost_pi_exp[i] = cost_pi
    #         cost_ilqr_exp[i] = cost_ilqr
        
    #     v_costs_pi[i_nsample] = cost_pi_exp
    #     v_costs_hilqr[i_nsample] = cost_ilqr_exp
    #     v_costs_pi[i_nsample] = cost_pi_exp
    #     v_costs_hilqr[i_nsample] = cost_ilqr_exp
        
    #     print("E[cost_pi]: ", np.mean(cost_pi_exp))
    #     print("E[cost_ilqr_exp]: ", np.mean(cost_ilqr_exp))
    #     print("improved: ", (np.mean(cost_ilqr_exp) - np.mean(cost_pi_exp)) / np.mean(cost_ilqr_exp))
    #     print("E[cost_pi]: ", np.mean(cost_pi_exp))
    #     print("E[cost_ilqr_exp]: ", np.mean(cost_ilqr_exp))
    #     print("improved: ", (np.mean(cost_ilqr_exp) - np.mean(cost_pi_exp)) / np.mean(cost_ilqr_exp))
        
    # # Plot the costs versus the number of samples
    # fig, ax = plt.subplots()
    # # Plot the costs versus the number of samples
    # fig, ax = plt.subplots()
    
    # avg_cost_pi = np.mean(v_costs_pi, axis=1)
    # std_cost_pi = np.std(v_costs_pi, axis=1)
    # avg_cost_pi = np.mean(v_costs_pi, axis=1)
    # std_cost_pi = np.std(v_costs_pi, axis=1)
    
    # avg_cost_hilqr = np.mean(v_costs_hilqr, axis=1)
    # std_cost_hilqr = np.std(v_costs_hilqr, axis=1)
    # avg_cost_hilqr = np.mean(v_costs_hilqr, axis=1)
    # std_cost_hilqr = np.std(v_costs_hilqr, axis=1)
    
    # # Plotting        
    # ax.grid(True)
    # # Plotting        
    # ax.grid(True)
    
    # # Mean as a solid line
    # ax.plot(v_nsamples, avg_cost_pi, 'r-', label='avg cost path integral')
    # ax.plot(v_nsamples, avg_cost_hilqr, 'b-', label='avg cost h-ilqr')
    # # Mean as a solid line
    # ax.plot(v_nsamples, avg_cost_pi, 'r-', label='avg cost path integral')
    # ax.plot(v_nsamples, avg_cost_hilqr, 'b-', label='avg cost h-ilqr')
    
    # # Shaded area for variability (e.g., ±1 standard deviation)
    # ax.fill_between(v_nsamples, avg_cost_pi - std_cost_pi, avg_cost_pi + std_cost_pi, color='red', alpha=0.1, label='±1 Std. cost path integral')
    # ax.fill_between(v_nsamples, avg_cost_hilqr - std_cost_hilqr, avg_cost_hilqr + std_cost_hilqr, color='blue', alpha=0.1, label='±1 Std. cost h-ilqr')
    # # Shaded area for variability (e.g., ±1 standard deviation)
    # ax.fill_between(v_nsamples, avg_cost_pi - std_cost_pi, avg_cost_pi + std_cost_pi, color='red', alpha=0.1, label='±1 Std. cost path integral')
    # ax.fill_between(v_nsamples, avg_cost_hilqr - std_cost_hilqr, avg_cost_hilqr + std_cost_hilqr, color='blue', alpha=0.1, label='±1 Std. cost h-ilqr')

    # ax.set_xlabel(r'Number of samples')
    # ax.set_ylabel(r'Cost')
    # ax.legend()
    # ax.set_xlabel(r'Number of samples')
    # ax.set_ylabel(r'Cost')
    # ax.legend()
    
    # fig.savefig(root_dir+f"/data/bouncing/ablation_nsamples.pdf", dpi=1100)
    
    # plt.show()
    
    
    # ============================
    # Ablation study for epsilon
    # ==========================
    n_exp = 100
    v_eps = [0.1, 0.5, 1.0, 1.5,
            2.0, 2.5, 3.0, 3.5, 
            4.0, 4.5, 5.0, 5.5,
            6.0, 6.5, 7.0, 7.5, 
            8.0, 8.5, 9.0, 9.5,
            10.0, 10.5, 11.0, 11.5, 
            12.0, 12.5, 13.0, 13.5,
            14.0, 14.5, 15.0, 15.5, 
            16.0, 16.5, 17.0, 17.5, 
            18.0, 18.5, 19.0, 19.5, 20.0]
    n_samples = 5000
    v_costs_pi = np.zeros((len(v_eps), n_exp))
    v_costs_hilqr = np.zeros((len(v_eps), n_exp))
    for i_eps, epsilon in enumerate(v_eps):
        filename = root_dir+f"/data/bouncing/data_{n_samples}samples_eps_{epsilon}_coupling.pickle"
    
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
            
    # ============================= Statistics ==============================
    avg_cost_pi = np.mean(v_costs_pi, axis=1)
    std_cost_pi = np.std(v_costs_pi, axis=1)
    
    avg_cost_hilqr = np.mean(v_costs_hilqr, axis=1)
    std_cost_hilqr = np.std(v_costs_hilqr, axis=1)
    
    # difference 
    v_costs_diff = (v_costs_hilqr - v_costs_pi) / v_costs_hilqr * 100.0
    avg_cost_diff = np.mean(v_costs_diff, axis=1)
    std_cost_diff = np.std(v_costs_diff, axis=1)
    
    # ============================= Plotting =============================       
    fig1, axes = plt.subplots(2,1, figsize=(10, 7))
    ax1, ax2 = axes.flatten()
    ax1.grid(True)
    ax2.grid(True)
    
    # # Mean as a solid line
    ax1.plot(v_eps, avg_cost_pi, 'r-', label='avg cost H-PI')
    ax1.plot(v_eps, avg_cost_hilqr, 'b-', label='avg cost H-iLQR')
    
    # Shaded area for variability (e.g., ±1 standard deviation)
    ax1.fill_between(v_eps, avg_cost_pi - std_cost_pi, avg_cost_pi + std_cost_pi, color='red', alpha=0.1, label='±1 Std.')
    ax1.fill_between(v_eps, avg_cost_hilqr - std_cost_hilqr, avg_cost_hilqr + std_cost_hilqr, color='blue', alpha=0.1, label='±1 Std.')

    ax1.legend(loc='upper left')

    ax1.set_title(r'Cost Mean and Std. Statistics', fontproperties=font_props)
    ax1.set_xlabel(r'$\epsilon$', fontproperties=font_props)
    ax1.set_ylabel(r'Cost', fontproperties=font_props)

    # ========= 
    # box plot
    # =========
    stats = []
    for dataset in v_costs_diff[::2]:
        quartiles = np.percentile(dataset, [25, 50, 75])
        whiskers = np.percentile(dataset, [5, 95])  # using percentiles for whiskers
        fliers = dataset[(dataset < whiskers[0]) | (dataset > whiskers[1])]
        
        stats.append({
            'med': quartiles[1],
            'q1': quartiles[0],
            'q3': quartiles[2],
            'whislo': whiskers[0],
            'whishi': whiskers[1],
            'fliers': fliers,
            'mean': np.mean(dataset)  # Optionally add mean
        })
    
    
    boxprops=dict(linewidth=1.5, color='black')
    flierprops=dict(marker='.', markerfacecolor='blue', markersize=5,
                                        linestyle='none', markeredgecolor='blue')
    meanprops=dict(linestyle='-', linewidth=1.5, color='green')
    medianprops=dict(linewidth=1.5, color='red')
    
    bxpstats = ax2.bxp(stats, vert=True, showmeans=True, meanline=True, meanprops=meanprops, showfliers=False, 
                       flierprops=flierprops, medianprops=medianprops, boxprops=boxprops)  
    
    ax2.set_title('Cost Improvement Statistics', fontproperties=font_props)
    ax2.set_xlabel(r'$\epsilon$', fontproperties=font_props)
    ax2.set_ylabel(r'$\frac{J_{\rm HiLQR} - J_{HPI}}{J_{\rm HiLQR}} (\%)$', fontproperties=font_props)

    # Customizing x-axis labels to show dataset numbers
    ax2.set_xticklabels([f'{eps}' for eps in v_eps[::2]])
    
    # mean and median line legned
    import matplotlib.lines as mlines
    mean_line = mlines.Line2D([], [], color='green', label='Mean', linestyle='-', linewidth=1.5)
    median_line = mlines.Line2D([], [], color='red', label='Median', linestyle='-', linewidth=1.5)

    ax2.legend(handles=[mean_line, median_line], loc='best')
    
    fig1.tight_layout()
    
    fig1.savefig(root_dir+f"/data/bouncing/ablation_epsilon_coupling.pdf", dpi=1100)
    
    plt.show()
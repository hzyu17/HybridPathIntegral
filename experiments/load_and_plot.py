from example_bouncingball import *
from matplotlib.patches import Circle

def compute_weights(PathCosts, epsilon):
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
    # pathcosts = np.exp(-(pathcosts - np.min(pathcosts))/epsilon)
    # weights = pathcosts / np.mean(pathcosts)
    
    # mean = np.mean(weights)
    # variance = np.sum((weights-mean)*(weights-mean)) / len(weights)
    
    weights = compute_weights(pathcosts, epsilon)
    
    variance = np.var(weights)

    # ------- Fraction of effective samples -------
    lbda = 1.0 / np.mean(weights**2)
    
    print("variance", variance)
    print("lbda", lbda)
    
    return variance, lbda*100.0

if __name__ == '__main__':
    
    exp_params = ExpParams()
    exp_data = ExpData(exp_params)

    filename = root_dir+"/data/bouncing/data_2024-04-12_12-41-43_hybrid_riccati.pickle"
    
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
    
    if exp_data.get_nominal_data():
        (states,inputs,k_feedforward,K_feedback,current_cost,states_iter,_,_) = exp_data.get_nominal_data()
    else:
        (states,inputs,k_feedforward,K_feedback,current_cost,states_iter,_,_) = solve_ilqr(exp_params)

    print("plotting")
    fig1, axes = plt.subplots(1, 2, figsize=(10, 8))
    (ax1, ax2) = axes.flatten()
    ax1.grid(True)
    ax2.grid(True)
    
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
        
        # dWs_zeros = np.zeros((nt, n_inputs))
        # cost_pi = compute_cost(trj_pi, u_star_pi, dWs_zeros, target_state, states, Q_k, R_k, Q_T, epsilon)
        # cost_ilqr = compute_cost(trj_ilqr, u_trj_ilqr, dWs_zeros, target_state, states, Q_k, R_k, Q_T, epsilon)
        
        # print("cost_pi:", cost_pi)
        # print("cost_ilqr:", cost_ilqr)
        
        cost_pi_exp[i] = cost_pi
        cost_ilqr_exp[i] = cost_ilqr
        
    print("E[cost_pi]: ", np.mean(cost_pi_exp))
    print("E[cost_ilqr_exp]: ", np.mean(cost_ilqr_exp))
    
    # # ----------- plot the path integral controlled trajectory -----------
    for i in range(n_exp):
        trj_ilqr = exp_data.get_data(i).x_trj_ilqr()
        trj_pi = exp_data.get_data(i).x_trj_pi()
        
        ax1.plot(time_span, trj_ilqr[:, 0], 'b', linewidth=0.8, alpha=0.2)
        ax2.plot(time_span, trj_ilqr[:, 1], 'b', linewidth=0.8, alpha=0.2)
        
        ax1.plot(time_span, trj_pi[:, 0], 'r', linewidth=0.8, alpha=0.6)
        ax2.plot(time_span, trj_pi[:, 1], 'r', linewidth=0.8, alpha=0.6)
        
    ax1.plot(time_span, trj_ilqr[:, 0], 'b', linewidth=0.8, alpha=0.2, label='iLQR')
    ax2.plot(time_span, trj_ilqr[:, 1], 'b', linewidth=0.8, alpha=0.2, label='iLQR')

    ax1.plot(time_span, trj_pi[:, 0], 'r', linewidth=0.8, alpha=0.6, label='Path Integral')
    ax2.plot(time_span, trj_pi[:, 1], 'r', linewidth=0.8, alpha=0.6, label='Path Integral')

    # ----------- Plot the reference -----------
    ax1.plot(time_span, states[:,0],'k',label='iLQR-deterministic')
    ax2.plot(time_span, states[:,1],'k',label='iLQR-deterministic')

    # ----------- Plot the start and goal states -----------
    ax1.scatter(time_span[-1], target_state[0], color='g', marker='o', s=50.0, linewidths=6, label='Target', zorder=2)
    ax1.scatter(time_span[0], init_state[0], color='r', marker='o', s=50.0, linewidths=6, label='Start', zorder=2)

    ax2.scatter(time_span[-1], target_state[1], color='g', marker='o', s=50.0, linewidths=6, label='Target', zorder=2)
    ax2.scatter(time_span[0], init_state[1], color='r', marker='o', s=50.0, linewidths=6, label='Start', zorder=2)
    
    ax1.set_xlabel(r"Time", fontsize=18)
    ax1.set_ylabel(r"$z$", fontsize=18)
    ax1.set_title("Vertical Position", fontsize=18)
    plt.tight_layout()

    ax2.set_xlabel(r"Time", fontsize=18)
    ax2.set_ylabel(r"$\dot z$", fontsize=18)
    ax2.set_title("Vertical Velocity", fontsize=18)
    plt.tight_layout()
    
    ax1.legend()
    ax2.legend()

    # save figures
    fig1.savefig(root_dir+'/data/figures/bouncing/bouncing_1D.pdf', format='pdf', dpi=2000)
    
    # =========== Plot the z-\dot_z figure ===========
    fig2, ax5 = plt.subplots(figsize=(10, 8))
    ax5.grid(True)

    # ----------- Plot the last iteration of iLQR controller ----------
    for i in range(n_exp):
        trj_ilqr = exp_data.get_data(i).x_trj_ilqr()
        trj_pi = exp_data.get_data(i).x_trj_pi()
        ax5.plot(trj_ilqr[:, 0], trj_ilqr[:, 1], 'b', linewidth=0.8, alpha=0.2)
        ax5.plot(trj_pi[:, 0], trj_pi[:, 1], 'r', linewidth=0.8, alpha=0.6)

    ax5.plot(trj_ilqr[:, 0], trj_ilqr[:, 1], 'b', linewidth=0.8, alpha=0.2, label='iLQR')
    ax5.plot(trj_pi[:, 0], trj_pi[:, 1], 'r', linewidth=0.8, alpha=0.6, label='Path Integral')
    ax5.plot(states[:,0], states[:,1],'k',label='iLQR-deterministic')
    
    # ----------- Plot the start and goal states -----------
    ax5.scatter(target_state[0], target_state[1], color='g', marker='o', s=50.0, linewidths=6, label='Target', zorder=2)
    ax5.scatter(init_state[0], init_state[1], color='r', marker='o', s=50.0, linewidths=6, label='Start', zorder=2)

    ax5.set_xlabel(r"z", fontsize=18)
    ax5.set_ylabel(r"$\dot z$", fontsize=18)
    ax5.set_title("Controlled Bouncing Ball Dynamics", fontsize=18)
    ax5.legend()
    fig2.tight_layout()
    fig2.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_zdotz.pdf', format='pdf', dpi=2000)

    # # plot control inputs
    # fig3, ax7 = plt.subplots(1, 1)
    # ax7.grid(True)
    # ax7.plot(time_span, inputs[:,0],'k',label='Final iteration ilqr')
    # ax7.plot(time_span, u_star_pi[:,0],'r',label='Path integral controller')
    # ax7.set_xlabel(r"Timestep")
    # ax7.set_ylabel(r"$u$")
    # ax7.set_title("Bouncing Ball Final Control Input")

    # ax7.legend()

    # plot PathCosts
    fig3, ax8 = plt.subplots(figsize=(8,6))
    ax8.grid(True)
    ax8.bar(range(n_exp), cost_ilqr_exp, width = 2, color='navy', alpha=0.1, label='Hybrid iLQR')
    ax8.bar(range(n_exp), cost_pi_exp, width = 2, color='red', alpha=0.5, label='Hybrid Path Integral Control')

    ax8.set_xlabel(r"Experiment ID", fontsize=14)
    ax8.set_ylabel(r"$Costs$", fontsize=14)
    ax8.legend()

    fig3.tight_layout()
    fig3.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_costs.pdf', format='pdf', dpi=2000)

    #--- plot the variances and useful portion
    # Calculating the mean and standard deviation along the repetitions
    avg_variances = np.mean(variances, axis=0)
    std_variances = np.std(avg_variances, axis=0)
    
    avg_lbdas = np.mean(lbdas, axis=0)
    std_lbdas = np.std(avg_lbdas, axis=0)
    
    # Plotting
    fig4, ax9 = plt.subplots(figsize=(8,6))
    fig5, ax10 = plt.subplots(figsize=(8,6))
    
    ax9.grid(True)
    ax10.grid(True)
    
    # Mean as a solid line
    ax9.plot(time_span[:-1], avg_variances, 'r-', label='Variance')
    ax10.plot(time_span[:-1], avg_lbdas, 'r-', label=r'$\lambda(\%)$')
    
    # Shaded area for variability (e.g., ±1 standard deviation)
    ax9.fill_between(time_span[:-1], avg_variances - std_variances, avg_variances + std_variances, color='gray', alpha=0.5, label='±1 Std.')
    ax10.fill_between(time_span[:-1], avg_lbdas - std_lbdas, avg_lbdas + std_lbdas, color='gray', alpha=0.5, label='±1 Std.')

    # ax9.set_title('Weight Variance')
    ax9.set_xlabel('Time', fontsize=14)
    ax9.set_ylabel(r'Var($\alpha$)', fontsize=14)
    fig4.tight_layout()
    
    # ax10.set_title('Effective Weights')
    ax10.set_xlabel('Time', fontsize=14)
    ax10.set_ylabel(r'$\lambda^u (\%)$', fontsize=14)
    ax10.set_ylim(0, 110)
    fig5.tight_layout()
    
    ax9.legend()
    ax10.legend()
    
    fig4.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_var.pdf', format='pdf', dpi=2000)
    fig5.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_lbda.pdf', format='pdf', dpi=2000)
    
    # plot step one cost distribution
    fig6, ax11 = plt.subplots(figsize=(8,6))
    ax11.grid(True)
    ax11.bar(range(allPathCosts.shape[1]), allPathCosts[500])
    ax11.set_title("Path Cost distribution")
    ax11.set_xlabel("Sample Number")
    ax11.set_ylabel("Costs")
    plt.show()
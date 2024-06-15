from example_bouncingball import *
from matplotlib.patches import Circle


if __name__ == '__main__':
    
    exp_params = ExpParams()
    exp_data = ExpData(exp_params)

    filename = root_dir+"/data/bouncing/data_5000samples_eps_15.0_coupling.pickle"
    
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
        (states,inputs,k_feedforward,K_feedback,current_cost,states_iter) = exp_data.get_nominal_data()
    else:
        (states,inputs,k_feedforward,K_feedback,current_cost,states_iter,_,_) = solve_ilqr(exp_params)

    print("===================== plotting =====================")
    fig1, axes = plt.subplots(1, 2, figsize=(10, 8))
    (ax1, ax2) = axes.flatten()
    ax1.grid(True)
    ax2.grid(True)
    
    # # compute the costs and variances
    cost_pi_exp = np.zeros(n_exp)
    cost_ilqr_exp = np.zeros(n_exp)
    variances, lbdas = np.zeros((n_exp, nt-1)), np.zeros((n_exp, nt-1))
    
    for i in range(n_exp):
        print(exp_data._data.keys())
        trj_ilqr = exp_data.get_data(i).x_trj_ilqr()
        trj_pi = exp_data.get_data(i).x_trj_pi()
        u_star_pi = exp_data.get_data(i).u_trj_pi()
        u_trj_ilqr = exp_data.get_data(i).u_trj_ilqr()
        
        allPathCosts = exp_data.get_data(i).allPathCosts()
        
        for j in range(nt -1):
            variances[i, j], lbdas[i, j] = variance_usefulportion(allPathCosts[j], epsilon)
        
        cost_pi = exp_data.get_data(i).cost_pi()
        cost_ilqr = exp_data.get_data(i).cost_ilqr()
        
        # print("cost_pi:", cost_pi)
        # print("cost_ilqr:", cost_ilqr)
        
        cost_pi_exp[i] = cost_pi
        cost_ilqr_exp[i] = cost_ilqr
        
    print("E[cost_pi]: ", np.mean(cost_pi_exp))
    print("E[cost_ilqr_exp]: ", np.mean(cost_ilqr_exp))
    print("improved: ", (np.mean(cost_ilqr_exp) - np.mean(cost_pi_exp)) / np.mean(cost_ilqr_exp))
    
    
    # ==============================================
    #                   Plottings
    # ============================================== 
    from matplotlib.font_manager import FontProperties
    # ---------------------------------------
    # Setting font properties using fontdict
    # ---------------------------------------
    font_props = FontProperties(family='serif', size=18, weight='normal')
    
    
    # --------------------------------------------
    # plot the path integral controlled trajectory
    # -------------------------------------------- 
    for i in range(n_exp):
        trj_ilqr = exp_data.get_data(i).x_trj_ilqr()
        trj_pi = exp_data.get_data(i).x_trj_pi()
        
        ax1.plot(time_span, trj_ilqr[:, 0], 'b', linewidth=0.8, alpha=0.2)
        ax2.plot(time_span, trj_ilqr[:, 1], 'b', linewidth=0.8, alpha=0.2)
        
        ax1.plot(time_span, trj_pi[:, 0], 'r', linewidth=0.8, alpha=0.6)
        ax2.plot(time_span, trj_pi[:, 1], 'r', linewidth=0.8, alpha=0.6)
        
    ax1.plot(time_span, trj_ilqr[:, 0], 'b', linewidth=0.8, alpha=0.2, label='H-iLQR')
    ax2.plot(time_span, trj_ilqr[:, 1], 'b', linewidth=0.8, alpha=0.2, label='H-iLQR')

    ax1.plot(time_span, trj_pi[:, 0], 'r', linewidth=0.8, alpha=0.6, label='H-PathIntegral')
    ax2.plot(time_span, trj_pi[:, 1], 'r', linewidth=0.8, alpha=0.6, label='H-PathIntegral')

    # ----------- Plot the reference -----------
    ax1.plot(time_span, states[:,0],'k',label='H-iLQR reference')
    ax2.plot(time_span, states[:,1],'k',label='H-iLQR reference')

    # ----------- Plot the start and goal states -----------
    ax1.scatter(time_span[-1], target_state[0], color='g', marker='o', s=50.0, linewidths=6, label='Target', zorder=2)
    ax1.scatter(time_span[0], init_state[0], color='r', marker='o', s=50.0, linewidths=6, label='Start', zorder=2)

    ax2.scatter(time_span[-1], target_state[1], color='g', marker='o', s=50.0, linewidths=6, label='Target', zorder=2)
    ax2.scatter(time_span[0], init_state[1], color='r', marker='o', s=50.0, linewidths=6, label='Start', zorder=2)
    
    ax1.set_xlabel(r"Time", fontproperties=font_props)
    ax1.set_ylabel(r"$z$", fontproperties=font_props)
    ax1.set_title("Vertical Position", fontproperties=font_props)
    plt.tight_layout()

    ax2.set_xlabel(r"Time", fontproperties=font_props)
    ax2.set_ylabel(r"$\dot z$", fontproperties=font_props)
    ax2.set_title("Vertical Velocity", fontproperties=font_props)
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

    ax5.plot(trj_ilqr[:, 0], trj_ilqr[:, 1], 'b', linewidth=0.8, alpha=0.2, label='H-iLQR')
    ax5.plot(trj_pi[:, 0], trj_pi[:, 1], 'r', linewidth=0.8, alpha=0.6, label='H-PathIntegral')
    ax5.plot(states[:,0], states[:,1],'k',label='H-iLQR')
    
    # ----------- Plot the start and goal states -----------
    ax5.scatter(target_state[0], target_state[1], color='g', marker='o', s=50.0, linewidths=6, label='Target', zorder=2)
    ax5.scatter(init_state[0], init_state[1], color='r', marker='o', s=50.0, linewidths=6, label='Start', zorder=2)

    ax5.set_xlabel(r"$z$", fontproperties=font_props)
    ax5.set_ylabel(r"$\dot z$", fontproperties=font_props)
    ax5.set_title("Controlled Bouncing Ball Dynamics", fontproperties=font_props)
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

    # ------------------------------------------ 
    # Bar Plot PathCosts for all experiments
    # ------------------------------------------
    fig3, ax8 = plt.subplots(figsize=(18,6))
    ax8.grid(True)
    
    # Set the bar width
    bar_width = 0.35
    # Set the opacity
    opacity = 0.8

    index = np.arange(n_exp)

    bars1 = ax8.bar(index, cost_ilqr_exp, bar_width, alpha=opacity, color='b', label='Hybrid iLQR')
    bars2 = ax8.bar(index + bar_width, cost_pi_exp, bar_width, alpha=opacity, color='r', label='Hybrid Path Integral')
    # bars2 = ax8.bar(index + bar_width, cost_pi_exp, bar_width, alpha=opacity, color='r', label='Hybrid Path Integral')

    # Add some labels, title and axes ticks
    ax8.set_xlabel(r"Experiment ID", fontproperties=font_props)
    ax8.set_ylabel(r"$Costs$", fontproperties=font_props)
    ax8.set_title('Comparison of Cost Statistics', fontproperties=font_props)
    ax8.set_xticks(index + bar_width / 2)  # Positioning the x-axis ticks in the middle of the two bars
    ax8.set_xticklabels(index)

    # Adding a legend
    ax8.legend()

    fig3.tight_layout()
    fig3.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_costs.pdf', dpi=2000)

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
    ax9.plot(time_span[:-1], avg_variances, 'r-', label='Weight Distribution Variance')
    ax10.plot(time_span[:-1], avg_lbdas, 'r-', label=r'Effective Samples $\lambda (\%)$')
    
    # Shaded area for variability (e.g., ±1 standard deviation)
    ax9.fill_between(time_span[:-1], avg_variances - std_variances, avg_variances + std_variances, color='gray', alpha=0.5, label='±1 Std. across experiments')
    ax10.fill_between(time_span[:-1], avg_lbdas - std_lbdas, avg_lbdas + std_lbdas, color='gray', alpha=0.5, label='±1 Std. across experiments')

    # ax9.set_title('Weight Variance')
    ax9.set_xlabel('Time', fontproperties=font_props)
    ax9.set_ylabel(r'Var($\alpha$)', fontproperties=font_props)
    fig4.tight_layout()
    
    # ax10.set_title('Effective Weights')
    ax10.set_xlabel('Time', fontproperties=font_props)
    ax10.set_ylabel(r'$\lambda^u (\%)$', fontproperties=font_props)
    ax10.set_ylim(0, 110)
    fig5.tight_layout()
    
    ax9.legend()
    ax10.legend()
    
    fig4.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_var.pdf', format='pdf', dpi=2000)
    fig5.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_lbda.pdf', format='pdf', dpi=2000)
    
    # plot step one cost distribution
    fig6, ax11 = plt.subplots(figsize=(8,6))
    ax11.grid(True)
    ax11.bar(range(allPathCosts.shape[1]), allPathCosts[-1])
    ax11.set_title("Path Cost distribution")
    ax11.set_xlabel("Sample Number", fontproperties=font_props)
    ax11.set_ylabel("Costs", fontproperties=font_props)
    
    # plt.show()
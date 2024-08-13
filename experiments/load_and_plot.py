from example_bouncingball import *
from matplotlib.font_manager import FontProperties
font_props = FontProperties(family='serif', size=12, weight='normal')


def compute_cvar(samples, confidence_level=0.95):
    # Step 1: Sort the samples
    samples_sorted = np.sort(samples)
    
    # Step 2: Find the VaR (confidence_level percentile)
    var_index = int(np.floor(confidence_level * len(samples_sorted)))
    var_value = samples_sorted[var_index]
    
    # Step 3: Calculate CVaR as the mean of the losses exceeding the VaR
    cvar_value = np.mean(samples_sorted[var_index:])
    
    return var_value, cvar_value

if __name__ == '__main__':
    
    exp_params = ExpParams()
    exp_data = ExpData(exp_params)

    # filename = root_dir+"/data/bouncing/ablation_study_nsamples/data_5000samples_eps_15.0_coupling.pickle"
    filename = root_dir+"/experiments/data/new_exp/bouncing/data_2024-08-07_18-46-25_example_bouncingball_jax_threading_5000samples_eps_10.0_coupling.pickle"
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
        (modes,states,inputs, 
         k_feedforward, K_feedback, current_cost, 
         states_iter, ref_modechanges,
         reference_extension_helper, ref_reset_args) = exp_data.get_nominal_data()
    else:
        (modes,states,inputs, 
         k_feedforward, K_feedback, current_cost, 
         states_iter, ref_modechanges,
         reference_extension_helper, ref_reset_args) = solve_ilqr(exp_params)
        
    plotting_function = exp_data.get_plotting_function()

    print("===================== plotting =====================")
    fig1, axes = plt.subplots(1, 2, figsize=(10, 8))
    (ax1, ax2) = axes.flatten()
    ax1.grid(True)
    ax2.grid(True)
    
    # # compute the costs and variances
    cost_pi_exp = np.zeros(n_exp)
    cost_ilqr_exp = np.zeros(n_exp)
    
    for i in range(n_exp):
        cost_pi = exp_data.get_data(i).cost_pi()
        cost_ilqr = exp_data.get_data(i).cost_ilqr()
        
        cost_pi_exp[i] = cost_pi
        cost_ilqr_exp[i] = cost_ilqr
        
    print("E[cost_pi]: ", np.mean(cost_pi_exp))
    print("E[cost_ilqr_exp]: ", np.mean(cost_ilqr_exp))
    print("improved: ", (np.mean(cost_ilqr_exp) - np.mean(cost_pi_exp)) / np.mean(cost_ilqr_exp))
    
    # ================================
    #  Sorting the cost improvements
    # ================================
    cost_diff = (cost_ilqr_exp - cost_pi_exp) / cost_ilqr_exp * 100
    sorted_indices = [index for index, value in sorted(enumerate(cost_diff), key=lambda x: x[1])]
    sorted_cost_diff = sorted(cost_diff)
    
    # Check the costs of the 10% tail of h-iLQR costs
    sorted_cost_ilqr_indices = [index for index, value in sorted(enumerate(cost_ilqr_exp), key=lambda x: x[1])]
    sorted_costilqr = sorted(cost_ilqr_exp)
    ilqr_tail_index = sorted_cost_ilqr_indices[int(np.floor(0.9 * len(sorted_cost_ilqr_indices))):]
    ilqr_best_index = sorted_cost_ilqr_indices[:int(np.floor(0.1 * len(sorted_cost_ilqr_indices)))]
    
    # ------------------------------------------------------------------
    #   Compute CVaR: check the cases where h-iLQR do not perform well
    # ------------------------------------------------------------------    
    for confidence in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        cVaR = compute_cvar(cost_diff, confidence)[1]
        print("cVaR at confidence level: ", confidence, " is ", cVaR)   
    
    cost_diff_tail = (cost_ilqr_exp[ilqr_tail_index] - cost_pi_exp[ilqr_tail_index]) / cost_ilqr_exp[ilqr_tail_index]
    mean_cost_diff_tail = np.mean(cost_diff_tail) * 100.0
    
    print("mean hilqg tail cost: ", np.mean(cost_ilqr_exp[ilqr_tail_index]))
    print("mean h pathintegral tail cost: ", np.mean(cost_pi_exp[ilqr_tail_index]))
    print("mean_cost_diff_tail: ", mean_cost_diff_tail)
    
    # ==============================================
    #                   Plottings
    # ============================================== 
    from matplotlib.font_manager import FontProperties
    font_props = FontProperties(family='serif', size=16, weight='normal')
    
    # ------------------------------------------------
    #   Plot the path integral controlled trajectory
    # ------------------------------------------------ 
    
    # plot the best 10%
    fig1, axes_12 = plt.subplots(1, 2, figsize=(10,6))
    fig2, ax3 = plt.subplots(figsize=(8,8))
    args = (fig1, axes_12, fig2, ax3)
    fig1, axes_12, fig2, ax3 =  plot_bouncingball_nexp(ilqr_best_index, exp_data, time_span, init_state, 
                                                        target_state, args=args)
    
    fig1.tight_layout()
    fig2.tight_layout()
        
    # plot the tail 10%
    fig3, axes_34 = plt.subplots(1, 2, figsize=(10,6))
    fig4, ax5 = plt.subplots(figsize=(8,8))
    args = (fig3, axes_34, fig4, ax5)
    fig3, axes_34, fig4, ax5 =  plot_bouncingball_nexp(ilqr_tail_index, exp_data, time_span, init_state, 
                                                        target_state, args=args)
    
    fig3.tight_layout()
    fig4.tight_layout()
    
    plt.show()
    
    # save figures
    # fig1.savefig(root_dir+'/data/figures/bouncing/bouncing_1D.pdf', format='pdf', dpi=2000)
    fig2.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_zdotz_best10.pdf', format='pdf', dpi=2000)
    fig4.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_zdotz_tail10.pdf', format='pdf', dpi=2000)

    # ------------------------------------------ 
    #   Bar Plot PathCosts for all experiments
    # ------------------------------------------
    fig3, ax8 = plt.subplots(figsize=(18,6))
    ax8.grid(True)
    
    # Set the bar width
    bar_width = 0.35
    # Set the opacity
    opacity = 0.
    index = np.arange(n_exp)

    bars1 = ax8.bar(index, cost_ilqr_exp, bar_width, alpha=opacity, color='b', label='Hybrid iLQG')
    bars2 = ax8.bar(index + bar_width, cost_pi_exp, bar_width, alpha=opacity, color='r', label='Hybrid Path Integral')

    # Add some labels, title and axes ticks
    ax8.set_xlabel(r"Experiment ID", fontproperties=font_props)
    ax8.set_ylabel(r"$Costs$", fontproperties=font_props)
    ax8.set_title('Comparison of Cost Statistics', fontproperties=font_props)
    ax8.set_xticks(index + bar_width / 2)  # Positioning the x-axis ticks in the middle of the two bars
    ax8.set_xticklabels(index)

    # Adding a legend
    ax8.legend(loc='best', prop={'family': 'serif', 'size': 12})

    fig3.tight_layout()
    fig3.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_costs.pdf', dpi=2000)

    #-----------------------------------------
    #  Plot the variances and useful portion
    #-----------------------------------------
    avg_variances, std_variances, avg_lbdas, std_lbdas = compute_var_lbd_nexp(n_exp, nt, exp_data)
    
    # Plotting
    fig4, ax9 = plt.subplots(figsize=(8,6))
    fig5, ax10 = plt.subplots(figsize=(8,6))
    
    ax9.grid(True)
    ax10.grid(True)
    
    # Mean as a solid line
    ax9.plot(time_span[:-1], avg_variances, 'r-', label=r'Weight Distribution Variance')
    ax9.fill_between(time_span[:-1], avg_variances - std_variances, avg_variances + std_variances, color='gray', alpha=0.5, label='±1 Std. across experiments')
    
    # Shaded area for variability (e.g., ±1 standard deviation)
    ax10.plot(time_span[:-1], avg_lbdas, 'r-', label=r'Effective Samples $(\%)$')
    ax10.fill_between(time_span[:-1], avg_lbdas - std_lbdas, avg_lbdas + std_lbdas, color='gray', alpha=0.5, label='±1 Std. across experiments')

    # ax9.set_title('Weight Variance')
    ax9.set_xlabel(r'Time', fontproperties=font_props)
    ax9.set_ylabel(r'Var (t)', fontproperties=font_props)
    fig4.tight_layout()
    
    # ax10.set_title('Effective Weights')
    ax10.set_xlabel(r'Time', fontproperties=font_props)
    ax10.set_ylabel(r'$\lambda$ (t) (\%)$', fontproperties=font_props)
    ax10.set_ylim(0, 110)
    fig5.tight_layout()
    
    ax9.legend(loc='best', prop={'family': 'serif', 'size': 12})
    ax10.legend(loc='best', prop={'family': 'serif', 'size': 12})
    
    fig4.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_var.pdf', format='pdf', dpi=2000)
    fig5.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_lbda.pdf', format='pdf', dpi=2000)
    
    # # Plot step one cost distribution
    # fig6, ax11 = plt.subplots(figsize=(8,6))
    # ax11.grid(True)
    # ax11.bar(range(allPathCosts.shape[1]), allPathCosts[-1])
    # ax11.set_title("Path Cost distribution")
    # ax11.set_xlabel("Sample Number", fontproperties=font_props)
    # ax11.set_ylabel("Costs", fontproperties=font_props)
    
    
    # # ----------------------------------------------------
    # # Plot the samples for the i-lqr tail performances
    # # ----------------------------------------------------
    # fig7, ax12 = plt.subplots(figsize=(8,6))
    # highcost_index = sorted_cost_ilqr_indices[-1]
    # highcost_Ksamples = exp_data.get_data(highcost_index).all_samples()
    
    # fig3, ax6 = plt.subplots()
    # ax6.grid(True)
    # for i_s in range(n_samples):
    #     ax6.plot(highcost_Ksamples[0, i_s,:,0], highcost_Ksamples[i_s,:,1],'b', alpha=0.2)
    
    # ax6.scatter(target_state[0], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
    # ax6.scatter(init_state[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')
    
    plt.show()
    
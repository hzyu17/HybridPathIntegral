import numpy as np

import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from hybrid_pathintegral.hybrid_pathintegral import *
from hybrid_ilqr.h_ilqr_bouncingball import *
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
    filename = root_dir+"/experiments/data/new_exp/bouncing/data_2024-08-23_11-05-08_h_PI_bouncingball_jax_threading_zero_control_2000samples_eps_5.0_coupling_zero_control.pickle"
    print("loading data: ", filename)
    exp_data.load(filename)
    
    exp_params = exp_data.get_params()
    
    dt = exp_params._dt
    n_exp = exp_params._n_exp
    n_samples = exp_params._n_samples
    time_span = np.arange(exp_params._start_time, exp_params._end_time, exp_params._dt).flatten()
    nt = len(time_span)
    epsilon = exp_params._epsilon
    init_state = exp_params._init_state
    target_state = exp_params._target_state
    
    if exp_data.get_nominal_data():
        (timespan, modes,states,inputs, 
         k_ff, K_fb, current_cost, 
         states_iter, ref_modechanges,
         ref_extension_helper, ref_reset_args) = exp_data.get_nominal_data()
    else:
        (timespan, modes,states,inputs, 
         k_ff, K_fb, current_cost, 
         states_iter, ref_modechanges,
         ref_extension_helper, ref_reset_args) = solve_ilqr(exp_params)
        
    plotting_function = exp_data.get_plotting_function()

    
    # ====================================
    #   Compute the costs and variances
    # ====================================
    v_cost_pi = np.zeros(n_exp)
    v_cost_ilqr = np.zeros(n_exp)
    
    for i in range(n_exp):
        cost_pi = exp_data.get_data(i).cost_pi()
        cost_ilqr = exp_data.get_data(i).cost_ilqr()
        
        v_cost_pi[i] = cost_pi
        v_cost_ilqr[i] = cost_ilqr
        
    print("E[v_cost_pi]: ", np.mean(v_cost_pi))
    print("E[v_cost_ilqr]: ", np.mean(v_cost_ilqr))
    print("improved: ", (np.mean(v_cost_ilqr) - np.mean(v_cost_pi)) / np.mean(v_cost_ilqr))
    
    # ================================
    #  Sorting the cost improvements
    # ================================
    cost_diff = (v_cost_ilqr - v_cost_pi) / v_cost_ilqr * 100
    sorted_indices = [index for index, _ in sorted(enumerate(cost_diff), key=lambda x: x[1])]
    sorted_cost_diff = sorted(cost_diff)
    
    # Check the costs of the 10% tail of h-iLQR costs
    sorted_indx_costilqr = [index for index, _ in sorted(enumerate(v_cost_ilqr), key=lambda x: x[1])]
    sorted_costilqr = sorted(v_cost_ilqr)
    ilqr_tail_index = sorted_indx_costilqr[int(np.floor(0.9 * len(sorted_indx_costilqr))):]
    ilqr_best_index = sorted_indx_costilqr[:int(np.floor(0.1 * len(sorted_indx_costilqr)))]
    
    # ------------------------------------------------------------------
    #   Compute CVaR: check the cases where h-iLQR do not perform well
    # ------------------------------------------------------------------    
    for confidence in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        cVaR = compute_cvar(cost_diff, confidence)[1]
        print("cVaR at confidence level: ", confidence, " is ", cVaR)   
    
    cost_diff_tail = (v_cost_ilqr[ilqr_tail_index] - v_cost_pi[ilqr_tail_index]) / v_cost_ilqr[ilqr_tail_index]
    mean_cost_diff_tail = np.mean(cost_diff_tail) * 100.0
    
    print("mean hilqg tail cost: ", np.mean(v_cost_ilqr[ilqr_tail_index]))
    print("mean h pathintegral tail cost: ", np.mean(v_cost_pi[ilqr_tail_index]))
    print("mean tail cost diff: ", mean_cost_diff_tail)
    
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
    
    # save figures
    fig2.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_zdotz_best10.pdf', format='pdf', dpi=2000)
    fig4.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_zdotz_tail10.pdf', format='pdf', dpi=2000)
    
    # fig2.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_zdotz_best10_zerocontrol.pdf', format='pdf', dpi=2000)
    # fig4.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_zdotz_tail10_zerocontrol.pdf', format='pdf', dpi=2000)

    # ------------------------------------------ 
    #   Bar Plot PathCosts for all experiments
    # ------------------------------------------
    fig5, ax6 = plt.subplots(figsize=(18,6))
    ax6.grid(True)
    
    # Set the bar width
    bar_width = 0.35
    # Set the opacity
    opacity = 0.9
    index = np.arange(n_exp)

    bars1 = ax6.bar(index, v_cost_ilqr, bar_width, alpha=opacity, color='b', label='Hybrid iLQG')
    bars2 = ax6.bar(index + bar_width, v_cost_pi, bar_width, alpha=opacity, color='r', label='Hybrid Path Integral')

    # Add some labels, title and axes ticks
    ax6.set_xlabel(r"Experiment ID", fontproperties=font_props)
    ax6.set_ylabel(r"Costs", fontproperties=font_props)
    ax6.set_title(r'Comparison of Cost Statistics', fontproperties=font_props)
    ax6.set_xticks(index + bar_width / 2)  # Positioning the x-axis ticks in the middle of the two bars
    ax6.set_xticklabels(index)

    # Adding a legend
    ax6.legend(loc='best', prop={'family': 'serif', 'size': 12})

    fig5.tight_layout()
    fig5.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_costs.pdf', dpi=2000)

    #-----------------------------------------
    #  Plot the variances and useful portion
    #-----------------------------------------
    (avg_var, std_var, lb_var, ub_var, 
    avg_lbdas, std_lbdas, lb_lbdas, ub_lbdas) = compute_var_lbd_nexp(n_exp, nt, exp_data)
    
    # --------------------------- Plotting --------------------------- 
    fig6, ax7 = plt.subplots(figsize=(8,6))
    fig7, ax8 = plt.subplots(figsize=(8,6))
    
    ax7.grid(True)
    ax8.grid(True)
    
    # Mean as a solid line
    ax7.plot(time_span[:-1], avg_var, 'r-', label=r'Mean Weight Distribution Variance')
    ax7.fill_between(time_span[:-1], lb_var, ub_var, color='gray', alpha=0.5, label='Varies across experiments')
    
    # ax9.set_title('Weight Variance')
    ax7.set_xlabel(r'Time', fontproperties=font_props)
    ax7.set_ylabel(r'Var$(\alpha)$ (t)', fontproperties=font_props)
    ax7.legend(loc='best', prop={'family': 'serif', 'size': 12})
    
    # Shaded area for variability (e.g., ±1 standard deviation)
    ax8.plot(time_span[:-1], avg_lbdas, 'r-', label=r'Mean Effective Samples $(\%)$')
    # ax8.fill_between(time_span[:-1], avg_lbdas-std_lbdas, avg_lbdas+std_lbdas, color='gray', alpha=0.5, label=r'$1$ StdV. across experiments')

    ax8.fill_between(time_span[:-1], lb_lbdas, ub_lbdas, color='gray', alpha=0.5, label='Varies across experiments')

    ax8.set_xlabel(r'Time', fontproperties=font_props)
    ax8.set_ylabel(r'$\lambda$ (t) (%)', fontproperties=font_props)
    ax8.set_ylim(0, 110)    
    ax8.legend(loc='best', prop={'family': 'serif', 'size': 12})
    
    fig6.tight_layout()
    fig7.tight_layout()
    fig6.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_var.pdf', format='pdf', dpi=2000)
    fig7.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_lbda.pdf', format='pdf', dpi=2000)
    
    
    # --------------------------- Compute the statistics ---------------------------
    jump_index = 585
    # total average
    ttl_avg_var = np.mean(avg_var)
    ttl_avg_lbda = np.mean(avg_lbdas)
    
    # begore jump
    pre_avg_var = np.mean(avg_var[:jump_index])
    pre_avg_lbda = np.mean(avg_lbdas[:jump_index])
    
    # after jump
    post_avg_var = np.mean(avg_var[jump_index:])
    post_avg_lbda = np.mean(avg_lbdas[jump_index:])
    
    print("Average variance: ", ttl_avg_var)
    print("Average variance before jump: ", pre_avg_var)
    print("Average variance after jump: ", post_avg_var)
    
    print("Average lambda: ", ttl_avg_lbda)
    print("Average lambda before jump: ", pre_avg_lbda)
    print("Average lambda after jump: ", post_avg_lbda)
    
    # # Plot step one cost distribution
    # fig6, ax11 = plt.subplots(figsize=(8,6))
    # ax11.grid(True)
    # ax11.bar(range(allPathCosts.shape[1]), allPathCosts[-1])
    # ax11.set_title("Path Cost distribution")
    # ax11.set_xlabel("Sample Number", fontproperties=font_props)
    # ax11.set_ylabel("Costs", fontproperties=font_props)
    
    plt.show()
    
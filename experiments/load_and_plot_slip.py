import os
import sys
file_path = os.path.abspath(__file__)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from dynamics.dynamics_slip import *
from hybrid_pathintegral.hybrid_pathintegral import *
from exp_params import *
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    exp_params = ExpParams()
    exp_data = ExpData(exp_params)

    # filename = root_dir+"/data/bouncing/ablation_study_nsamples/data_5000samples_eps_15.0_coupling.pickle"
    filename = root_dir+"/experiments/data/new_exp/slip/data_h_PI_slip_jax_threading_100exp_5000samples_eps_0.006_coupling.pickle"
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
    
    print("dt: ", exp_params._dt)
    
    (timespan,modes,states,inputs, 
        k_feedforward, K_feedback, current_cost, 
        states_iter, ref_modechanges,
        reference_extension_helper, ref_reset_args) = exp_data.get_nominal_data()
    
        
    plotting_function = exp_data.get_plotting_function()
    
    # # compute the costs and variances
    cost_pi_exp = np.zeros(n_exp)
    cost_ilqr_exp = np.zeros(n_exp)
    
    for i in range(n_exp):        
        cost_pi = exp_data.get_data(i).cost_pi()
        cost_ilqr = exp_data.get_data(i).cost_ilqr()
        
        cost_pi_exp[i] = cost_pi
        cost_ilqr_exp[i] = cost_ilqr
    
    print("================= Expected Cost Improvement =================")
    print("E[cost_pi]: ", np.mean(cost_pi_exp))
    print("E[cost_ilqr_exp]: ", np.mean(cost_ilqr_exp))
    print("improved: ", (np.mean(cost_ilqr_exp) - np.mean(cost_pi_exp)) / np.mean(cost_ilqr_exp))
    
    
    # ================================
    #  Sorting the cost improvements
    # ================================
    cost_diff = (cost_ilqr_exp - cost_pi_exp) / cost_ilqr_exp * 100
    sorted_indices = [index for index, _ in sorted(enumerate(cost_diff), key=lambda x: x[1])]
    sorted_cost_diff = sorted(cost_diff)
    largest_improve_index = sorted_indices[-1:]
    
    # Check the costs of the 10% tail of h-iLQR costs
    sorted_cost_ilqr_indices = [index for index, _ in sorted(enumerate(cost_ilqr_exp), key=lambda x: x[1])]
    sorted_costilqr = sorted(cost_ilqr_exp)
    ilqr_tail_index = sorted_cost_ilqr_indices[int(np.floor(0.9 * len(sorted_cost_ilqr_indices))):]
    ilqr_best_index = sorted_cost_ilqr_indices[:int(np.floor(0.1 * len(sorted_cost_ilqr_indices)))]
    
    # ------------------------------------------------------------------
    #   Compute CVaR: check the cases where h-iLQR do not perform well
    # ------------------------------------------------------------------
    def compute_cvar(samples, confidence_level=0.95):
        # Step 1: Sort the samples
        samples_sorted = np.sort(samples)
        
        # Step 2: Find the VaR (confidence_level percentile)
        var_index = int(np.floor(confidence_level * len(samples_sorted)))
        var_value = samples_sorted[var_index]
        
        # Step 3: Calculate CVaR as the mean of the losses exceeding the VaR
        cvar_value = np.mean(samples_sorted[var_index:])
        
        return var_value, cvar_value
    
    print("================= Expected Cost Improvement cVaR =================")
    for confidence in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        cVaR = compute_cvar(cost_diff, confidence)[1]
        print("cVaR at confidence level: ", confidence, " is ", cVaR)   
    
    cost_diff_tail = (cost_ilqr_exp[ilqr_tail_index] - cost_pi_exp[ilqr_tail_index]) / cost_ilqr_exp[ilqr_tail_index]
    mean_cost_diff_tail = np.mean(cost_diff_tail) * 100.0
    
    print("================= H-iLQG tail Cost =================")
    print("mean 10% tail H-iLQG cost, H-iLQG: ", np.mean(cost_ilqr_exp[ilqr_tail_index]))
    print("mean 10% tail H-iLQG cost, H-PI: ", np.mean(cost_pi_exp[ilqr_tail_index]))
    print("mean 10% tail H-iLQG cost, Improvement (%): ", mean_cost_diff_tail)
    
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
    
    time_span = np.arange(exp_params._start_time, exp_params._end_time, exp_params._dt).flatten()
    
    figs1, axes1 = plot_slip_nexp(ilqr_best_index, exp_data, time_span, init_state, target_state, args=None)
    figs2, axes2 = plot_slip_nexp(ilqr_tail_index, exp_data, time_span, init_state, target_state, args=None)
    
    fig11, fig12, fig13, fig14, fig15 = figs1.flatten()
    fig21, fig22, fig23, fig24, fig25 = figs2.flatten()
    
    fig11.tight_layout()
    fig12.tight_layout()
    fig13.tight_layout()
    fig21.tight_layout()
    fig22.tight_layout()
    fig23.tight_layout()
    
    # save figures
    fig11.savefig(root_dir+'/data/figures/slip/slip_mode0_best10_1.pdf', format='pdf', dpi=2000)
    fig12.savefig(root_dir+'/data/figures/slip/slip_mode0_best10_2.pdf', format='pdf', dpi=2000)
    fig13.savefig(root_dir+'/data/figures/slip/slip_mode1_best10.pdf', format='pdf', dpi=2000)
    fig21.savefig(root_dir+'/data/figures/slip/slip_mode0_tail10_1.pdf', format='pdf', dpi=2000)
    fig22.savefig(root_dir+'/data/figures/slip/slip_mode0_tail10_2.pdf', format='pdf', dpi=2000)
    fig23.savefig(root_dir+'/data/figures/slip/slip_mode1_tail10.pdf', format='pdf', dpi=2000)
    
    plt.show()
    
    
    # -------------------------------------------------
    #  Animate a few trajectories fot the worst cases
    # -------------------------------------------------
    for i_exp in largest_improve_index:
        
        modes_pi = exp_data.get_data(i_exp).mode_trj_pi()
        states_pi = exp_data.get_data(i_exp).x_trj_pi()
        inputs_pi = exp_data.get_data(i_exp).u_trj_pi()
        states_pi = exp_data.get_data(i_exp).x_trj_pi()
        states_pi = np.array(states_pi)
        
        modes_ilqg = exp_data.get_data(i_exp).mode_trj_ilqr()
        states_ilqg = exp_data.get_data(i_exp).x_trj_ilqr()
        inputs_ilqg = exp_data.get_data(i_exp).u_trj_ilqr()
        
        init_mode = 1
        target_mode = 0
        nt = len(time_span)
        reset_args = [np.array([0.0]) for _ in range(nt)]
        target_reset_args = reset_args
        states_pi = unpad_state_slip(modes_pi, states_pi)
        
        # Draw reference
        # fig8, ax9 = animate_slip(modes, states, init_mode, 
        #                         exp_params._init_state, target_mode, exp_params._target_state, 
        #                         nt, reset_args, target_reset_args, step=20)
        # ax9.set_title(r"H-iLQG reference", fontproperties=font_props)
        # plt.show()
        # ax9.set_title(r"H-PI controlled SLIP under uncertainty", fontproperties=font_props)
        
        
        fig8, ax9 = animate_slip(modes_pi, states_pi, init_mode, 
                                exp_params._init_state, target_mode, exp_params._target_state, 
                                nt, reset_args, target_reset_args, step=10)
        ax9.set_title(r"H-PI controlled SLIP under uncertainty", fontproperties=font_props)
        
        
        fig9, ax10 = animate_slip(modes_ilqg, states_ilqg, init_mode, 
                                exp_params._init_state, target_mode, exp_params._target_state, 
                                nt, reset_args, target_reset_args, step=10)
        ax10.set_title(r"H-iLQG controlled SLIP under uncertainty", fontproperties=font_props)
        
        fig8.tight_layout()
        fig9.tight_layout()
        
        plt.show()
        
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

    bars1 = ax8.bar(index, cost_ilqr_exp, bar_width, alpha=opacity, color='b', label='Hybrid iLQG')
    bars2 = ax8.bar(index + bar_width, cost_pi_exp, bar_width, alpha=opacity, color='r', label='Hybrid Path Integral')

    # Add some labels, title and axes ticks
    ax8.set_xlabel(r"Experiment ID", fontproperties=font_props)
    ax8.set_ylabel(r"$Costs$", fontproperties=font_props)
    ax8.set_title('Comparison of Cost Statistics', fontproperties=font_props)
    ax8.set_xticks(index + bar_width / 2)  # Positioning the x-axis ticks in the middle of the two bars
    ax8.set_xticklabels(index)

    # Adding a legend
    ax8.legend()

    fig3.tight_layout()
    fig3.savefig(root_dir+'/data/figures/slip/slip_costs.pdf', dpi=2000)

    #--- plot the variances and useful portion
    
    (avg_variances, std_variances, lowerbound_variances, upperbound_variances, 
            avg_lbdas, std_lbdas, lowerbound_lbdas, upperbound_lbdas) = compute_var_lbd_nexp(n_exp, nt, exp_data)
    
    # --------------------------- Plotting --------------------------- 
    fig6, ax7 = plt.subplots(figsize=(8,6))
    fig7, ax8 = plt.subplots(figsize=(8,6))
    
    ax7.grid(True)
    ax8.grid(True)
    
    # Mean as a solid line
    ax7.plot(time_span[:-1], avg_variances, 'r-', label=r'Mean Weight Distribution Variance')
    ax7.fill_between(time_span[:-1], lowerbound_variances, upperbound_variances, color='gray', alpha=0.5, label='Varies across experiments')
    
    # ax9.set_title('Weight Variance')
    ax7.set_xlabel(r'Time', fontproperties=font_props)
    ax7.set_ylabel(r'Var$(\alpha)$ (t)', fontproperties=font_props)
    ax7.legend(loc='best', prop={'family': 'serif', 'size': 12})
    
    # Shaded area for variability (e.g., ±1 standard deviation)
    ax8.plot(time_span[:-1], avg_lbdas, 'r-', label=r'Mean Effective Samples $(\%)$')
    # ax8.fill_between(time_span[:-1], avg_lbdas-std_lbdas, avg_lbdas+std_lbdas, color='gray', alpha=0.5, label=r'$1$ StdV. across experiments')
    ax8.fill_between(time_span[:-1], lowerbound_lbdas, upperbound_lbdas, color='gray', alpha=0.5, label='Varies across experiments')

    ax8.set_xlabel(r'Time', fontproperties=font_props)
    ax8.set_ylabel(r'$\lambda$ (t) (%)', fontproperties=font_props)
    ax8.set_ylim(0, 110)    
    ax8.legend(loc='best', prop={'family': 'serif', 'size': 12})
    
    fig6.tight_layout()
    fig7.tight_layout()
    fig6.savefig(root_dir+'/data/figures/slip/slip_var.pdf', format='pdf', dpi=2000)
    fig7.savefig(root_dir+'/data/figures/slip/slip_lbda.pdf', format='pdf', dpi=2000)
        
    # --------------------------- Compute the statistics ---------------------------
    jump_indexes = np.zeros(n_exp)
    for i_exp in range(n_exp):
        
        modes_pi = exp_data.get_data(i_exp).mode_trj_pi()
        mode0_index = np.where(modes_pi==0)
        jump_indexes[i_exp] = mode0_index[0][0]
        
    jump_index = int(np.mean(jump_indexes))
    print("jump_index: ", jump_index)
    
    # total average
    ttl_avg_var = np.mean(avg_variances)
    ttl_avg_lbda = np.mean(avg_lbdas)
    
    # begore jump
    pre_avg_var = np.mean(avg_variances[:jump_index])
    pre_avg_lbda = np.mean(avg_lbdas[:jump_index])
    
    # after jump
    post_avg_var = np.mean(avg_variances[jump_index:])
    post_avg_lbda = np.mean(avg_lbdas[jump_index:])
    
    print("================= Variance and Lambda statistics =================")
    print("Average variance: ", ttl_avg_var)
    print("Average variance before jump: ", pre_avg_var)
    print("Average variance after jump: ", post_avg_var)
    
    print("Average lambda: ", ttl_avg_lbda)
    print("Average lambda before jump: ", pre_avg_lbda)
    print("Average lambda after jump: ", post_avg_lbda)
    
    
    # plt.show()
    
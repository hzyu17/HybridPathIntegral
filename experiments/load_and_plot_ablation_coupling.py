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


if __name__ == '__main__':
    
    exp_params = ExpParams()
    exp_data = ExpData(exp_params)

    # filename = root_dir+"/data/bouncing/ablation_study_nsamples/data_5000samples_eps_15.0_coupling.pickle"
    filename = root_dir+"/experiments/data/new_exp/slip/data_2024-10-27_10-22-50_ablation_coupling_bouncing_3exp_500samples_eps_2.0_coupling_ablation.pickle"
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
    
    
    for i_exp in range(n_exp):
        
        modes_pi = exp_data.get_data(i_exp).mode_trj_pi()
        mode0_index = np.where(modes_pi==0)

    #-----------------------------------------
    #  Plot the variances and useful portion
    #-----------------------------------------
    [(avg_var, std_var, lb_var, ub_var, 
     avg_lbdas, std_lbdas, lb_lbdas, ub_lbdas),
     (avg_var_uncoupled, std_var_uncoupled, 
      lb_var_uncoupled, ub_var_uncoupled, 
     avg_lbdas_uncoupled, std_lbdas_uncoupled, 
     lb_lbdas_uncoupled, ub_lbdas_uncoupled)] = compute_var_lbd_nexp_ablation_coupling(n_exp, nt, exp_data)
    
    # --------------------------- Compute the statistics ---------------------------
    jump_index = 212
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
    
    # =====================================
    #               Uncoupled
    # =====================================
    print("====================== uncoupled results ======================")
    ttl_avg_var_uncoupled = np.mean(avg_var_uncoupled)
    ttl_avg_lbda_uncoupled = np.mean(avg_lbdas_uncoupled)
    
    # begore jump
    pre_avg_var_uncoupled = np.mean(avg_var_uncoupled[:jump_index])
    pre_avg_lbda_uncoupled = np.mean(avg_lbdas_uncoupled[:jump_index])
    
    # after jump
    post_avg_var_uncoupled = np.mean(avg_var_uncoupled[jump_index:])
    post_avg_lbda_uncoupled = np.mean(avg_lbdas_uncoupled[jump_index:])
    
    
    print("Average variance uncoupled: ", ttl_avg_var_uncoupled)
    print("Average variance before jump uncoupled: ", pre_avg_var_uncoupled)
    print("Average variance after jump uncoupled: ", post_avg_var_uncoupled)
    
    print("Average lambda uncoupled: ", ttl_avg_lbda_uncoupled)
    print("Average lambda before jump uncoupled: ", pre_avg_lbda_uncoupled)
    print("Average lambda after jump uncoupled: ", post_avg_lbda_uncoupled)
    
    print("Improved var: ", (ttl_avg_var_uncoupled - ttl_avg_var) / ttl_avg_var_uncoupled)
    print("Improved lambda: ", (ttl_avg_lbda-ttl_avg_lbda_uncoupled) / ttl_avg_lbda)
    
    
    # # --------------------------- Plotting --------------------------- 
    # fig6, ax7 = plt.subplots(figsize=(8,6))
    # fig7, ax8 = plt.subplots(figsize=(8,6))
    
    # ax7.grid(True)
    # ax8.grid(True)
    
    # # Mean as a solid line
    # ax7.plot(time_span[:-1], avg_var, 'r-', label=r'Mean Weight Distribution Variance')
    # # ax7.fill_between(time_span[:-1], lb_var, ub_var, color='gray', alpha=0.5, label='Varies across experiments')
    
    # # ax9.set_title('Weight Variance')
    # ax7.set_xlabel(r'Time', fontproperties=font_props)
    # ax7.set_ylabel(r'Var$(\alpha)$ (t)', fontproperties=font_props)
    # ax7.legend(loc='best', prop={'family': 'serif', 'size': 12})
    
    # # Shaded area for variability (e.g., ±1 standard deviation)
    # ax8.plot(time_span[:-1], avg_lbdas, 'r-', label=r'Mean Effective Samples $(\%)$')
    # ax8.fill_between(time_span[:-1], avg_lbdas-std_lbdas, avg_lbdas+std_lbdas, color='gray', alpha=0.5, label=r'$1$ StdV. across experiments')

    # # ax8.fill_between(time_span[:-1], lb_lbdas, ub_lbdas, color='gray', alpha=0.5, label='Varies across experiments')

    # ax8.set_xlabel(r'Time', fontproperties=font_props)
    # ax8.set_ylabel(r'$\lambda$ (t) (%)', fontproperties=font_props)
    # ax8.set_ylim(0, 110)    
    # ax8.legend(loc='best', prop={'family': 'serif', 'size': 12})
    
    # fig6.tight_layout()
    # fig7.tight_layout()
    # fig6.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_var.pdf', format='pdf', dpi=2000)
    # fig7.savefig(root_dir+'/data/figures/bouncing/bouncing_1D_lbda.pdf', format='pdf', dpi=2000)    
    
    plt.show()
    
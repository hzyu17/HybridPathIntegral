from example_bouncingball import *
from matplotlib.patches import Circle

if __name__ == '__main__':
    
    exp_params = ExpParams(init_state, target_state, start_time, end_time, dt, dt_pathintegral, epsilon, n_samples, Q_k, R_k, Q_T)
    exp_data = ExpData(exp_params)
    
    filename = root_dir+"/data/bouncing/data_2024-03-22_14-31-11_example_bouncingball.pickle"
    print("loading data")
    exp_data.load(filename)

    print("plotting")
    fig1, axes = plt.subplots(1, 2)
    (ax1, ax2) = axes.flatten()
    ax1.grid(True)
    ax2.grid(True)
    
    # # compute the costs
    cost_pi_exp = np.zeros(n_exp)
    cost_ilqr_exp = np.zeros(n_exp)
    for i in range(n_exp):
        trj_ilqr = exp_data.get_data(i).x_trj_ilqr()
        trj_pi = exp_data.get_data(i).x_trj_pi()
        u_star_pi = exp_data.get_data(i).u_trj_pi()
        u_trj_ilqr = exp_data.get_data(i).u_trj_ilqr()
        
        dWs_zeros = np.zeros((nt, n_inputs))
        cost_pi = compute_cost(trj_pi, u_star_pi, dWs_zeros, target_state, states, Q_k, R_k, Q_T, epsilon)
        cost_ilqr = compute_cost(trj_ilqr, u_trj_ilqr, dWs_zeros, target_state, states, Q_k, R_k, Q_T, epsilon)
        
        print("cost_pi:", cost_pi)
        print("cost_ilqr:", cost_ilqr)
        
        cost_pi_exp[i] = cost_pi
        cost_ilqr_exp[i] = cost_ilqr
    
    # # ----------- plot the path integral controlled trajectory -----------
    for i in range(n_exp):
        trj_ilqr = exp_data.get_data(i).x_trj_ilqr()
        trj_pi = exp_data.get_data(i).x_trj_pi()
        
        ax1.plot(time_span, trj_ilqr[:, 0], 'b', alpha=0.2)
        ax2.plot(time_span, trj_ilqr[:, 1], 'b', alpha=0.2)
        
        ax1.plot(time_span, trj_pi[:, 0], 'r', alpha=0.6)
        ax2.plot(time_span, trj_pi[:, 1], 'r', alpha=0.6)
        
    ax1.plot(time_span, trj_ilqr[:, 0], 'b', alpha=0.2, label='iLQR')
    ax2.plot(time_span, trj_ilqr[:, 1], 'b', alpha=0.2, label='iLQR')

    ax1.plot(time_span, trj_pi[:, 0], 'r', alpha=0.6, label='Path Integral')
    ax2.plot(time_span, trj_pi[:, 1], 'r', alpha=0.6, label='Path Integral')

    # ----------- Plot the reference -----------
    ax1.plot(time_span[1:], states[1:-1,0],'k',label='iLQR-deterministic')
    ax2.plot(time_span[1:], states[1:-1,1],'k',label='iLQR-deterministic')

    # ----------- Plot the start and goal states -----------
    ax1.scatter(time_span[-1], target_state[0], color='g', marker='o', s=50.0, linewidths=6, label='Target', zorder=2)
    ax1.scatter(time_span[0], init_state[0], color='r', marker='o', s=50.0, linewidths=6, label='Start', zorder=2)

    ax2.scatter(time_span[-1], target_state[1], color='g', marker='o', s=50.0, linewidths=6, label='Target', zorder=2)
    ax2.scatter(time_span[0], init_state[1], color='r', marker='o', s=50.0, linewidths=6, label='Start', zorder=2)
    
    ax1.set_xlabel(r"Time")
    ax1.set_ylabel(r"$z$")
    ax1.set_title("Vertical Position")
    plt.tight_layout()

    ax2.set_xlabel(r"Time")
    ax2.set_ylabel(r"$\dot z$")
    ax2.set_title("Vertical Velocity")
    plt.tight_layout()
    
    ax1.legend()
    ax2.legend()

    # save figures
    fig1.savefig(root_dir+'/hybrid_pathintegral/bouncing_1D.pdf', format='pdf', dpi=1000)
    
    # =========== Plot the z-\dot_z figure ===========
    fig2, ax5 = plt.subplots()
    ax5.grid(True)

    # ----------- Plot the last iteration of iLQR controller ----------
    for i in range(n_exp):
        trj_ilqr = exp_data.get_data(i).x_trj_ilqr()
        trj_pi = exp_data.get_data(i).x_trj_pi()
        ax5.plot(trj_ilqr[:, 0], trj_ilqr[:, 1], 'b', alpha=0.2)
        ax5.plot(trj_pi[:, 0], trj_pi[:, 1], 'r', alpha=0.6)

    ax5.plot(trj_ilqr[:, 0], trj_ilqr[:, 1], 'b', alpha=0.2, label='iLQR')
    ax5.plot(trj_pi[:, 0], trj_pi[:, 1], 'r', alpha=0.6, label='Path Integral')
    ax5.plot(states[1:-1,0], states[1:-1,1],'k',label='iLQR-deterministic')
    
    # ----------- Plot the start and goal states -----------
    ax5.scatter(target_state[0], target_state[1], color='g', marker='o', s=50.0, linewidths=6, label='Target', zorder=2)
    ax5.scatter(init_state[0], init_state[1], color='r', marker='o', s=50.0, linewidths=6, label='Start', zorder=2)

    ax5.set_xlabel(r"z")
    ax5.set_ylabel(r"$\dot z$")
    ax5.set_title("Controlled Bouncing Ball Dynamics")
    ax5.legend()
    plt.tight_layout()
    fig2.savefig(root_dir+'/hybrid_pathintegral/bouncing_1D_zdotz.pdf', format='pdf', dpi=1000)

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
    fig3, ax8 = plt.subplots()
    ax8.grid(True)
    ax8.bar(range(n_exp), cost_ilqr_exp, width = 2, color='navy', alpha=0.1, label='Cost iLQR')
    ax8.bar(range(n_exp), cost_pi_exp, width = 2, color='red', alpha=0.5, label='Cost PathIntegralControl')

    ax8.set_xlabel(r"Experiment ID")
    ax8.set_ylabel(r"$Costs$")
    ax8.legend()

    plt.tight_layout()
    fig3.savefig(root_dir+'/hybrid_pathintegral/bouncing_1D_costs.pdf', format='pdf', dpi=1000)
    
    plt.show()


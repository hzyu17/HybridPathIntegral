import numpy as np
import os
import sys

file_path = os.path.abspath(__file__)
exp_dir = os.path.dirname(file_path)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
root_dir = os.path.abspath(os.path.join(exp_dir, '..'))
sys.path.append(root_dir)


# Import iLQR class
from hybrid_ilqr.h_ilqr_jax import *
# Import 3 link walker dynamics
from walking_3link import *
# Import experiment parameter class
from experiments.exp_params import *


if __name__ == '__main__':
    # ---------------- solve for the nominals -----------------
    tout, xout, uout, t_events, x_events, saltations = solve_limcycle_3link()
    
    show_nominals = False
    if show_nominals:
        
        # Plotting states
        fig1 = plt.figure(figsize=(16, 9))
        plt.subplot(3, 4, 1)
        plt.plot(tout, xout[:, 0], label=r'$\theta_1$')
        plt.plot(tout, xout[:, 1], '--', label=r'$\theta_2$')
        plt.plot(tout, xout[:, 2], '-.', label=r'$\theta_3$')
        plt.legend(loc="best", fontsize=10)
        plt.title('Joint Positions')
        plt.grid()

        plt.subplot(3, 4, 2)
        plt.plot(tout, xout[:, 3], label=r'$\dot{\theta}_1$')
        plt.plot(tout, xout[:, 4], '--', label=r'$\dot{\theta}_2$')
        plt.plot(tout, xout[:, 5], '-.', label=r'$\dot{\theta}_3$')
        plt.legend(loc="best", fontsize=10)
        plt.title('Joint Velocities')
        plt.xlabel('Time (sec)')
        plt.grid()

        plt.subplot(3, 4, 3)
        plt.plot(tout, uout[:, 0], label=r'$u_1$')
        plt.plot(tout, uout[:, 1], label=r'$u_2$')
        plt.legend(loc="best", fontsize=10)
        plt.title('Control Input Torque')
        plt.xlabel('Time (sec)')
        plt.grid()

        plt.subplot(3, 4, 4)
        swingfoot_height = swingfoot_height_jax(xout)
        plt.plot(tout, swingfoot_height, label=r"Swing Foot Height")
        plt.legend(loc="best", fontsize=10)
        plt.title('Swing Foot Height')
        plt.grid()

        plt.subplot(3, 4, 5)
        swingfoot_vertical_vel = swingfoot_vel_vertical_jax(xout)
        plt.plot(tout, swingfoot_vertical_vel, label=r"Swing Foot Vertical Velocity")
        plt.legend(loc="best", fontsize=10)
        plt.title('Swing Foot Vertical Velocity')
        plt.grid()

        # Position and velocity
        plt.subplot(3, 4, 6)
        hip_trj = hipheight_jax(xout)
        plt.plot(tout, hip_trj, label=r'Hip height')
        plt.legend(loc="best", fontsize=10)
        plt.title('Hip Heights')
        plt.grid()

        plt.subplot(3, 4, 7)
        hipvel_trj = hipvel_H_jax(xout)
        plt.plot(tout, hipvel_trj, label=r'Hip horizontal vel')
        plt.legend(loc="best", fontsize=10)
        plt.title('Hip Horizontal Velocities')
        plt.xlabel('Time (sec)')
        plt.grid()

        plt.subplot(3, 4, 8)
        hipvel_trj_v = hipvel_V_jax(xout)
        plt.plot(tout, hipvel_trj_v, label=r'Hip vertical vel')
        plt.legend(loc="best", fontsize=10)
        plt.title('Hip Vertical Velocities')
        plt.xlabel('Time (sec)')
        plt.grid()
        
        # 2D limit cycle plot
        plt.subplot(3, 4, 11)
        plt.plot(xout[:, 0], xout[:, 1], linewidth=2)
        plt.xlabel(r'$\theta_1$')
        plt.ylabel(r'$\theta_2$')
        plt.title('2D Limit Cycle')
        plt.grid()
        plt.tight_layout()
        
        # # 3D limit cycle plot
        # fig2 = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')  
        # ax.plot3D(xout[:, 0], xout[:, 1], xout[:, 2], linewidth=2)
        # ax.set_xlabel(r'$\theta_1$')
        # ax.set_ylabel(r'$\theta_2$')
        # ax.set_zlabel(r'$\theta_3$')
        # plt.title('3D Limit Cycle')
        # plt.grid()
        # fig.tight_layout()
        
        # Animation
        anim(tout, xout, 1/30, speed=1, fig=fig1)
        
        plt.show()

    # ---------------- H-iLQR 3link walking -----------------
    epsilon = 2.0
    dt_shrink = 0.95
    
    time_span = tout
    
    start_time = tout[0]
    end_time = tout[-1]
    
    dt = tout[1:] - tout[:-1]
    nt = len(time_span)

    # generate initial state
    omega_1 = 1.55
    init_state = sigma_three_link(omega_1, a)
    init_state = resetmap_3link_12(0.0,init_state)[0].T

    target_state = init_state  # A limit cycle hopes to go back to the initial state

    # Number of modes: 2
    n_modes = 2
    
    # Both modes share the same dynamics
    n_states = [6, 6]
    n_inputs = [2, 2]

    # ---------------------------- 
    # Define weighting matrices
    # ----------------------------
    Q_k = [np.eye(n_states[0]), np.eye(n_states[1])] 
    R_k = [np.eye(n_inputs[0]), np.eye(n_inputs[1])]

    # ---------------------------- Set the terminal cost ----------------------------
    Q_T = 200*np.eye(n_states[0])
    # Q_T[0,0] = 2000.0

    n_exp = 1
    n_samples = 10
    
    init_reset_args = [np.array([0.0]) for _ in range(nt)]
    target_reset_args = [np.array([0.0]) for _ in range(nt)]
    
    # ====================================
    #    Solve for hybrid ilqr proposal
    # ====================================
    
    initial_guess = [uout, uout]
    
    # fig = plt.figure()
    # plt.subplot(1, 1, 1)
    # plt.plot(tout, uout[:,0], label=r'$u_0 mode 0$')
    # plt.legend(loc="best", fontsize=10)
    # plt.title('Initial Control GUess')
    # plt.xlabel('Time (sec)')
    # plt.grid()
    # plt.show()
    
    smooth_flow = dyn_control_3link_discrete_jax
    
    target_hipvel = 2.0
    running_cost_arg = target_hipvel
    terminalcost_arg = init_state
    
    niters = 20

    hilqr_obj = hybrid_ilqr_jax(n_states, n_inputs,
                                init_state, target_state, 
                                initial_guess, 
                                time_span,
                                niters, 
                                is_detect=True, 
                                detect_func=detect_3link, 
                                smooth_dynamics=smooth_flow, 
                                running_cost=hip_moving_cost, 
                                cost_args=running_cost_arg,
                                terminal_cost=deltx_norm_cost, 
                                terminal_cost_args=target_state)
    
    hybrid_ilqr_result = hilqr_obj.solve()
    
    (modes,states,inputs,
     k_feedforward,K_feedback,
     current_cost,states_iter,
     ref_modechanges,ref_ext_helper, ref_reset_args) = hybrid_ilqr_result
    
    exp_data.add_nominal_data(hybrid_ilqr_result)


    show_results = True
    if show_results:
        plot_bouncingball(time_span, modes, states, inputs, init_state, target_state, nt, color='k')
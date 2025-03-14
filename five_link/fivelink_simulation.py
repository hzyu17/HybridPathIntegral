import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)


import jax.numpy as jnp
import matplotlib.pyplot as plt
from five_link.rabbit_kinematics import *
from five_link.rabbit_dynamics import *
from dynamics.saltation_matrix import compute_saltation

import numpy as np


class FiveLinkSimulator:
    def __init__(self, x0, u_trj, dts):
        
        self.nx = len(x0)
        self.nu = u_trj.shape[1]
        self.nt = dts.shape[0] + 1 
        
        self.x0 = x0
        self.u_trj = u_trj
        self.dts = dts
        self.t_trj = np.zeros(self.nt)
        
        self.x_trj = np.zeros((self.nt, self.nx))
        self.x_trj[0] = x0
        
        self.n_impact = 0
        self.impact_time = []
        self.saltation_matrices = []
        
        # information for plotting 
        self.com_trj = np.zeros((self.nt, 2))
        self.v_com_trj = np.zeros((self.nt, 2))
        
        self.pos_com_st_foot_trj = np.zeros((self.nt, 2))
        self.vel_com_st_foot_trj = np.zeros((self.nt, 2))
        
        self.st_foot_wrench = np.zeros((self.nt, 2))
        self.sw_foot_pos_trj = np.zeros((self.nt, 2))
        self.sw_foot_vel_trj = np.zeros((self.nt, 2))
        
        self.st_foot_pos_trj = np.zeros((self.nt, 2))
        self.st_foot_vel_trj = np.zeros((self.nt, 2))
        
        self.sw_foot_wrench = np.zeros((self.nt, 2))
        
    
    def simulate(self):
        """
        Simulate the 5-link robot dynamics forward under the control u.
        We use discretized integration and perform guard condition checking and reset map.
        Args:
            x0 (jnp.array): initial state, in shape (nx, )
            u (jnp.array): control inputs, in shape (nt, nu)
            dts (jnp.array): discrete time intervals, in shape (nt, )
        """
        
        for i_t in range(self.nt-1):
            # print("Integrating time step: ", i_t)
            xt = self.x_trj[i_t]
            ut = self.u_trj[i_t]
            dt = self.dts[i_t]
            
            # ------------------------ 
            #    Forward integration 
            # ------------------------ 
            # x_next = f_euler_fivelink(xt, ut, dt) # Euler
            x_next = f_rk4_fivelink(xt, ut, dt) # RK4
            
            # -------------------- 
            #   Check for impact 
            # --------------------
            if sw_foot_ground_touching_event(x_next):
                # ------ bisection -----
                print("Guard function activated")
                
                # -------------------
                #     Bi-section 
                # -------------------
                tol = 1e-8
                t_left = 0.0
                x_left = xt
                t_right = 0.0+dt
                
                while (t_right - t_left) / 2.0 > tol:
                    t_mid = (t_left + t_right) / 2.0
                    dt_new = t_mid - t_left
                    
                    # x_mid = smooth_integration_fun(x_left, u, t_span, epsilon, dW_new)
                    x_mid = f_rk4_fivelink(xt, ut, dt_new)
                    
                    if sw_foot_ground_touching_event(x_mid):
                        x_left = x_mid
                        break
                    
                    elif guard_12_5link(t_left, x_left) * guard_12_5link(t_mid, x_mid) < 0:
                        t_right = t_mid  # The root is in the left half
                        # x_right = x_mid
                    else:
                        t_left = t_mid  # The root is in the right half
                        x_left = x_mid
                
                t_event = t_left
                x_event = x_left
                
                print("----- Guard function at the event state: ", guard_12_5link(t_event, x_event), "-----")
                
                # Apply reset map
                x_next = impact_map(x_event)
                
                self.n_impact += 1
                self.impact_time.append(self.t_trj[i_t] + dt)
                
                # Compute saltation matrix
                
                F_1 = f_NL_fivelink(t_event, x_event, ut)
                F_2 = f_NL_fivelink(t_event, x_next, ut) # Important, the F2 is evaluated at the resetted state!
                Rt = Rt_5link_12(x_event)
                Rx = Rx_5link_12(x_event)
                gt = gt_12_5link(t_event, x_event)
                gx = gx_12_5link(t_event, x_event)
                saltation = compute_saltation(F_1, F_2, Rt, Rx, gt, gx)
                self.saltation_matrices.append(saltation)
            
            self.x_trj[i_t+1] = x_next
            
            self.t_trj[i_t+1] = self.t_trj[i_t] + dt
            
        return self.x_trj
    
    
    def test_detect_func(self):
        """
        Test the detect function for five link walker.
        """
        from rabbit_dynamics import detect_fivelink
        
        modes = np.zeros(self.nt)
        for ii in range(self.nt-1):
            
            x_i = self.x_trj[ii]
            mode_i = modes[ii]
            
            # ------------------- 
            # Get the references 
            # ------------------- 
            u_i = self.u_trj[ii]
            
            # =================
            # Simulate forward
            # =================
            t_ii = self.t_trj[ii]
            dt_ii = self.dts[ii]
            t_ii_plus = t_ii + dt_ii
            
            (next_state, saltation, 
             mode_change, t_event, x_event, 
             x_reset, _) = detect_fivelink(x_i, u_i, t_ii, t_ii_plus, mode_i, reset_args=None)

            # ------------------------------
            # Update the hybrid information
            # ------------------------------
            if saltation is not None:
                timespan = np.concatenate(
                    (np.concatenate(
                        (timespan[:ii+1], np.array([t_event]))), timespan[ii+1:]))
                nt += 1
                
                modes = np.concatenate(
                    (np.concatenate(
                        (modes[:ii+1], np.array([mode_change[1]]))), modes[ii+1:]))
                
                states = np.concatenate(
                    (np.concatenate(
                        (states[:ii+1], np.array([x_reset]))), states[ii+1:]))
                
                next_state = x_reset
            
            self.x_trj[ii+1] = next_state
            
        return self.x_trj
    
    
    def compute_info(self):
        for i_t in range(self.nt-1):
            xt = self.x_trj[i_t]
            ut = self.u_trj[i_t]

            # ----------- for plotting ----------- 
            w_st = wrench_st(xt, ut)
            self.st_foot_wrench[i_t] = w_st
            self.com_trj[i_t] = COM_Position(xt[0:7])
            self.v_com_trj[i_t] = vel_com_world(xt[0:7], xt[7:])
            
            self.pos_com_st_foot_trj[i_t] = pos_com_right_foot(xt[:7])
            self.vel_com_st_foot_trj[i_t] = vel_com_right_foot(xt[:7], xt[7:])
            
            self.sw_foot_pos_trj[i_t] = Left_Swing_Foot_Position(xt[:7])
            self.sw_foot_vel_trj[i_t] = vel_left_foot(xt[:7], xt[7:])
            
            self.st_foot_pos_trj[i_t] = Right_Stance_Foot_Position(xt[:7])
            self.st_foot_vel_trj[i_t] = vel_right_foot(xt[:7], xt[7:])
                
                
    def plot_results(self):
        self.compute_info()
        
        _, axs = plt.subplots(3, 2, figsize=(10, 8))
        axs[0, 0].plot(self.t_trj, self.u_trj[:, 0], label='Input Torque u1')
        axs[0, 0].plot(self.t_trj, self.u_trj[:, 1], label='Input Torque u2')
        axs[0, 0].plot(self.t_trj, self.u_trj[:, 2], label='Input Torque u3')
        axs[0, 0].plot(self.t_trj, self.u_trj[:, 3], label='Input Torque u4')

        axs[0, 1].plot(self.t_trj, self.st_foot_wrench[:, 0], label='Stance Foot Wrench, x')
        axs[0, 1].plot(self.t_trj, self.st_foot_wrench[:, 1], label='Stance Foot Wrench, z')
        
        axs[1, 0].plot(self.t_trj, self.sw_foot_wrench[:, 0], label='Swing Foot Wrench, x')
        axs[1, 0].plot(self.t_trj, self.sw_foot_wrench[:, 1], linestyle='--', label='Swing Foot Wrench, z')
        
        axs[2, 0].plot(self.t_trj, self.x_trj[:, 0], label='State x1')
        axs[2, 0].plot(self.t_trj, self.x_trj[:, 1], label='State x2')
        axs[2, 0].plot(self.t_trj, self.x_trj[:, 2], label='State x3')
        axs[2, 0].plot(self.t_trj, self.x_trj[:, 3], label='State x4')
        axs[2, 0].plot(self.t_trj, self.x_trj[:, 4], label='State x5')
        axs[2, 0].plot(self.t_trj, self.x_trj[:, 5], label='State x6')
        axs[2, 0].plot(self.t_trj, self.x_trj[:, 6], label='State x7')

        axs[2, 1].plot(self.t_trj, self.x_trj[:, 7], label='State dx1')
        axs[2, 1].plot(self.t_trj, self.x_trj[:, 8], label='State dx2')
        axs[2, 1].plot(self.t_trj, self.x_trj[:, 9], label='State dx3')
        axs[2, 1].plot(self.t_trj, self.x_trj[:, 10], label='State dx4')
        axs[2, 1].plot(self.t_trj, self.x_trj[:, 11], label='State dx5')
        axs[2, 1].plot(self.t_trj, self.x_trj[:, 12], label='State dx6')
        axs[2, 1].plot(self.t_trj, self.x_trj[:, 13], label='State dx7')
        
        axs[0, 0].grid(True)
        axs[0, 1].grid(True)        
        axs[1, 0].grid(True)
        axs[1, 1].grid(True)
        axs[2, 0].grid(True)
        axs[2, 1].grid(True)

        axs[0, 0].legend()
        axs[0, 1].legend()
        axs[1, 0].legend()
        axs[1, 1].legend()
        axs[2, 0].legend()
        axs[2, 1].legend()
        
        plt.tight_layout()
    
        _, axs1 = plt.subplots(3, 2, figsize=(8, 10))
        axs1[0, 0].plot(self.t_trj, self.sw_foot_pos_trj[:, 0], label='Swing Foot Position, x')
        axs1[0, 0].plot(self.t_trj, self.st_foot_pos_trj[:, 0], label='Stance Foot Position, x')
        
        axs1[0, 1].plot(self.t_trj, self.sw_foot_pos_trj[:, 1], label='Swing Foot Position, z')
        axs1[0, 1].plot(self.t_trj, self.st_foot_pos_trj[:, 1], label='Stance Foot Position, z')
        
        axs1[1, 0].plot(self.t_trj, self.sw_foot_vel_trj[:, 0], label='Swing Foot Velocity, x')
        axs1[1, 0].plot(self.t_trj, self.st_foot_vel_trj[:, 0], label='Stance Foot Velocity, x')
        
        axs1[1, 1].plot(self.t_trj, self.sw_foot_vel_trj[:, 1], label='Swing Foot Velocity, z')
        axs1[1, 1].plot(self.t_trj, self.st_foot_vel_trj[:, 1], label='Stance Foot Velocity, z')
        
        # axs1[2, 0].plot(self.t_trj, self.com_trj[:, 0], label='CoM World Pos trj, x')
        # axs1[2, 0].plot(self.t_trj, self.pos_com_st_foot_trj[:, 0], label='CoM Stance Foot Pos trj, x')
        
        axs1[2, 0].plot(self.t_trj, self.com_trj[:, 1], label='CoM World Pos trj, z')
        axs1[2, 0].plot(self.t_trj, self.pos_com_st_foot_trj[:, 1], label='CoM Stance Foot Pos trj, z')
        
        axs1[2, 1].plot(self.t_trj, self.v_com_trj[:, 0], label='CoM World Vel trj, x')
        axs1[2, 1].plot(self.t_trj, self.v_com_trj[:, 1], label='CoM World Vel trj, z')
        
        
        axs1[0, 0].grid(True)
        axs1[0, 1].grid(True)  
        axs1[1, 0].grid(True)      
        axs1[1, 1].grid(True)
        axs1[2, 0].grid(True)
        axs1[2, 1].grid(True)
        
        axs1[0, 0].legend()
        axs1[0, 1].legend()
        axs1[1, 0].legend()
        axs1[1, 1].legend()
        axs1[2, 0].legend()
        axs1[2, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        
def draw_5link(q, ax=None, legend=True):
    
    # pos_hip = hip_position(q)
    pos_hip = jnp.array([q[0], 0.0, q[1]])
    pos_com = COM_Position(q)
    pos_left_swingfoot = p_LeftToe(q)
    pos_right_stancefoot = p_RightToe(q)
    pos_right_knee = p_q2_right(q)
    pos_left_knee = p_q2_left(q)
    pos_torso = p_Torso(q)
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax.clear()
        
    # Plot the right ground line:
    ax.plot([0, 2], [0, 0], linewidth=4, color='k')
    # Plot the left ground line:
    ax.plot([0, -1], [0, 0], linewidth=4, color='k')
    
    circle = plt.Circle((pos_com[0], pos_com[1]), radius=0.01, color='green', fill=True, linewidth=3)

    # Add the circle to the axes
    ax.add_patch(circle)

    # ax.scatter(pos_com[0], pos_com[1], s=6.0, color='g', label='CoM')
    # ax.scatter(pos_hip[0], pos_hip[2], s=10.0, color='r', label='Hip')
    # ax.scatter(pos_torso[0], pos_torso[1], s=6.0, color='c', label='Torso')
    
    # ax.scatter(pos_left_swingfoot[0], pos_left_swingfoot[2], s=6.0, color='b', label='Left Swing Foot')
    # ax.scatter(pos_left_knee[0], pos_left_knee[2], s=6.0, color='b', label='Left Knee')
    
    # ax.scatter(pos_right_stancefoot[0], pos_right_stancefoot[2], s=6.0, color='g', label='Right Stance Foot')
    # ax.scatter(pos_right_knee[0], pos_right_knee[2], s=6.0, color='g', label='Right Knee')
    
    # Draw torso
    ax.plot([pos_torso[0], pos_hip[0]], [pos_torso[2], pos_hip[2]], color='k', linewidth=2, label='Torso')
    
    # Draw fem 1
    ax.plot([pos_right_knee[0], pos_hip[0]], [pos_right_knee[2], pos_hip[2]], color='r', linewidth=2, label='Right Stance thigh')

    # Draw fem 2
    ax.plot([pos_left_knee[0], pos_hip[0]], [pos_left_knee[2], pos_hip[2]], color='b', linewidth=2, label='Left Swing thigh')

    # Draw tib 1
    ax.plot([float(pos_right_knee[0]), pos_right_stancefoot[0]], [float(pos_right_knee[2]), pos_right_stancefoot[1]], color='r', linewidth=2, label='Right Stance Leg')
    
    # Draw tib 2
    ax.plot([float(pos_left_knee[0]), pos_left_swingfoot[0]], [float(pos_left_knee[2]), pos_left_swingfoot[2]], color='b', linewidth=2, label='Left Swing Leg')
    
    ax.set_ylim(-0.2, 1.5)
    
    # plt.axis('equal')
    if legend:
        ax.legend()
        

def animate_trj(x_trj, step=5):
    fig, ax = plt.subplots()
    
    plt.ion()
    
    n_time_steps = x_trj.shape[0]
    
    for i in range(0, n_time_steps, step):
        x_i = x_trj[i, :]
        q_i = x_i[:7]
        
        draw_5link(q_i, ax, legend=True)
        plt.pause(0.005)
        ax.clear()  
        
    plt.ioff()
    
    draw_5link(x_trj[-1, 0:7], ax, legend=True)
    plt.show()
        

if __name__ == '__main__':
    # q = [xbar, zbar, rotY, q1R, q2R, q1L, q2L]
    
    q_init = jnp.array([0, 0.658, 0, -0.6828+jnp.pi, 1.20, -0.6489+jnp.pi, 1.281])    
    qdot_init = jnp.zeros(7)
    x_init = jnp.concatenate([q_init, qdot_init])
    
    print("Initial swing foot height: ", Left_Swing_Foot_Position(q_init)[1])
    print("Initial stance foot height: ", Right_Stance_Foot_Position(q_init)[1])
    
    fig, ax = plt.subplots()
    draw_5link(q_init, ax)
    plt.show()
    
    nt = 240
    dt_trj = np.ones(nt)*0.001
    nu = 4
    u_trj = np.zeros((nt, nu))
    u_trj[:30, 0] = 11 # u_1R
    u_trj[:30, 1] = -2 # u_2R
    u_trj[:30, 2] = -11 # u_1L
    u_trj[:30, 3] = -2.5 # u_2L

    u_trj[30:100, 0] = -8 # u_1R
    u_trj[30:100, 1] = 1 # u_2R
    u_trj[30:100, 2] = 8 # u_1L
    u_trj[30:100, 3] = 1.5 # u_2L
    
    u_trj[100:135, 0] = 13.5 # u_1R
    u_trj[100:135, 1] = -2 # u_2R
    u_trj[100:135, 2] = -11 # u_1L
    u_trj[100:135, 3] = -3.5 # u_2L
    
    u_trj[135:, 0] = -8 # u_1R
    u_trj[135:, 1] = 1 # u_2R
    u_trj[135:, 2] = 8 # u_1L
    u_trj[135:, 3] = 2.5 # u_2L
    
    
    fivelink_simulator = FiveLinkSimulator(x_init, u_trj, dt_trj)
    
    x_trj = fivelink_simulator.test_detect_func()
    
    # x_trj = fivelink_simulator.simulate()
    # fivelink_simulator.plot_results()
    
    # for i in range(3):
    #     animate_trj(x_trj, step=5)
    
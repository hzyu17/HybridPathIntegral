import jax.numpy as jnp
import jax
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from rabbit_kinematics import *
from rabbit_dynamics import *

import numpy as np

def check_swing_foot_clearance(p_sw_world):
    return p_sw_world[1] > 0



def draw_5link(q, ax=None):
    
    # Draw ground
    kx = 0.0  # example value

    # Plot the right ground line:
    ground_right, = plt.plot([0, 2], [0, 2 * kx], linewidth=4, color='k')
    # Plot the left ground line:
    ground_left, = plt.plot([0, -1], [0, -1 * kx], linewidth=4, color='k')

    
    # pos_hip = hip_position(q)
    pos_hip = jnp.array([q[0], 0.0, q[1]])
    pos_com = com_position(q)
    pos_left_swingfoot = p_LeftToe(q)
    pos_right_stancefoot = p_RightToe(q)
    pos_right_knee = p_q2_right(q)
    pos_left_knee = p_q2_left(q)
    pos_torso = p_Torso(q)
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # ax.scatter(pos_com[0], pos_com[1], s=6.0, color='k', label='CoM')
    ax.scatter(pos_hip[0], pos_hip[2], s=10.0, color='r', label='Hip')
    # ax.scatter(pos_torso[0], pos_torso[1], s=6.0, color='c', label='Torso')
    
    # ax.scatter(pos_left_swingfoot[0], pos_left_swingfoot[2], s=6.0, color='b', label='Left Swing Foot')
    # ax.scatter(pos_left_knee[0], pos_left_knee[2], s=6.0, color='b', label='Left Knee')
    
    # ax.scatter(pos_right_stancefoot[0], pos_right_stancefoot[2], s=6.0, color='g', label='Right Stance Foot')
    # ax.scatter(pos_right_knee[0], pos_right_knee[2], s=6.0, color='g', label='Right Knee')
    
    # Draw torso
    ax.plot([pos_torso[0], pos_hip[0]], [pos_torso[2], pos_hip[2]], color='k', linewidth=2, label='Torso')
    
    # Draw fem 1
    ax.plot([pos_right_knee[0], pos_hip[0]], [pos_right_knee[2], pos_hip[2]], color='b', linewidth=2, label='Right thigh')

    # Draw fem 2
    ax.plot([pos_left_knee[0], pos_hip[0]], [pos_left_knee[2], pos_hip[2]], color='r', linewidth=2, label='Left thigh')

    # Draw tib 1
    ax.plot([float(pos_right_knee[0]), pos_right_stancefoot[0]], [float(pos_right_knee[2]), pos_right_stancefoot[1]], color='b', linewidth=2, label='Leg 1')
    
    # Draw tib 2
    ax.plot([float(pos_left_knee[0]), pos_left_swingfoot[0]], [float(pos_left_knee[2]), pos_left_swingfoot[2]], color='r', linewidth=2, label='Leg 2')
    
    # plt.axis('equal')
    ax.legend()
        
        
if __name__ == '__main__':
    # q = [xbar, zbar, rotY, q1R, q2R, q1L, q2L]
    
    q_init = jnp.array([0, 0.658, 0, -0.6828+jnp.pi, 1.168, -0.6489+jnp.pi, 1.281])
    qdot_init = jnp.zeros(7)
    
    x_init = jnp.concatenate([q_init, qdot_init])
    
    fig, ax = plt.subplots()
    
    draw_5link(q_init, ax)
    
    plt.show()
import jax.numpy as jnp
import jax
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from rabbit_kinematics import *
from rabbit_dynamics import *

import numpy as np

def hip_position(q):
    # Unpack the 7 generalized coordinates (MATLAB q(1) ... q(7) -> Python q[0] ... q[6])
    xbar  = q[0]
    zbar  = q[1]
    rotY  = q[2]
    
    # Compute the hip position
    posHip = jnp.array([xbar + (63/100) * jnp.sin(rotY),
                        zbar + (63/100) * jnp.cos(rotY)])
    return posHip

def hip_jacobian(xb, zb, rotYb, q1R, q2R, q1L, q2L):
    JacHip = jnp.array([
        [1, 0, (63/100)*jnp.cos(rotYb), 0, 0, 0, 0],
        [0, 1, (-63/100)*jnp.sin(rotYb), 0, 0, 0, 0]
    ])
    return JacHip


def com_position(q):
    # Unpack the 7 generalized coordinates
    xbar  = q[0]
    zbar  = q[1]
    rotY  = q[2]
    q1R   = q[3]
    q2R   = q[4]
    q1L   = q[5]
    q2L   = q[6]
    
    # Precompute common trigonometric values
    sin_rotY = jnp.sin(rotY)
    cos_rotY = jnp.cos(rotY)
    
    sin_q1L = jnp.sin(q1L)
    cos_q1L = jnp.cos(q1L)
    
    sin_q1R = jnp.sin(q1R)
    cos_q1R = jnp.cos(q1R)
    
    # -----------------------------
    # Compute the x-coordinate of COM
    # -----------------------------
    # Term A: contribution from the trunk
    termA_x = 12 * (xbar + (6/25) * sin_rotY)
    
    # Term B: left leg, first term
    termB_x = (34/5) * (xbar + (11/100) * (cos_rotY * sin_q1L + cos_q1L * sin_rotY))
    
    # Term C: right leg, first term
    termC_x = (34/5) * (xbar + (11/100) * (cos_rotY * sin_q1R + cos_q1R * sin_rotY))
    
    # Term D: left leg swing contribution
    termD_x = (16/5) * (
        xbar +
        (2/5) * (1 - jnp.cos(q2L)) * (cos_rotY * sin_q1L + cos_q1L * sin_rotY) +
        (-2/5) * jnp.sin(q2L) * (cos_q1L * cos_rotY - sin_q1L * sin_rotY) +
        (16/25) * (
            jnp.cos(q2L) * (cos_rotY * sin_q1L + cos_q1L * sin_rotY) +
            jnp.sin(q2L) * (cos_q1L * cos_rotY - sin_q1L * sin_rotY)
        )
    )
    
    # Term E: right leg swing contribution
    termE_x = (16/5) * (
        xbar +
        (2/5) * (1 - jnp.cos(q2R)) * (cos_rotY * sin_q1R + cos_q1R * sin_rotY) +
        (-2/5) * jnp.sin(q2R) * (cos_q1R * cos_rotY - sin_q1R * sin_rotY) +
        (16/25) * (
            jnp.cos(q2R) * (cos_rotY * sin_q1R + cos_q1R * sin_rotY) +
            jnp.sin(q2R) * (cos_q1R * cos_rotY - sin_q1R * sin_rotY)
        )
    )
    
    posCOM_x = (1/32) * (termA_x + termB_x + termC_x + termD_x + termE_x)
    
    # -----------------------------
    # Compute the z-coordinate of COM
    # -----------------------------
    # Term A: contribution from the trunk
    termA_z = 12 * (zbar + (6/25) * cos_rotY)
    
    # Term B: left leg, first term
    termB_z = (34/5) * (zbar + (11/100) * (cos_q1L * cos_rotY - sin_q1L * sin_rotY))
    
    # Term C: right leg, first term
    termC_z = (34/5) * (zbar + (11/100) * (cos_q1R * cos_rotY - sin_q1R * sin_rotY))
    
    # For the swing contributions below note:
    #   (-1).*cos(rotY).*sin(q1) becomes -cos(rotY)*sin(q1)
    #   and (cos(q1).*cos(rotY)+(-1).*sin(q1).*sin(rotY)) becomes (cos(q1)*cos(rotY) - sin(q1)*sin(rotY))
    
    # Term D: left leg swing contribution for z
    termD_z = (16/5) * (
        zbar +
        (-2/5) * jnp.sin(q2L) * (- (cos_rotY * sin_q1L + cos_q1L * sin_rotY)) +
        (2/5) * (1 - jnp.cos(q2L)) * (cos_q1L * cos_rotY - sin_q1L * sin_rotY) +
        (16/25) * (
            jnp.sin(q2L) * (- (cos_rotY * sin_q1L + cos_q1L * sin_rotY)) +
            jnp.cos(q2L) * (cos_q1L * cos_rotY - sin_q1L * sin_rotY)
        )
    )
    # Simplify the negatives in termD_z:
    #   -2/5 * sin(q2L) * ( -(...) ) = (2/5)* sin(q2L)*(cos_rotY*sin_q1L+cos_q1L*sin_rotY)
    #   and similarly for the (16/25) term.
    termD_z = (16/5) * (
        zbar +
        (2/5) * jnp.sin(q2L) * (cos_rotY * sin_q1L + cos_q1L * sin_rotY) +
        (2/5) * (1 - jnp.cos(q2L)) * (cos_q1L * cos_rotY - sin_q1L * sin_rotY) +
        (16/25) * (
            - jnp.sin(q2L) * (cos_rotY * sin_q1L + cos_q1L * sin_rotY) +
            jnp.cos(q2L) * (cos_q1L * cos_rotY - sin_q1L * sin_rotY)
        )
    )
    
    # Term E: right leg swing contribution for z
    termE_z = (16/5) * (
        zbar +
        (-2/5) * jnp.sin(q2R) * (- (cos_rotY * sin_q1R + cos_q1R * sin_rotY)) +
        (2/5) * (1 - jnp.cos(q2R)) * (cos_q1R * cos_rotY - sin_q1R * sin_rotY) +
        (16/25) * (
            jnp.sin(q2R) * (- (cos_rotY * sin_q1R + cos_q1R * sin_rotY)) +
            jnp.cos(q2R) * (cos_q1R * cos_rotY - sin_q1R * sin_rotY)
        )
    )
    termE_z = (16/5) * (
        zbar +
        (2/5) * jnp.sin(q2R) * (cos_rotY * sin_q1R + cos_q1R * sin_rotY) +
        (2/5) * (1 - jnp.cos(q2R)) * (cos_q1R * cos_rotY - sin_q1R * sin_rotY) +
        (16/25) * (
            - jnp.sin(q2R) * (cos_rotY * sin_q1R + cos_q1R * sin_rotY) +
            jnp.cos(q2R) * (cos_q1R * cos_rotY - sin_q1R * sin_rotY)
        )
    )
    
    posCOM_z = (1/32) * (termA_z + termB_z + termC_z + termD_z + termE_z)
    
    # Return the COM position as a 2-element vector [x, z]
    return jnp.array([posCOM_x, posCOM_z])


def fxgu_floating_base(t, x, u, params):
    q = x[0:7]
    dq = x[7:]
    B = B_matrix()
    C = C_matrix(q, dq)
    D = D_matrix(q)
    G = G_vector(q)
    
    # Compute Fx and Gx
    Fx = jnp.linalg.solve(D, -C @ x[5:10] - G.flatten())
    Gx = jnp.linalg.solve(D, B)
    
    # Compute state derivatives
    dx = jnp.zeros_like(x)
    dx[:5] = x[5:10]
    dx[5:10] = Fx + Gx @ u
    
    return dx


def foot_touching_event(t,x):
    q = x[0:7]
    return left_swing_foot_position(q)[1]


def check_swing_foot_clearance(p_sw_world):
    return p_sw_world[1] > 0


def integrate_fxgu_euler(x0, u, dt):
    return x0 + fxgu_floating_base(0.0, x0, u) * dt


# def integrate_fxgu(x0, u, params, event=True, tstart=0.0, tfinal=0.2):
#     print("Integrating the 5-link model with the given initial state.")    
#     num_t_eval = 300
#     t_eval = np.linspace(tstart, tfinal, num_t_eval)
#     t_span = [tstart, tfinal]
    
#     if event:
#       event_func = foot_touching_event
#     else:
#       event_func = None
    
#     options = {
#         'rtol': 1e-5,
#         'atol': 1e-6,
#         'events': event_func
#     }
    
#     # Solve until the first terminal event
#     sol = solve_ivp(
#        fun=lambda t, x: fxgu_floating_base(t, x, u, params),
#         t_span=t_span,
#         t_eval=t_eval,
#         y0=x0,
#         # method='RK45', 
#         # rtol=options['rtol'],
#         # atol=options['atol'],
#         events=options['events']
#     )
    
#     t_trj = sol.t
#     x_trj = sol.y.T
    
#     if sol.t_events:
#       te = sol.t_events[0][0]
#       xe = sol.y_events[0][0]
      
#       print("Swing foot guard time:")
#       print(te)
      
#       print("Swing foot guard state:")
#       print(xe)
      
#       print("Swing foot guard function value:")
#       print(foot_touching_event(te, xe))

#       # Create a mask for all integrated times strictly less than te.
#       mask = t_trj <= te
      
#       # Select all times and states up until the event
#       t_trj = sol.t[mask]
#       x_trj = x_trj[mask]
    
#     return t_trj, x_trj


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
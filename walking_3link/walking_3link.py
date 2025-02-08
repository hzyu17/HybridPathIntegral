import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jacfwd 
jax.config.update("jax_enable_x64", True)

import os
import sys
file_path = os.path.abspath(__file__)
exp_dir = os.path.dirname(file_path)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
root_dir = os.path.abspath(os.path.join(exp_dir, '..'))
sys.path.append(root_dir)

from dynamics import *
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from dynamics.saltation_matrix import compute_saltation
from dynamics.dynamics import *


# ================ dynamics parameters ================
l = 0.5 # torso length
r = 1.0 # leg length
MT = 10.0 # torso mass
MH = 15.0 # hip mass
m = 5.0 # leg mass
g0 = 9.8 # gravity

# Optimization parameters
a = [0.512, 0.073, 0.035, -0.819, -2.27, 3.26, 3.11, 1.89]

# Define global variables
t_2 = []
torque = []
y = []
force = []
u_trj = []

# mode 1: (vertical) hip velocity < 0, and before the swing foot touches the ground
# mode 2: from the touching to (vertical) hip velocity < 0.
# guard function: swing leg touches the ground, and the swing leg is in front of the stance leg.
# p2_v(q) = 0; assuming r = 1

# Guard function from mode 1 to mode 2 (direction = 1 for the 3-link)
def guard_3link_12(t, x_event):
    th1, th2 = x_event[0], x_event[1]
    # swing_foot_vV = -x_event[3]*jnp.sin(x_event[0]) + x_event[4]*jnp.sin(x_event[1])
    # # The swing foot reach the ground from under the ground for 3-link, due to scuffing... For 5-link will be normal guard.
    return np.cos(th1) - np.cos(th2)

def guard_3link_12_jax(t, x_event):
    th1, th2 = x_event[0], x_event[1]
    # swing_foot_vV = -x_event[3]*jnp.sin(x_event[0]) + x_event[4]*jnp.sin(x_event[1])
    # # The swing foot reach the ground from under the ground for 3-link due to scuffing... For 5-link will be normal guard.
    return jnp.cos(th1) - jnp.cos(th2)

# Guard function from mode 2 to mode 1 (direction = -1)
def guard_3link_21(t, x_event):
    return hipvel_V_pt(x_event)

def guard_3link_21_jax(t, x_event):
    return hipvel_V_pt_jax(x_event)

gt_3link_12 = jax.jit(jacfwd(lambda t, x: guard_3link_12_jax(t, x), 0))
gx_3link_12 = jax.jit(jacfwd(lambda t, x: guard_3link_12_jax(t, x), 1))

gt_3link_21 = jax.jit(jacfwd(lambda t, x: guard_3link_21_jax(t, x), 0))
gx_3link_21 = jax.jit(jacfwd(lambda t, x: guard_3link_21_jax(t, x), 1))


def resetmap_3link_21(t,x,cur_mode=1,args=None):
    return x, 0, None

def resetmap_3link_21_jax(t,x,cur_mode=1,args=None):
    return x

def resetmap_3link_12(t,x,cur_mode=0,reset_args=None): 
    # Unpack state variables
    th1, th2, th3 = x[:3]

    # De matrix
    De = np.zeros((5, 5))
    De[0, 0] = (r**2 * (4 * MH + 4 * MT + 5 * m)) / 4
    De[0, 1] = -(m * r**2 * (np.cos(th1) * np.cos(th2) + np.sin(th1) * np.sin(th2))) / 2
    De[0, 2] = l * MT * r * (np.cos(th1) * np.cos(th3) + np.sin(th1) * np.sin(th3))
    De[0, 3] = (r * np.cos(th1) * (2 * MH + 2 * MT + 3 * m)) / 2
    De[0, 4] = -(r * np.sin(th1) * (2 * MH + 2 * MT + 3 * m)) / 2
    De[1, 0] = De[0, 1]
    De[1, 1] = (m * r**2) / 4
    De[1, 3] = -(m * r * np.cos(th2)) / 2
    De[1, 4] = (m * r * np.sin(th2)) / 2
    De[2, 0] = De[0, 2]
    De[2, 2] = l**2 * MT
    De[2, 3] = l * MT * np.cos(th3)
    De[2, 4] = -l * MT * np.sin(th3)
    De[3, 0] = De[0, 3]
    De[3, 1] = De[1, 3]
    De[3, 2] = De[2, 3]
    De[3, 3] = MH + MT + 2 * m
    De[4, 0] = De[0, 4]
    De[4, 1] = De[1, 4]
    De[4, 2] = De[2, 4]
    De[4, 4] = MH + MT + 2 * m

    # E matrix
    E = np.zeros((2, 5))
    E[0, 0] = r * np.cos(th1)
    E[0, 1] = -r * np.cos(th2)
    E[0, 3] = 1
    E[1, 0] = -r * np.sin(th1)
    E[1, 1] = r * np.sin(th2)
    E[1, 4] = 1

    # Solve for the transition using the equation from Grizzle's paper
    A = np.block([[De, -E.T], [E, np.zeros((2, 2))]])
    b = np.hstack([De @ np.hstack([x[3:6], [0, 0]]), [0, 0]])
    tmp_vec = np.linalg.solve(A, b)

    # Update state vector after impact
    x_new = np.zeros(6)
    x_new[0] = x[1]
    x_new[1] = x[0]
    x_new[2] = x[2]
    x_new[3] = tmp_vec[1]
    x_new[4] = tmp_vec[0]
    x_new[5] = tmp_vec[2]
    # x_new[6] = tmp_vec[5]
    # x_new[7] = tmp_vec[6]

    z2_new = tmp_vec[4]
    nextmode = 1
    byproduct = None

    return x_new, nextmode, byproduct


def resetmap_3link_12_jax(t, x, cur_mode=0, reset_args=None): 
    # Unpack state variables
    th1, th2, th3 = x[:3]

    # De matrix
    De = jnp.zeros((5, 5))
    De.at[0, 0].set((r**2 * (4 * MH + 4 * MT + 5 * m)) / 4)
    De.at[0, 1].set(-(m * r**2 * (jnp.cos(th1) * jnp.cos(th2) + jnp.sin(th1) * jnp.sin(th2))) / 2)
    De.at[0, 2].set(l * MT * r * (jnp.cos(th1) * jnp.cos(th3) + jnp.sin(th1) * jnp.sin(th3)))
    De.at[0, 3].set((r * jnp.cos(th1) * (2 * MH + 2 * MT + 3 * m)) / 2)
    De.at[0, 4].set(-(r * jnp.sin(th1) * (2 * MH + 2 * MT + 3 * m)) / 2)
    De.at[1, 0].set(De[0, 1])
    De.at[1, 1].set((m * r**2) / 4)
    De.at[1, 3].set(-(m * r * jnp.cos(th2)) / 2)
    De.at[1, 4].set((m * r * jnp.sin(th2)) / 2)
    De.at[2, 0].set(De[0, 2])
    De.at[2, 2].set(l**2 * MT)
    De.at[2, 3].set(l * MT * jnp.cos(th3))
    De.at[2, 4].set(-l * MT * jnp.sin(th3))
    De.at[3, 0].set(De[0, 3])
    De.at[3, 1].set(De[1, 3])
    De.at[3, 2].set(De[2, 3])
    De.at[3, 3].set(MH + MT + 2 * m)
    De.at[4, 0].set(De[0, 4])
    De.at[4, 1].set(De[1, 4])
    De.at[4, 2].set(De[2, 4])
    De.at[4, 4].set(MH + MT + 2 * m)

    # E matrix
    E = jnp.zeros((2, 5))
    
    E.at[0, 0].set(r * jnp.cos(th1))
    E.at[0, 1].set(-r * jnp.cos(th2))
    E.at[0, 3].set(1)
    E.at[1, 0].set(-r * jnp.sin(th1))
    E.at[1, 1].set(r * jnp.sin(th2))
    E.at[1, 4].set(1)

    # Solve for the transition using the equation from Grizzle's paper
    A = jnp.block([[De, -E.T], [E, jnp.zeros((2, 2))]])
    b = jnp.hstack([De @ jnp.hstack([x[3:6], [0, 0]]), [0, 0]])
    tmp_vec = jnp.linalg.solve(A, b)

    # Update state vector after impact
    x_new = jnp.zeros(6)
    
    x_new.at[0].set(x[1])
    x_new.at[1].set(x[0])
    x_new.at[2].set(x[2])
    x_new.at[3].set(tmp_vec[1])
    x_new.at[4].set(tmp_vec[0])
    x_new.at[5].set(tmp_vec[2])
    # x_new.at[6].set(tmp_vec[5])
    # x_new.at[7].set(tmp_vec[6])

    # z2_new = tmp_vec[4]

    return x_new

Rt_3link_12 = jax.jit(jacfwd(lambda t, x, mode, byproduct: resetmap_3link_12_jax(t, x), 0))
Rx_3link_12 = jax.jit(jacfwd(lambda t, x, mode, byproduct: resetmap_3link_12_jax(t, x), 1))

Rt_3link_21 = jax.jit(jacfwd(lambda t, x, mode, byproduct: resetmap_3link_21_jax(t, x), 0))
Rx_3link_21 = jax.jit(jacfwd(lambda t, x, mode, byproduct: resetmap_3link_21_jax(t, x), 1))


def detect_3link(x0, u, t0, tf, current_mode, reset_args, detect=True, backwards=False):
    guard_3link_12.terminal=True
    guard_3link_12.direction=1
    
    guard_3link_21.terminal=True
    guard_3link_21.direction=-1
    
    smoothdyn_3link = {0:dyn_control_3link, 1:dyn_control_3link}
    
    Rxs_3link = {0:Rx_3link_12, 1:Rx_3link_21}
    Rts_3link = {0:Rt_3link_12, 1:Rt_3link_21}
    
    gxs_3link = {0:gx_3link_12, 1:gx_3link_21}
    gts_3link = {0:gt_3link_12, 1:gt_3link_21}
    
    guards_3link = {0:guard_3link_12, 1: guard_3link_21}
    resetmaps_3link = {0:resetmap_3link_12, 1:resetmap_3link_21}
    
    return event_detect_onestep(x0, u, 
                                t0, tf, 
                                current_mode, 
                                smoothdyn_3link, 
                                guards_3link, gxs_3link, gts_3link,
                                resetmaps_3link, Rxs_3link, Rts_3link,
                                reset_args, detect, backwards)


def sysmat_3link(x, a):
    """
    Model of a three-link biped walker.

    Parameters:
        x : array-like
            State vector (size 6).
        a : array-like
            Coefficients for the dynamics.

    Returns:
        D, C, G, B, K, dV, dVl, Al, Bl, H, LfH, dLfH : tuple of arrays
            Matrices and vectors representing the system dynamics.
    """
    # Retrieve model and control parameters
    th3d, th1d, alpha, epsilon = control_params_three_link()

    # Unpack state variables
    th1, th2, th3 = x[0], x[1], x[2]
    dth1, dth2, dth3 = x[3], x[4], x[5]

    # D matrix
    D = np.zeros((3, 3))
    D[0, 0] = MH * r**2 + MT * r**2 + (5 * m * r**2) / 4
    D[0, 1] = -(m * r**2 * np.cos(th1 - th2)) / 2
    D[0, 2] = l * MT * r * np.cos(th1 - th3)
    D[1, 0] = D[0, 1]
    D[1, 1] = (m * r**2) / 4
    D[2, 0] = D[0, 2]
    D[2, 2] = l**2 * MT

    # C matrix
    C = np.zeros((3, 3))
    C[0, 1] = -(dth2 * m * r**2 * np.sin(th1 - th2)) / 2
    C[0, 2] = l * MT * dth3 * r * np.sin(th1 - th3)
    C[1, 0] = (dth1 * m * r**2 * np.sin(th1 - th2)) / 2
    C[2, 0] = -l * MT * dth1 * r * np.sin(th1 - th3)

    # G matrix
    G = np.zeros(3)
    G[0] = -MH * g0 * r * np.sin(th1) - MT * g0 * r * np.sin(th1) - (3 * g0 * m * r * np.sin(th1)) / 2
    G[1] = (g0 * m * r * np.sin(th2)) / 2
    G[2] = -l * MT * g0 * np.sin(th3)

    # B matrix
    B = np.zeros((3, 2))
    B[0, 0] = -1
    B[1, 1] = -1
    B[2, 0] = 1
    B[2, 1] = 1

    # K matrix
    K = np.zeros((2, 4))
    K[0, 0] = 156
    K[0, 2] = 25
    K[1, 1] = 110
    K[1, 3] = 21

    # dV matrix
    dV = np.zeros((1, 6))
    dV[0, 0] = (184 * th1) / 77 - 10 * dth2 - 10 * dth1 + (184 * th2) / 77
    dV[0, 1] = dV[0, 0]
    dV[0, 2] = (4515147318722739 * th3) / 2251799813685248 - 10 * dth3 - \
               (4515147318722739 * th3d) / 4503599627370496 - \
               (4515147318722739 * th3d) / 4503599627370496
    dV[0, 3] = (370 * dth1) / 7 + (370 * dth2) / 7 - 10 * th1 - 10 * th2
    dV[0, 4] = dV[0, 3]
    dV[0, 5] = (4419157134357289 * dth3) / 70368744177664 - 10 * th3 + 5 * th3d + 5 * th3d

    # dVl matrix
    dVl = np.zeros((1, 4))
    dVl[0, 0] = (4515147318722739 * th3) / 2251799813685248 - 10 * dth3 - \
                (4515147318722739 * th3d) / 2251799813685248
    dVl[0, 1] = dV[0, 0]
    dVl[0, 2] = (4419157134357289 * dth3) / 70368744177664 - 10 * th3 + 10 * th3d
    dVl[0, 3] = (370 * dth1) / 7 + (370 * dth2) / 7 - 10 * th1 - 10 * th2

    # Al matrix
    Al = np.zeros((4, 4))
    Al[0, 2] = 1
    Al[1, 3] = 1

    # Bl matrix
    Bl = np.zeros((4, 2))
    Bl[2, 0] = 1
    Bl[3, 1] = 1

    # Extract coefficients
    a01, a11, a21, a31 = a[:4]
    a02, a12, a22, a32 = a[4:]

    # H matrix
    H = np.zeros((2, 1))
    H[0, 0] = th3 - a01 - a11 * th1 - a21 * th1**2 - a31 * th1**3
    H[1, 0] = th1 + th2 - (th1 + th1d) * (th1 - th1d) * (a02 + a12 * th1 + a22 * th1**2 + a32 * th1**3)

    # LfH matrix
    LfH = np.zeros((2, 1))
    LfH[0, 0] = dth3 - dth1 * (a11 + 2 * a21 * th1 + 3 * a31 * th1**2)
    LfH[1, 0] = dth2 - dth1 * ((th1 - th1d) * (a02 + a12 * th1 + a22 * th1**2 + a32 * th1**3) +
                               (th1 + th1d) * (a02 + a12 * th1 + a22 * th1**2 + a32 * th1**3) +
                               (th1 + th1d) * (th1 - th1d) * (a12 + 2 * a22 * th1 + 3 * a32 * th1**2) - 1)

    # dLfH matrix
    dLfH = np.zeros((2, 6))
    dLfH[0, 0] = -dth1 * (2 * a21 + 6 * a31 * th1)
    dLfH[0, 3] = -(a11 + 2 * a21 * th1 + 3 * a31 * th1**2)
    dLfH[0, 5] = 1
    dLfH[1, 0] = -dth1 * (2 * a02 + 2 * (th1 + th1d) * (a12 + 2 * a22 * th1 + 3 * a32 * th1**2) +
                          2 * a12 * th1 + 2 * (th1 - th1d) * (a12 + 2 * a22 * th1 + 3 * a32 * th1**2) +
                          2 * a22 * th1**2 + 2 * a32 * th1**3 +
                          (th1 + th1d) * (2 * a22 + 6 * a32 * th1) * (th1 - th1d))
    dLfH[1, 3] = 1 - (th1 + th1d) * (a02 + a12 * th1 + a22 * th1**2 + a32 * th1**3) - \
                 (th1 + th1d) * (th1 - th1d) * (a12 + 2 * a22 * th1 + 3 * a32 * th1**2) - \
                 (th1 - th1d) * (a02 + a12 * th1 + a22 * th1**2 + a32 * th1**3)
    dLfH[1, 4] = 1

    return D, C, G, B, K, dV, dVl, Al, Bl, H, LfH, dLfH


def sysmat_3link_jax(x, a):
    """
    Model of a three-link biped walker.

    Parameters:
        x : array-like
            State vector (size 6).
        a : array-like
            Coefficients for the dynamics.

    Returns:
        D, C, G, B, K, dV, dVl, Al, Bl, H, LfH, dLfH : tuple of arrays
            Matrices and vectors representing the system dynamics.
    """
    # Retrieve model and control parameters
    th3d, th1d, alpha, epsilon = control_params_three_link()
    
    # Unpack state variables
    th1, th2, th3 = x[0], x[1], x[2]
    dth1, dth2, dth3 = x[3], x[4], x[5]

    # D matrix
    D = jnp.zeros((3, 3))
    D = D.at[0, 0].set(MH * r**2 + MT * r**2 + (5 * m * r**2) / 4)
    D = D.at[0, 1].set(-(m * r**2 * jnp.cos(th1 - th2)) / 2)
    D = D.at[0, 2].set(l * MT * r * jnp.cos(th1 - th3))
    D = D.at[1, 0].set(D[0, 1])
    D = D.at[1, 1].set((m * r**2) / 4)
    D = D.at[2, 0].set(D[0, 2])
    D = D.at[2, 2].set(l**2 * MT)

    # C matrix
    C = jnp.zeros((3, 3))
    C = C.at[0, 1].set(-(dth2 * m * r**2 * jnp.sin(th1 - th2)) / 2)
    C = C.at[0, 2].set(l * MT * dth3 * r * jnp.sin(th1 - th3))
    C = C.at[1, 0].set((dth1 * m * r**2 * jnp.sin(th1 - th2)) / 2)
    C = C.at[2, 0].set(-l * MT * dth1 * r * jnp.sin(th1 - th3))

    # G matrix
    G = jnp.zeros(3)
    G = G.at[0].set(-MH * g0 * r * jnp.sin(th1) - MT * g0 * r * jnp.sin(th1) - (3 * g0 * m * r * jnp.sin(th1)) / 2)
    G = G.at[1].set((g0 * m * r * jnp.sin(th2)) / 2)
    G = G.at[2].set(-l * MT * g0 * jnp.sin(th3))

    # B matrix
    B = jnp.zeros((3, 2))
    B = B.at[0, 0].set(-1)
    B = B.at[1, 1].set(-1)
    B = B.at[2, 0].set(1)
    B = B.at[2, 1].set(1)

    # K matrix
    K = jnp.zeros((2, 4))
    K = K.at[0, 0].set(156)
    K = K.at[0, 2].set(25)
    K = K.at[1, 1].set(110)
    K = K.at[1, 3].set(21)

    # dV matrix
    dV = jnp.zeros((1, 6))
    dV = dV.at[0, 0].set((184 * th1) / 77 - 10 * dth2 - 10 * dth1 + (184 * th2) / 77)
    dV = dV.at[0, 1].set(dV[0, 0])
    dV = dV.at[0, 2].set((4515147318722739 * th3) / 2251799813685248 - 10 * dth3 - \
                    (4515147318722739 * th3d) / 4503599627370496 - \
                    (4515147318722739 * th3d) / 4503599627370496)
    dV = dV.at[0, 3].set((370 * dth1) / 7 + (370 * dth2) / 7 - 10 * th1 - 10 * th2)
    dV = dV.at[0, 4].set(dV[0, 3])
    dV = dV.at[0, 5].set((4419157134357289 * dth3) / 70368744177664 - 10 * th3 + 5 * th3d + 5 * th3d)

    # dVl matrix
    dVl = jnp.zeros((1, 4))
    dVl = dVl.at[0, 0].set((4515147318722739 * th3) / 2251799813685248 - 10 * dth3 - \
                (4515147318722739 * th3d) / 2251799813685248)
    dVl = dVl.at[0, 1].set(dV[0, 0])
    dVl = dVl.at[0, 2].set((4419157134357289 * dth3) / 70368744177664 - 10 * th3 + 10 * th3d)
    dVl = dVl.at[0, 3].set((370 * dth1) / 7 + (370 * dth2) / 7 - 10 * th1 - 10 * th2)

    # Al matrix
    Al = jnp.zeros((4, 4))
    Al = Al.at[0, 2].set(1)
    Al = Al.at[1, 3].set(1)

    # Bl matrix
    Bl = jnp.zeros((4, 2))
    Bl = Bl.at[2, 0].set(1)
    Bl = Bl.at[3, 1].set(1)

    # Extract coefficients
    a01, a11, a21, a31 = a[:4]
    a02, a12, a22, a32 = a[4:]

    # H matrix
    H = jnp.zeros((2, 1))
    H = H.at[0, 0].set(th3 - a01 - a11 * th1 - a21 * th1**2 - a31 * th1**3)
    H = H.at[1, 0].set(th1 + th2 - (th1 + th1d) * (th1 - th1d) * (a02 + a12 * th1 + a22 * th1**2 + a32 * th1**3))

    # LfH matrix
    LfH = jnp.zeros((2, 1))
    LfH = LfH.at[0, 0].set(dth3 - dth1 * (a11 + 2 * a21 * th1 + 3 * a31 * th1**2))
    LfH = LfH.at[1, 0].set(dth2 - dth1 * ((th1 - th1d) * (a02 + a12 * th1 + a22 * th1**2 + a32 * th1**3) +
                    (th1 + th1d) * (a02 + a12 * th1 + a22 * th1**2 + a32 * th1**3) +
                    (th1 + th1d) * (th1 - th1d) * (a12 + 2 * a22 * th1 + 3 * a32 * th1**2) - 1))

    # dLfH matrix
    dLfH = jnp.zeros((2, 6))
    dLfH = dLfH.at[0, 0].set(-dth1 * (2 * a21 + 6 * a31 * th1))
    dLfH = dLfH.at[0, 3].set(-(a11 + 2 * a21 * th1 + 3 * a31 * th1**2))
    dLfH = dLfH.at[0, 5].set(1)
    dLfH = dLfH.at[1, 0].set(-dth1 * (2 * a02 + 2 * (th1 + th1d) * (a12 + 2 * a22 * th1 + 3 * a32 * th1**2) +
                          2 * a12 * th1 + 2 * (th1 - th1d) * (a12 + 2 * a22 * th1 + 3 * a32 * th1**2) +
                          2 * a22 * th1**2 + 2 * a32 * th1**3 +
                          (th1 + th1d) * (2 * a22 + 6 * a32 * th1) * (th1 - th1d)))
    dLfH = dLfH.at[1, 3].set(1 - (th1 + th1d) * (a02 + a12 * th1 + a22 * th1**2 + a32 * th1**3) - \
                 (th1 + th1d) * (th1 - th1d) * (a12 + 2 * a22 * th1 + 3 * a32 * th1**2) - \
                 (th1 - th1d) * (a02 + a12 * th1 + a22 * th1**2 + a32 * th1**3))
    dLfH = dLfH.at[1, 4].set(1)

    return D, C, G, B, K, dV, dVl, Al, Bl, H, LfH, dLfH


def Ds(x):
    th1, th2, th3, _, _, _ = x
    D_s = np.zeros((6, 6))
    D_s[0, 0] = (1.25*m + MH + MT)*r**2
    D_s[0, 1] = -0.5*m*r**2*np.cos(th1-th2)
    D_s[0, 2] = MT*r*l*np.cos(th1-th3)
    D_s[1, 1] = 0.25*m*r**2
    D_s[2, 2] = MT*l*l
    
    # Get other entries by symmetry
    D_s = (D_s + D_s.T) / 2
    
    return D_s
    
    
def Cs(x):
    th1, th2, th3, dth1, dth2, dth3 = x
    C_s = np.zeros((6, 6))
    C_s[0,1] = -0.5*m*r**2*np.sin(th1-th2)*dth2
    C_s[0,2] = -MT*r*l*np.sin(-th1+th3)*dth3
    C_s[1,0] = 1/2*r^2*np.sin(th1-th2)*m*dth1
    C_s[2,0] = MT*r*l*np.sin(-th1+th3)*dth1
    
    return C_s


def Gs(x):
    th1, th2, th3, dth1, dth2, dth3 = x
    G = np.zeros((3,1))
    G[0] = -3/2*r*np.sin(th1)*m*g0-r*np.sin(th1)*MH*g0-r*np.sin(th1)*MT*g0
    G[1] = 1/2*r*np.sin(th2)*m*g0
    G[2] = -l*np.sin(th3)*MT*g0
    
    return G


def Bs(x):
    # th1, th2, th3, dth1, dth2, dth3 = x
    B=np.zeros((3,2))
    B[0,0] = -1
    B[1,1] = -1
    B[2,0] = 1
    B[2,1] = 1
    
    return B

def control_params_three_link():
    # Replace with the actual implementation
    th3d=np.pi/6.0   
    th1d = np.pi/8.0 
    alpha=0.9  
    epsilon=0.1

    return th3d, th1d, alpha, epsilon


def switching_leg_events(t, x, args):
    # Retrieve parameters
    th3d, th1d, alpha, epsilon = control_params_three_link()
    
    return th1d - x[0]
switching_leg_events.terminal = True 
switching_leg_events.direction = 0

def control_three_link(H, LfH):
    """
    Calculate the control for the feedback linearized three-link biped walker.

    Parameters:
        H : array-like
            Output function.
        LfH : array-like
            Derivative of the output function.

    Returns:
        v : ndarray
            Control vector.
    """
    # Retrieve control parameters
    th3d, th1d, alpha, epsilon = control_params_three_link()

    # Scale LfH
    LfH = epsilon * np.array(LfH)

    # Phi functions
    phi1 = H[0] + 1 / (2 - alpha) * np.sign(LfH[0]) * abs(LfH[0])**(2 - alpha)
    phi2 = H[1] + 1 / (2 - alpha) * np.sign(LfH[1]) * abs(LfH[1])**(2 - alpha)

    # Psi functions
    psi = np.zeros((2, 1))
    psi[0, 0] = -np.sign(LfH[0]) * abs(LfH[0])**alpha - np.sign(phi1) * abs(phi1)**(alpha / (2 - alpha))
    psi[1, 0] = -np.sign(LfH[1]) * abs(LfH[1])**alpha - np.sign(phi2) * abs(phi2)**(alpha / (2 - alpha))

    # Calculate control
    v = psi / epsilon**2

    return v

# compute u inside the function using feedback linearization
def fxgu_nom(t, x, a):
    """
    Compute the state derivative (dx/dt) for the three-link walker.

    Parameters:
        t : float
            Current time.
        x : array-like
            Current state vector.
        a : array-like
            Parameters for the dynamics.

    Returns:
        dx : ndarray
            Time derivative of the state vector.
    """
    global t_2, torque, y, force, u_trj

    # Extract dynamics matrices and control parameters
    D, C, G, B, K, dV, dVl, Al, Bl, H, LfH, dLfH = sysmat_3link(x[:6], a)
    
    # Compute Fx and Gx
    Fx = np.linalg.solve(D, -C @ x[3:6] - G)
    Gx = np.linalg.solve(D, B)
    v = control_three_link(H, LfH)

    # Compute control signal (u) using feedback linearization
    u = np.linalg.solve(dLfH @ np.vstack([np.zeros((3, 2)), Gx]), v.flatten() - dLfH @ np.hstack([x[3:6], Fx]).transpose())

    # Compute state derivatives
    dx = np.zeros_like(x)
    dx[:3] = x[3:6]
    dx[3:6] = Fx + Gx @ u

    # Update global variables
    torque.append(u)
    t_2.append(t)
    y.append(H)
    f_tan, f_norm = stance_force_three_link(x[:6], dx[:6], u)
    force.append([f_tan, f_norm])

    return dx


# fx+gu 
def dyn_control_3link(t, x, u):
    # Extract dynamics matrices and control parameters
    D, C, G, B, K, dV, dVl, Al, Bl, H, LfH, dLfH = sysmat_3link(x[:6], a)
    
    # Compute Fx and Gx
    Fx = np.linalg.solve(D, -C @ x[3:6] - G)
    Gx = np.linalg.solve(D, B)

    # Compute state derivatives
    dx = np.zeros_like(x)
    dx[:3] = x[3:6]
    dx[3:6] = Fx + Gx @ u

    return dx

# x + f*dt
def dyn_control_3link_discrete_jax(x, u, dt):
    # Extract dynamics matrices and control parameters
    D, C, G, B, K, dV, dVl, Al, Bl, H, LfH, dLfH = sysmat_3link_jax(x[:6], a)
    
    # Compute Fx and Gx
    Fx = jnp.linalg.solve(D, -C @ x[3:6] - G)
    Gx = jnp.linalg.solve(D, B)

    # Compute state derivatives
    f = jnp.zeros_like(x)
    f = f.at[:3].set(x[3:6])
    f = f.at[3:6].set(Fx + Gx @ u)

    return x + f*dt


def fxgu_3link_jax(x, u):
    # Extract dynamics matrices and control parameters
    D, C, G, B, K, dV, dVl, Al, Bl, H, LfH, dLfH = sysmat_3link_jax(x[:6], a)
    
    # Compute Fx and Gx
    Fx = jnp.linalg.solve(D, -C @ x[3:6] - G)
    Gx = jnp.linalg.solve(D, B)

    # Compute state derivatives
    Fx = jnp.zeros_like(x)
    Fx = Fx.at[:3].set(x[3:6])
    Fx = Fx.at[3:6].set(Fx + Gx @ u)
    
    return Fx

# linearizations and jacobians
jac_fxgu_x = jax.jit(jax.jacobian(lambda t, x, u, a: fxgu_3link_jax(t, x, u, a), argnums=1))
jac_fxgu_u = jax.jit(jax.jacobian(lambda t, x, u, a: fxgu_3link_jax(t, x, u, a), argnums=2))


def hip_moving_cost(x, u, target_hipvel= 2.0):
    hipvelocity = hipvel_H_pt_jax(x)
    return 0.01*jnp.linalg.norm(hipvelocity-target_hipvel) + u.T@u/2

def deltx_norm_cost(x, x_tar):
    return jnp.linalg.norm(x-x_tar)

def stance_force_three_link(x, dx, u):
    """
    Calculate the forces on the stance leg during impact.

    Parameters:
        x : array-like
            State vector (size 6).
        dx : array-like
            Derivative of the state vector (size 6).
        u : array-like
            Control input vector (size 2).

    Returns:
        f_tan : float
            Tangential force on the stance leg.
        f_norm : float
            Normal force on the stance leg.
    """
    # Unpack state variables
    th1, th2, th3 = x[:3]
    dth1, dth2, dth3 = x[3:6]

    # De11 matrix
    De11 = np.zeros((3, 3))
    De11[0, 0] = (r**2 * (4 * MH + 4 * MT + 5 * m)) / 4
    De11[0, 1] = -(m * r**2 * (np.cos(th1) * np.cos(th2) + np.sin(th1) * np.sin(th2))) / 2
    De11[0, 2] = l * MT * r * (np.cos(th1) * np.cos(th3) + np.sin(th1) * np.sin(th3))
    De11[1, 0] = De11[0, 1]
    De11[1, 1] = (m * r**2) / 4
    De11[2, 0] = De11[0, 2]
    De11[2, 2] = l**2 * MT

    # De12 matrix
    De12 = np.zeros((3, 2))
    De12[0, 0] = (r * np.cos(th1) * (2 * MH + 2 * MT + 3 * m)) / 2
    De12[0, 1] = -(r * np.sin(th1) * (2 * MH + 2 * MT + 3 * m)) / 2
    De12[1, 0] = -(m * r * np.cos(th2)) / 2
    De12[1, 1] = (m * r * np.sin(th2)) / 2
    De12[2, 0] = l * MT * np.cos(th3)
    De12[2, 1] = -l * MT * np.sin(th3)

    # De22 matrix
    De22 = np.zeros((2, 2))
    De22[0, 0] = MH + MT + 2 * m
    De22[1, 1] = MH + MT + 2 * m

    # Ce11 matrix
    Ce11 = np.zeros((3, 3))
    Ce11[0, 1] = (dth2 * m * r**2 * (np.cos(th1) * np.sin(th2) - np.cos(th2) * np.sin(th1))) / 2
    Ce11[0, 2] = -l * MT * dth3 * r * (np.cos(th1) * np.sin(th3) - np.cos(th3) * np.sin(th1))
    Ce11[1, 0] = -(dth1 * m * r**2 * (np.cos(th1) * np.sin(th2) - np.cos(th2) * np.sin(th1))) / 2
    Ce11[2, 0] = l * MT * dth1 * r * (np.cos(th1) * np.sin(th3) - np.cos(th3) * np.sin(th1))

    # Ce21 matrix
    Ce21 = np.zeros((2, 3))
    Ce21[0, 0] = -(dth1 * r * np.sin(th1) * (2 * MH + 2 * MT + 3 * m)) / 2
    Ce21[0, 1] = (dth2 * m * r * np.sin(th2)) / 2
    Ce21[0, 2] = -l * MT * dth3 * np.sin(th3)
    Ce21[1, 0] = -(dth1 * r * np.cos(th1) * (2 * MH + 2 * MT + 3 * m)) / 2
    Ce21[1, 1] = (dth2 * m * r * np.cos(th2)) / 2
    Ce21[1, 2] = -l * MT * dth3 * np.cos(th3)

    # Ge1 matrix
    Ge1 = np.zeros((3, 1))
    Ge1[0, 0] = -MH * g0 * r * np.sin(th1) - MT * g0 * r * np.sin(th1) - (3 * g0 * m * r * np.sin(th1)) / 2
    Ge1[1, 0] = (g0 * m * r * np.sin(th2)) / 2
    Ge1[2, 0] = -l * MT * g0 * np.sin(th3)

    # Ge2 matrix
    Ge2 = np.zeros((2, 1))
    Ge2[1, 0] = MH * g0 + MT * g0 + 2 * g0 * m

    # B matrix
    B = np.zeros((3, 2))
    B[0, 0] = -1
    B[1, 1] = -1
    B[2, 0] = 1
    B[2, 1] = 1

    # Compute forces
    De22_inv = np.linalg.inv(De22)
    De12_inv_De22 = De12 @ De22_inv
    DD = np.linalg.inv((De12_inv_De22.T @ De12_inv_De22)) @ De12_inv_De22.T

    F = DD @ (-(De11 - De12 @ De22_inv @ De12.T) @ dx[3:] +
              (De12 @ De22_inv @ Ce21 - Ce11) @ dx[:3] +
              De12 @ De22_inv @ Ge2.flatten() - Ge1.flatten() + B @ u)

    f_tan = F[0]
    f_norm = F[1]

    return f_tan, f_norm


def fxgudw(x, u, dt, eps):
    D = Ds(x)
    C = Cs(x)
    G = Gs(x)
    B = Bs(x)
    
    Fx = np.linalg.inv(D)@(-C*x[3:]-G)
    Gx = np.linalg.inv(D)@B
    
    dx = np.zeros(6)
    
    nu=u.shape[0]
    dw = np.sqrt(dt*eps)*np.random.randn(nu)
    
    dx[0:2] = x[3:]
    dx[3:] = Fx+Gx@u
    
    x_next = x + dx*dt + Gx@dw
    
    return x_next


def sigma_three_link(omega_1_minus, a):
    """
    Maps the velocity of the stance leg just before impact to the state of the system just before impact.

    Parameters:
        omega_1_minus : float
            Angular velocity of the stance leg just before impact.
        a : list or array-like
            Coefficients of the polynomial control parameters (length 8).

    Returns:
        x : list
            State of the system just before impact.
    """
    # Retrieve control parameters
    th3d, th1d, alpha, epsilon = control_params_three_link()

    # Extract coefficients
    a01, a11, a21, a31 = a[0], a[1], a[2], a[3]
    a02, a12, a22, a32 = a[4], a[5], a[6], a[7]

    # Define state variables
    th1 = th1d
    dth1 = omega_1_minus

    th3 = a01 + a11 * th1 + a21 * th1**2 + a31 * th1**3
    dth2 = dth1 * (
        (th1 - th1d) * (a02 + a12 * th1 + a22 * th1**2 + a32 * th1**3) +
        (th1 + th1d) * (a02 + a12 * th1 + a22 * th1**2 + a32 * th1**3) +
        (th1 + th1d) * (th1 - th1d) * (a12 + 2 * a22 * th1 + 3 * a32 * th1**2) - 1
    )
    dth3 = dth1 * (a11 + 2 * a21 * th1 + 3 * a31 * th1**2)

    # State vector
    x = np.array([th1, -th1, th3, dth1, dth2, dth3])

    return x


def limb_position(q, pH_horiz):
    # Define the position of stance leg foot
    pFoot1 = np.array([pH_horiz[0], 0])
    pH = np.array([pFoot1[0] + r * np.sin(q[0]), pFoot1[1] + r * np.cos(q[0])])
    pFoot2 = np.array([pH[0] - r * np.sin(q[1]), pH[1] - r * np.cos(q[1])])
    pT = np.array([pH[0] + l * np.sin(q[2]), pH[1] + l * np.cos(q[2])])

    return pFoot1, pFoot2, pH, pT

def hipheight_pt_jax(x):
    # Estimate the height of hips, assuming r=1
    pV = jnp.cos(x[0])
    return pV

def hipheight_jax(x):
    # Estimate the height of hips, assuming r=1
    pV = jnp.cos(x[:, 0])
    return pV

# Convert angle velocities to hip velocities
def hip_vel(x):
    vV = -np.sin(x[:, 0]) * x[:, 3]
    # Estimate the horizontal velocity of hips, assuming r=1
    vH = np.cos(x[:, 0]) * x[:, 3]
    return vV, vH

def hipvel_H_jax(x):
    # Estimate the horizontal velocity of hips, assuming r=1
    return jnp.cos(x[:, 0]) * x[:, 3]

def hipvel_V_jax(x):
    # Estimate the vertical velocity of hips, assuming r=1
    return -jnp.sin(x[:, 0]) * x[:, 3]

def hipvel_H_pt_jax(x):
    # Estimate the horizontal velocity of hips, assuming r=1
    vH = jnp.cos(x[0]) * x[3]
    return vH

def hipvel_V_pt(x):
    # Estimate the vertical velocity of hips, assuming r=1
    return -np.sin(x[0]) * x[3]

def hipvel_V_pt_jax(x):
    # Estimate the vertical velocity of hips, assuming r=1
    return -jnp.sin(x[0]) * x[3]

def swingfoot_height_jax(x):
    return jnp.cos(x[:, 0]) - jnp.cos(x[:, 1])

def swingfoot_vel_vertical_jax(x):
    return -x[:,3]*jnp.sin(x[:, 0]) + x[:,4]*jnp.sin(x[:, 1])

def anim(t, x, ts, speed, fig=None, loop=1):
    for _ in range(loop):
        # Retrieve the size of x
        n, m = x.shape

        # Calculate hip velocity
        vV, vH = hip_vel(x)  # convert angles to horizontal position of hips
        pH_horiz = np.zeros(n)
    
        # Estimate hip horizontal position by estimating integral of hip velocity
        for j in range(1, n):
            pH_horiz[j] = pH_horiz[j-1] + (t[j] - t[j-1]) * vH[j-1]

        # Evenly sample time and hip position
        te, pH_horiz = even_sample(t, pH_horiz.reshape(-1, 1), 1 / ts)
        te, xe = even_sample(t, x, 1 / ts)
        n, m = xe.shape

        # Set initial limb position
        q = xe[0, :3]
        pFoot1, pFoot2, pH, pT = limb_position(q, pH_horiz[0])

        # Set up the plot
        if fig is None:
            fig, ax = plt.subplots(figsize=(5, 4))
            
        else:
            ax = plt.subplot(3,4,12)
            
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.axis('off')
        ax.grid()

        # Model parameters
        scl = 0.04  # scaling factor for masses
        mr_legs = m**(1/3) * scl  # radius of mass for legs
        mr_torso = MT**(1/3) * scl  # radius of mass for torso

        # Draw ground
        buffer = 5
        ground = Line2D([-buffer, pH_horiz[-1][0] + buffer], [0, 0], color='k', linewidth=2)
        ax.add_line(ground)

        # Draw tick marks and labels
        ref_tick = []
        ref_label = []
        for k in range(-buffer, int(np.floor(pH_horiz[-1]) + buffer)):
            tick = Line2D([k, k], [-0.1, 0], color='k')
            label = ax.text(k, -0.2, str(k), ha='center', va='top', fontsize=8)
            ref_tick.append(tick)
            ref_label.append(label)
            ax.add_line(tick)

        # Draw leg one
        param = np.linspace(0, 2 * np.pi, 50)
        xmass_legs = mr_legs * np.cos(param)
        ymass_legs = mr_legs * np.sin(param)

        xmass_torso = mr_torso * np.cos(param)
        ymass_torso = mr_torso * np.sin(param)

        leg1_color = 'green'  # Color for leg one
        leg1, = ax.plot([pFoot1[0], pH[0]], [pFoot1[1], pH[1]], color='g', linewidth=2)
        mass1_x = xmass_legs + (pH[0] - pFoot1[0]) / 2
        mass1_y = ymass_legs + (pH[1] - pFoot1[1]) / 2
        mass1 = Polygon(np.column_stack((mass1_x, mass1_y)), closed=True, color=leg1_color)
        ax.add_patch(mass1)

        # Draw leg two
        leg2_color = 'red'  # Color for leg two
        leg2, = ax.plot([pFoot2[0], pH[0]], [pFoot2[1], pH[1]], color='r', linewidth=2)
        mass2_x = xmass_legs + pH[0] - (pH[0] - pFoot2[0]) / 2
        mass2_y = ymass_legs + pH[1] - (pH[1] - pFoot2[1]) / 2
        mass2 = Polygon(np.column_stack((mass2_x, mass2_y)), closed=True, color=leg2_color)
        ax.add_patch(mass2)

        # Draw torso
        torso_color = 'blue' 
        torso, = ax.plot([pH[0], pT[0]], [pH[1], pT[1]], color='b', linewidth=2)
        # Add torso mass
        mass_torso_x = xmass_torso + pT[0]
        mass_torso_y = ymass_torso + pT[1]
        torso_mass = Polygon(np.column_stack((mass_torso_x, mass_torso_y)), closed=True, color=torso_color)
        ax.add_patch(torso_mass)

        # Animation loop
        for k in range(1, n):
            q = xe[k, :3]
            pFoot1, pFoot2, pH, pT = limb_position(q, pH_horiz[k])

            # Update positions
            leg1.set_data([pFoot1[0], pH[0]], [pFoot1[1], pH[1]])
            mass1.set_xy(np.array([(pH[0] - pFoot1[0]) / 2 + pH_horiz[k][0], (pH[1] - pFoot1[1]) / 2]).reshape((-1, 2)))

            leg2.set_data([pFoot2[0], pH[0]], [pFoot2[1], pH[1]])
            mass2.set_xy(np.array([pH[0] - (pH[0] - pFoot2[0]) / 2, pH[1] - (pH[1] - pFoot2[1]) / 2]).reshape((-1,2)))

            torso.set_data([pH[0], pT[0]], [pH[1], pT[1]])
            torso_mass.set_xy(np.array([pT[0], pT[1]]).reshape((-1,2)))

            # Update axis and labels
            plt.xlim(-2.2 + pH[0], 2.2 + pH[0])

            for j, (label, tick) in enumerate(zip(ref_label, ref_tick)):
                if j - buffer - 1.05 < ax.get_xlim()[0] or j - buffer - 1 > ax.get_xlim()[1]:
                    label.set_visible(False)
                    tick.set_visible(False)
                else:
                    label.set_visible(True)
                    tick.set_visible(True)

            plt.title(f"T_est = {te[k]:.1f}")

            plt.draw()
            plt.pause(ts * speed)

        plt.show()
    
    
    
def events(t, x):
    """
    Event function to handle simulation events.

    Parameters:
        t : float
            Current time.
        x : array-like
            Current state vector.

    Returns:
        value : list
            Event conditions.
        is_terminal : list
            Indicates if the simulation should terminate when an event occurs.
        direction : list
            Specifies the direction of zero-crossing to detect.
    """
    # Persistent variable to count calls
    if not hasattr(events, "control_call_cnt") or t == 0:
        events.control_call_cnt = 0
    else:
        events.control_call_cnt += 1

    # Retrieve parameters
    th3d, th1d, alpha, epsilon = control_params_three_link()

    # Define event conditions
    value = [
        th1d - x[0],          # When stance leg attains angle of th1d
        r * np.cos(x[0]) - 0.5 * r  # Hips get too close to the ground
    ]
    is_terminal = [1, 1]  
    direction = [-1, -1] 

    return value, is_terminal, direction
    

def solve_limcycle_3link(n_steps=2):
    global t_2, torque, y, force, u_trj

    # Reset global variables
    torque = []
    t_2 = []
    y = []
    force = []

    tstart = 0
    tfinal = 13
    
    omega_1 = 1.55
    x0 = sigma_three_link(omega_1, a)
    x0,_,_ = resetmap_3link_12(tstart,x0)

    options = {
        'events': switching_leg_events, 
        'rtol': 1e-5,
        'atol': 1e-6,
        
    }

    tout = [tstart]
    xout = [x0]
    uout = []
    teout = []
    xeout = []
    ieout = []

    print("(impact ratio is the ratio of tangential to normal forces of the tip of the swing leg at impact)")

    t_events = []
    x_events = []
    x_resets = []
    saltations = []
    
    # Run five steps
    for i in range(n_steps):  

        # Solve until the first terminal event
        sol = solve_ivp(
            fxgu_nom,
            [tstart, tfinal],
            x0,
            args=(a,),
            events=options['events'],
            rtol=options['rtol'],
            atol=options['atol']
        )

        t = sol.t
        x = sol.y.T
        u = []
        # -------- Recover the control u --------
        for k in range(len(sol.t)):
            t_k = sol.t[k]
            x_k = sol.y[:,k]
            # Extract dynamics matrices and control parameters
            D, C, G, B, K, dV, dVl, Al, Bl, H, LfH, dLfH = sysmat_3link(x_k[:6], a)
            
            # Compute Fx and Gx
            Fx = np.linalg.solve(D, -C @ x_k[3:6] - G)
            Gx = np.linalg.solve(D, B)
            v = control_three_link(H, LfH)

            # Compute control signal (u) using feedback linearization
            uk = np.linalg.solve(dLfH @ np.vstack([np.zeros((3, 2)), Gx]), v.flatten() - dLfH @ np.hstack([x_k[3:6], Fx]).transpose())

            u.append(uk)
        
        # If a foot-touching event happened: compute the saltation matrix
        if sol.t_events:
            
            # Compute the saltation matrix
            te = sol.t_events[0][0]
            xe = sol.y_events[0][0]
            
            print("Swing foot guard function value:")
            print(guard_3link_12_jax(te, xe))
            
            F_1 = fxgu_nom(te, xe, a)
            x_re, _, _ = resetmap_3link_12(te,xe)
            F_2 = fxgu_nom(te, x_re, a) # Important, the F2 is evaluated at the reseted state!
            Rt = Rt_3link_12(te, xe, 0, None)
            Rx = Rx_3link_12(te, xe, 0, None)
            gt = gt_3link_12(te, xe)
            gx = gx_3link_12(te, xe)
            saltation = compute_saltation(F_1, F_2, Rt, Rx, gt, gx)
            saltations.append(saltation)
            
            t_events.append(te)
            x_events.append(xe)
            
        # down sampling and interpolation
        _, x = down_sample(np.array(t), np.array(x), Fs=20)
        t, u = down_sample(np.array(t), np.array(u), Fs=20)
        t, x, u = t.tolist(), x.tolist(), u.tolist()

        # Set new initial conditions after impact
        x0,_,_ = resetmap_3link_12(t[-1],x[-1])
        x[-1] = x0

        x_resets.append(sol.y_events[0])

        tout.extend(t[1:])
        xout.extend(x[1:])
        uout.extend(u)
        
        # print(f"Step: {i + 1}, Impact ratio: {x0[6] / x0[7]}")
        plt.show()

        tstart = t[-1]
        if tstart >= tfinal:
            break

    # Convert to NumPy arrays for plotting
    tout = np.array(tout)
    xout = np.array(xout)
    uout = np.array(uout[:-1])

    return tout, xout, uout, t_events, x_events, saltations


def plot_3link_states(tout, xout, uout):
    # Plotting states
    
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

    # plt.figure(figsize=(6, 6))
    plt.subplot(3, 4, 3)
    plt.plot(tout, uout[:, 0], label=r'$u_1$')
    plt.plot(tout, uout[:, 1], label=r'$u_2$')
    plt.legend(loc="best", fontsize=10)
    plt.title('Control Input Torque')
    plt.xlabel('Time (sec)')
    plt.grid()

    # plt.figure(figsize=(6, 9))
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

    # Plotting hip pos and vel
    # plt.figure(figsize=(6, 9))
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
    
def demo():

    global t_2, torque, y, force

    tout, xout, uout, t_events, x_events, saltations = solve_limcycle_3link()

    fig1 = plt.figure(figsize=(16, 9))
    
    plot_3link_states(tout, xout, uout)
    
    # Create the figure
    # fig_hl = plt.figure(figsize=(6, 8)) 
    # fig_hl.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.1)

    # First subplot: Forces on the stance leg
    # ax1 = fig_hl.add_subplot(211)
    force = np.asarray(force)
    ax1 = plt.subplot(3, 4, 9)
    ax1.plot(t_2, force[:, 0], '-b', label=r'$F_{tan}, (N)$') 
    ax1.plot(t_2, force[:, 1], '--r', label=r'$F_{norm}, (N)$') 
    ax1.legend(loc='best')
    ax1.grid(True)
    ax1.set_title('Forces on End of Stance Leg')

    # Second subplot: Ratio of forces
    # ax2 = fig_hl.add_subplot(212) 
    ax2 = plt.subplot(3, 4, 10)
    ax2.plot(t_2, force[:, 0] / force[:, 1])
    ax2.set_ylabel(r'$F_{tan} / F_{norm}$')  
    ax2.set_xlabel('time (sec)') 
    ax2.grid(True)  
    
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
    
    anim(tout, xout, 1/30, speed=1, fig=fig1)
    
    plt.show()

from scipy.interpolate import interp1d

def even_sample(t, x, Fs):
    # Obtain the process related parameters
    N = x.shape[1]
    M = t.shape[0]
    t0 = t[0]
    tf = t[-1]
    EM = int((tf - t0) * Fs)
    Et = np.linspace(t0, tf, EM)

    # Re-sample each signal using linear interpolation
    Ex = np.zeros((EM, N))  # initialize the output array
    for s in range(N):
        interp_func = interp1d(t[:], x[:, s], kind='linear')
        Ex[:, s] = interp_func(Et)

    return Et, Ex

# Down sampling between every 2 neighbouring time stamps
def down_sample(t, x, Fs):
    # Obtain the process related parameters
    N = x.shape[1]
    M = t.shape[0]

    Et = np.array([t[0]])
    Ex = x[0]

    for ii in range(M-1):
        t0 = t[ii]
        tf = t[ii+1]

        Et_i = np.linspace(t0, tf, Fs)

        # Re-sample each signal using linear interpolation
        Ex_i = np.zeros((Fs, N))  # initialize the output array
        for s in range(N):
            interp_func = interp1d(t[:], x[:, s], kind='linear')
            Ex_i[:, s] = interp_func(Et_i)

        # Concatenate
        Et = np.concatenate((Et, Et_i[1:]))
        Ex = np.vstack((Ex, Ex_i[1:]))

    # Et = Et[1:]
    # Ex = Ex[1:]

    return Et, Ex


if __name__ == '__main__':
    demo()
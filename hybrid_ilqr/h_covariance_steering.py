import numpy as np

import os
import sys
file_path = os.path.abspath(__file__)
exp_dir = os.path.dirname(file_path)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
root_dir = os.path.abspath(os.path.join(exp_dir, '..'))
sys.path.append(root_dir)

# Import iLQR class and reference extension handler
from hybrid_ilqr.h_ilqr_discrete import solve_ilqr, extract_extensions
from experiments.exp_params import *
from dynamics.dynamics_bouncing import *
from dynamics.dynamics_discrete_bouncing import *
from tools.plot_ellipsoid import *


if __name__=='__main__':
    # ======================== H-iLQR ===========================
    n_exp = 1
        
    # Set desired state
    n_modes = 2

    # the state and control dimensions, mode-dependent
    n_states = [2, 2]
    n_inputs = [1, 1]

    # ---------------- multiple bouncing example -----------------
    dt = 0.0015
    epsilon = 1.0
    dt_shrink = 0.95
    n_samples = 0

    # ------------------ one bounce --------------------
    # dt = 0.005
    dt_shrink = 0.9
    start_time = 0
    end_time = 2.0
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)

    Q_T = 60*np.eye(n_states[0])
    init_state = np.array([5, 1.5], dtype=np.float64)    # Define the initial state to be the origin with no velocity
    target_state = np.array([2.5, 0], dtype=np.float64)  # Swing pendulum upright

    init_mode = 0
    target_mode = 0
    initial_guess = [0.5*np.ones((np.shape(time_span)[0],n_inputs[0])), 0.5*np.ones((np.shape(time_span)[0],n_inputs[1]))]

    # ---------------- / bouncing example -----------------

    # ---------------------------- 
    # Define weighting matrices
    # ----------------------------
    Q_k = [np.zeros((n_states[0],n_states[0]), dtype=np.float64), np.zeros((n_states[1],n_states[1]), dtype=np.float64)] # zero weight to penalties along a strajectory since we are finding a trajectory
    R_k = [np.eye(n_inputs[0], dtype=np.float64), np.eye(n_inputs[1], dtype=np.float64)]
    
    init_reset_args = [np.array([0.0]) for _ in range(nt)]
    target_reset_args = [np.array([0.0]) for _ in range(nt)]
    
    # ============================================================================================================
    #                                       Solve for hybrid ilqg proposal
    # ============================================================================================================
    exp_params = ExpParams()
    
    flow_dynamics = [symbolic_dynamics_bouncing, symbolic_dynamics_bouncing]
    
    exp_params.update_params(n_modes, init_mode, target_mode, n_states, init_state, target_state, 
                             start_time, end_time, dt, dt_shrink, initial_guess, 
                             epsilon, n_exp, n_samples, 
                             Q_k, R_k, Q_T, flow_dynamics, 
                             event_detect_bouncing_discrete, 
                             plot_bouncingball, 
                             convert_state_21_bouncing, 
                             init_reset_args, target_reset_args)
    exp_data = ExpData(exp_params)
    
    print("===================== Solving for h-iLQG proposal controller =====================")
    hybrid_ilqr_result = solve_ilqr(exp_params, detect=True, verbose=False)
    
    (timespan,modes,states,inputs,
     k_feedforward,K_feedback,
     current_cost,states_iter,
     ref_modechanges,reference_extension_helper, ref_reset_args) = hybrid_ilqr_result
    
    show_hilqr_results = True
    if show_hilqr_results:
        time_span = np.arange(start_time, end_time, dt).flatten()
        plot_bouncingball(time_span, modes, states, inputs, init_state, target_state, nt, trj_labels='iLQG-stochastic')
        plt.show()
    
    # ==================== covariance steering ====================

    (v_mode_change_ref, v_ext_trj_bwd_ref, v_ext_trj_fwd_ref, 
             v_Kfb_ext_trj_bwd_ref, v_Kfb_ext_trj_fwd_ref, 
             v_kff_ext_trj_bwd_ref, v_kff_ext_trj_fwd_ref, v_tevents_ref) = extract_extensions(reference_extension_helper)
    
    print("v_tevents_ref: ", v_tevents_ref)

    # The event time at which the system enters mode 2 from mode 1.
    t_event = v_tevents_ref[0]

    Sig0 = 0.5*np.eye(n_states[0])
    SigT = 0.1*np.eye(n_states[0])

    E_linear = np.array([[1.0, 0.0], [0.0, -0.6]], dtype=np.float64)

    A = np.array([[0, 1], [0, 0]], dtype=np.float64)
    B = np.array([[0],[1.0]], dtype=np.float64)
    Q = np.zeros((2, 2))

    A = np.tile(A, (nt, 1, 1))
    B = np.tile(B, (nt, 1, 1))
    Q = np.tile(Q, (nt, 1, 1))

    print("A shape: ", A.shape)

    M = np.zeros((nt, 2*n_states[0], 2*n_states[0]), dtype=np.float64)

    for i in range(nt):
        top_row = np.concatenate((A[i], -B[i] @ B[i].T), axis=1)
        bottom_row = np.concatenate((-Q[i], -A[i].T), axis=1)
        M[i] = np.concatenate((top_row, bottom_row), axis=0)

    # the Phi at hybrid event
    Phi_miuns = np.zeros((2*n_states[0], 2*n_states[0]), dtype=np.float64)
    Phi_miuns_top_row = np.concatenate((E_linear, np.zeros((n_states[0], n_states[0]))), axis=1)
    Phi_miuns_bottom_row = np.concatenate((np.zeros((n_states[0], n_states[0])), np.linalg.inv(E_linear).T), axis=1)
    Phi_miuns = np.concatenate((Phi_miuns_top_row, Phi_miuns_bottom_row), axis=0)
    
    # integrate Phi
    Phi = np.eye(2*n_states[0])
    for i in range(0, t_event):
        Phi_next = Phi + dt * (M[i] @ Phi)
        Phi = Phi + (M[i] @ Phi + M[i+1] @ Phi_next) * (dt/2.0)
        # Phi = Phi + dt * Phi@M

    Phi = Phi_miuns@Phi

    for i in range(t_event+1, nt-1):
        Phi_next = Phi + dt * (M[i] @ Phi)
        Phi = Phi + (M[i] @ Phi + M[i+1] @ Phi_next) * (dt/2.0)
        # Phi = Phi + dt * Phi@M

    Phi_11 = Phi[0:n_states[0], 0:n_states[0]]
    Phi_12 = Phi[0:n_states[0], n_states[0]:]
    Phi_21 = Phi[n_states[0]:, 0:n_states[0]]
    Phi_22 = Phi[n_states[0]:, n_states[0]:]

    I = np.eye(n_states[0])

    inv_Phi12 = np.linalg.solve(Phi_12, np.eye(n_states[0]))
    invSig0 = np.linalg.solve(Sig0, np.eye(n_states[0]))

    eval_invSig0, evec_invSig0 = np.linalg.eigh(invSig0)
    sqrtInvSig0 = evec_invSig0 @ np.diag(np.sqrt(eval_invSig0)) @ evec_invSig0.T

    eval_Sig0, evec_Sig0 = np.linalg.eigh(Sig0)
    sqrtSig0 = evec_Sig0 @ np.diag(np.sqrt(eval_Sig0)) @ evec_Sig0.T

    tmp = epsilon**2 * I/4 + sqrtSig0 @ inv_Phi12 @ SigT @ inv_Phi12.T @ sqrtSig0
    tmp = (tmp + tmp.T) / 2
        
    eval_tmp, evec_tmp = np.linalg.eigh(tmp)
    sqrt_tmp = evec_tmp @ np.diag(np.sqrt(eval_tmp)) @ evec_tmp.T

    # ==================== Solve for Pi(t) ====================
    Pi0 = epsilon*invSig0/2 - inv_Phi12@Phi_11 - sqrtInvSig0@sqrt_tmp@sqrtInvSig0

    Pi = np.zeros((nt, n_states[0], n_states[0]), dtype=np.float64)
    Pi[0] = (Pi0 + Pi0.T) / 2
    v_XY = np.zeros((nt, 2*n_states[0], n_states[0]), dtype=np.float64)
    v_XY[0, :n_states[0], :n_states[0]] = np.eye(n_states[0])
    v_XY[0, n_states[0]:, :n_states[0]] = Pi[0]

    for i in range(nt - 1):
        dXY = M[i]@v_XY[i]
        next_XY = v_XY[i] + dXY*dt
        dXY_next = M[i+1]@next_XY
        
        v_XY[i+1] = v_XY[i] + (dXY + dXY_next)*(dt/2.0) 
        X_next = v_XY[i+1,:n_states[0],:n_states[0]]
        Y_next = v_XY[i+1,n_states[0]:,:n_states[0]]
        inv_X_next = np.linalg.solve(X_next, np.eye(n_states[0]))
        Pi[i+1] = Y_next@inv_X_next

    K = np.zeros((nt, n_inputs[0], n_states[0]), dtype=np.float64)
    
    for i in range(nt):
        K[i] = -B[i].T @ Pi[i]
    
    # ==================== Propagate controlled Sigma(t) ====================
    cov_trj = np.zeros((nt, n_states[0], n_states[0]))
    cov_trj[0] = Sig0
    for i in range(0, t_event):
        Acl_i = A[i] + B[i]@K[i]
        cov_trj[i+1] = (Acl_i@cov_trj[i] + cov_trj[i]@Acl_i.T + B[i]@B[i].T) * dt
    
    # hybrid time
    cov_trj[t_event] = E_linear@cov_trj[t_event-1]@E_linear.T

    for i in range(t_event, nt-1):
        Acl_i = A[i] + B[i]@K[i]
        cov_trj[i+1] = (Acl_i@cov_trj[i] + cov_trj[i]@Acl_i.T + B[i]@B[i].T) * dt

    # ----------------- plot -----------------
    fig, ax = plt.subplots(1, 1)
    # plot covariance trajecotry
    for i in range(0, nt, 5):
        ellipse_boundary, ax = plot_2d_ellipsoid_boundary(states[i], cov_trj[i], ax, 'r')

    ax.grid(True)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\dot z$')
    plt.show()
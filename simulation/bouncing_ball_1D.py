import scipy
import matplotlib.pyplot as plt

import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(root_dir)

from dynamics.bouncing_ball_1D import *
from tools.plot_ellipsoid import *

from tools.propagate_covariance import *

def ode_bouncing_ball_1d(z0, vz0, t0, tf, nt, Modes, n_events=None, return_saltation=False, Sig0=None):
    """Solving the ODE with hybrid events.

    Args:
        z0 (numpy array): Initial height
        vz0 (numpy array): Initial velocity in z direction
        t0 (float): Initial time
        tf (float): Terminal time
        Modes (list(functions)): The collection of all the mode functions.
        n_events (int): Number of events before stopping, 
                        if set to None, then the simulation will continue until tf.
        return_saltation (bool): If return the saltation matrices.

    Returns:
        t_ttl: The total time span.
        x_trj: The whole trajectory including the hybrid events.
        tevents: Time of the hybrid event happening.
        xevents: States at the hybrid event.
        xresets: States after the reset map, for each hybrid event.
        saltations: Collection of saltation matrices at the hybrid events.
    """
    x_trj = []
    tevents = []
    xevents = []
    xresets = []
    saltations = []
    current_mode = 0
    num_events = 0
    x0 = np.array([z0, vz0], dtype=np.float64)    
    
    guard_bouncing.terminal=True
    guard_bouncing.direction=-1
    
    while (num_events < n_events):   
        dyn = Modes[current_mode]
        
        t_span = (t0, tf)
        t_eval = np.linspace(t0, tf, 1000)
        
        if return_saltation: # Compute saltation matrix and solve for covariance matrix
            initial_conditions = np.concatenate([x0, Sig0.flatten()])  # Combine initial conditions
            
            # ---------- Solve ODE for mean and covariance jointly, with hybrid event detection ---------- 
            args = (dyn, linearize_bouncing, 2)    
            solution = scipy.integrate.solve_ivp(fun=lambda t, y: dxdX_solve_ivp(t, y, *args), 
                                                t_span=t_span, 
                                                y0=initial_conditions, method='RK45', 
                                                t_eval=t_eval, 
                                                dense_output=True, 
                                                events=guard_bouncing)
            t_event = solution.t_events[0][0]
            xX_event = solution.y_events[0][0]
            x_event = xX_event[0:2]
            x_reset = reset_map_bouncing(t_event, x_event)
            
            # ---------- Compute saltation matrix ---------- 
            R_x = Rx_bouncing(t_event, x_event)
            R_t = Rt_bouncing(t_event, x_event)
            g_x = gx_bouncing(t_event, x_event)
            g_t = gt_bouncing(t_event, x_event)
            F_1 = dyn(t_event, x_event)
            F_2 = dyn(t_event, x_reset) # Important, the F2 is evaluated at the reseted state!
            saltation = saltation_matrix(F_1, F_2, R_t, R_x, g_t, g_x)
            saltations.append(saltation)
            
            X_event = xX_event[2:6].reshape([2, 2])
            Sig0 = saltation @ X_event @ saltation.transpose()
            
        else: # Do not compute saltation matrix and do not solve for covariance matrix
            # ---------- Solve ODE with hybrid event detection ---------- 
            solution = scipy.integrate.solve_ivp(dyn, t_span, x0, method='RK45', 
                                                t_eval=t_eval, dense_output=True, 
                                                events=guard_bouncing, vectorized=False, args=None)
            t_event = solution.t_events[0][0]
            x_event = solution.y_events[0][0]

        current_mode = mode_jump_bouncing(current_mode)
        x_reset = reset_map_bouncing(t_event, x_event)
        x0 = x_reset
        
        # Solve for the continuous trajectory before the contact 
        t = np.linspace(t0, t_event, nt).flatten()
        
        # The solved trajecoty, in shape (nx+nx*nx, nt)
        x_trj_i = solution.sol(t)      
        
        # ----------- Add the piece-wise trajectory
        if len(x_trj) == 0:
            x_trj = x_trj_i
            tevents = np.array([t_event])
            xevents = x_event
            xresets = x_reset
            t_ttl = t
        else:
            x_trj = np.concatenate([x_trj, x_trj_i], axis=1)
            tevents = np.concatenate([tevents, np.array([t_event])], axis=0)
            xevents = np.concatenate([xevents, x_event], axis=0)
            xresets = np.concatenate([xresets, x_reset], axis=0)
            t_ttl = np.concatenate([t_ttl, t]).flatten()
        
        t0 = t_event
        tf = t0 + 5.0
        
        num_events += 1
    
    return t_ttl, x_trj, tevents, xevents, xresets, saltations


## A collection of 1D bouncing balls which are sampled from a Gaussian distribution
#  N(m0, Sig0), m0=[z0, vz0]^T.

def bouncing_ball_1d_samples(z0, vz0, Sig0, t0, tf, nt, Modes, n_samples, n_events=None):
    """
    The bouncing ball solution for a set of sampled trajectories.
    """
    t_collections = []
    trj_collections = []
    tevent_collections = []
    xevent_collections = []
    xreset_collections = []
    
    m0 = np.array([z0, vz0], dtype=np.float64)
    
    for _ in range(n_samples):
        x0 = m0 + scipy.linalg.sqrtm(Sig0)@np.random.randn(2)
        z0_i, vz0_i = x0[0], x0[1]
        
        t_i, trj_i, tevents_i, xevents_i, xresets_i, _ = ode_bouncing_ball_1d(z0_i, vz0_i, t0, tf, nt, Modes, n_events, return_saltation=False, Sig0=None)
        
        t_collections.append(t_i)
        trj_collections.append(trj_i)
        tevent_collections.append(tevents_i)
        xevent_collections.append(xevents_i)
        xreset_collections.append(xresets_i)
        
    return t_collections, trj_collections, tevent_collections, xevent_collections, xreset_collections


if __name__ == '__main__':
    z0 = 10.0
    vz0 = 0.0
    t0 = 0.0
    tf = 5.0
    nt = 300
    n_events = 2
    Modes = [dyn_bouncing]
    
    fig1, axs = plt.subplots(2, 2)
    ax1, ax2, ax3, ax4 = axs.flatten()
    
    # Solve for samples
    sig0 = 0.1
    Sig0 = sig0*np.eye(2)
    Sig0[1, 1] = 1.0
    
    n_samples = 500
    t_collections, trj_collections, tevent_collections, xevent_collections, xreset_collections = bouncing_ball_1d_samples(z0, vz0, Sig0, t0, tf, nt, Modes, n_samples, n_events)
    
    # Solve for mean trajectory
    t_mean, xX_mean, t_event, x_event, x_reset, saltations = ode_bouncing_ball_1d(z0, vz0, t0, tf, nt, Modes, n_events, return_saltation=True, Sig0=Sig0)
    x_mean = xX_mean[0:2, :]
    
    # ========================== ax1: Plot z-t ========================== 
    # -------------------------- Plot samples --------------------------
    for t_i, trj_i in zip(t_collections, trj_collections):
        ax1.plot(t_i, trj_i[0,:].T, '-.', alpha=0.1)
        
    # -------------------------- Plot mean --------------------------
    ax1.plot(t_mean, x_mean[0,:].T, 'r')
        
    ax1.grid(True)
    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$z$')
    ax1.set_title('1D Bouncing ball')
    
    # ========================== ax2: z - vz plot ========================== 
    # -------------------------- Plot samples --------------------------
    for t_i, trj_i in zip(t_collections, trj_collections):
        ax2.plot(trj_i[0,:].T, trj_i[1,:].T, '-.', alpha=0.1)
        
    # -------------------------- Plot mean --------------------------
    ax2.plot(x_mean[0,:].T, x_mean[1,:].T, 'r')
    
    # -------------------------- Plot covariance --------------------------   
    for i in range(0, n_events*nt, 20):
        _, ax2 = plot_2d_ellipsoid_boundary(xX_mean[0:2,i], xX_mean[2:6,i].reshape([2,2]), ax2, 'k')
    
    ax2.grid(True)
    ax2.set_xlabel('z')
    ax2.set_ylabel(r'$\dot z$')
    
    # ========================= plot the pre-contact samples and post-contact samples =========================    
    # ======================= ax3: pre-contact samples =======================
    ## ------ find the earliest contact time -----
    tevent_min = np.min(np.array(tevent_collections), axis=0)
    
    print("tevent_min")
    print(tevent_min)
    
    # ------ find the timestamp for other samples that is close to the earliest contact time ------    
    pre_contact_index = np.argmax(np.array(t_collections) > tevent_min[0], axis=1)
    
    # ------ plot the distribution of the samples right before the ealiest contact -----
    for sample_i in range(n_samples):
        ax3.scatter(trj_collections[sample_i][0, pre_contact_index[sample_i]], 
                    trj_collections[sample_i][1, pre_contact_index[sample_i]], s=1.2, c='b', alpha=0.5)

    # ----- plot the covariance ellipsoid -----
    # find the mean state at the first pre-contact time
    pre_contact_index_mean = np.argmax(t_mean > tevent_min[0])
    pre_contact_mean = xX_mean[0:2, pre_contact_index_mean]
    pre_contact_cov = xX_mean[2:6, pre_contact_index_mean].reshape([2,2])
    
    _, ax3 = plot_2d_ellipsoid_boundary(pre_contact_mean, pre_contact_cov, ax3, 'r')
    
    # plot the pre-contact time covariance in the covariance trajectory
    _, ax2 = plot_2d_ellipsoid_boundary(pre_contact_mean, pre_contact_cov, ax2, 'r')
    
    # ======================= ax4: post-contact samples =======================
    ## ------ find the latest contact time -----
    tevent_max = np.max(np.array(tevent_collections), axis=0)
        
    # ------ find the timestamp for other samples that is close to the earliest contact time ------      
    post_contact_index = np.argmax(np.array(t_collections) > tevent_max[0] + 0.1, axis=1)
    
    # ------ plot the distribution of the samples right before the ealiest contact -----
    for sample_i in range(n_samples):
        ax4.scatter(trj_collections[sample_i][0, post_contact_index[sample_i]], 
                    trj_collections[sample_i][1, post_contact_index[sample_i]], s=1.2, c='b', alpha=0.5)
        
    # ----- plot the covariance ellipsoid -----
    # find the mean state at the last post-contact time
    post_times = t_mean > tevent_max[0] + 0.1
    
    post_contact_index_mean = np.argmax(t_mean > tevent_max[0] + 0.1)
    post_contact_mean = xX_mean[0:2, post_contact_index_mean]
    post_contact_cov = xX_mean[2:6, post_contact_index_mean].reshape([2,2])
    
    _, ax4 = plot_2d_ellipsoid_boundary(post_contact_mean, post_contact_cov, ax4, 'g')
    
    # plot the post-time covariance in the covariance trajectory
    _, ax2 = plot_2d_ellipsoid_boundary(post_contact_mean, post_contact_cov, ax2, 'g')
    
    ax3.set_title('Pre-contact')
    ax4.set_title('Post-contact')
    
    ax3.grid(True)
    ax4.grid(True)
    
    # ax3.axis('equal')
    ax3.set_xlabel(r'$z$')
    ax3.set_ylabel(r'$\dot z$')
    
    # ax4.axis('equal')
    ax4.set_xlabel(r'$z$')
    ax4.set_ylabel(r'$\dot z$')
    
    plt.show()
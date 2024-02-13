import scipy
import os
import sys
import matplotlib.pyplot as plt


file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(root_dir)
from tools.plot_ellipsoid import *
from dynamics.constant_flow import *

def ode_constant_flow(x1, x2, t0, tf, n_events=None, return_saltation=False):
    """Solving the ODE with hybrid events.

    Args:
        x1 (numpy array): Initial position
        x2 (numpy array): Initial velocity
        t0 (float): Initial time
        tf (float): Terminal time
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
    mode_1 = True
    event_cst_flow.terminal=True
    event_cst_flow.direction=1
    
    x0 = np.array([x1, x2], dtype=np.float64)    
    while (True):
        t_span = (t0, tf)
        if mode_1:
            solution = scipy.integrate.solve_ivp(dyn_f1, t_span, x0, method='RK45', 
                                                t_eval=None, dense_output=True, 
                                                events=event_cst_flow, vectorized=False, args=None)
            
            t_event = solution.t_events[0][0]
            x_event = solution.y_events[0][0]
            
            if return_saltation:
                R_x = Rx(t_event, x_event)
                R_t = Rt(t_event, x_event)
                g_x = gx(t_event, x_event)
                g_t = gt(t_event, x_event)
                F_1 = dyn_f1(t_event, x_event)
                F_2 = dyn_f2(t_event, x_event)
                saltations.append(saltation_matrix(F_1, F_2, R_t, R_x, g_t, g_x))
            
            x_reset = reset_map(t_event, x_event)
            
            mode_1 = False
            
            t = np.linspace(t0, t_event, 300).flatten()
            x_trj_i = solution.sol(t)        
               
            x_trj = x_trj_i
            tevents = np.array([t_event])
            xevents = x_event
            xresets = x_reset
            
            # reset the conditions for the solver
            x0 = x_reset
            t0 = t_event
            tf = t0 + 10.0
        
        ## Mode 2
        else:
            solution = scipy.integrate.solve_ivp(dyn_f2, t_span, x0, method='RK45', 
                                                t_eval=None, dense_output=True, 
                                                events=event_cst_flow, vectorized=False, args=None)
            
            t = np.linspace(t0, tf, 300).flatten()
            x_trj_i = solution.sol(t)        
                    
            x_trj = np.concatenate([x_trj, x_trj_i], axis=1)
            
            break
                    
    t_ttl = np.linspace(0.0, tf, x_trj.shape[1]).flatten()
    
    return t_ttl, x_trj, tevents, xevents, xresets, saltations


## A collection of 1D bouncing balls which are sampled from a Gaussian distribution
#  N(m0, Sig0), m0=[x1, x2]^T.

def constflow_samples(x1, x2, Sig0, t0, tf, n_samples, n_events=None):
    """
    The bouncing ball solution for a set of sampled trajectories.
    """
    t_collections = []
    trj_collections = []
    tevent_collections = []
    xevent_collections = []
    xreset_collections = []
    
    m0 = np.array([x1, x2], dtype=np.float64)
    
    for _ in range(n_samples):
        x0 = m0 + scipy.linalg.sqrtm(Sig0)@np.random.randn(2)
        z0_i, vz0_i = x0[0], x0[1]
        t0_i = t0
        tf_i = tf
        
        t_i, trj_i, tevents_i, xevents_i, xresets_i, _ = ode_constant_flow(z0_i, vz0_i, t0_i, tf_i, n_events, False)
        
        t_collections.append(t_i)
        trj_collections.append(trj_i)
        tevent_collections.append(tevents_i)
        xevent_collections.append(xevents_i)
        xreset_collections.append(xresets_i)
             
    return t_collections, trj_collections, tevent_collections, xevent_collections, xreset_collections


if __name__ == '__main__':
    x1 = 0.0
    x2 = 0.0
    t0 = 0.0
    tf = 20.0
        
    # ========================== Plot collection movements ========================== 
    t_mean, x_mean, t_event, x_event, x_reset, saltations = ode_constant_flow(x1, x2, t0, tf, n_events=None, return_saltation=True)
    fig1, axs = plt.subplots(2, 2)
    
    ax1, ax2, ax3, ax4 = axs.flatten()
    
    # -------------------------- Plot samples --------------------------
    sig0 = 0.1
    Sig0 = sig0*np.eye(2)
    
    n_samples = 500
    t_collections, trj_collections, tevent_collections, xevent_collections, xreset_collections = constflow_samples(x1, x2, Sig0, t0, tf, n_samples, n_events=None)

    for t_i, trj_i in zip(t_collections, trj_collections):
        ax1.plot(t_i, trj_i[0,:].T, '-.', alpha=0.1)
        
    # -------------------------- Plot mean --------------------------
    ax1.plot(t_mean, x_mean[0,:].T, 'r')
    
    ax1.grid(True)
    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$z$')
    ax1.set_title('Constant flow')
    
    # ========================== z - vz plot ==========================     
    # -------------------------- Plot samples --------------------------
    for t_i, trj_i in zip(t_collections, trj_collections):
        ax2.plot(trj_i[0,:].T, trj_i[1,:].T, '-.', alpha=0.1)
        
    # -------------------------- Plot mean --------------------------
    ax2.plot(x_mean[0,:].T, x_mean[1,:].T, 'r')
    
    ax2.grid(True)
    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$\dot z$')
    ax2.axis('equal')
    
    # ========================= plot the pre-contact samples and post-contact samples =========================    
    # ======================= pre-contact samples =======================
    ## ------ find the earliest contact time -----
    tevent_min = np.min(np.array(tevent_collections))
    
    print("tevent_min")
    print(tevent_min)
    
    # ------ find the timestamp for other samples that is close to the earliest contact time ------    
    pre_contact_index = np.argmax(np.array(t_collections) > tevent_min - 0.1, axis=1)
    
    # ------ plot the distribution of the samples right before the ealiest contact -----
    for sample_i in range(n_samples):
        ax3.scatter(trj_collections[sample_i][0, pre_contact_index[sample_i]], 
                    trj_collections[sample_i][1, pre_contact_index[sample_i]], s=1.2, c='b', alpha=0.5)
    
    # ----- plot the covariance ellipsoid -----
    # find the mean state at the first pre-contact time
    pre_contact_index_mean = np.argmax(t_mean > tevent_min - 0.1)
    pre_contact_mean = x_mean[:, pre_contact_index_mean]
    ellipse_boundary, ax3 = plot_2d_ellipsoid_boundary(pre_contact_mean, Sig0, ax3, 'r')
    
    # ======================= post-contact samples =======================
    ## ------ find the latest contact time -----
    tevent_max = np.max(np.array(tevent_collections))
    
    print("tevent_max")
    print(tevent_max)
    
    # ------ find the timestamp for other samples that is close to the earliest contact time ------      
    post_contact_index = np.argmax(np.array(t_collections) > tevent_max + 0.1, axis=1)
    
    # ------ plot the distribution of the samples right before the ealiest contact -----
    for sample_i in range(n_samples):
        ax4.scatter(trj_collections[sample_i][0, post_contact_index[sample_i]], 
                    trj_collections[sample_i][1, post_contact_index[sample_i]], s=1.2, c='b', alpha=0.5)
        
    # ----- plot the covariance ellipsoid -----
    # find the mean state at the last post-contact time
    post_contact_index_mean = np.argmax(t_mean > tevent_max + 0.1)
    post_contact_mean = x_mean[:, post_contact_index_mean]
    print("saltations[0]")
    print(saltations[0])
    Cov_plus = saltations[0] @ Sig0 @ saltations[0].transpose()
    print(Cov_plus)
    _, ax4 = plot_2d_ellipsoid_boundary(post_contact_mean, Cov_plus, ax4, 'g')
    
    ax3.set_title('Pre-contact')
    ax4.set_title('Post-contact')
    
    ax3.grid(True)
    ax4.grid(True)
    
    ax3.axis('equal')
    ax3.set_xlabel(r'$z$')
    ax3.set_ylabel(r'$\dot z$')
    
    ax4.axis('equal')
    ax4.set_xlabel(r'$z$')
    ax4.set_ylabel(r'$\dot z$')
    
    plt.show()
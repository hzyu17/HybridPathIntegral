import scipy
import os
import sys
import matplotlib.pyplot as plt


file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(root_dir)

from dynamics.bouncing_ball_1D import *

def ode_bouncing_ball_1d(z0, vz0, t0, tf, n_events=None, return_saltation=False):
    """Solving the ODE with hybrid events.

    Args:
        z0 (numpy array): Initial height
        vz0 (numpy array): Initial velocity in z direction
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
    num_events = 0
    x0 = np.array([z0, vz0], dtype=np.float64)    
    
    while (abs(vz0) > 5.0) or (z0>2.0):
        event_bouncing.terminal=True
        event_bouncing.direction=-1
        
        t_span = (t0, tf)
        solution = scipy.integrate.solve_ivp(dyn_f, t_span, x0, method='RK45', 
                                            t_eval=None, dense_output=True, 
                                            events=event_bouncing, vectorized=False, args=None)
        
        t_event = solution.t_events[0][0]
        x_event = solution.y_events[0][0]
        
        if return_saltation:
            R_x = Rx(t_event, x_event)
            R_t = Rt(t_event, x_event)
            g_x = gx(t_event, x_event)
            g_t = gt(t_event, x_event)
            F_1 = dyn_f(t_event, x_event)
            F_2 = dyn_f(t_event, x_event)
            saltation = saltation_matrix(F_1, F_2, R_t, R_x, g_t, g_x)
            saltations.append(saltation)
        
        x_reset = reset_map(t_event, x_event)
        x0 = x_reset
        
        t = np.linspace(t0, t_event, 300).flatten()
        x_trj_i = solution.sol(t)        
                
        if len(x_trj) == 0:
            x_trj = x_trj_i
            tevents = np.array([t_event])
            xevents = x_event
            xresets = x_reset
        else:
            x_trj = np.concatenate([x_trj, x_trj_i], axis=1)
            tevents = np.concatenate([tevents, np.array([t_event])], axis=0)
            xevents = np.concatenate([xevents, x_event], axis=0)
            xresets = np.concatenate([xresets, x_reset], axis=0)
        
        t0 = t_event
        tf = t0 + 5.0
        
        num_events += 1
        
        if (n_events is not None) and (num_events == n_events):
            break
        
    t_ttl = np.linspace(0.0, tf, x_trj.shape[1]).flatten()
    
    return t_ttl, x_trj, tevents, xevents, xresets, saltations


## A collection of 1D bouncing balls which are sampled from a Gaussian distribution
#  N(m0, Sig0), m0=[z0, vz0]^T.

def bouncing_ball_1d_samples(z0, vz0, Sig0, t0, tf, n_samples, n_events=None):
    """
    The bouncing ball solution for a set of sampled trajectories.
    """
    t_collections = []
    trj_collections = []
    tevent_collections = []
    xevent_collections = []
    xreset_collections = []
    
    m0 = np.array([z0, vz0], dtype=np.float64)
    
    for i_sample in range(n_samples):
        x0 = m0 + scipy.linalg.sqrtm(Sig0)@np.random.randn(2)
        z0_i, vz0_i = x0[0], x0[1]
        t0_i = t0
        tf_i = tf
        
        t_i, trj_i, tevents_i, xevents_i, xresets_i, _ = ode_bouncing_ball_1d(z0_i, vz0_i, t0_i, tf_i, n_events, False)
        
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
    
    n_events = 2
    
    # ========================== Plot collection movements ========================== 
    # -------------------------- Plot mean --------------------------
    t_mean, x_mean, t_event, x_event, x_reset, saltations = ode_bouncing_ball_1d(z0, vz0, t0, tf, n_events, True)
    plt.plot(t_mean, x_mean[0,:].T, 'r')
    
    # -------------------------- Plot samples --------------------------
    sig0 = 0.1
    Sig0 = sig0*np.eye(2)
    Sig0[1, 1] = 1.0
    
    n_samples = 500
    t_collections, trj_collections, tevent_collections, xevent_collections, xreset_collections = bouncing_ball_1d_samples(z0, vz0, Sig0, t0, tf, n_samples, n_events)

    for t_i, trj_i in zip(t_collections, trj_collections):
        plt.plot(t_i, trj_i[0,:].T, '-.', alpha=0.1)
    plt.grid(True)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$z$')
    plt.title('1D Bouncing ball')
    plt.show()
    
    # ========================== z - vz plot ========================== 
    # -------------------------- Plot mean --------------------------
    plt.plot(x_mean[0,:].T, x_mean[1,:].T, 'r')
    
    # -------------------------- Plot samples --------------------------
    for t_i, trj_i in zip(t_collections, trj_collections):
        plt.plot(trj_i[0,:].T, trj_i[1,:].T, '-.', alpha=0.1)
        
    plt.grid(True)
    plt.xlabel('z')
    plt.ylabel(r'$\dot z$')
    plt.show()
    
    # ========================= plot the pre-contact samples and post-contact samples =========================    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Pre-contact')
    ax2.set_title('Post-contact')
    
    ax1.grid(True)
    ax2.grid(True)
    
    # -------------------------- pre-contact samples --------------------------
    for precontact_i in xevent_collections:
        ax1.scatter(precontact_i[0], precontact_i[1], s=0.7, c='b')
    
    # -------------------------- post-contact samples --------------------------
    for postcontact_i in xreset_collections:
        ax2.scatter(postcontact_i[0], postcontact_i[1], s=0.7, c='b')

    ax1.set_xlabel(r'$z$')
    ax1.set_ylabel(r'$\dot z$')
    
    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$\dot z$')
    
    plt.show()
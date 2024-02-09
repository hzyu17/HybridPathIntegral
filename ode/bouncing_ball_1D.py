import scipy
import os
import sys
import matplotlib.pyplot as plt


file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(root_dir)

from dynamics.bouncing_ball_1D import *

def ode_bouncing_ball_1d(z0, vz0, t0, tf, n_events=None):
    """Solving the ODE with hybrid events.

    Args:
        z0 (numpy array): Initial height
        vz0 (numpy array): Initial velocity in z direction
        t0 (float): Initial time
        tf (float): Terminal time
        n_events (int): Number of events before stopping, 
                        if set to None, then the simulation will continue until tf.

    Returns:
        t_ttl: The total time span.
        x_trj: The whole trajectory including the hybrid events.
        tevents: Time of the hybrid event happening.
        xevents: States at the hybrid event.
    """
    x_trj = []
    tevents = []
    xevents = []
    num_events = 0
    while (abs(vz0) > 5.0) or (z0>4.0):
        event_bouncing.terminal=True
        event_bouncing.direction=-1
        
        x0 = np.array([z0, vz0], dtype=np.float64)    
        t_span = (t0, tf)
        solution = scipy.integrate.solve_ivp(dyn_f, t_span, x0, method='RK45', 
                                            t_eval=None, dense_output=True, 
                                            events=event_bouncing, vectorized=False, args=None)
        
        t_event = solution.t_events[0]
        x_event = solution.y_events[0]
        
        x_reset = reset_map(t_event, x_event[0])
        z0, vz0 = x_reset[0], x_reset[1]
        
        t = np.linspace(t0, t_event[0], 300).flatten()
        x_trj_i = solution.sol(t)        
                
        if len(x_trj) == 0:
            x_trj = x_trj_i
            tevents = t_event
            xevents = x_event.T
        else:
            x_trj = np.concatenate([x_trj, x_trj_i], axis=1)
            tevents = np.concatenate([tevents, t_event], axis=0)
            xevents = np.concatenate([xevents, x_event.T], axis=1)
        
        t0 = t_event[0]
        tf = t0 + 5.0
        
        num_events += 1
        
        if (n_events is not None) and (num_events == n_events):
            break
        
    t_ttl = np.linspace(0.0, tf, x_trj.shape[1]).flatten()
    
    return t_ttl, x_trj, tevents, xevents


## A collection of 1D bouncing balls which are sampled from a Gaussian distribution
#  N(m0, Sig0), m0=[z0, vz0]^T, Sig0=sig0*eye(2).

def bouncing_ball_1d_samples(z0, vz0, sig0, t0, tf, n_samples, n_events=None):
    t_collections = []
    trj_collections = []
    tevent_collections = []
    xevent_collections = []
    Sig0 = sig0*np.eye(2)
    m0 = np.array([z0, vz0], dtype=np.float64)
    
    for i_sample in range(n_samples):
        x0 = m0 + scipy.linalg.sqrtm(Sig0)@np.random.randn(2)
        z0_i, vz0_i = x0[0], x0[1]
        t0_i = t0
        tf_i = tf
        
        t_i, trj_i, tevents_i, xevents_i = ode_bouncing_ball_1d(z0_i, vz0_i, t0_i, tf_i, n_events)
        
        t_collections.append(t_i)
        trj_collections.append(trj_i)
        tevent_collections.append(tevents_i)
        xevent_collections.append(xevents_i)
        
    return t_collections, trj_collections, tevent_collections, xevent_collections


if __name__ == '__main__':
    z0 = 5.0
    vz0 = 0.0
    t0 = 0.0
    tf = 5.0
    
    n_events = 2
    
    # ========================== Plot collection movements ========================== 
    # -------------------------- Plot mean --------------------------
    t_mean, x_mean, t_event, x_event = ode_bouncing_ball_1d(z0, vz0, t0, tf, n_events)
    plt.plot(t_mean, x_mean[0,:].T, 'r')
    
    # -------------------------- Plot samples --------------------------
    sig0 = 0.1
    n_samples = 20
    t_collections, trj_collections, tevent_collections, xevent_collections = bouncing_ball_1d_samples(z0, vz0, sig0, t0, tf, n_samples, n_events)
    
    for t_i, trj_i in zip(t_collections, trj_collections):
        plt.plot(t_i, trj_i[0,:].T, '-.', alpha=0.3)
    plt.grid(True)
    plt.xlabel('t')
    # plt.legend(['z', r'$\dot z$'], shadow=True)
    plt.title('1D Bouncing ball')
    plt.show()
    
    # ========================== z - vz plot ========================== 
    # -------------------------- Plot mean --------------------------
    plt.plot(x_mean[0,:].T, x_mean[1,:].T, 'r')
    
    # -------------------------- Plot samples --------------------------
    for t_i, trj_i in zip(t_collections, trj_collections):
        plt.plot(trj_i[0,:].T, trj_i[1,:].T, '-.', alpha=0.3)
        
    plt.grid(True)
    plt.xlabel('z')
    plt.ylabel(r'$\dot z$')
    plt.show()
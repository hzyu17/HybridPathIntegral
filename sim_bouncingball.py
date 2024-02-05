import scipy
import os
import sys
import matplotlib.pyplot as plt

file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
dynamics_dir = os.path.abspath(os.path.join(os.path.join(current_dir, '..'), 'dynamics'))

sys.path.append(dynamics_dir)

from dynamics.bouncing_ball import *

if __name__ == '__main__':
    y0 = 0.0
    z0 = 10.0
    vy0 = 5.0
    vz0 = 0.0
    t0 = 0.0
    tf = 5.0
    x_trj = []
    
    while vy0 > 0.01:
        event_bouncing.terminal=True
        event_bouncing.direction=-1
        
        x0 = np.array([y0, z0, vy0, vz0], dtype=np.float64)    
        t_span = (t0, tf)
        solution = scipy.integrate.solve_ivp(dyn_f, t_span, x0, method='RK45', 
                                            t_eval=None, dense_output=True, 
                                            events=event_bouncing, vectorized=False, args=None)
        
        t_event = solution.t_events[0]
        x_event = solution.y_events[0]
                
        y0 = x_event[0, 0]
        z0 = x_event[0, 1]
        vy0 = x_event[0, 2]
        vz0 = -0.9*x_event[0, 3] 
        
        t = np.linspace(t0, t_event[0], 300).flatten()
        x_trj_i = solution.sol(t)        
                
        if len(x_trj) == 0:
            x_trj = x_trj_i
        else:
            x_trj = np.concatenate([x_trj, x_trj_i], axis=1)
        
        t0 = t_event[0]
        tf = t0 + 5.0
        
    t_ttl = np.linspace(0.0, tf, x_trj.shape[1]).flatten()
    
    plt.plot(t_ttl, x_trj.T)
    plt.grid(True)
    plt.xlabel('t')
    plt.legend(['y', 'z', r'$\dot y$', r'$\dot z$'], shadow=True)
    plt.title('Bouncing ball simulation')
    plt.show()
    

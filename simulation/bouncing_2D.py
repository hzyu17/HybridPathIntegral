import scipy
import os
import sys
import matplotlib.pyplot as plt

file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
dynamics_dir = os.path.abspath(os.path.join(os.path.join(current_dir, '..'), 'dynamics'))

sys.path.append(root_dir)
sys.path.append(dynamics_dir)

from dynamics.bouncing_2D import *

def ode_bouncing_ball_2d(y0, z0, vy0, vz0):
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
        
        x_reset = reset_map(t_event, x_event[0])
        y0, z0, vy0, vz0 = x_reset[0], x_reset[1], x_reset[2], x_reset[3]
        
        t = np.linspace(t0, t_event[0], 300).flatten()
        x_trj_i = solution.sol(t)        
                
        if len(x_trj) == 0:
            x_trj = x_trj_i
        else:
            x_trj = np.concatenate([x_trj, x_trj_i], axis=1)
        
        t0 = t_event[0]
        tf = t0 + 5.0
        
    t_ttl = np.linspace(0.0, tf, x_trj.shape[1]).flatten()
    
    return t_ttl, x_trj


if __name__ == '__main__':
    y0 = 0.0
    z0 = 10.0
    vy0 = 5.0
    vz0 = 0.0
    t0 = 0.0
    tf = 5.0
    
    t_ttl, x_trj = ode_bouncing_ball_2d(y0, z0, vy0, vz0)
    
    
    plt.plot(t_ttl, x_trj.T)
    plt.grid(True)
    plt.xlabel('t')
    plt.legend(['y', 'z', r'$\dot y$', r'$\dot z$'], shadow=True)
    plt.title('Bouncing ball simulation')
    plt.show()
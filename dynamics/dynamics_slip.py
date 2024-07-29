## 2-dimensional SLIP dynamics
# mode 1 (flight): x = [px, vx, pz, vz, theta], u = [theta_dot]
# mode 2 (stance): x = [theta, theta_dot, r, r_dot], u = [r_delta, \tau_hip]
# reset maps: identity

from dynamics.dynamics import *
from dynamics.guard_reset_slip import *

def dyn_slip(t, x, *args):
    """
    Args:
        t (_type_): time variable
        x (_type_): state
        args[0]: control input
    """
   
    if len(args) == 0:
        u = np.array([0.0])
    else:
        u = args[0]
    return np.array([x[1], u[0]-g])


def gdWt_slip(dWt, eps):
    B = np.array([[0],[1.0]], dtype=np.float64)
    return np.sqrt(eps) * B@dWt
    

# =================================
# Flight dynamics Definitions
# =================================
def symbolic_flight_dynamics_slip_continuoustime():
    g = 9.81
    x,x_dot,z,z_dot,theta,u = sp.symbols('x x_dot z z_dot theta u')

    # Define the states and inputs
    inputs = Matrix([u])
    states = Matrix([x, x_dot, z, z_dot, theta])
    
    # Defining the dynamics of the system
    f_cont = Matrix([x_dot, 
                    0,
                    z_dot,
                    -g,
                    u])
    
    # Take the jacobian with respect to states and inputs
    A_disc = f_cont.jacobian(states)
    B_disc = f_cont.jacobian(inputs)

    f_cont_func = sp.lambdify((states,inputs),f_cont)
    A_cont_func = sp.lambdify((states,inputs),A_disc)
    B_cont_func = sp.lambdify((states,inputs),B_disc)
    return (f_cont_func,A_cont_func,B_cont_func)

f_flight_cont_func, _, _ = symbolic_flight_dynamics_slip_continuoustime()

# ------------------------------------------------------------
# function definition for numerical integration in scipy
# ------------------------------------------------------------
def dyn_flight_slip(t, x, *args):
    """
    Args:
        t (_type_): time variable
        x (_type_): state
        args[0]: control input
    """
   
    if len(args) == 0:
        u = np.array([0.0])
    else:
        u = args[0]
    
    return f_flight_cont_func(x, u).flatten()

# ---------------------------
# Discrete-time definition
# ---------------------------
def symbolic_flight_dynamics_slip():
    g = 9.81
    x,x_dot,z,z_dot,theta,u,dt = sp.symbols('x x_dot z z_dot theta u dt')

    # Define the states and inputs
    inputs = Matrix([u])
    states = Matrix([x, x_dot, z, z_dot, theta])
    
    # Defining the dynamics of the system
    f = Matrix([x_dot, 
                0,
                z_dot,
                -g,
                u])

    # Discretize the dynamics usp.sing euler integration
    f_disc = states+f*dt
    
    # Take the jacobian with respect to states and inputs
    A_disc = f_disc.jacobian(states)
    B_disc = f_disc.jacobian(inputs)

    f_disc_func = sp.lambdify((states,inputs,dt),f_disc)
    A_disc_func = sp.lambdify((states,inputs,dt),A_disc)
    B_disc_func = sp.lambdify((states,inputs,dt),B_disc)
    
    return (f_disc_func,A_disc_func,B_disc_func)


# =================================
# Stance dynamics Definitions
# =================================

# ---------------------------
# Continuous-time definition
# ---------------------------
def symbolic_stance_dynamics_slip_continuoustimes():
    g = 9.81
    k = 15.0
    m = 0.2
    r0 = 1
    theta,theta_dot,r,r_dot,u1,u2 = sp.symbols('theta theta_dot r r_dot u1 u2')

    # Define the states and inputs
    inputs = Matrix([u1, u2])
    states = Matrix([theta, theta_dot, r, r_dot])
    
    # Defining the stance dynamics of the system
    f_stance_cont = Matrix([theta_dot, 
                            -2*theta_dot*r_dot/r-g*sp.cos(theta)/r,
                            r_dot + u2/m/r/r,
                            k/m*(r0-r) - g*sp.sin(theta) + theta_dot*theta_dot*r + k*u1/m
                            ])
    
    # Take the jacobian with respect to states and inputs
    A_cont = f_stance_cont.jacobian(states)
    B_cont = f_stance_cont.jacobian(inputs)

    f_cont_func = sp.lambdify((states,inputs),f_stance_cont)
    A_cont_func = sp.lambdify((states,inputs),A_cont)
    B_cont_func = sp.lambdify((states,inputs),B_cont)
    return (f_cont_func, A_cont_func, B_cont_func)

f_stance_cont_func, _, _ = symbolic_stance_dynamics_slip_continuoustimes()

# ------------------------------------------------------------
# function definition for numerical integration in scipy
# ------------------------------------------------------------
def dyn_stance_slip(t, x, *args):
    """
    Args:
        t (_type_): time variable
        x (_type_): state.
        args[0]: control input
    """
        
    if len(args) == 0:
        u = np.array([0.0, 0.0])
    else:
        u = args[0]
    
    return f_stance_cont_func(x, u).flatten()

# ---------------------------
# Discrete-time definition
# ---------------------------
def symbolic_stance_dynamics_slip():
    g = 9.81
    k = 15.0
    m = 0.2
    r0 = 1
    theta,theta_dot,r,r_dot,u1,u2,dt = sp.symbols('theta theta_dot r r_dot u1 u2 dt')

    # Define the states and inputs
    inputs = Matrix([u1, u2])
    states = Matrix([theta, theta_dot, r, r_dot])
    
    # Defining the stance dynamics of the system
    f = Matrix([theta_dot, 
                -2*theta_dot*r_dot/r-g*sp.cos(theta)/r,
                r_dot + u2/m/r/r,
                k/m*(r0-r) - g*sp.sin(theta) + m*r*r+k/m*u1
                ])

    # Discretize the dynamics usp.sing euler integration
    f_disc = states+f*dt
    
    # Take the jacobian with respect to states and inputs
    A_disc = f_disc.jacobian(states)
    B_disc = f_disc.jacobian(inputs)

    f_disc_func = sp.lambdify((states,inputs,dt),f_disc)
    A_disc_func = sp.lambdify((states,inputs,dt),A_disc)
    B_disc_func = sp.lambdify((states,inputs,dt),B_disc)
    return (f_disc_func,A_disc_func,B_disc_func)


def event_detect_slip(x0, u, t0, tf, current_mode, reset_args, detection=True, backwards=False):
    guard_slip_12.terminal=True
    guard_slip_12.direction=1
    
    guard_slip_21.terminal=True
    guard_slip_21.direction=1
    
    guards_slip = {0:guard_slip_12, 1: guard_slip_21}
    reset_maps_slip = {0:reset_map_slip_12, 1:reset_map_slip_21}
    
    reset_controls_slip = {0:reset_control_slip_12, 1:reset_control_slip_21}
    
    Rxs_slip = {0:Rx_slip_12, 1:Rx_slip_21}
    Rts_slip = {0:Rt_slip_12, 1:Rt_slip_21}
    
    gxs_slip = {0:gx_slip_12, 1:gx_slip_21}
    gts_slip = {0:gt_slip_12, 1:gt_slip_21}
    
    smooth_dynamics_slip = {0:dyn_flight_slip, 1:dyn_stance_slip}
    
    return event_detect_onestep(x0, 
                                u, 
                                t0, 
                                tf, 
                                current_mode, 
                                smooth_dynamics_slip, 
                                guards_slip,
                                gxs_slip,
                                gts_slip,
                                reset_maps_slip,
                                reset_controls_slip,
                                Rxs_slip,
                                Rts_slip,
                                reset_args, detection, backwards)
    

def plot_slip(time_span, modes, states, inputs, init_state, target_state, nt, stance_xp):
    print("Plotting SLIP state and input trajectory")
    # =============== plotting ===============
    fig1, axes = plt.subplots(2, 7, figsize=(15, 10))
    (ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax21, ax22, ax23, ax24, ax25, ax26, ax27) = axes.flatten()
    ax11.grid(True)
    ax12.grid(True)
    ax13.grid(True)
    ax14.grid(True)
    ax15.grid(True)
    ax16.grid(True)
    ax17.grid(True)
    
    ax21.grid(True)
    ax22.grid(True)
    ax23.grid(True)
    ax24.grid(True)
    ax25.grid(True)
    ax26.grid(True)
    ax27.grid(True)

    # ----------- Plot the start and goal states -----------
    if (modes[0] == 1):
        init_state = convert_state_21_slip(init_state, np.array([0.0]))
        
    ax12.scatter(time_span[-1], target_state[0], color='g', marker='x', s=50.0, linewidths=6, label='Target')
    ax12.scatter(time_span[0], init_state[0], color='r', marker='x', s=50.0, linewidths=6, label='Start')

    ax13.scatter(time_span[-1], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
    ax13.scatter(time_span[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')
    
    ax14.scatter(time_span[-1], target_state[2], color='g', marker='x', s=50.0, linewidths=6, label='Target')
    ax14.scatter(time_span[0], init_state[2], color='r', marker='x', s=50.0, linewidths=6, label='Start')
    
    ax15.scatter(time_span[-1], target_state[3], color='g', marker='x', s=50.0, linewidths=6, label='Target')
    ax15.scatter(time_span[0], init_state[3], color='r', marker='x', s=50.0, linewidths=6, label='Start')
    
    ax16.scatter(time_span[-1], target_state[4], color='g', marker='x', s=50.0, linewidths=6, label='Target')
    ax16.scatter(time_span[0], init_state[4], color='r', marker='x', s=50.0, linewidths=6, label='Start')

    # ----------- Plot the reference -----------
    for i in range(nt-1):
        if modes[i] == 0:
            ax11.scatter(time_span[i], modes[i], s=0.8, color='k')   
            ax12.scatter(time_span[i], states[i][0], s=0.8, color='k')
            ax13.scatter(time_span[i], states[i][1], s=0.8, color='k')
            ax14.scatter(time_span[i], states[i][2], s=0.8, color='k')
            ax15.scatter(time_span[i], states[i][3], s=0.8, color='k')
            ax16.scatter(time_span[i], states[i][4], s=0.8, color='k')
            ax17.scatter(time_span[i], inputs[modes[i]][i][0], s=0.8, color='k')
            
        if modes[i] == 1:
            assert len(states[i]) == 4
            # Plot the state in mode 1
            ax21.scatter(time_span[i], modes[i], s=0.8, color='k')   
            ax22.scatter(time_span[i], states[i][0], s=0.8, color='k')
            ax23.scatter(time_span[i], states[i][1], s=0.8, color='k')
            ax24.scatter(time_span[i], states[i][2], s=0.8, color='k')
            ax25.scatter(time_span[i], states[i][3], s=0.8, color='k')
            
            ax26.scatter(time_span[i], inputs[modes[i]][i][0], s=0.8, color='b')
            ax26.scatter(time_span[i], inputs[modes[i]][i][1], s=0.8, color='r')
            
            # Plot the converted state in mode 0
            equivalent_state = convert_state_21_slip(states[i], stance_xp[i][0])
            
            ax11.scatter(time_span[i], modes[i], s=0.8, color='k')   
            ax12.scatter(time_span[i], equivalent_state[0], s=0.8, color='k')
            ax13.scatter(time_span[i], equivalent_state[1], s=0.8, color='k')
            ax14.scatter(time_span[i], equivalent_state[2], s=0.8, color='k')
            ax15.scatter(time_span[i], equivalent_state[3], s=0.8, color='k')
            ax16.scatter(time_span[i], equivalent_state[4], s=0.8, color='k')
            
    
    ax11.set_xlabel(r"Time")
    ax11.set_ylabel(r"mode")
    ax11.set_title(r"SLIP mode")
    
    ax12.set_xlabel(r"Time")
    ax12.set_ylabel(r"$x$")
    ax12.set_title(r"SLIP x")

    ax13.set_xlabel(r"Time")
    ax13.set_ylabel(r"$\dot x$")
    ax13.set_title(r"SLIP $\dot x$")
    
    ax14.set_xlabel(r"Time")
    ax14.set_ylabel(r"$z$")
    ax14.set_title(r"SLIP $z$")
    
    ax15.set_xlabel(r"Time")
    ax15.set_ylabel(r"$\dot z$")
    ax15.set_title(r"SLIP $\dot z$")
    
    ax16.set_xlabel(r"Time")
    ax16.set_ylabel(r"$\theta$")
    ax16.set_title(r"SLIP $\theta $")
    
    ax17.set_xlabel(r"Time")
    ax17.set_ylabel(r"Inputs")
    ax17.set_title(r"SLIP Inputs")
    
    ax21.set_xlabel(r"Time")
    ax21.set_ylabel(r"mode")
    ax21.set_title(r"SLIP mode")
    
    ax22.set_xlabel(r"Time")
    ax22.set_ylabel(r"$\theta$")
    ax22.set_title(r"SLIP $\theta$")
    
    ax23.set_xlabel(r"Time")
    ax23.set_ylabel(r"$\dot \theta$")
    ax23.set_title(r"SLIP $\dot \theta$")
    
    ax24.set_xlabel(r"Time")
    ax24.set_ylabel(r"$r$")
    ax24.set_title(r"SLIP $r$")

    ax25.set_xlabel(r"Time")
    ax25.set_ylabel(r"$\dot r$")
    ax25.set_title(r"SLIP $\dot r$")
    
    ax26.set_xlabel(r"Time")
    ax26.set_ylabel(r"Inputs")
    ax26.set_title(r"SLIP Inputs")

    # ax11.legend()
    # ax12.legend()
    # ax13.legend()
    # ax14.legend()
    # ax15.legend()
    # ax16.legend()
    
    # ax21.legend()
    # ax22.legend()
    # ax23.legend()
    # ax24.legend()
    # ax25.legend()
    ax26.legend()
    
    plt.tight_layout()

    plt.show()
    

def plot_slip_flight_animate(state_flight, r0, ax=None, spring_color='c-'):
    if ax is None:
        fig, ax = plt.subplots()
        ax.grid(True)

    # Parameters
    x, _, z, _, theta = state_flight[0], state_flight[1], state_flight[2], state_flight[3], state_flight[4]
    
    spring_coils = 15
    spring_amplitude = 0.02
    ball_radius = 0.02
    theta_spring = np.pi / 2 - theta
    
    # Generate spring data
    t = np.linspace(0, 2 * np.pi * spring_coils, 1000)
    x_spring = spring_amplitude * np.sin(t)
    y_spring = np.linspace(0, r0, 1000)
    x_spring_rot = x_spring * np.cos(theta_spring) + y_spring * np.sin(theta_spring)
    z_spring_rot = -x_spring * np.sin(theta_spring) + y_spring * np.cos(theta_spring)

    x_spring_rot = x_spring_rot + x - r0*np.cos(theta)
    z_spring_rot = z_spring_rot + z - r0*np.sin(theta)
    
    # Plot the spring
    ax.plot(x_spring_rot, z_spring_rot, spring_color, lw=2)

    # Calculate spring end position
    spring_end_x = x
    spring_end_z = z
    
    ball_x = x 
    ball_y = z 
    
    ball = plt.Circle((ball_x, ball_y), ball_radius, color='k')
    ax.add_patch(ball)
    
    colors = ['r', 'g', 'b', 'c']
    labels = ['Start', 'Goal', 'Stance', 'Flight']
    proxy_artists = [plt.Line2D([0], [0], color=color, lw=2) for color in colors]
    ax.legend(proxy_artists, labels)
    
    plt.tight_layout()
    

def plot_slip_stance_animate(state_stance, xp, ax=None, spring_color='b-'):
    if ax is None:
        fig, ax = plt.subplots()
        ax.grid(True)

    # Parameters
    theta, theta_dot, r, rdot = state_stance[0], state_stance[1], state_stance[2], state_stance[3]
    
    spring_coils = 15
    spring_amplitude = 0.02
    ball_radius = 0.02
    theta_spring = np.pi / 2 - theta
    
    # Generate spring data
    t = np.linspace(0, 2 * np.pi * spring_coils, 1000)
    x_spring = spring_amplitude * np.sin(t)
    y_spring = np.linspace(0, r, 1000)
    x_spring_rot = x_spring * np.cos(theta_spring) + y_spring * np.sin(theta_spring)
    z_spring_rot = -x_spring * np.sin(theta_spring) + y_spring * np.cos(theta_spring)

    x_spring_rot = x_spring_rot + xp
    
    # Plot the spring
    ax.plot(x_spring_rot, z_spring_rot, spring_color, lw=2)

    # Calculate spring end position
    spring_end_x = xp + r * np.cos(theta)
    spring_end_z = r * np.sin(theta)
    
    ball_x = spring_end_x
    ball_y = spring_end_z
    
    ball = plt.Circle((ball_x, ball_y), ball_radius, color='k')
    ax.add_patch(ball)
    
    plt.tight_layout()

    
if __name__ == '__main__':
    r0 = 1.0
    xp = 1.0
    stance_state = np.array([np.pi/4, 0.0, 0.9*r0, 0.0], dtype=np.float64)
    plot_slip_stance_animate(stance_state, xp)
    
    flight_state = np.array([3.0, 0.0, 3.0, 0.0, np.pi/3], dtype=np.float64)
    plot_slip_flight_animate(flight_state, r0)
    
    plt.show()
    
    
def animate_slip(modes, states, init_mode, init_state, target_mode, target_state, nt, reset_args, target_reset_args):
    r0 = 1
    fig, ax = plt.subplots()
    ax.grid(True)
    for ii in range(nt):
        if modes[ii] == 0:
            plot_slip_flight_animate(states[ii].flatten(), r0, ax)
        elif modes[ii] == 1:
            plot_slip_stance_animate(states[ii].flatten(), reset_args[ii][0], ax)
    
    # Plot start and goal 

    if init_mode == 0:
        plot_slip_flight_animate(init_state, r0, ax, 'r-')
    elif init_mode == 1:
        plot_slip_stance_animate(init_state, reset_args[0], ax, 'r-')
        
        
    if target_mode == 0:
        plot_slip_flight_animate(target_state, r0, ax, 'g-')
    elif target_mode == 1:
        plot_slip_stance_animate(target_state, target_reset_args, ax, 'g-')
    
    plt.show()
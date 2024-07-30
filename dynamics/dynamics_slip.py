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

def stochastic_integration_slip(x0, u, t_span, epsilon, dW):
    return stochastic_integration(x0, u, t_span, epsilon, dW, dyn_slip, gdWt_slip)


# ----------------------------- 
# Define condition functions
# -----------------------------
# --------------------------------- Condition: mode mismatch --------------------------------- 
def cond_mode_mismatch_slip(current_mode, ref_current_mode): 
    return (current_mode != ref_current_mode)

# --------------------------------- Condition: early arrival ---------------------------------
def cond_early_arrival_slip(current_mode, ref_current_mode): 
    return (current_mode==1) and (ref_current_mode==0) 

# Condition: guard function hit
def cond_guard_function_hit_slip(xt, xt_next, guard_func): 
    return ((guard_func(0.0, xt)>0) and (guard_func(0.0, xt_next)<=0))


guard_slip_12.terminal=True
guard_slip_12.direction=1

guard_slip_21.terminal=True
guard_slip_21.direction=1

guards_slip = {0:guard_slip_12, 1: guard_slip_21}
reset_maps_slip = {0:reset_map_slip_12, 1:reset_map_slip_21}


def stochastic_feedback_rollout_slip(init_mode, x0, n_inputs, xt_ref, ref_modechanges, 
                                    ut, Kt, kt, target_state, Q_T, t0, tf, 
                                    epsilon, GaussianNoise, dt_shrinkingrate, 
                                    reference_extension_helper, init_reset_args):

    (_, v_ref_ext_bwd, v_ref_ext_fwd, 
    v_Kfb_ref_ext_bwd, v_Kfb_ref_ext_fwd, 
    v_kff_ref_ext_bwd, v_kff_ref_ext_fwd, _) = extract_extensions(reference_extension_helper, start_index = 0)
    
    n_timestamps = xt_ref.shape[0]
    
    dt = (tf - t0) / n_timestamps
    dt_int = dt
    
    # returning trajectory
    mode_trj = np.zeros(n_timestamps, dtype=np.int64)
    mode_trj[0] = init_mode
    
    xt_trj = [np.array([0.0]) for _ in range(n_timestamps)]
    xt_trj[0] = x0   
    
    # closed-loop controls 
    ut_cl_trj = [np.zeros((n_timestamps, n_inputs[0])), np.zeros((n_timestamps, n_inputs[1]))]
    
    # only consider the 1->2 reset for now 
    current_guard = guards_slip[init_mode]
    
    cnt_mismatch = 0
    xt_ref_actual = np.zeros_like(xt_ref)
    
    # path cost
    Sk = 0
    
    # hybrid event related 
    cnt_event = 0
    reset_args = init_reset_args
    event_args = [init_reset_args[0]]
    
    # -------------- roullout function --------------
    for ii_t in range(n_timestamps-1):   

        t0_i = t0 + ii_t*dt   
        
        current_mode = mode_trj[ii_t]
        xt = xt_trj[ii_t]
        
        ref_current_mode = ref_modechanges[ii_t][0]
        reset_args[ii_t] = event_args[cnt_event]
        
        # ======== Handle mode mismatch ========
        K_fb_i = Kt[ii_t]
        k_ff_i = kt[ii_t]
        
        xref_i = xt_ref[ii_t] 
        if cond_mode_mismatch_slip(current_mode, ref_current_mode):
            xref_i, K_fb_i, k_ff_i, cnt_mismatch = reaction_mode_mismatch(cond_early_arrival_slip, ii_t, current_mode, ref_current_mode, 
                                                                        v_ref_ext_fwd[0], v_ref_ext_bwd[0], 
                                                                        v_Kfb_ref_ext_fwd[0], v_kff_ref_ext_fwd[0],
                                                                        v_Kfb_ref_ext_bwd[0], v_kff_ref_ext_bwd[0],
                                                                        cnt_mismatch)
            
        xt_ref_actual[ii_t] = xref_i
        
        delta_xt_i = xt_trj[ii_t] - xref_i
        u = ut[current_mode][ii_t] + K_fb_i@delta_xt_i + k_ff_i
        ut_cl_trj[current_mode][ii_t] = u
        
        dW_i = np.sqrt(dt_int)*GaussianNoise[current_mode][ii_t]
        
        # ============================== One step integration ==============================        
        # ---- solver for the deterministic part
        t_span = (t0_i, t0_i + dt_int)
        
        xt_next = stochastic_integration_slip(xt, u, t_span, epsilon, dW_i).flatten()
        
        current_guard = guards_slip[current_mode]
        next_mode = current_mode
        # Condition: Hit the guard function.  
        if cond_guard_function_hit_slip(xt, xt_next, current_guard): 
            
            args = (xt, current_mode, u, t0_i, t0_i+dt_int, xt_next, 
                    dt_int, dt_shrinkingrate, GaussianNoise[current_mode][ii_t], epsilon, 
                    stochastic_integration_slip, guards_slip, reset_maps_slip, reset_args[ii_t])
            
            xt_next, next_mode, dW_i, new_reset_args = event_reactive_fun(args)
            dt_int = dt
            
            event_args.append(new_reset_args)
            cnt_event += 1
        
        # Collect cost: consider only the terminal state cost for now.
        Sk += u.T@u/2.0 * dt + np.sqrt(epsilon) * np.dot(u.T, dW_i)
        
        # Update trajectories
        xt_trj[ii_t+1] = xt_next
        mode_trj[ii_t+1] = next_mode
    
    xt_ref_actual[-1] = xt_ref[-1]
    
    # Terminal cost
    Sk += (xt-target_state)@Q_T@(xt-target_state) / 2.0
    
    show_mismatch = False
    if show_mismatch:
        # ======== Show mode mismatch ======== 
        fig2, axes = plt.subplots(1,2, figsize=(9, 6))
        ax5, ax6 = axes.flatten()
        ax5.grid(True)
        ax6.grid(True)
        
        ax5.plot(xt_trj[:,0], xt_trj[:,1],color='b',linewidth=1.5,label='Rollout')
        ax5.plot(xt_ref[:,0], xt_ref[:,1],color='k',linewidth=2.5,label='Reference')
        ax5.plot(xt_ref_actual[:,0], xt_ref_actual[:,1],color='r',linewidth=1.5,linestyle='--', label='Modified Reference')
        
        ax5.set_xlabel(r"z", fontsize=14)
        ax5.set_ylabel(r"$\dot z$", fontsize=14)
        ax5.legend(loc='upper right')
        plt.tight_layout()
        
        ax6.plot(xt_trj[:,0], xt_trj[:,1],color='b',linewidth=1.5,label='Rollout')
        ax6.plot(xt_ref[:,0], xt_ref[:,1],color='k',linewidth=2.5,label='Reference')
        ax6.plot(xt_ref_actual[:,0], xt_ref_actual[:,1],color='r',linewidth=1.5,linestyle='--',label='Modified Reference')
        ax6.set_xlabel(r"z", fontsize=14)
        ax6.set_ylabel(r"$\dot z$", fontsize=14)
        ax6.legend(loc='upper right')
        plt.tight_layout()
        
        plt.show()
    
    return mode_trj, xt_trj, ut_cl_trj, Sk, xt_ref_actual


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
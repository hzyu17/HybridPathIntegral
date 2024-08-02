## 2-dimensional SLIP dynamics
# mode 1 (flight): x = [px, vx, pz, vz, theta], u = [theta_dot]
# mode 2 (stance): x = [theta, theta_dot, r, r_dot], u = [r_delta, \tau_hip]
# reset maps: identity

import matplotlib.pyplot as plt
from dynamics.dynamics import *
from dynamics.guard_reset_slip import *

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

def gdWt_flight_slip(x0, dWt, eps):
    B = np.array([[0],[0],[0],[0],[1.0]], dtype=np.float64)
    return np.sqrt(eps) * B@dWt

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
    k = 25.0
    m = 0.5
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


def gdWt_stance_slip(xt, dWt, eps):
    k = 25.0
    m = 0.5
    
    r = xt[2]
    B = np.array([[0.0, 0.0],[0.0, 0.0],[0.0, 1/m/r/r],[k/m, 0.0]], dtype=np.float64)
    
    return np.sqrt(eps) * B@dWt


# ---------------------------
# Discrete-time definition
# ---------------------------
def symbolic_stance_dynamics_slip():
    g = 9.81
    k = 25.0
    m = 0.5
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

def stochastic_integration_slip(mode, x0, u, t_span, epsilon, dW):
    if (mode==0):
        return stochastic_integration(x0, u, t_span, epsilon, dW, dyn_flight_slip, gdWt_flight_slip)
    elif (mode==1):
        return stochastic_integration(x0, u, t_span, epsilon, dW, dyn_stance_slip, gdWt_stance_slip)


# ----------------------------- 
# Define condition functions
# -----------------------------
# --------------------------------- Condition: mode mismatch --------------------------------- 
def cond_mode_mismatch_slip(current_mode, ref_current_mode): 
    return (current_mode != ref_current_mode)

# --------------------------------- Condition: early arrival ---------------------------------
def cond_early_arrival_slip(current_mode, ref_current_mode, event_modechange): 
    return (current_mode==event_modechange[1]) and (ref_current_mode==event_modechange[0]) 

# Condition: guard function hit
def cond_guard_function_hit_slip(xt, xt_next, guard_func): 
    return (guard_func(0.0, xt)>0) and (guard_func(0.0, xt_next)<=0)


guards_slip = {0:guard_slip_12, 1: guard_slip_21}
reset_maps_slip = {0:reset_map_slip_12, 1:reset_map_slip_21}

def hybrid_stochastic_feedback_rollout_slip(init_mode, x0, n_inputs, xt_ref, ref_modes, 
                                            ut, Kt, kt, target_state, Q_T, t0, tf, 
                                            epsilon, GaussianNoise, dt_shrinkingrate, 
                                            reference_extension_helper, init_reset_args):

    (v_event_modechange, v_ref_ext_bwd, v_ref_ext_fwd, 
    v_Kfb_ref_ext_bwd, v_Kfb_ref_ext_fwd, 
    v_kff_ref_ext_bwd, v_kff_ref_ext_fwd, _) = extract_extensions(reference_extension_helper, start_index = 0)
    
    n_timestamps = len(xt_ref)
    
    dt = (tf - t0) / n_timestamps
    dt_int = dt
    
    # returning trajectory    
    xt_trj = [np.array([0.0]) for _ in range(n_timestamps)]
    xt_trj[0] = x0  
    
    mode_trj = np.zeros((n_timestamps), dtype=np.int64) 
    mode_trj[0] = init_mode
    
    # closed-loop controls 
    ut_cl_trj = [np.zeros((n_timestamps, n_inputs[0])), np.zeros((n_timestamps, n_inputs[1]))]
    
    # only consider the 1->2 reset for now 
    current_guard = guards_slip[init_mode]
    
    cnt_mismatch = 0
    xt_ref_actual = [np.array([0.0]) for _ in range(n_timestamps)]
    
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
        
        ref_current_mode = ref_modes[ii_t]
        reset_args[ii_t] = event_args[cnt_event]
        
        # ======== Handle mode mismatch ========
        K_fb_i = Kt[ii_t]
        k_ff_i = kt[ii_t]
        xref_i = xt_ref[ii_t] 
        
        if cond_mode_mismatch_slip(current_mode, ref_current_mode):
            xref_i, K_fb_i, k_ff_i, cnt_mismatch = reaction_mode_mismatch(cond_early_arrival_slip, ii_t, 
                                                                          current_mode, ref_current_mode, 
                                                                            v_ref_ext_fwd[0], v_ref_ext_bwd[0], 
                                                                            v_event_modechange[0],
                                                                            v_Kfb_ref_ext_fwd[0], v_kff_ref_ext_fwd[0],
                                                                            v_Kfb_ref_ext_bwd[0], v_kff_ref_ext_bwd[0],
                                                                            cnt_mismatch)
        
        xt_ref_actual[ii_t] = xref_i
        
        delta_xt_i = xt_trj[ii_t] - xref_i
        current_u = ut[current_mode][ii_t] + K_fb_i@delta_xt_i + k_ff_i
        ut_cl_trj[current_mode][ii_t] = current_u
        
        noise_i = GaussianNoise[current_mode][ii_t]
        dW_i = np.sqrt(dt_int)*noise_i
        
        # ============================== One step integration ==============================        
        # ---- solver for the deterministic part
        t_span = (t0_i, t0_i + dt_int)
        
        xt_next = stochastic_integration_slip(current_mode, xt, current_u, t_span, epsilon, dW_i).flatten()
        next_mode = current_mode
        
        # Condition: Hit the guard function.  
        current_guard = guards_slip[current_mode]
        if cond_guard_function_hit_slip(xt, xt_next, current_guard): 
            
            args = (xt, current_mode, current_u, t0_i, t0_i+dt_int, xt_next, 
                    dt_int, dt_shrinkingrate, GaussianNoise[current_mode][ii_t], epsilon, 
                    stochastic_integration_slip, guards_slip, reset_maps_slip, reset_args[ii_t])
            
            xt_next, next_mode, dW_i, new_reset_args = event_reactive_fun(args)
            dt_int = dt
            
            event_args.append(new_reset_args)
            cnt_event += 1
        
        # ============================== // One step integration // ==============================     
        
        # Collect cost: consider only the terminal state cost for now.
        Sk += current_u.T@current_u/2.0 * dt + np.sqrt(epsilon) * np.dot(current_u.T, dW_i)
        
        # Update trajectories
        xt_trj[ii_t+1] = xt_next
        mode_trj[ii_t+1] = next_mode
    
    xt_ref_actual[-1] = xt_ref[-1]
    
    fig, ax = plt.subplots()
    time_span = np.arange(0, n_timestamps)
    plot_slip(time_span, mode_trj, xt_trj, ut_cl_trj, 
              x0, target_state, n_timestamps, reset_args, 
              fig=None, axes=None, color='k', alpha=1.0, step=2)
    
    ax.legend()
    plt.show()
    
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
    # guard_slip_12.terminal=True
    # guard_slip_12.direction=1
    
    # guard_slip_21.terminal=True
    # guard_slip_21.direction=1
    
    # guards_slip = {0:guard_slip_12, 1: guard_slip_21}
    # reset_maps_slip = {0:reset_map_slip_12, 1:reset_map_slip_21}
    
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
    

def plot_slip(time_span, modes, states, inputs, 
              init_state, target_state, nt, reset_args, 
              fig=None, axes=None, color='k', alpha=1.0, step=2):
    # =============== plotting ===============
    if (fig is None) and (axes is None):
        fig, axes = plt.subplots(2, 7, figsize=(15, 10))
        
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

    # convert the stance mode states to the flight mode states
    flight_mode_states = np.zeros((nt, 5))
    for i in range(nt):
        if modes[i] == 0:
            flight_mode_states[i] = states[i].flatten()
        elif modes[i] == 1:
            flight_mode_states[i] = convert_state_21_slip(states[i], reset_args[i][0]).flatten()

    
    ax11.plot(time_span[0:-1:step], modes[0:-1:step], color=color)   
    ax12.plot(time_span[0:-1:step], flight_mode_states[0:-1:step,0], color=color, alpha=alpha)
    ax13.plot(time_span[0:-1:step], flight_mode_states[0:-1:step,1], color=color, alpha=alpha)
    ax14.plot(time_span[0:-1:step], flight_mode_states[0:-1:step,2], color=color, alpha=alpha)
    ax15.plot(time_span[0:-1:step], flight_mode_states[0:-1:step,3], color=color, alpha=alpha)
    ax16.plot(time_span[0:-1:step], flight_mode_states[0:-1:step,4], color=color, alpha=alpha)
    
    
    # --------------------------------------- 
    # collect the mode 1 states and inputs
    # ---------------------------------------
    mode1_timestamps = []
    mode1_states = []
    mode1_inputs = []
    mode1_modes = []
    
    # --------------------------------------- 
    # collect the mode 0 states and inputs
    # ---------------------------------------
    mode0_timestamps = []
    mode0_states = []
    mode0_inputs = []
    mode0_modes = []
    
    
    for i in range(0,nt-1,step):
        if modes[i] == 0:
            mode0_timestamps.append(time_span[i])
            mode0_states.append(states[i])
            mode0_inputs.append(inputs[modes[i]][i])
            mode0_modes.append(modes[i])
            
        if modes[i] == 1:
            assert len(states[i]) == 4
            mode1_timestamps.append(time_span[i])
            mode1_states.append(states[i])
            mode1_inputs.append(inputs[modes[i]][i])
            mode1_modes.append(modes[i])
        
    
    mode0_timestamps = np.array(mode0_timestamps)
    mode0_states = np.array(mode0_states)
    mode0_inputs = np.array(mode0_inputs)
    mode0_modes = np.array(mode0_modes)
    
    mode1_timestamps = np.array(mode1_timestamps)
    mode1_states = np.array(mode1_states)
    mode1_inputs = np.array(mode1_inputs)
    mode1_modes = np.array(mode1_modes)
    
    # plot mode 0 control input
    ax17.plot(mode0_timestamps[0:-1:step], mode0_inputs[0:-1:step], color='b', label=r'$u$')
    
    # plot mode 1
    ax21.plot(mode1_timestamps[0:-1:step], mode1_modes[0:-1:step], color=color, alpha=alpha)   
    ax22.plot(mode1_timestamps[0:-1:step], mode1_states[0:-1:step, 0], color=color, alpha=alpha)
    ax23.plot(mode1_timestamps[0:-1:step], mode1_states[0:-1:step, 1], color=color, alpha=alpha)
    ax24.plot(mode1_timestamps[0:-1:step], mode1_states[0:-1:step, 2], color=color, alpha=alpha)
    ax25.plot(mode1_timestamps[0:-1:step], mode1_states[0:-1:step, 3], color=color, alpha=alpha)
    
    ax26.plot(mode1_timestamps[0:-1:step], mode1_inputs[0:-1:step, 0], color='b', label=r'$u_1$')
    ax26.plot(mode1_timestamps[0:-1:step], mode1_inputs[0:-1:step, 1], color='r', label=r'$u_2$')
    
    
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
    
    if ax26.get_legend() is None:
        ax26.legend()
        
    if ax17.get_legend() is None:
        ax17.legend()
    
    return fig, np.array([[ax11, ax12, ax13, ax14, ax15, ax16, ax17], [ax21, ax22, ax23, ax24, ax25, ax26, ax27]], dtype=object)  


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
    

def unpad_control_slip(modes, inputs_padded):
    nt = modes.shape[0]
    inputs = [np.zeros((nt, 1)), np.zeros((nt, 2))]
    
    for i in range(nt):
        
        if modes[i] == 0:
            input_i = inputs_padded[i, :1]
        elif modes[i] == 1:
            input_i = inputs_padded[i]
            
        inputs[modes[i]][i] = input_i
    
    return inputs

def unpad_state_slip(modes, states_padded):
    
    nt = modes.shape[0]
    states = [np.array([0.0]) for _ in range(nt)]
    
    for i in range(nt):
        if modes[i] == 0:
            state_i = states_padded[i]
        elif modes[i] == 1:
            state_i = states_padded[i, 0:4]
        states[i] = state_i
    
    return states

    
if __name__ == '__main__':
    r0 = 1.0
    xp = 1.0
    stance_state = np.array([np.pi/4, 0.0, 0.9*r0, 0.0], dtype=np.float64)
    plot_slip_stance_animate(stance_state, xp)
    
    flight_state = np.array([3.0, 0.0, 3.0, 0.0, np.pi/3], dtype=np.float64)
    plot_slip_flight_animate(flight_state, r0)
    
    plt.show()
    
    
def animate_slip(modes, states, init_mode, init_state, target_mode, target_state, nt, reset_args, target_reset_args,step=1):
    r0 = 1
    fig, ax = plt.subplots()
    ax.grid(True)
    for ii in range(0,nt,step):
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
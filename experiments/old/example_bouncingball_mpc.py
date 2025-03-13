import numpy as np

import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

# Import Path Integral Control functions
from example_bouncingball import process_sampling_feedback, process_compute_costs
# Import iLQR class
from hybrid_ilqr import h_ilqr
# Importing path integral control
from hybrid_pathintegral.hybrid_pathintegral import update_control_pathintegral
# Import pendulum dynamics
from dynamics.ode_solver.dynamics_bouncing import *
# Import plotting
import matplotlib.pyplot as plt

# for paralle sampling on cpu
from joblib import Parallel, delayed

# Import dynamics
(f,A,B) = sym_dyn_bouncing()

# # ==================== Define necessary functions for sampling and compute costs ====================
# def compute_cost(states,inputs,target_state,trj_ref, Qk, Rk, QT):
#     nt= trj_ref.shape[0]
#     # Initialize cost
#     total_cost = 0.0
#     for ii in range(nt):
#         current_x = states[ii] # Not being used currently
#         # current_x_ref = trj_ref[ii]
#         # trj_difference =  current_x_ref - current_x
#         # current_cost_xref = trj_difference.T@Qk@trj_difference
        
#         current_u = inputs[ii,:].flatten()
        
#         trj_difference =  np.sum((trj_ref - current_x)*(trj_ref - current_x), axis=1)
#         min_value = np.min(trj_difference)
#         # print(min_value)
#         if min_value > 0.001:
#             current_cost_xref = 1000
#         else:
#             min_indx = np.argmin(trj_difference)
#             min_difference =  trj_ref[min_indx] - current_x
#             current_cost_xref = min_difference.T@Qk@min_difference
        
#         current_cost = current_u.T@Rk@current_u # Right now only considering cost in input
        
#         total_cost = total_cost+current_cost+current_cost_xref
        
#     # Compute terminal cost
#     terminal_difference = (target_state-states[-1,:]).flatten()
#     terminal_cost = terminal_difference.T@QT@terminal_difference
#     total_cost = total_cost+terminal_cost
#     return total_cost

# # Define sampling function
# def process_sampling(sample_i, init_state, inputs, start_time, end_time, epsilon, RandN, i):
#     # print("Sampling trajectory: ", i)
#     sample_i = rollout_bouncing_stochastic(init_state, inputs, start_time, end_time, epsilon, RandN[i])
#     return sample_i, i

# # Compute path costs function
# def process_compute_costs(sample_i, inputs, target_state, ref_states, i, Qk, Rk, QT):
#     # print("Computing costs: ", i)
#     costs_i = compute_cost(sample_i, inputs, target_state, ref_states, Qk, Rk, QT)
#     return costs_i, i

# ==================== Initialize timings ====================
# Initialize timings
dt = 0.05
start_time = 0
end_time = 2.0
time_span = np.arange(start_time, end_time, dt).flatten()

# Set desired state
n_states = 2
n_inputs = 1
init_state = np.array([5, 1.5])    # Define the initial state to be the origin with no velocity
target_state = np.array([2.5, 0])  # Swing pendulum upright

# Initial guess of zeros, but you can change it to any guess
initial_guess = 0.5*np.ones((np.shape(time_span)[0],n_inputs))
# Define weighting matrices
Q_k = np.zeros((n_states,n_states)) # zero weight to penalties along a strajectory since we are finding a trajectory
R_k = 0.01*np.eye(n_inputs)

# Set the terminal cost
Q_T = 10*np.eye(n_states)

# Set the physical parameters of the system
mass = 1
gravity = 9.8
parameters = np.array([mass,gravity])

# Specify max number of iterations
n_iterations = 10

# Initialize hybrid ilqr object
ilqr_ = h_ilqr(init_state,target_state,initial_guess,dt,start_time,end_time,detect_bouncing,f,A,B,Q_k,R_k,Q_T,parameters,n_iterations)

# Solve for swing up
(states,inputs,k_feedforward,K_feedback,current_cost,states_iter) = ilqr_.solve()

states[0] = init_state

# ====================================== Path Integral Control ====================================== 
# Define the cost
Q_k = 0.5*Q_T
R_k = np.eye(1)
# Q_T = 0.0*Q_T

# Horizon
nt = len(time_span)
Horizon = 30 
n_samples = 100
epsilon = 0.3

plot_samples = True
if plot_samples:
    # Create an interactive figure to plot iterative samples and trajectories
    plt.ion() # Enable interactive mode

    # Initial setup: create a figure and axes
    fig0, axes = plt.subplots(1, 2)
    (ax0, ax1) = axes.flatten()
    ax0.grid(True)
    ax1.grid(True)
    ax0.scatter(time_span, states[:-1,0], color='r', s=2.0)
    ax1.scatter(time_span, states[:-1,1], color='r', s=2.0)
    line_objects_0 = []
    line_objects_1 = []

inputs_i = inputs[0: Horizon]
planning_length = Horizon

# Actual optimal control and trajectory 
xt = init_state
xt_trj = np.zeros((nt, n_states))
xt_trj[0] = xt
xt_ilqr = xt
u_star_mpc = np.zeros((nt, n_inputs))

for i_iter in range(nt-1):        
    # When we reach the end 
    if i_iter >= nt-Horizon:
        planning_length = nt - i_iter 
        
    inputs_i = inputs[i_iter: i_iter+planning_length]
    K_feedback_i = K_feedback[i_iter: i_iter+planning_length]
    k_feedforward_i = k_feedforward[i_iter: i_iter+planning_length]
        
    # The reference 
    x_ref_i = states[i_iter: i_iter+planning_length]
    target_state_i = states[i_iter+planning_length]
    start_time_i = time_span[i_iter]
    end_time_i = time_span[i_iter+planning_length-1]
    
    # Rollouts
    sampled_trjs = np.zeros((n_samples, planning_length, n_states))
    sampled_controls = np.zeros((n_samples, planning_length, n_inputs))
    PathCosts_i = np.zeros(n_samples, dtype=np.float64)

    # Generate the randomness
    GaussianNoise = np.random.randn(n_samples, planning_length, n_inputs)

    # ------------- ilqr --------------
    # samples_index = Parallel(n_jobs=-1)(delayed(process_sampling)(sampled_trjs[i,:,:], xt, inputs_i, start_time_i, end_time_i, epsilon, GaussianNoise, i) for i in range(n_samples))
    samples_index = Parallel(n_jobs=-1)(delayed(process_sampling_feedback)(sampled_trjs[i,:,:], xt, states_i, inputs_i, K_feedback_i, k_feedforward_i, start_time_i, end_time_i, epsilon, GaussianNoise, i) for i in range(n_samples))

    for sample_i, sample_input_i, index in samples_index:
        sampled_trjs[index] = sample_i
        sampled_controls[index] = sample_input_i
            
    costs_index = Parallel(n_jobs=-1)(delayed(process_compute_costs)(sampled_trjs[i], sampled_controls[i,:,:], target_state_i, x_ref_i, i, Q_k, R_k, Q_T) for i in range(n_samples))
    for cost_i, index in costs_index:
        PathCosts_i[index] = cost_i

    # ------------- E{cost_ilqr} -------------
    cost_ilqr = np.mean(PathCosts_i)

    # Update the control proposal using path integral 
    u_star = update_control_pathintegral(inputs_i, PathCosts_i, epsilon, dt)
    
    # --------------- Send the control to actuator ---------------
    RndN_actual = np.random.randn(n_inputs)
    dW_actual = np.sqrt(dt)*RndN_actual
    
    t_span = (start_time_i, start_time_i+dt)
    t_eval = np.linspace(start_time_i, start_time_i+dt, planning_length)
    
    u_star_mpc[i_iter] = u_star[0]
    
    xt_next = stochastic_integration_bouncing(xt, u_star[0], t_span, t_eval, epsilon, dW_actual, dt, planning_length)
    
    
    if plot_samples:
        # Update the data of the plot object
        for line in line_objects_0:
            line.remove()
        line_objects_0.clear()  # Clear the list of line objects
        
        for line in line_objects_1:
            line.remove()
        line_objects_1.clear()  # Clear the list of line objects
        
        t_span_horizon = np.linspace(start_time_i, end_time_i, planning_length)
        n_samples_plotted = 100
        plot_step = n_samples // n_samples_plotted
        for i_s in range(0, n_samples, plot_step):
            (line_0,) = ax0.plot(t_span_horizon, sampled_trjs[i_s,:,0], 'b', alpha=0.1)
            line_objects_0.append(line_0)
            (line_1,) = ax1.plot(t_span_horizon, sampled_trjs[i_s,:,1], 'b', alpha=0.1)
            line_objects_1.append(line_1)
        
        ax0.scatter(start_time_i, xt[0], color='g', s=2.0)
        ax0.scatter(start_time_i, xt_ilqr[0], color='k', s=2.0)
        
        ax1.scatter(start_time_i, xt[1], color='g', s=2.0)
        ax1.scatter(start_time_i, xt_ilqr[1], color='k', s=2.0)
        
        plt.draw()
        plt.pause(0.1)  # Pause to update the plot and process GUI events
        
    xt_ilqr = stochastic_integration(xt, inputs_i[0], t_span, t_eval, epsilon, dW_actual, planning_length)
    xt = xt_next
    xt_trj[i_iter+1] = xt
        
u_star_mpc[-1] = np.zeros(n_inputs)

# ========================== plot results ==========================
plt.ioff()
fig1, axes = plt.subplots(1, 2)
(ax2, ax3) = axes.flatten()
ax2.grid(True)
ax3.grid(True)

# ----------- plot the path integral controlled trajectory -----------
ax2.plot(time_span, xt_trj[:, 0], 'r', label='Actual Position')
ax2.plot(time_span, states[:-1,0], label='Reference Position')
ax3.plot(time_span, xt_trj[:, 1], 'r', label='Actual Velocity')
ax3.plot(time_span, states[:-1,1], label='Reference Velocity')

# ----------- Plot the start and goal states -----------
ax2.scatter(time_span[-1], target_state[0], color='g', marker='x', s=50.0, linewidths=6, label='Target')
ax2.scatter(time_span[0], init_state[0], color='r', marker='x', s=50.0, linewidths=6, label='Start')

ax3.scatter(time_span[-1], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
ax3.scatter(time_span[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')

# -------- Labels and Legends --------
ax2.set_xlabel(r"Time")
ax2.set_ylabel(r"$z$")
ax2.set_title("Bouncing Ball Vertical Position")

ax3.set_xlabel(r"Time")
ax3.set_ylabel(r"$\dot z$")
ax3.set_title("Bouncing Ball Vertical Velocity")

ax2.legend()
ax3.legend()

plt.show()

# ----------- Plot the last iteration of iLQR controller ----------
# states = states_iter[-1]
# ax3.plot(time_span, states[:-1,0],'k',label='iLQR')
# ax3.set_xlabel(r"Time")
# ax3.set_ylabel(r"$z$")
# ax3.set_title("Bouncing Ball Vertical Position")

# # ----------- Plot the start and goal states -----------
# ax3.scatter(time_span[-1], target_state[0], color='g', marker='x', s=50.0, linewidths=6, label='Target')
# ax3.scatter(time_span[0], init_state[0], color='r', marker='x', s=50.0, linewidths=6, label='Start')

# ax3.legend()

# # ----------- Plot the last iteration of iLQR controller ----------
# ax4.plot(time_span, states[:-1,1],'k',label='iLQR')

# # ----------- Plot the start and goal states -----------
# ax4.scatter(time_span[-1], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
# ax4.scatter(time_span[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')

# ax4.set_xlabel(r"Time")
# ax4.set_ylabel(r"$\dot z$")
# ax4.set_title("Bouncing Ball Vertical Velocity")

# ax4.legend()

# plot control inputs
# fig3, ax7 = plt.subplots(1, 1)
# ax7.grid(True)
# ax7.plot(time_span, inputs[:,0],'k',label='Final iteration ilqr')
# ax7.plot(time_span, u_star_mpc[:,0],'r',label='Path integral controller')
# ax7.set_xlabel(r"Timestep")
# ax7.set_ylabel(r"$u$")
# ax7.set_title("Bouncing Ball Final Control Input")

# ax7.legend()


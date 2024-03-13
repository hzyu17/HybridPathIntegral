import numpy as np

import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

# Import iLQR class
from hybrid_ilqr import hybrid_ilqr
# Importing path integral control
from hybrid_pathintegral.hybrid_pathintegral import update_control_pathintegral
# Import pendulum dynamics
from dynamics.symbolic_bouncing_1D import *
# Import plotting
import matplotlib.pyplot as plt

# for paralle sampling on cpu
from joblib import Parallel, delayed

# Import dynamics
(f,A,B) = symbolic_dynamics_bouncing()

# Initialize timings
dt = 0.01
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
ilqr_ = hybrid_ilqr(init_state,target_state,initial_guess,dt,start_time,end_time,detect_bouncing,f,A,B,Q_k,R_k,Q_T,parameters,n_iterations)

# Solve for swing up
(states,inputs,k_feedforward,K_feedback,current_cost,states_iter) = ilqr_.solve()

states[0] = init_state

# ====================================== Path Integral Control ====================================== 
# Define the cost
Q_k = 0.1*Q_T

def compute_cost(states,inputs,target_state,trj_ref, Qk, Rk, QT):
    nt, _ = trj_ref.shape
    # Initialize cost
    total_cost = 0.0
    for ii in range(nt):
        current_x = states[ii] # Not being used currently
        current_x_ref = trj_ref[ii]
        current_u = inputs[ii,:].flatten()
        
        trj_difference =  current_x_ref - current_x

        current_cost_xref = trj_difference.T@Qk@trj_difference
        
        current_cost = current_u.T@Rk@current_u # Right now only considering cost in input
        total_cost = total_cost+current_cost+current_cost_xref
    # Compute terminal cost
    terminal_difference = (target_state-states[-1,:]).flatten()
    terminal_cost = terminal_difference.T@QT@terminal_difference
    total_cost = total_cost+terminal_cost
    return total_cost

# Define sampling function
def process_sampling(sample_i, init_state, inputs, start_time, end_time, epsilon, RandN, i):
    print("Sampling trajectory: ", i)
    sample_i = rollout_bouncing_stochastic(init_state, inputs, start_time, end_time, epsilon, RandN[i])
    return sample_i, i

# Compute path costs function
def process_compute_costs(sample_i, inputs, target_state, ref_states, i):
    print("Computing costs: ", i)
    costs_i = compute_cost(sample_i, inputs, target_state, ref_states, Q_k, R_k, Q_T)
    return costs_i, i

# Horizon
H_size = 20 
nt = len(time_span)
n_samples = 200
epsilon = 0.2

# Actual trajectory
xt = init_state
xt_trj = np.zeros((nt, n_states))
xt_trj[0] = xt

inputs_i = inputs[0: 0+H_size]
    
for i_iter in range(nt-H_size):
    # The reference state trajectory and proposal control
    
    x_ref_i = states[i_iter: i_iter+H_size]
    target_state_i = states[i_iter+H_size]
    start_time_i = 0.0
    end_time_i = time_span[i_iter+H_size] - time_span[i_iter]
    
    # sampling stochastic rollouts
    sampled_trjs = np.zeros((n_samples, H_size, n_states))
    PathCosts_i = np.zeros(n_samples, dtype=np.float64)

    # Generate the randomness
    GaussianNoise = np.random.randn(n_samples, H_size, n_inputs)

    # ------------- ilqr --------------
    samples_index = Parallel(n_jobs=-1)(delayed(process_sampling)(sampled_trjs[i,:,:], xt, inputs_i, start_time_i, end_time_i, epsilon, GaussianNoise, i) for i in range(sampled_trjs.shape[0]))

    for sample_i, index in samples_index:
        sampled_trjs[index] = sample_i

    costs_index = Parallel(n_jobs=-1)(delayed(process_compute_costs)(sampled_trjs[i], inputs_i, target_state_i, x_ref_i, i) for i in range(sampled_trjs.shape[0]))
    for cost_i, index in costs_index:
        PathCosts_i[index] = cost_i

    # ------------- E{cost_ilqr} -------------
    cost_ilqr = np.mean(PathCosts_i)

    # update the control proposal using path integral 
    u_star = update_control_pathintegral(inputs_i, PathCosts_i, epsilon, dt)
    
    # Send the first control to actuator
    RndN_actual = np.random.randn(nu)
    dW_actual = np.sqrt(dt)*RndN_actual
    
    t_span = (start_time_i, end_time_i)
    t_eval = np.linspace(start_time_i, end_time_i, nt)
    
    xt = stochastic_integration(x0, u_star, t_span, t_eval, epsilon, dW_actual, nt)
    xt_trj[i_iter+1] = xt
    
    # Update the proposal control
    inputs_i[0:-1] = u_star[1:]
    inputs_i[-1] = np.zeros(nu)
    

# # # ================== compare the cost: path integral V.S. ilqr proposal ==================
# sampled_trjs_PI = np.zeros((n_samples, nt, n_states))
# PathCosts_PI = np.zeros(n_samples, dtype=np.float64)

# # ------------- sampling PI controller -------------
# GaussianNoise_PI = np.random.randn(n_samples, nt, n_inputs)
# samples_index_PI = Parallel(n_jobs=-1)(delayed(process_sampling)(sampled_trjs_PI[i,:,:], u_star, GaussianNoise_PI, i) for i in range(sampled_trjs_PI.shape[0]))

# for sample_i, index in samples_index_PI:
#     sampled_trjs_PI[index] = sample_i
    
# # ------------- E{cost_pi} -------------
# costs_index_PI = Parallel(n_jobs=-1)(delayed(process_compute_costs)(sampled_trjs_PI[i], u_star, i) for i in range(sampled_trjs_PI.shape[0]))
# for cost_i, index in costs_index_PI:
#     PathCosts_PI[index] = cost_i
    
# cost_pi = np.mean(PathCosts_PI)

# print("Cost iLQR: ", cost_ilqr)
# print("Cost Path Integral Controller: ", cost_pi)

# # actual trajectory using the path integral control
GaussianNoise_new = np.random.randn(nt, n_inputs)
trj_pi = rollout_bouncing_stochastic(init_state, u_star, start_time, end_time, epsilon, GaussianNoise_new)

# Animate
# animate_pendulum(states,inputs,dt,parameters)

fig1, axes = plt.subplots(2, 2)
(ax1, ax2, ax3, ax4) = axes.flatten()
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)

# for i in range(len(states_iter)):
#     states = states_iter[i]
#     ax1.plot(time_span, states[:-1,0],label='Iteration {}'.format(i))
#     ax1.set_xlabel(r"Time")
#     ax1.set_ylabel(r"$z$")
#     ax1.set_title("Bouncing Ball Vertical Position")
    
#     ax2.plot(time_span, states[:-1,1],label='Iteration {}'.format(i))
#     ax2.set_xlabel(r"Time")
#     ax2.set_ylabel(r"$\dot z$")
#     ax2.set_title("Bouncing Ball Vertical Velocity")
    
# ----------- plot the stochastic sampled trajectory -----------
n_samples_plotted = 100
plot_step = n_samples // n_samples_plotted
for i_s in range(0, n_samples, plot_step):
    ax1.plot(time_span, sampled_trjs[i_s, :, 0], 'b', alpha=0.1)
ax1.plot(time_span, sampled_trjs[-1, :, 0], 'b', alpha=0.1, label='Samples')

# # ----------- plot the path integral controlled trajectory -----------
# ax1.plot(time_span, trj_pi[:, 0], 'r', label='Path Integral')

# ----------- Plot the last iteration of iLQR controller ----------
states = states_iter[-1]
ax1.plot(time_span, states[:-1,0],'k',label='iLQR')
ax1.set_xlabel(r"Time")
ax1.set_ylabel(r"$z$")
ax1.set_title("Bouncing Ball Vertical Position")

# ----------- Plot the start and goal states -----------
ax1.scatter(time_span[-1], target_state[0], color='g', marker='x', s=50.0, linewidths=6, label='Target')
ax1.scatter(time_span[0], init_state[0], color='r', marker='x', s=50.0, linewidths=6, label='Start')

ax1.legend()

# ----------- plot the stochastic sampled trajectory -----------
for i_s in range(0, n_samples, plot_step):
    ax2.plot(time_span, sampled_trjs[i_s, :, 1], 'b', alpha=0.15)
ax2.plot(time_span, sampled_trjs[-1, :, 1], 'b', alpha=0.15, label='Samples')

# # ----------- plot the path integral controlled trajectory -----------
# ax2.plot(time_span, trj_pi[:, 1], 'r', label='Path Integral')

# ----------- Plot the last iteration of iLQR controller ----------
ax2.plot(time_span, states[:-1,1],'k',label='iLQR')

# ----------- Plot the start and goal states -----------
ax2.scatter(time_span[-1], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
ax2.scatter(time_span[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')

ax2.set_xlabel(r"Time")
ax2.set_ylabel(r"$\dot z$")
ax2.set_title("Bouncing Ball Vertical Velocity")

ax2.legend()


# =========== samples for path integral control ===========
# ----------- plot the stochastic sampled trajectory -----------
n_samples_plotted = 100
plot_step = n_samples // n_samples_plotted
for i_s in range(0, n_samples, plot_step):
    ax3.plot(time_span, sampled_trjs_PI[i_s, :, 0], 'r', alpha=0.1)
ax3.plot(time_span, sampled_trjs_PI[-1, :, 0], 'r', alpha=0.1, label='Samples')

# ----------- plot the path integral controlled trajectory -----------
ax3.plot(time_span, trj_pi[:, 0], 'r', label='Path Integral')

# ----------- Plot the last iteration of iLQR controller ----------
states = states_iter[-1]
ax3.plot(time_span, states[:-1,0],'k',label='iLQR')
ax3.set_xlabel(r"Time")
ax3.set_ylabel(r"$z$")
ax3.set_title("Bouncing Ball Vertical Position")

# ----------- Plot the start and goal states -----------
ax3.scatter(time_span[-1], target_state[0], color='g', marker='x', s=50.0, linewidths=6, label='Target')
ax3.scatter(time_span[0], init_state[0], color='r', marker='x', s=50.0, linewidths=6, label='Start')

ax3.legend()

# ----------- plot the stochastic sampled trajectory -----------
for i_s in range(0, n_samples, plot_step):
    ax4.plot(time_span, sampled_trjs_PI[i_s, :, 1], 'b', alpha=0.15)
ax4.plot(time_span, sampled_trjs_PI[-1, :, 1], 'b', alpha=0.15, label='Samples')

# ----------- plot the path integral controlled trajectory -----------
ax4.plot(time_span, trj_pi[:, 1], 'r', label='Path Integral')

# ----------- Plot the last iteration of iLQR controller ----------
ax4.plot(time_span, states[:-1,1],'k',label='iLQR')

# ----------- Plot the start and goal states -----------
ax4.scatter(time_span[-1], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
ax4.scatter(time_span[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')

ax4.set_xlabel(r"Time")
ax4.set_ylabel(r"$\dot z$")
ax4.set_title("Bouncing Ball Vertical Velocity")

ax4.legend()

# =========== Plot the z-\dot_z figure ===========
fig2, (ax5, ax6) = plt.subplots(1, 2)
ax5.grid(True)
ax6.grid(True)

# ----------- plot the stochastic sampled trajectory -----------
for i_s in range(0, n_samples, plot_step):
    ax5.plot(sampled_trjs[i_s, :, 0], sampled_trjs[i_s, :, 1], 'b', alpha=0.1)
ax5.plot(sampled_trjs[-1, :, 0], sampled_trjs[i_s, :, 1], 'b', alpha=0.1, label='Samples')

# ----------- Plot the last iteration of iLQR controller ----------
ax5.plot(states[:-1,0], states[:-1,1],'k',label='iLQR')

# ----------- plot the path integral controlled trajectory -----------
ax5.plot(trj_pi[:, 0], trj_pi[:, 1], 'r', label='Path Integral')

# ----------- Plot the start and goal states -----------
ax5.scatter(target_state[0], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
ax5.scatter(init_state[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')

ax5.legend()

# =========== Plot the z-\dot_z figure: PI controller ===========
# ----------- plot the stochastic sampled trajectory -----------
for i_s in range(0, n_samples, plot_step):
    ax6.plot(sampled_trjs_PI[i_s, :, 0], sampled_trjs_PI[i_s, :, 1], 'b', alpha=0.1)
ax6.plot(sampled_trjs_PI[-1, :, 0], sampled_trjs_PI[i_s, :, 1], 'b', alpha=0.1, label='Samples')

# ----------- Plot the last iteration of iLQR controller ----------
ax6.plot(states[:-1,0], states[:-1,1],'k',label='iLQR')

# ----------- plot the path integral controlled trajectory -----------
ax6.plot(trj_pi[:, 0], trj_pi[:, 1], 'r', label='Path Integral')

# ----------- Plot the start and goal states -----------
ax6.scatter(target_state[0], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
ax6.scatter(init_state[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')

ax6.legend()

# plot control inputs
fig3, ax7 = plt.subplots(1, 1)
ax7.grid(True)
ax7.plot(time_span, inputs[:,0],'k',label='Final iteration ilqr')
ax7.plot(time_span, u_star[:,0],'r',label='Path integral controller')
ax7.set_xlabel(r"Timestep")
ax7.set_ylabel(r"$u$")
ax7.set_title("Bouncing Ball Final Control Input")

ax7.legend()

# plot PathCosts
fig3, ax7 = plt.subplots()
ax7.grid(True)
ax7.bar(range(n_samples), PathCosts, width = 2, color='navy')
ax7.set_xlabel(r"Sample ID")
ax7.set_ylabel(r"$Path Cost$")
ax7.set_title("Path Cost of stochastic rollouts")

plt.show()

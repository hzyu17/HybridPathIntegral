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
dt = 0.001
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
pendulum_length = 1
parameters = np.array([mass,gravity,pendulum_length])

# Specify max number of iterations
n_iterations = 10

# Initialize hybrid ilqr object
ilqr_ = hybrid_ilqr(init_state,target_state,initial_guess,dt,start_time,end_time,detect_bouncing,f,A,B,Q_k,R_k,Q_T,parameters,n_iterations)

# Solve for swing up
(states,inputs,k_feedforward,K_feedback,current_cost,states_iter) = ilqr_.solve()
    
# ====================================== Path Integral Control ====================================== 
nt = len(time_span)

# The same cost function in ilqr
def compute_cost(states,inputs):
    # Initialize cost
    total_cost = 0.0
    for ii in range(nt):
        current_x = states[ii,:] # Not being used currently
        current_u = inputs[ii,:].flatten()

        current_cost = current_u.T@R_k@current_u # Right now only considering cost in input
        total_cost = total_cost+current_cost
    # Compute terminal cost
    terminal_difference = (target_state-states[-1,:]).flatten()
    terminal_cost = terminal_difference.T@Q_T@terminal_difference
    total_cost = total_cost+terminal_cost
    return total_cost

# sampling stochastic rollouts
n_samples = 100
epsilon = 0.01
sampled_trjs = np.zeros((n_samples, nt, n_states))
PathCosts = np.zeros(n_samples, dtype=np.float64)

def process_sampling(sample_i, i):
    print("Sampling trajectory: ", i)
    sample_i = rollout_bouncing_stochastic(init_state, inputs, start_time, end_time, epsilon)
    return sample_i, i

samples_index = Parallel(n_jobs=-1)(delayed(process_sampling)(sampled_trjs[i,:,:], i) for i in range(sampled_trjs.shape[0]))

for sample_i, index in samples_index:
    sampled_trjs[index] = sample_i
    
# Compute path costs
def process_compute_costs(sample_i, inputs, i):
    print("Computing costs: ", i)
    costs_i = compute_cost(sample_i, inputs)
    return costs_i, i

costs_index = Parallel(n_jobs=-1)(delayed(process_compute_costs)(sampled_trjs[i], inputs, i) for i in range(sampled_trjs.shape[0]))
for cost_i, index in costs_index:
    PathCosts[index] = cost_i
    
# sampled_trjs = np.zeros((n_samples, nt, n_states))
# PathCosts = np.zeros(n_samples, dtype=np.float64)
# print("Stochastic rollout")
# epsilon = 0.01
# for i_s in range(n_samples):
#     print("Sample trajectory: ", i_s)
#     sampletrj_i = rollout_bouncing_stochastic(init_state, inputs, start_time, end_time, epsilon)
#     sampled_trjs[i_s] = sampletrj_i
#     PathCosts[i_s] = compute_cost(sampletrj_i, inputs)

# update the control proposal using path integral 
u_star = update_control_pathintegral(inputs, PathCosts, epsilon, dt=0.0001)

# rollout using the path integral control
trj_pi = rollout_bouncing_stochastic(init_state, u_star, start_time, end_time, epsilon)

# compare the cost: path integral controller V.S. ilqr proposal
cost_PI = compute_cost(trj_pi, u_star)
cost_ilqr = compute_cost(states, inputs)

print("Cost Path Integral Controller: ", cost_PI)
print("Cost iLQR: ", cost_ilqr)

# Animate
# animate_pendulum(states,inputs,dt,parameters)

fig1, (ax1, ax2) = plt.subplots(1, 2)
ax1.grid(True)
ax2.grid(True)

for i in range(len(states_iter)):
    states = states_iter[i]
    ax1.plot(time_span, states[:-1,0],label='Iteration {}'.format(i))
    ax1.set_xlabel(r"Time")
    ax1.set_ylabel(r"$z$")
    ax1.set_title("Bouncing Ball Vertical Position")
    
    # ----------- plot the stochastic sampled trajectory -----------
    for i_s in range(n_samples):
        ax1.plot(time_span, sampled_trjs[i_s, :, 0], 'b', alpha=0.2)
    
    # ----------- plot the path integral controlled trajectory -----------
    ax1.plot(time_span, trj_pi[:, 0], 'r')
    
    ax2.plot(time_span, states[:-1,1],label='Iteration {}'.format(i))
    ax2.set_xlabel(r"Time")
    ax2.set_ylabel(r"$\dot z$")
    ax2.set_title("Bouncing Ball Vertical Velocity")

states = states_iter[-1]
ax1.plot(time_span, states[:-1,0],'k',label='Final iteration')
ax1.set_xlabel(r"Time")
ax1.set_ylabel(r"$z$")
ax1.set_title("Bouncing Ball Vertical Position")
plt.legend()

ax2.plot(time_span, states[:-1,1],'k',label='Final iteration')
ax2.set_xlabel(r"Time")
ax2.set_ylabel(r"$\dot z$")
ax2.set_title("Bouncing Ball Vertical Velocity")

# plot control inputs
fig2, ax3 = plt.subplots(1, 1)
ax3.grid(True)
ax3.plot(time_span, inputs[:,0],'k',label='Final iteration ilqr')
ax3.plot(time_span, u_star[:,0],'r',label='Path integral controller')
ax3.set_xlabel(r"Timestep")
ax3.set_ylabel(r"$u$")
ax3.set_title("Bouncing Ball Final Control Input")

plt.legend()

# plot PathCosts
fig3, ax4 = plt.subplots()
ax4.grid(True)
ax4.bar(range(n_samples), PathCosts, width = 2, color='navy')

plt.show()

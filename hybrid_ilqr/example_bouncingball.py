import numpy as np

import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

# Import iLQR class
from hybrid_ilqr import hybrid_ilqr
# Import pendulum dynamics
from dynamics.symbolic_bouncing_1D import *

# Import plotting
import matplotlib.pyplot as plt

# Import dynamics
(f,A,B) = symbolic_dynamics_bouncing()

# Initialize timings
dt = 0.005
start_time = 0
end_time = 2.0
time_span = np.arange(start_time, end_time, dt).flatten()

# Set desired state
n_states = 2
n_inputs = 1
init_state = np.array([5, 1.0])    # Define the initial state to be the origin with no velocity
target_state = np.array([2.5, 0])  # Swing pendulum upright

# Initial guess of zeros, but you can change it to any guess
initial_guess = 0.5*np.ones((np.shape(time_span)[0],n_inputs))
# Define weighting matrices
Q_k = np.zeros((n_states,n_states)) # zero weight to penalties along a strajectory since we are finding a trajectory
R_k = 0.001*np.eye(n_inputs)

# Set the terminal cost
Q_T = 10*np.eye(n_states)

# Set the physical parameters of the system
mass = 1
gravity = 9.8
pendulum_length = 1
parameters = np.array([mass,gravity,pendulum_length])

# Specify max number of iterations
n_iterations = 50

# Initialize hybrid ilqr object
ilqr_ = hybrid_ilqr(init_state,target_state,initial_guess,dt,start_time,end_time,detect_bouncing,f,A,B,Q_k,R_k,Q_T,parameters,n_iterations)

# Solve for swing up
(states,inputs,k_feedforward,K_feedback,current_cost, states_iter) = ilqr_.solve()

# Animate
# animate_pendulum(states,inputs,dt,parameters)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.grid(True)
ax2.grid(True)

for i in range(len(states_iter)):
    states = states_iter[i]
    ax1.plot(states[:,0],label='Iteration {}'.format(i))
    ax1.set_xlabel(r"Timestep")
    ax1.set_ylabel(r"$z$")
    # ax1.legend(['Iteration {}'.format(i)])
    ax1.set_title("Bouncing Ball Vertical Position")

    plt.plot(states[:,1],label='Iteration {}'.format(i))
    ax2.set_xlabel(r"Timestep")
    ax2.set_ylabel(r"$\dot z$")
    # ax2.legend(['Iteration {}'.format(i)])
    ax2.set_title("Bouncing Ball Vertical Velocity")

states = states_iter[-1]
ax1.plot(states[:,0],'k',label='Final iteration')
ax1.set_xlabel(r"Timestep")
ax1.set_ylabel(r"$z$")
ax1.set_title("Bouncing Ball Vertical Position")
plt.legend()

ax2.plot(states[:,1],'k',label='Final iteration')
ax2.set_xlabel(r"Timestep")
ax2.set_ylabel(r"$\dot z$")
ax2.set_title("Bouncing Ball Vertical Velocity")

fig, ax3 = plt.subplots(1, 1)
ax3.grid(True)
ax3.plot(inputs[:,0],'k',label='Final iteration')
ax3.set_xlabel(r"Timestep")
ax3.set_ylabel(r"$u$")
ax3.set_title("Bouncing Ball Final Control Input")

plt.legend()
plt.show()
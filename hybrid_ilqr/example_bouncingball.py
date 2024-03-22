import numpy as np

import os
import sys
file_path = os.path.abspath(__file__)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

# Import iLQR class
from hybrid_ilqr import hybrid_ilqr
# Importing path integral control
from hybrid_pathintegral.hybrid_pathintegral import *
# Import pendulum dynamics
from dynamics.symbolic_bouncing_1D import *
# Import plotting
import matplotlib.pyplot as plt

# for paralle sampling on cpu
from joblib import Parallel, delayed

import numba

import pickle as pkl

class DataOneSample():
    def __init__(self, x_trj_pi, u_trj_pi, x_trj_ilqr, u_trj_ilqr):
        self._x_trj_pi = x_trj_pi
        self._u_trj_pi = u_trj_pi
        self._x_trj_ilqr = x_trj_ilqr
        self._u_trj_ilqr = u_trj_ilqr
        
    def x_trj_pi(self):
        return self._x_trj_pi
    
    def u_trj_pi(self):
        return self._u_trj_pi
    
    def x_trj_ilqr(self):
        return self._x_trj_ilqr
    
    def u_trj_ilqr(self):
        return self._u_trj_ilqr
        
        
class ExpParams():
    def __init__(self, init_state, target_state, start_time, end_time, dt, dt_pathintegral, epsilon, n_samples, Q_k, R_k, Q_T):
        self._init_state = init_state
        self._target_state = target_state
        self._start_time = start_time
        self._end_time = end_time
        self._dt = dt
        self._dt_pathintegral = dt_pathintegral
        self._epsilon = epsilon
        self._n_samples = n_samples
        self._Q_k = Q_k
        self._R_k = R_k
        self._Q_T = Q_T

class ExpData():
    def __init__(self, params):
        self._data = {}
        self._data['params'] = params
    
    def add_data(self, iter, iter_data):
        self._data[str(iter)] = iter_data
        
    def dump(self, file_name):
        with open(file_name, 'wb') as f:
            pkl.dump(self._data, f, pkl.HIGHEST_PROTOCOL)
            
    def load(self, file_name):
        with open(file_name, 'rb') as f:
            self._data = pkl.load(f)
            
    def get_data(self, iter):
        return self._data[str(iter)]

# Import dynamics
(f,A,B) = symbolic_dynamics_bouncing()

# Initialize timings
dt = 0.01
# dt_pathintegral = dt / 50.0
dt_pathintegral = dt
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
Q_T = 20*np.eye(n_states)
Q_T[0,0] = 400.0

# Set the physical parameters of the system
mass = 1
gravity = 9.8
parameters = np.array([mass,gravity])

# Specify max number of iterations
n_iterations = 10

ilqr_ = hybrid_ilqr(init_state,target_state,initial_guess,dt,start_time,end_time,detect_bouncing,f,A,B,Q_k,R_k,Q_T,parameters,n_iterations)
(states,inputs,k_feedforward,K_feedback,current_cost,states_iter) = ilqr_.solve()


# ====================================== Path Integral Control ====================================== 
# parameters
epsilon = 0.2
n_samples = 100
n_exp = 100
nt = len(time_span)

@njit(float64(
    float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64), parallel=True)
def compute_cost(states,inputs,dWs,target_state,trj_ref, Qk, Rk, QT, epsilon):
    
    states = np.ascontiguousarray(states)
    trj_ref = np.ascontiguousarray(trj_ref)
    target_state = np.ascontiguousarray(target_state)
    inputs = np.ascontiguousarray(inputs)
    Qk = np.ascontiguousarray(Qk)
    Rk = np.ascontiguousarray(Rk)
    QT = np.ascontiguousarray(QT)
    dWs = np.ascontiguousarray(dWs)
    
    # Initialize cost
    total_cost = 0.0
    for ii in range(states.shape[0]):
        current_x = states[ii] # Not being used currently
        current_x_ref = trj_ref[ii]
        current_u = inputs[ii,:].flatten()
        dW = dWs[ii]
        
        trj_difference =  current_x_ref - current_x

        current_cost_xref = trj_difference.T@Qk@trj_difference
        
        current_cost = current_u.T@Rk@current_u # Right now only considering cost in input
        
        noise_cost = np.sqrt(epsilon)*current_u.T@dW
        
        total_cost = total_cost+current_cost+current_cost_xref+noise_cost
    # Compute terminal cost
    terminal_difference = (target_state-states[-1,:]).flatten()
    terminal_cost = terminal_difference.T@QT@terminal_difference
    total_cost = total_cost+terminal_cost
    return total_cost

# Define sampling function
def process_sampling(sample_i, init_state, inputs, start_time, end_time, epsilon, RandN, i):
    # print("Sampling trajectory: ", i)
    sample_i = rollout_bouncing_stochastic(init_state, inputs, start_time, end_time, epsilon, RandN[i])
    return sample_i, i

def process_sampling_feedback(sample_i, init_state, xt_ref, ut, K_feedback, k_feedforward, start_time, end_time, epsilon, RandN, i):
    # print("Sampling trajectory: ", i)
    sample_i, ut_cl_i = rollout_bouncing_stochastic_feedback(init_state, xt_ref, ut, K_feedback, k_feedforward, start_time, end_time, epsilon, RandN[i])
    return sample_i, ut_cl_i, i

# Compute path costs function
@njit(numba.types.Tuple((float64, int32))(
    float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:,:], int32, float64[:,:], float64[:,:], float64[:,:], float64))
def process_compute_costs(sample_i, inputs, dWs, target_state, ref_states, i, Q_k, R_k, Q_T, epsilon):
    # print("Computing costs: ", i)
    costs_i = compute_cost(sample_i, inputs, dWs, target_state, ref_states, Q_k, R_k, Q_T, epsilon)
    return costs_i, i

if __name__ == '__main__':
    # === Do N experiments and compare the expected costs ===
    cost_pi_exp = np.zeros(n_exp)
    cost_ilqr_exp = np.zeros(n_exp)

    # Horizon
    nt_ode_solve = 1000 # number of points used to solve the ode
    
    exp_params = ExpParams(init_state, target_state, start_time, end_time, dt, dt_pathintegral, epsilon, n_samples, Q_k, R_k, Q_T)
    exp_data = ExpData(exp_params)

    for i_exp in prange(n_exp):

        # The reference state trajectory and proposal control
        xt = init_state
        xt_ilqr = init_state
        trj_pi = np.zeros((nt, n_states))
        trj_ilqr = np.zeros((nt, n_states))

        trj_pi[0] = xt
        trj_ilqr[0] = xt_ilqr
        u_star_pi = np.zeros((nt, n_inputs))
        u_trj_ilqr = np.zeros((nt, n_inputs))

        start_time_i = start_time
        end_time_i = end_time
        target_state_i = target_state

        for i in range(nt-1):
            print("time index: ", i)
            
            time_span_i = np.arange(start_time_i, end_time_i, dt).flatten()
            nt_i = nt - i + 1
            
            states_i = states[i:,:]
            inputs_i = inputs[i:,:]
            K_feedback_i = K_feedback[i:,:]
            k_feedforward_i = k_feedforward[i:,:]
            
            # sampling stochastic rollouts
            sampled_trjs = np.zeros((n_samples, nt_i, n_states))
            sampled_controls = np.zeros((n_samples, nt_i, n_inputs))
            PathCosts = np.zeros(n_samples, dtype=np.float64)

            # Generate the randomness
            GaussianNoise = np.random.randn(n_samples, nt_i, n_inputs)
            
            dWs_cost = np.sqrt(dt_pathintegral)*GaussianNoise

            # ------------- ilqr --------------
            samples_index = Parallel(n_jobs=-1)(delayed(process_sampling_feedback)(sampled_trjs[i,:,:], xt, states_i, inputs_i, K_feedback_i, k_feedforward_i, start_time_i, end_time_i, epsilon, GaussianNoise, i) for i in range(n_samples))

            for sample_i, sample_input_i, index in samples_index:
                sampled_trjs[index] = sample_i
                sampled_controls[index] = sample_input_i

            costs_index = Parallel(n_jobs=-1)(delayed(process_compute_costs)(sampled_trjs[i,:,:], sampled_controls[i,:,:], dWs_cost[i,:,:], target_state_i, states_i, i, Q_k, R_k, Q_T, epsilon) for i in range(n_samples))
            for cost_i, index in costs_index:
                PathCosts[index] = cost_i
            
            # update the control proposal using path integral 
            u0_star = update_u0_pathintegral(sampled_controls[0, 0, :], PathCosts, epsilon, dt_pathintegral)
            u_star_pi[i] = u0_star
            
            # go to the next state
            RndN_actual = np.random.randn(n_inputs)
            
            t_span = (start_time_i, start_time_i+dt)
            t_eval = np.linspace(start_time_i, start_time_i+dt, nt_ode_solve)
            t_next = t_eval[-1]
                
            xt = stochastic_integration_bouncing(xt, u0_star, t_span, t_eval, epsilon, RndN_actual, dt, nt_ode_solve)
            trj_pi[i+1] = xt
            
            # ilqr for comparison
            xt_ilqr = stochastic_integration_bouncing(xt_ilqr, sampled_controls[0, 0], t_span, t_eval, epsilon, RndN_actual, dt, nt_ode_solve)
            trj_ilqr[i+1] = xt_ilqr
            u_trj_ilqr[i] = sampled_controls[0, 0]
            
            
        # Compare cost
        dWs_zeros = np.zeros((nt, n_inputs))
        cost_pi = compute_cost(trj_pi, u_star_pi, dWs_zeros, target_state, states, Q_k, R_k, Q_T, epsilon)
        cost_ilqr = compute_cost(trj_ilqr, u_trj_ilqr, dWs_zeros, target_state, states, Q_k, R_k, Q_T, epsilon)
        
        # ---- Record data ----
        data_i = DataOneSample(trj_pi, u_star_pi, trj_ilqr, u_trj_ilqr)
        exp_data.add_data(i_exp, data_i)

        print("cost_pi:", cost_pi)
        print("cost_ilqr:", cost_ilqr)
        
        cost_pi_exp[i_exp] = cost_pi
        cost_ilqr_exp[i_exp] = cost_ilqr
        
    print("E[cost_pi]: ", np.mean(cost_pi_exp))
    print("E[cost_ilqr_exp]: ", np.mean(cost_ilqr_exp))

    # =========== save data ===========
    from datetime import datetime
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"data_{formatted_datetime}_{script_filename}.pickle"
    exp_data.dump(f"{root_dir}/data/bouncing/{filename}")

    # =============== plotting ===============
    fig1, axes = plt.subplots(1, 2)
    (ax1, ax2) = axes.flatten()
    ax1.grid(True)
    ax2.grid(True)

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
    # n_samples_plotted = 100
    # plot_step = n_samples // n_samples_plotted
    # for i_s in range(0, n_samples, plot_step):
    #     ax1.plot(time_span, sampled_trjs[i_s, :, 0], 'b', alpha=0.1)
    # ax1.plot(time_span, sampled_trjs[-1, :, 0], 'b', alpha=0.1, label='Samples')

    # # ----------- plot the path integral controlled trajectory -----------
    for i in range(n_exp):
        trj_ilqr = exp_data.get_data(i).x_trj_ilqr()
        trj_pi = exp_data.get_data(i).x_trj_pi()
        
        ax1.plot(time_span, trj_ilqr[:, 0], 'b', alpha=0.2)
        ax2.plot(time_span, trj_ilqr[:, 1], 'b', alpha=0.2)
        
        ax1.plot(time_span, trj_pi[:, 0], 'r', alpha=0.8)
        ax2.plot(time_span, trj_pi[:, 1], 'r', alpha=0.8)
        
    ax1.plot(time_span, trj_ilqr[:, 0], 'b', alpha=0.2, label='iLQR')
    ax2.plot(time_span, trj_ilqr[:, 1], 'b', alpha=0.2, label='iLQR')

    ax1.plot(time_span, trj_pi[:, 0], 'r', alpha=0.8, label='Path Integral')
    ax2.plot(time_span, trj_pi[:, 1], 'r', alpha=0.8, label='Path Integral')

    # ----------- Plot the start and goal states -----------
    ax1.scatter(time_span[-1], target_state[0], color='g', marker='x', s=50.0, linewidths=6, label='Target')
    ax1.scatter(time_span[0], init_state[0], color='r', marker='x', s=50.0, linewidths=6, label='Start')

    ax2.scatter(time_span[-1], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
    ax2.scatter(time_span[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')

    # ----------- Plot the reference -----------
    ax1.plot(time_span[1:], states[1:-1,0],'k',label='iLQR-deterministic')
    ax2.plot(time_span[1:], states[1:-1,1],'k',label='iLQR-deterministic')

    ax1.set_xlabel(r"Time")
    ax1.set_ylabel(r"$z$")
    ax1.set_title("Bouncing Ball Vertical Position")

    ax2.set_xlabel(r"Time")
    ax2.set_ylabel(r"$\dot z$")
    ax2.set_title("Bouncing Ball Vertical Velocity")

    ax1.legend()
    ax2.legend()


    # # =========== samples for path integral control ===========
    # # ----------- plot the stochastic sampled trajectory -----------
    # n_samples_plotted = 100
    # plot_step = n_samples // n_samples_plotted
    # for i_s in range(0, n_samples, plot_step):
    #     ax3.plot(time_span, sampled_trjs_PI[i_s, :, 0], 'r', alpha=0.1)
    # ax3.plot(time_span, sampled_trjs_PI[-1, :, 0], 'r', alpha=0.1, label='Samples')

    # # ----------- plot the path integral controlled trajectory -----------
    # ax3.plot(time_span, trj_pi[:, 0], 'r', label='Path Integral')

    # # ----------- Plot the last iteration of iLQR controller ----------
    # states = states_iter[-1]
    # ax3.plot(time_span, states[:-1,0],'k',label='iLQR')
    # ax3.set_xlabel(r"Time")
    # ax3.set_ylabel(r"$z$")
    # ax3.set_title("Bouncing Ball Vertical Position")

    # # ----------- Plot the start and goal states -----------
    # ax3.scatter(time_span[-1], target_state[0], color='g', marker='x', s=50.0, linewidths=6, label='Target')
    # ax3.scatter(time_span[0], init_state[0], color='r', marker='x', s=50.0, linewidths=6, label='Start')

    # ax3.legend()

    # # ----------- plot the stochastic sampled trajectory -----------
    # for i_s in range(0, n_samples, plot_step):
    #     ax4.plot(time_span, sampled_trjs_PI[i_s, :, 1], 'b', alpha=0.15)
    # ax4.plot(time_span, sampled_trjs_PI[-1, :, 1], 'b', alpha=0.15, label='Samples')

    # # ----------- plot the path integral controlled trajectory -----------
    # ax4.plot(time_span, trj_pi[:, 1], 'r', label='Path Integral')

    # # ----------- Plot the last iteration of iLQR controller ----------
    # ax4.plot(time_span, states[:-1,1],'k',label='iLQR')

    # # ----------- Plot the start and goal states -----------
    # ax4.scatter(time_span[-1], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
    # ax4.scatter(time_span[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')

    # ax4.set_xlabel(r"Time")
    # ax4.set_ylabel(r"$\dot z$")
    # ax4.set_title("Bouncing Ball Vertical Velocity")

    # ax4.legend()

    # =========== Plot the z-\dot_z figure ===========
    fig2, ax5 = plt.subplots()
    ax5.grid(True)

    # ----------- Plot the last iteration of iLQR controller ----------
    for i in range(n_exp):
        trj_ilqr = exp_data.get_data(i).x_trj_ilqr()
        trj_pi = exp_data.get_data(i).x_trj_pi()
        ax5.plot(trj_ilqr[:, 0], trj_ilqr[:, 1], 'b', alpha=0.2)
        ax5.plot(trj_pi[:, 0], trj_pi[:, 1], 'r', alpha=0.8)

    ax5.plot(trj_ilqr[:, 0], trj_ilqr[:, 1], 'b', alpha=0.2, label='iLQR')
    ax5.plot(trj_pi[:, 0], trj_pi[:, 1], 'r', alpha=0.8, label='Path Integral')
    ax5.plot(states[1:-1,0], states[1:-1,1],'k',label='iLQR-deterministic')

    # ----------- Plot the start and goal states -----------
    ax5.scatter(target_state[0], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
    ax5.scatter(init_state[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')

    ax5.legend()

    # =========== Plot the z-\dot_z figure: PI controller ===========
    # # ----------- plot the stochastic sampled trajectory -----------
    # for i_s in range(0, n_samples, plot_step):
    #     ax6.plot(sampled_trjs_PI[i_s, :, 0], sampled_trjs_PI[i_s, :, 1], 'b', alpha=0.1)
    # ax6.plot(sampled_trjs_PI[-1, :, 0], sampled_trjs_PI[i_s, :, 1], 'b', alpha=0.1, label='Samples')

    # ----------- Plot the last iteration of iLQR controller ----------
    # ax6.plot(states[:-1,0], states[:-1,1],'k',label='iLQR')

    # ----------- plot the path integral controlled trajectory -----------
    # ax6.plot(trj_pi[:, 0], trj_pi[:, 1], 'r', label='Path Integral')

    # for i in range(len(x_trj_pi_exp)):
    #     trj_pi = x_trj_pi_exp[i]
    #     trj_ilqr = x_trj_ilqr_exp[i]
    #     ax5.plot(trj_pi[:, 0], trj_pi[:, 1], 'r', alpha=0.3)
    #     ax5.plot(trj_ilqr[:, 0], trj_ilqr[:, 1], 'b', alpha=0.3)
    # ax5.plot(trj_pi[:, 0], trj_pi[:, 1], 'r', alpha=0.3, label='Path Integral')
    # ax5.plot(trj_ilqr[:, 0], trj_ilqr[:, 1], 'b', alpha=0.3, label='iLQR')
    # ax5.plot(states[:-1,0], states[:-1,1],'k',label='iLQR-deterministic')


    # # ----------- Plot the start and goal states -----------
    # ax6.scatter(target_state[0], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
    # ax6.scatter(init_state[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')

    # ax6.legend()

    # plot control inputs
    fig3, ax7 = plt.subplots(1, 1)
    ax7.grid(True)
    ax7.plot(time_span, inputs[:,0],'k',label='Final iteration ilqr')
    ax7.plot(time_span, u_star_pi[:,0],'r',label='Path integral controller')
    ax7.set_xlabel(r"Timestep")
    ax7.set_ylabel(r"$u$")
    ax7.set_title("Bouncing Ball Final Control Input")

    ax7.legend()

    # plot PathCosts
    fig3, ax8 = plt.subplots()
    ax8.grid(True)
    ax8.bar(range(n_exp), cost_ilqr_exp, width = 2, color='navy', alpha=0.1, label='Cost iLQR')
    ax8.bar(range(n_exp), cost_pi_exp, width = 2, color='red', alpha=0.5, label='Cost PathIntegralControl')

    ax8.set_xlabel(r"Experiment ID")
    ax8.set_ylabel(r"$Costs$")
    ax8.legend()

    plt.show()

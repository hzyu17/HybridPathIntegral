from hybrid_ilqr import *
from hybrid_riccati import *


if __name__ == '__main__':
    # === ilqr parameters ===
    # Initialize timings
    dt = 0.01
    # dt_pathintegral = dt / 50.0
    dt_pathintegral = dt

    # Set desired state
    n_states = 2
    n_inputs = 1
    
    # Define weighting matrices
    Q_k = np.zeros((n_states,n_states)) # zero weight to penalties along a strajectory since we are finding a trajectory
    # R_k = 0.01*np.eye(n_inputs)
    R_k = 0.1*np.eye(n_inputs)

    # Set the terminal cost
    Q_T = 1000*np.eye(n_states)
    # Q_T[0,0] = 200.0
    
    # === path integral parameters ===
    epsilon = 1
    n_samples = 10
    n_exp = 5
    
    # ------------- verification with no contact ------------- 
    start_time = 0
    end_time = 0.75
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)
    
    init_state = np.array([5, 1.5])    # Define the initial state to be the origin with no velocity
    # target_state = np.array([4.0, -5.0])  # Swing pendulum upright
    target_state = np.array([0.0, 0.0])
    
    # ------------- /verification with no contact ------------- 
    # === Do N experiments and compare the expected costs ===
    cost_pi_exp = np.zeros(n_exp)
    cost_ilqr_exp = np.zeros(n_exp)

    # Horizon
    nt_ode_solve = 1000 # number of points used to solve the ode
    
    exp_params_ilqr = ExpParams()
    exp_params_ilqr.update_params(init_state, target_state, start_time, end_time, dt, dt_pathintegral, epsilon, n_exp, n_samples, Q_k, R_k, Q_T, sym_dyn_bouncing,detect_bouncing)
    
    exp_params_riccati = ExpParams()
    exp_params_riccati.update_params(init_state, target_state, start_time, end_time, dt, dt_pathintegral, epsilon, n_exp, n_samples, Q_k, R_k, Q_T, symbolic_dynamics_bouncing_continuoustime,detect_bouncing)
    
    exp_data = ExpData(exp_params_riccati)
    
    # === solve for ilqr ===
    print("================ riccati results ================")
    (states, inputs, K, k, PI, q) = solve_riccati(exp_params_riccati)
    print("================ ilqr results ================")
    (states_ilqr,inputs_ilqr,k_feedforward,K_feedback,current_cost,states_iter) = solve_ilqr(exp_params_ilqr)
    
    print("aa")
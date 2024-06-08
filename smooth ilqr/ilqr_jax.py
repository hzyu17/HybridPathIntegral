import jax
import jax.numpy as jnp
from jax import grad, jit
from scipy.linalg import solve_continuous_are

def pendulum_dynamics(state, u, dt):
    """Dynamics of the pendulum."""
    g = 9.81  # gravity
    m = 1.0   # mass
    l = 1.0   # length
    b = 0.1   # damping coefficient
    
    theta, theta_dot = state
    theta_ddot = (-b * theta_dot - m * g * l * jnp.sin(theta) + u) / (m * l ** 2)
    theta_dot_new = theta_dot + theta_ddot * dt
    theta_new = theta + theta_dot_new * dt
    return jnp.array([theta_new, theta_dot_new])

def cost_function(states, actions):
    """Cost function for the pendulum."""
    Q = jnp.array([[1.0, 0.0], [0.0, 0.1]])  # state cost
    R = jnp.array([[0.01]])                  # control cost
    return jnp.sum(jnp.einsum('ti,ij,tj->t', states, Q, states) + jnp.einsum('ti,ij,tj->t', actions, R, actions))

@jit
def rollout(initial_state, actions, dynamics, dt):
    """Rollout the system with given actions."""
    states = [initial_state]
    for u in actions:
        states.append(dynamics(states[-1], u, dt))
    return jnp.stack(states)

def ilqr_control(initial_state, horizon, Q, R, Qf, dynamics, cost, dt, max_iter=100, eps=1e-5):
    """
    Iterative Linear Quadratic Regulator (iLQR) control.

    Args:
        initial_state (array): Initial state of the system.
        horizon (int): Number of time steps to optimize over.
        Q (array): State cost matrix.
        R (array): Control cost matrix.
        Qf (array): Final state cost matrix.
        dynamics (function): Function representing system dynamics.
        cost (function): Function representing cost.
        dt (float): Time step.
        max_iter (int): Maximum number of iterations.
        eps (float): Convergence threshold.

    Returns:
        tuple: Trajectory (states, actions) and costs.
    """
    state_dim = initial_state.shape[0]
    action_dim = 1  # scalar control input

    # Initialize controls randomly
    actions = jax.random.uniform(jax.random.PRNGKey(0), shape=(horizon, action_dim))

    # Initialize derivatives
    fx = grad(dynamics, 0)
    fu = grad(dynamics, 1)
    lx = grad(cost, 0)
    lu = grad(cost, 1)
    lxx = jax.hessian(cost, 0)
    luu = jax.hessian(cost, 1)
    lux = jax.jacfwd(jax.jacrev(cost, 1), 0)

    Vx = jnp.zeros((horizon, state_dim))
    Vxx = jnp.zeros((horizon, state_dim, state_dim))

    # Iterate
    for _ in range(max_iter):
        # Forward pass
        states = rollout(initial_state, actions, dynamics, dt)

        # Compute derivatives along trajectory
        l = cost(states, actions)
        l_terminal = Qf @ states[-1]  # Final state cost
        V = jnp.concatenate([l, l_terminal])
        
        for t in range(horizon - 1, -1, -1):
            Qx = lx(states[t], actions[t]) + fx(states[t], actions[t], dt).T @ Vx[t]
            Qu = lu(states[t], actions[t]) + fu(states[t], actions[t], dt).T @ Vx[t]
            Qxx = lxx(states[t], actions[t]) + fx(states[t], actions[t], dt).T @ Vxx[t] @ fx(states[t], actions[t], dt)
            Quu = luu(states[t], actions[t]) + fu(states[t], actions[t], dt).T @ Vxx[t] @ fu(states[t], actions[t], dt)
            Qux = lux(states[t], actions[t]) + fu(states[t], actions[t], dt).T @ Vxx[t] @ fx(states[t], actions[t], dt)
            
            Quu_inv = jnp.linalg.inv(Quu)
            k = -Quu_inv @ Qu
            K = -Quu_inv @ Qux

            Vx[t] = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
            Vxx[t] = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K

        # Backward pass
        alpha = 0.1
        beta = 0.5
        actions_new = jnp.zeros_like(actions)
        for t in range(horizon):
            actions_new[t] = actions[t] + alpha * k[t] + K[t] @ (states[t] - initial_state)
            # Check if new action violates constraints
            # Update trajectory
            states = rollout(initial_state, actions_new, dynamics, dt)
            # Compute new cost
            l_new = cost(states, actions_new)
            if l_new < l:
                actions = actions_new
                break
            else:
                alpha *= beta
        else:
            # If no valid trajectory found, break
            break

        # Check for convergence
        if jnp.max(jnp.abs(actions_new - actions)) < eps:
            break

        actions = actions_new

    return actions #rollout(initial_state, actions, dynamics, dt), cost(rollout(initial_state, actions, dynamics, dt), actions)

# Define problem parameters
initial_state = jnp.array([jnp.pi, 0.0])  # Initial angle and angular velocity
horizon = 100  # Horizon length
dt = 0.1  # Time step

# Define cost matrices
Q = jnp.array([[1.0, 0.0], [0.0, 0.1]])  # state cost
R = jnp.array([[0.01]])                  # control cost
Qf = jnp.array([[100.0, 0.0], [0.0, 0.1]])  # final state cost

# Solve the iLQR problem
trajectory, cost = ilqr_control(initial_state, horizon, Q, R, Qf, pendulum_dynamics, cost_function, dt)

print("Final Cost:", cost)

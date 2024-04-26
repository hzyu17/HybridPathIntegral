import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from dynamics.bouncing_guard_reset import *

# numpy and scipy
import scipy
import sympy as sp
from sympy.matrices import Matrix
import numpy as np

# plotting
import matplotlib.pyplot as plt

# jax import
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from functools import partial
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
print("Devices:", jax.devices())


def dyn_bouncing(t, x, *args):
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

def dyn_bouncing_euler(x, u):
    return jnp.array([x[1], u[0]-9.81], dtype=jnp.float64)


def gdWt_bouncing(dWt, eps):
    B = np.array([[0],[1.0]], dtype=np.float64)
    return np.sqrt(eps) * B@dWt
    
    
def symbolic_dynamics_bouncing_continuoustime():
    g = 9.81
    z,z_dot,u,dt = sp.symbols('z z_dot u dt')

    # Define the states and inputs
    inputs = Matrix([u])
    states = Matrix([z, z_dot])
    # Defining the dynamics of the system
    f_contin = Matrix([z_dot, u-g])
    
    A_contin = f_contin.jacobian(states)
    B_contin = f_contin.jacobian(inputs)

    A_contin_func = sp.lambdify((states,inputs),A_contin)
    B_contin_func = sp.lambdify((states,inputs),B_contin)
    
    return (f_contin,A_contin_func,B_contin_func)


def symbolic_dynamics_bouncing():
    g = 9.81
    z,z_dot,u,dt = sp.symbols('z z_dot u dt')

    # Define the states and inputs
    inputs = Matrix([u])
    states = Matrix([z, z_dot])
    # Defining the dynamics of the system
    f = Matrix([z_dot, u-g])

    # Discretize the dynamics usp.sing euler integration
    f_disc = states+f*dt
    
    # Take the jacobian with respect to states and inputs
    A_disc = f_disc.jacobian(states)
    B_disc = f_disc.jacobian(inputs)

    f_disc_func = sp.lambdify((states,inputs,dt),f_disc)
    A_disc_func = sp.lambdify((states,inputs,dt),A_disc)
    B_disc_func = sp.lambdify((states,inputs,dt),B_disc)
    return (f_disc_func,A_disc_func,B_disc_func)


def bouncing_event_condition_jax(xt, xt_next, guard):
    # assume time invariant guard for now
    return jnp.logical_and(guard(0.0,xt)>0, guard(0.0,xt_next)<=0) 

def bouncing_cond_true_fun_jax(args):
    (xt_current, current_mode, u, t, t_next, xt_next, next_mode, dt_int, dt_shrinkingrate, RandN, epsilon) = args
    
    def while_loop_cond(vars):
        (xt_current, xt_swch, u, t, dt_int, dt_shrinkingrate, RandN, epsilon, cnt, can_continue) = vars
        return can_continue
    
    def while_loop_body(vars):
        (xt_current, xt_swch, u, t, dt_int, dt_shrinkingrate, RandN, epsilon, cnt, can_continue) = vars
        cnt += 1
        
        # Too far from the guard, shrink the step size.
        dt_int = dt_int * dt_shrinkingrate
        dW_new = jnp.sqrt(dt_int)*RandN
        
        xt_swch = stochastic_integration_euler(xt_current, u, dt_int, epsilon, dW_new)
        
        new_condition = jnp.logical_not(jnp.logical_or(guard_bouncing_12(t, xt_swch)>0, cnt==10))
        new_vars = (xt_current, xt_swch, u, t, dt_int, dt_shrinkingrate, RandN, epsilon, cnt, new_condition)
        
        return new_vars
    
    init_condition = True
    init_vars = (xt_current, xt_next, u, t, dt_int, dt_shrinkingrate, RandN, epsilon, 0, init_condition)
    final_vars = jax.lax.while_loop(while_loop_cond, while_loop_body, init_val=init_vars)
    
    (xt_current, xt_swch, u, t, dt_int, dt_shrinkingrate, RandN, epsilon, cnt, can_continue) = final_vars
    xt_next, next_mode = reset_map_bouncing_12_jax(t, xt_swch, current_mode)
    dW_new = jnp.sqrt(dt_int)*RandN
    
    return xt_next, next_mode, dW_new
            

def bouncing_cond_true_fun(args):
    (xt_current, current_mode, u, t, t_next, xt_next, next_mode, dt_int, dt_shrinkingrate, RandN, epsilon) = args
    xt_swch = xt_next
        
    # Sandwich rule to find finer grind 
    cnt = 0
    while (True):
        cnt += 1
        # xt_last = xt_swch
        
        # Too far from the guard, shrink the step size.
        dt_int = dt_int * dt_shrinkingrate
        dW_new = jnp.sqrt(dt_int)*RandN
        
        # # ---- solver for the deterministic part
        # t_span = (t, t_next)
        # t_eval = np.linspace(t, t_next, nt)
        
        xt_swch = stochastic_integration_euler(xt_current, u, dt_int, epsilon, dW_new)

        # /---- solver for the deterministic part
        if (guard_bouncing_12(t, xt_swch)>0) or (cnt==10): # Until the guard condition is no longer met.
            # The reset map is called
            xt_next, next_mode = reset_map_bouncing_12(t, xt_swch, current_mode)
            dW = dW_new
            # dt_int = dt
            break
        
    return xt_next, next_mode, dW


def bouncing_cond_false_fun(args):
    (xt_current, current_mode, u, t, t_next, xt_next, next_mode, dt_int, dt_shrinkingrate, RandN, epsilon) = args
    dW = jnp.sqrt(dt_int)*RandN
    return xt_next, next_mode, dW
    

def stochastic_integration_euler(x0, u, dt, epsilon, dW):
    B = jnp.array([[0],[1.0]], dtype=jnp.float64)
    xt_next = x0 + jnp.array([x0[1], u[0]-9.81], dtype=jnp.float64) * dt + jnp.sqrt(epsilon) * B@dW
    # xt_next = x0 + dyn_bouncing_euler(x0, u)*dt + gdWt_bouncing(dW, epsilon)
    return xt_next


def roullout_bouncing_stochastic_feedback_jax(n_samples, x0, cur_mode_change, xt_ref, ref_modechanges, 
                                                ut_ref, K_feedback, k_feedforward, target_state, R_k, Q_T, t0, dt, tf, dt_shrinkingrate, 
                                                epsilon, GaussianNoise, cond_guard, guard_fun, reset_map_fun):
    
    # ============================================= jax function definitions ============================================= 
    from functools import partial

    def feedback_cost_jax(carry, inputs, epsilon, dt):
        xt, current_mode_change, St = carry
        ut_ref, K, k, randN, xt_ref = inputs
        
        # -------- compute control input --------
        delta_xt = xt - xt_ref
        ut = ut_ref + jnp.dot(K, delta_xt) + k    
        dW = jnp.sqrt(dt)*randN

        # -------- propagate dynamics --------
        xt_next = stochastic_integration_euler(xt, ut, dt, epsilon, dW)
        
        # -------- change mode --------
        current_mode, next_mode = current_mode_change[0], current_mode_change[1]
        args = (xt, current_mode, ut, t0, tf, xt_next, next_mode, dt, dt_shrinkingrate, randN, epsilon)
        guard_hit = cond_guard(xt, xt_next, guard_fun)
        xt, next_mode, dW = jax.lax.cond(guard_hit, bouncing_cond_true_fun_jax, bouncing_cond_false_fun, args)

        # Collect cost: consider only the terminal state cost for now.
        St += jnp.array([jnp.dot(ut.T, ut)/2.0 * dt + jnp.sqrt(epsilon) * jnp.dot(ut.T, dW)])
        
        next_mode_change = jnp.array([current_mode, next_mode])
        
        return (xt, next_mode_change, St), (xt, next_mode_change, St)

    
    # # terminal cost function in jax
    # def terminal_cost_jax(xT, cost_xT, Qf, rf, constant_tf):
    #     return jnp.array(cost_xT(xT, Qf, rf, constant_tf))

    # terminal_cost_xQrx_jax = partial(terminal_cost_jax, cost_xT=xQrx, Qf=Qf, rf=rf, constant_tf=constant_tf)
    # parallel_terminal_cost_xQrx = jax.vmap(terminal_cost_xQrx_jax, in_axes=0)

    fb_cost_scanfunc = partial(feedback_cost_jax, epsilon=epsilon, dt=dt)

    def feedbackcost_jax_updaterow(x_mdchg_S_row, u_randN_row):
        initial_carry = (x_mdchg_S_row[0], x_mdchg_S_row[1], x_mdchg_S_row[2])
        _, updated_row = jax.lax.scan(fb_cost_scanfunc, initial_carry, u_randN_row)
        return updated_row

    parallel_update_sde_randN = jax.vmap(feedbackcost_jax_updaterow, in_axes=(0,0))

    # ============================================= / jax function definitions ============================================= 
    
    # ========== jax parallel sampling =========
    xt_jax = jnp.asarray(x0)
    v_x0 = jnp.tile(xt_jax, (n_samples, 1))
    v_cur_mode_change = jnp.tile(cur_mode_change, (n_samples, 1))
    v_xt_ref = jnp.tile(xt_ref, (n_samples, 1, 1))
    v_ut_ref = jnp.tile(ut_ref, (n_samples, 1, 1))
    v_S0 = jnp.zeros((n_samples, 1), dtype=jnp.float64)
    K_feedback_i_jax = jnp.asarray(K_feedback)
    k_feedforward_i_jax = jnp.asarray(k_feedforward)
    
    v_Ks = jnp.tile(K_feedback_i_jax, (n_samples, 1, 1, 1))
    v_ds = jnp.tile(k_feedforward_i_jax, (n_samples, 1, 1))
    
    v_randN = jnp.asarray(GaussianNoise)
    
    v_initial_carry = (v_x0, v_cur_mode_change, v_S0)
    v_inputs = (v_ut_ref, v_Ks, v_ds, v_randN, v_xt_ref)
    v_xt_St_randN = parallel_update_sde_randN(v_initial_carry, v_inputs)
    
    Ksamples_jax, mode_chage_jax, PathCosts_jax = v_xt_St_randN
    
    # Move the samples forward by 1 place and add xt to the front, to keep the same with numpy results.
    Ksamples_jax = jnp.concatenate((v_x0.reshape((n_samples, 1, -1)), Ksamples_jax[:,0:-1,:]), axis=1)
    PathCosts_jax = PathCosts_jax[:,-2,1]
    
    return Ksamples_jax, PathCosts_jax
    
    # xT_samples = Ksamples_jax[:,-1,:]
    # v_S_xT = parallel_terminal_cost_xQrx(xT_samples)
    # PathCosts_jax = PathCosts_jax + v_S_xT
    
    # allPathCosts[i] = np.array(PathCosts_jax)
        
    # ============================================== / jax parallel sampling ====================================
    

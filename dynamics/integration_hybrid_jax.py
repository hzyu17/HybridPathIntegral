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

# -------- mode mismatch condition handling -------- 
def cond_mode_mismatch_jax(next_mode, ref_next_mode, mode_exttrjs_maps):
    next_mode_diff = (next_mode == ref_next_mode)
    return jnp.logical_and(jnp.logical_not(next_mode_diff), len(mode_exttrjs_maps) > 0)

def mode_mismatch_true_fun_jax(args):
    (next_mode, ref_next_mode, ref_modechanges, selected_extended_trj, xref_i, cnt_mismatch, cnt_indx) = args
    # Take the first hybrid event for now. Needs to find the correct corresponding one among all hybrid events.
    
    # Condition: if it is the first time early arrival: find and reverse the extended reference trajectory
    is_early = cond_early_arrival_jax(next_mode, ref_next_mode, cnt_mismatch)
    
    # jax.debug.print("Condition is_early: {}", is_early)
    args = (next_mode, ref_next_mode, ref_modechanges, selected_extended_trj, cnt_indx)
    exttrj_starting, extended_trj = jax.lax.cond(is_early, early_arrival_true_fun_jax, early_arrival_false_fun_jax, args)
    
    xref_modified_i = extended_trj[exttrj_starting+cnt_mismatch[0]]
    # jax.debug.print("exttrj_starting: {}", exttrj_starting)
    # jax.debug.print("extended_trj: {}", extended_trj)
    # jax.debug.print("xref_i: {}", xref_modified_i)
    
    cnt_mismatch += 1
    
    return xref_modified_i, cnt_mismatch

def mode_mismatch_false_fun_jax(args):
    (next_mode, ref_next_mode, ref_modechanges, selected_extended_trj, xref_i, cnt_mismatch, cnt_indx) = args
    return xref_i, cnt_mismatch

# -------- early arrival condition handling -------- 
def cond_early_arrival_jax(next_mode, ref_next_mode, cnt_mismatch):
    # res = jnp.logical_and(jnp.logical_and(next_mode==2, ref_next_mode==1), cnt_mismatch[0]==0)
    if_mismatched = jnp.logical_and(next_mode==2, ref_next_mode==1)
    if_firstarrival = (cnt_mismatch[0]==0)
    return jnp.logical_and(if_mismatched, if_firstarrival)

def early_arrival_true_fun_jax(args):
    (next_mode, ref_next_mode, ref_modechanges, extended_trajectory, starting_index) = args
    def while_loop_cond(vars):
        (next_mode, ref_modechanges, extended_trajectory, starting_index, cnt_ext, can_continue) = vars
        return can_continue

    def while_loop_body(vars):
        (next_mode, ref_modechanges, extended_trajectory, starting_index, cnt_ext, can_continue) = vars
        
        # ------------- first time visit: revert the extended trajectory ------------- 
        def first_visit_true(ext_trj):
            # jax.debug.print("debug extended_trj {}", ext_trj)
            reverted_ext_trj = ext_trj[::-1]
            # jax.debug.print("debug reverted_ext_trj {}", reverted_ext_trj)
            # jax.debug.print("debug reverted_ext_trj")
            return reverted_ext_trj
        def first_visit_false(ext_trj):
            return ext_trj
        
        extended_trj = jax.lax.cond(cnt_ext==0, first_visit_true, first_visit_false, extended_trajectory)
        # jax.debug.print("debug extended_trj {}", extended_trj)
        # ------------- / first time visit ------------- 
        
        cnt_ext += 1
        cur_index = starting_index+cnt_ext
        cur_ref_next_mode = ref_modechanges[cur_index][1]
        
        # loop until the reference next mode is the same as the next mode of the true state
        new_condition = jnp.logical_not(cur_ref_next_mode == next_mode)
        
        new_vars = (next_mode, ref_modechanges, extended_trj, starting_index, cnt_ext, new_condition)
        
        return new_vars
    
    # Have to re-define the condition for entering the while loop. Under the vmap, JAX will execute 
    # both branches of the condition for each element in the batch at compile time, 
    # but only the relevant branch's result will be computed at runtime for each element based on its condition.
    init_condition = jnp.logical_and(next_mode==2, ref_next_mode==1)
    init_vars = (next_mode, ref_modechanges, extended_trajectory, starting_index, 0, init_condition)
    final_vars = jax.lax.while_loop(while_loop_cond, while_loop_body, init_val=init_vars)
    
    (next_mode, ref_modechanges, extended_trajectory, starting_index, len_ref, _) = final_vars
    
    num_rows = extended_trajectory.shape[0] 
    return num_rows-len_ref, extended_trajectory
    # return 0, extended_trj

def early_arrival_false_fun_jax(args):
    (next_mode, ref_next_mode, ref_modechanges, extended_trajectory, starting_index) = args
    return 0, extended_trajectory

# # -------- bouncing condition handling -------- 
# def cond_firsttime_mismatch_jax(mismatched_states):
#     return mismatched_states is None

# # -------- bouncing condition handling -------- 
# def cond_empty_modified_refs_jax(modified_refs):
#     return modified_refs is None

# -------- bouncing condition handling -------- 
def bouncing_event_condition_jax(xt, xt_next, guard):
    # assume time invariant guard for now
    return jnp.logical_and(guard(0.0,xt)>0, guard(0.0,xt_next)<=0) 

def bouncing_cond_true_fun_jax(args):
    print("bouncing_cond: True")
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
        
        # Too far from the guard, shrink the step size.
        dt_int = dt_int * dt_shrinkingrate
        dW_new = jnp.sqrt(dt_int)*RandN
        
        xt_swch = stochastic_integration_euler(xt_current, u, dt_int, epsilon, dW_new)

        # /---- solver for the deterministic part
        if (guard_bouncing_12(t, xt_swch)>0) or (cnt==10): # Until the guard condition is no longer met.
            # The reset map is called
            xt_next, next_mode = reset_map_bouncing_12(t, xt_swch, current_mode)
            dW = dW_new
            break
        
    return xt_next, next_mode, dW


def bouncing_cond_false_fun(args):
    print("bouncing_cond: False")
    (xt_current, current_mode, u, t, t_next, xt_next, next_mode, dt_int, dt_shrinkingrate, RandN, epsilon) = args
    dW = jnp.sqrt(dt_int)*RandN
    return xt_next, next_mode, dW
    

def stochastic_integration_euler(x0, u, dt, epsilon, dW):
    B = jnp.array([[0],[1.0]], dtype=jnp.float64)
    xt_next = x0 + jnp.array([x0[1], u[0]-9.81], dtype=jnp.float64) * dt + jnp.sqrt(epsilon) * B@dW
    # xt_next = x0 + dyn_bouncing_euler(x0, u)*dt + gdWt_bouncing(dW, epsilon)
    return xt_next


# ============================================= jax function definitions ============================================= 
from functools import partial

# terminal cost function in jax
def terminal_cost_jax(xT, cost_xT, args):
    return jnp.array(cost_xT(xT, args))

def quadratic_terminal_cost(xT, args):
    target_state, QT =  args
    return (xT-target_state)@QT@(xT-target_state) / 2.0


# ============================================= / jax function definitions ============================================= 

def roullout_bouncing_stochastic_feedback_jax(n_samples, x0, cur_mode_change, xt_ref, 
                                              cur_ref_modechange, ref_modechanges, 
                                              ut_ref, K_feedback, k_feedforward, target_state, R_k, Q_T, 
                                              t0, dt, tf, dt_shrinkingrate, 
                                              epsilon, GaussianNoise, guard_fun, mode_exttrjs_maps):
    
    def feedback_cost_jax(carry, inputs, epsilon, dt, mode_exttrjs_maps, ref_modechanges):
        cur_xt, cur_mode_change, cur_St, cur_cnt_mismatch, cur_cnt_indx = carry
        cur_ut_ref, cur_K, cur_k, cur_randN, cur_xt_ref, cur_ref_modechange = inputs
            
        current_mode, next_mode = cur_mode_change[0], cur_mode_change[1]
        next_mode_ref = cur_ref_modechange[1]
        
        cond_mode_mismatch = cond_mode_mismatch_jax(next_mode, next_mode_ref, mode_exttrjs_maps)
            
        # -------- mode mismatch --------        
        mode_keys = sorted(mode_exttrjs_maps[0][0])
        map_trjs = mode_exttrjs_maps[0][1]
        len_trj1 = map_trjs[mode_keys[0]].shape[0]
        len_trj2 = map_trjs[mode_keys[1]].shape[0]
        starts = jnp.array([0, len_trj1])
        ext_trjs_stack = jnp.vstack((map_trjs[mode_keys[0]], map_trjs[mode_keys[1]]))
        starting_index = starts[jnp.floor_divide(next_mode, len(mode_keys))]
        
        num_rows = max(len_trj1, len_trj2)
        num_columns = ext_trjs_stack.shape[1]
        slice_sizes = (num_rows, num_columns) 

        selected_extended_trj = jax.lax.dynamic_slice(ext_trjs_stack, (starting_index, 0), slice_sizes)
        
        args_mm = (next_mode, next_mode_ref, ref_modechanges, selected_extended_trj, cur_xt_ref, cur_cnt_mismatch, cur_cnt_indx[0])
        cur_xt_ref, cur_cnt_mismatch = jax.lax.cond(cond_mode_mismatch, mode_mismatch_true_fun_jax, mode_mismatch_false_fun_jax, args_mm)    
        
        # -------- compute control input --------
        delta_xt = cur_xt - cur_xt_ref
        ut = cur_ut_ref + jnp.dot(cur_K, delta_xt) + cur_k    
        dW = jnp.sqrt(dt)*cur_randN

        # -------- propagate dynamics --------
        xt_next = stochastic_integration_euler(cur_xt, ut, dt, epsilon, dW)
        
        # -------- change mode --------
        args = (cur_xt, current_mode, ut, t0, tf, xt_next, next_mode, dt, dt_shrinkingrate, cur_randN, epsilon)
        guard_hit = bouncing_event_condition_jax(cur_xt, xt_next, guard_fun)
        cur_xt, next_mode, dW = jax.lax.cond(guard_hit, bouncing_cond_true_fun_jax, bouncing_cond_false_fun, args)

        # Collect cost: consider only the terminal state cost for now.
        cur_St += jnp.array([jnp.dot(ut.T, ut)/2.0 * dt + jnp.sqrt(epsilon) * jnp.dot(ut.T, dW)])
        cur_mode_change = jnp.array([current_mode, next_mode])
        cur_cnt_indx = cur_cnt_indx + jnp.array([1])
        
        return (cur_xt, cur_mode_change, cur_St, cur_cnt_mismatch, cur_cnt_indx), (cur_xt, cur_St, cur_xt_ref)  
    
    fb_cost_scanfunc = partial(feedback_cost_jax, epsilon=epsilon, dt=dt, mode_exttrjs_maps=mode_exttrjs_maps, ref_modechanges=ref_modechanges)
    
    def feedbackcost_jax_updaterow(x_mdchg_S_row, u_randN_row):
        initial_carry = (x_mdchg_S_row[0], x_mdchg_S_row[1], x_mdchg_S_row[2], x_mdchg_S_row[3], x_mdchg_S_row[4])
        _, updated_row = jax.lax.scan(fb_cost_scanfunc, initial_carry, u_randN_row)
        return updated_row

    parallel_update_sde_randN = jax.vmap(feedbackcost_jax_updaterow, in_axes=(0,0))
    
    args_terminal_cost = (target_state, Q_T)
    terminal_cost_xQrx_jax = partial(terminal_cost_jax, cost_xT=quadratic_terminal_cost, args=args_terminal_cost)
    parallel_terminal_cost_xQrx = jax.vmap(terminal_cost_xQrx_jax, in_axes=0)
    
    # ========== jax parallel sampling =========
    xt_jax = jnp.asarray(x0)
    
    # vectorized carry
    v_x0 = jnp.tile(xt_jax, (n_samples, 1))
    v_cur_mode_change = jnp.tile(cur_mode_change, (n_samples, 1))
    v_S0 = jnp.zeros((n_samples, 1), dtype=jnp.float64)
    v_cnt_mismatch = jnp.zeros((n_samples, 1), dtype=jnp.int64)
    v_current_index = jnp.tile(0, (n_samples, 1))
    
    # vectorized inputs
    v_xt_ref = jnp.tile(xt_ref, (n_samples, 1, 1))
    v_ut_ref = jnp.tile(ut_ref, (n_samples, 1, 1))
    K_feedback_i_jax = jnp.asarray(K_feedback)
    k_feedforward_i_jax = jnp.asarray(k_feedforward)
    v_Ks = jnp.tile(K_feedback_i_jax, (n_samples, 1, 1, 1))
    v_ds = jnp.tile(k_feedforward_i_jax, (n_samples, 1, 1))
    v_randN = jnp.asarray(GaussianNoise)
    v_ref_modechanges = jnp.tile(ref_modechanges, (n_samples, 1, 1))
    
    v_initial_carry = (v_x0, v_cur_mode_change, v_S0, v_cnt_mismatch, v_current_index)
    v_inputs = (v_ut_ref, v_Ks, v_ds, v_randN, v_xt_ref, v_ref_modechanges)
    v_xt_St_randN = parallel_update_sde_randN(v_initial_carry, v_inputs)
    
    Ksamples_jax, PathCosts_jax, actual_ref_jax = v_xt_St_randN
    
    # Move the samples forward by 1 place and add xt to the front, to keep the same with numpy results.
    Ksamples_jax = jnp.concatenate((v_x0.reshape((n_samples, 1, -1)), Ksamples_jax[:,0:-1,:]), axis=1)
    PathCosts_jax = PathCosts_jax[:,-2,1]
    
    # Terminal cost
    xT_samples = Ksamples_jax[:,-1,:]
    v_S_xT = parallel_terminal_cost_xQrx(xT_samples)
    PathCosts_jax = PathCosts_jax + v_S_xT
    
    return Ksamples_jax, PathCosts_jax, actual_ref_jax
        
    # ============================================== / jax parallel sampling ====================================
    

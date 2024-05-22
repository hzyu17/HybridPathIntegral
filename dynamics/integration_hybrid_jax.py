import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from dynamics.bouncing_guard_reset import *

# numpy
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
print("jax.default_backend()", jax.default_backend())

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
    
def stochastic_integration_euler(x0, u, dt, eps, dW):
    B = jnp.array([[0],[1.0]], dtype=jnp.float64)
    xt_next = x0 + jnp.array([x0[1], u[0]-9.81], dtype=jnp.float64) * dt + jnp.sqrt(eps) * B@dW
    return xt_next

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

# ------------------------------------- terminal loss ------------------------------------- 
from functools import partial

def quadratic_terminal_cost(xT, args):
    x_tar, QT =  args
    return (xT-x_tar)@QT@(xT-x_tar) / 2.0
quadratic_terminal_cost_jit = jax.jit(quadratic_terminal_cost)

# terminal cost function in jax
def terminal_cost_jax(xT, args):
    return jnp.array(quadratic_terminal_cost(xT, args))
terminal_cost_jit = jax.jit(terminal_cost_jax)

# ------------------------------------- / terminal loss ------------------------------------- 


# ===============================================================================================================
#                                       Mode mismatch condition handling
# ===============================================================================================================
"""
MM: mode mismatch
EArr: early arrival
"""
# -----------------------------------
# Mode Mismatch condition handling
# -----------------------------------

def MM_true_fun_jax(args):
    
    (next_mode, ref_next_mode, ext_trj_fwd, ext_trj_bwd, xref_i, cnt_indx, cnt_MM)= args
    # TODO: Take the first hybrid event for now. 
    # Need to find the correct corresponding one among all hybrid events.
    
    # # ------------------------------------------------------------ 
    # #                        EArr handling
    # # ------------------------------------------------------------
    
    is_early = cond_EArr_jax(next_mode, ref_next_mode)
    args_EArr = (ext_trj_fwd, ext_trj_bwd)
    extended_trj = jax.lax.cond(is_early, EArr_true_fun_jax, EArr_false_fun_jax, args_EArr)
    
    # # ------------------------------------------------------------ 
    # #                     // End of EArr handling
    # # ------------------------------------------------------------
    
    xref_i = extended_trj[cnt_indx].flatten()
    cnt_MM += 1
    
    return xref_i, cnt_MM

def MM_false_fun_jax(args):
    # (next_mode, ref_next_mode, ref_MC_seq, extended_trj, xref_i, cnt_MM, cnt_indx) = args
    (next_mode, ref_next_mode, ext_trj_fwd, ext_trj_bwd, xref_i, cnt_indx, cnt_MM) = args
    
    return xref_i, cnt_MM

# ================================================================================================================
#                               Early arrival condition handling function definitions
# ================================================================================================================
# ----------------------------------------------------
# first time visit: revert the extended trajectory
# ---------------------------------------------------- 
def first_visit_true(ext_trj):
    return ext_trj[::-1]

def first_visit_false(ext_trj):
    return ext_trj    
    
def cond_EArr_jax(next_mode, ref_next_mode):
    # hand picked, need to be automatic
    is_EArr = jnp.logical_and(next_mode==2, ref_next_mode==1)
    # is_firstarrival = (cnt_MM[0]==0)
    # return jnp.logical_and(is_EArr, is_firstarrival)
    return is_EArr

def EArr_true_fun_jax(args):
    (ext_trj_fwd, ext_trj_bwd) = args
    return ext_trj_bwd

def EArr_false_fun_jax(args):
    (ext_trj_fwd, ext_trj_bwd) = args
    return ext_trj_fwd

# ============================ // End of Mode Mismatch condition handling //=======================================


# ===============================================================================================================
#                                        Bouncing condition handling 
# ===============================================================================================================
def bouncing_event_condition_jax(xt, xt_next):
    # assume time invariant guard for now
    return jnp.logical_and(guard_bouncing_12_jit(0.0,xt)>0, guard_bouncing_12_jit(0.0,xt_next)<=0) 
bouncing_event_condition_jit = jax.jit(bouncing_event_condition_jax)

def bouncing_cond_true_fun_jax(args):
    print("bouncing_cond: True")
    (xt_current, current_mode, u, t, t_next, xt_next, next_mode, dt_int, dt_shr, RandN, eps) = args
    
    def while_cond(vars):
        (xt_current, xt_swch, u, t, dt_int, dt_shr, RandN, eps, cnt, can_continue) = vars
        return can_continue
    
    def while_loop_body(vars):
        (xt_current, xt_swch, u, t, dt_int, dt_shr, RandN, eps, cnt, can_continue) = vars
        cnt += 1
        
        # Too far from the guard, shrink the step size.
        dt_int = dt_int * dt_shr
        dW_new = jnp.sqrt(dt_int)*RandN
        
        xt_swch = stochastic_integration_euler(xt_current, u, dt_int, eps, dW_new)
        
        new_condition = jnp.logical_not(jnp.logical_or(guard_bouncing_12(t, xt_swch)>0, cnt==10))
        new_vars = (xt_current, xt_swch, u, t, dt_int, dt_shr, RandN, eps, cnt, new_condition)
        
        return new_vars
    
    init_condition = True
    init_vars = (xt_current, xt_next, u, t, dt_int, dt_shr, RandN, eps, 0, init_condition)
    final_vars = jax.lax.while_loop(while_cond, while_loop_body, init_val=init_vars)
    
    (xt_current, xt_swch, u, t, dt_int, dt_shr, RandN, eps, cnt, can_continue) = final_vars
    xt_next, next_mode = reset_map_bouncing_12_jax(t, xt_swch, current_mode)
    dW_new = jnp.sqrt(dt_int)*RandN
    
    return xt_next, next_mode, dW_new

bouncing_true_fun_jit = jax.jit(bouncing_cond_true_fun_jax)

def bouncing_cond_false_fun(args):
    print("bouncing_cond: False")
    (xt_current, current_mode, u, t, t_next, xt_next, next_mode, dt_int, dt_shr, RandN, eps) = args
    dW = jnp.sqrt(dt_int)*RandN
    return xt_next, next_mode, dW
bouncing_false_fun_jit = jax.jit(bouncing_cond_false_fun)

# ===============================================================================================================
#                                       // End of Bouncing condition handling //
# ===============================================================================================================


"""
One step cost
"""
def cost_i(xt, current_mode, next_mode, cur_St, xref, uref, K, k, randN, eps, dt, dt_shrink, t0, tf):
    # -------- compute control input --------
    delta_xt = xt - xref
    ut = uref + jnp.dot(K, delta_xt) + k    
    dW = jnp.sqrt(dt)*randN
    
    # -------- propagate dynamics --------
    # stochastic_integration_euler_jit = jax.jit(stochastic_integration_euler)
    xt_next = stochastic_integration_euler(xt, ut, dt, eps, dW)
    
    # ----------------
    # change mode
    # ----------------
    args_guard = (xt, current_mode, ut, t0, tf, xt_next, next_mode, dt, dt_shrink, randN, eps)
    guard_hit = bouncing_event_condition_jit(xt, xt_next)
    xt, next_mode, dW = jax.lax.cond(guard_hit, bouncing_true_fun_jit, bouncing_false_fun_jit, args_guard)

    # Collect cost: consider only the terminal state cost for now.
    cur_St += jnp.array([jnp.dot(ut.T, ut)/2.0 * dt + jnp.sqrt(eps) * jnp.dot(ut.T, dW)])
    cur_MC = jnp.array([current_mode, next_mode])
    
    return xt, cur_MC, cur_St

# collect_cost_jitted = jax.jit(collect_cost)

def feedback_cost_jax(carry, inputs, eps, dt, dt_shrink, t0, tf, v_mode_change, v_ext_trj_fwd, v_ext_trj_bwd):
    xt, MC, St, cnt_MM, indx = carry
    uref, K, k, randN, xref, ref_MC = inputs
    
    # the current and next mode of the stochastic state
    current_mode, next_mode = MC[0], MC[1]
    
    # the next mode of the reference state
    next_mode_ref = ref_MC[1]
    
    # -------------------------------
    # Get the trajectory extensions 
    # -------------------------------
    # put to 0 first, need to consider more than one bouncing in the future.
    cnt_event = 0
    ext_trj_fwd = v_ext_trj_fwd[0]
    ext_trj_bwd = v_ext_trj_bwd[0]
    # mode_change_i, mode_exttrjs_i = mode_exttrjs_maps[cnt_event]
    # extended_trj = mode_exttrjs_i[next_mode]
    
    # selected_exttrj = select_MM_ref(mode_exttrjs_maps, next_mode)
    
    is_MM = (next_mode != next_mode_ref)    
    args_MM = (next_mode, next_mode_ref, ext_trj_fwd, ext_trj_bwd, xref, indx, cnt_MM)
    xref, cnt_MM = jax.lax.cond(is_MM, MM_true_fun_jax, MM_false_fun_jax, args_MM)
   
    ## testing comments
    xt, MC, St = cost_i(xt, current_mode, next_mode, St, xref, uref, 
                        K, k, randN, eps, dt, dt_shrink, t0, tf)
    indx = indx + 1
    
    return (xt, MC, St, cnt_MM, indx), (xt, St, xref) 

    
"""
Synonyms:
MC: mode change
MM: mismatch
shr: shrinking ratio
fb: feedback
ff: feedforward
x_tar: target state
eps: epsilon
"""
def roullout_bouncing_jax(n_samples, x0, cur_MC, 
                            xref_trj, ref_MC_seq, uref_trj, 
                            K_fb, k_ff, 
                            x_tar, Q_T, 
                            t0, dt, tf, dt_shr, 
                            eps, noise, v_mode_change, v_ext_trj_fwd, v_ext_trj_bwd):
    
    # -----------------------------
    # move the variables onto GPU
    # ----------------------------- 
    xref_trj = jnp.asarray(xref_trj)
    uref_trj = jnp.asarray(uref_trj)
    ref_MC_seq = jnp.asarray(ref_MC_seq)
    K_fb = jnp.asarray(K_fb)
    k_ff = jnp.asarray(k_ff)
    noise = jnp.asarray(noise)
            
    # ================================
    # Define scan and vmap functions.
    # ================================ 
    feedback_cost_scan_fun = partial(feedback_cost_jax, eps=eps, dt=dt, 
                                     dt_shrink=dt_shr, t0=t0, tf=tf, 
                                     v_mode_change=v_mode_change, 
                                     v_ext_trj_fwd=v_ext_trj_fwd, 
                                     v_ext_trj_bwd=v_ext_trj_bwd)
    
    def feedbackcost_onerow(carrys, inputs):
        initial_carry = (carrys[0], carrys[1], carrys[2], carrys[3], carrys[4])
        _, updated_row = jax.lax.scan(feedback_cost_scan_fun, initial_carry, inputs)
        return updated_row
    
    # feedbackcost_updaterow_jit = jax.jit(feedbackcost_onerow)
    feedbackcost_vmap = jax.vmap(feedbackcost_onerow, in_axes=(0,0))
    
    args_terminal_cost = (x_tar, Q_T)
    terminal_cost_xQrx_vmap = jax.vmap(partial(quadratic_terminal_cost_jit, args=args_terminal_cost), in_axes=0)
    
    # ============================
    # start jax sampling process
    # ===========================
    xt_jax = jnp.asarray(x0)
    
    # -----------------------------------
    # vectorizing carrys and inputs
    # -----------------------------------
    
    # ----- carrys: (x0, current mode change, mismatch counter, timestep index, Path Cost) ----- 
    v_x0 = jnp.tile(xt_jax, (n_samples, 1))
    v_cur_MC = jnp.tile(cur_MC, (n_samples, 1))
    v_cnt_MM = jnp.zeros((n_samples, 1), dtype=jnp.int64)
    v_index = jnp.tile(0, (n_samples, 1))
    v_St = jnp.zeros((n_samples, 1), dtype=jnp.float64)
    # --------------- // carrys // --------------- 
    
    # --------------- inputs --------------------
    # (reference_trj_x, reference_trj_u, feedback gain, feed forward gain, 
    # Gaussian randomness, reference mode change sequence) ----- 
    # --------------- / inputs --------------------
    v_xref = jnp.tile(xref_trj, (n_samples, 1, 1))
    v_uref = jnp.tile(uref_trj, (n_samples, 1, 1))

    v_Ks = jnp.tile(K_fb, (n_samples, 1, 1, 1))
    v_ds = jnp.tile(k_ff, (n_samples, 1, 1))
    
    v_randN = jnp.asarray(noise)
    
    v_ref_MC = jnp.tile(ref_MC_seq, (n_samples, 1, 1))
    
    v_initial_carry = (v_x0, v_cur_MC, v_St, v_cnt_MM, v_index)
    v_inputs = (v_uref, v_Ks, v_ds, v_randN, v_xref, v_ref_MC)
    
    # -------------------- // inputs // ------------------------- 
    
    # =================
    # Sampling process
    # =================
    # ---------------------- with profiler -----------------------
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    #     v_sample_results = feedbackcost_vmap(v_initial_carry, v_inputs)
    
    # ---------------------- with NO profiler -----------------------
    v_sample_results = feedbackcost_vmap(v_initial_carry, v_inputs)

    # --------------------------
    # results and terminal loss 
    # --------------------------
    Ksamples_jax, PathCosts_jax, actual_ref_jax = v_sample_results
    
    # Move the samples forward by 1 place and add xt to the front, to keep the same with numpy results.
    Ksamples_jax = jnp.concatenate((v_x0.reshape((n_samples, 1, -1)), Ksamples_jax[:,0:-1,:]), axis=1)
    PathCosts_jax = PathCosts_jax[:,-2,1]
    
    # ------------ Terminal cost ------------
    xT_samples = Ksamples_jax[:,-1,:]
    v_S_xT = terminal_cost_xQrx_vmap(xT_samples)
    PathCosts_jax = PathCosts_jax + v_S_xT
    
    return Ksamples_jax, PathCosts_jax, actual_ref_jax
    
    # ============================================== / jax parallel sampling ====================================
    

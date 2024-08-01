import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

import jax
import jax.numpy as jnp
import numpy as np
from dynamics.dynamics_slip import *
from dynamics.dynamics_discrete import *

# @jax.jit
def stochastic_integration_euler_SLIP(mode, x0, u, dt, eps, dW):   
    def mode0_dynamics_true_func_slip(args):
        (x0, u, dW, eps) = args
        # flight mode
        # [x, x_dot, z, z_dot, theta] = x0
        
        B = np.array([[0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [1.0]], dtype=np.float64)
        
        return x0 + np.array([x0[1], 0, x0[3], -9.81, u[0]], dtype=np.float64) * dt + np.sqrt(eps) * B@dW
    
    def mode0_dynamics_false_func_slip(args):
        (x0, u, dW, eps) = args
        # stance mode
        # [theta, theta_dot, r, r_dot] = x0
        
        g = 9.81
        k = 25.0
        m = 0.5
        r0 = 1
        
        theta, theta_dot, r, r_dot = x0[0], x0[1], x0[2], x0[3]
        
        # Defining the stance dynamics of the system
        B = np.array([[0.0, 0.0], 
                        [0.0, 0.0], 
                        [0.0, 1/m/r/r], 
                        [k/m, 0.0]], dtype=np.float64)
    
        xt_next = x0 + np.array([theta_dot, 
                                -2*theta_dot*r_dot/r-g*np.cos(theta)/r, 
                                r_dot + u[1]/m/r/r, 
                                k/m*(r0-r) - g*np.sin(theta) + theta_dot*theta_dot*r + k*u[0]/m], dtype=np.float64) * dt + np.sqrt(eps) * B@dW
        
        return xt_next
    
    xt_next = None
    args_choose_dynamics = (x0, u, dW, eps)
    if (mode == 0):
        xt_next = mode0_dynamics_true_func_slip(args_choose_dynamics)
    elif(mode == 1):
        xt_next = mode0_dynamics_false_func_slip(args_choose_dynamics)

    return xt_next


# ===============================================================================================================
#                                        SLIP Guard condition handling 
# ===============================================================================================================
def guard_condition_slip(xt, xt_next, current_mode):
    # assume time invariant guard for now
    return (current_mode==1) and (guard_slip_21(0.0,xt)<0) and (guard_slip_21(0.0,xt_next)>=0)

def guard_true_func_slip(args):
    print("slip_cond: True")
    (xt_current, current_mode, u, t, xt_next, dt_int, dt_shrinkrate, RandN, eps, reset_arg) = args
    
    def while_loop_body(xt, u, t, dt_int, dt_shrinkrate, RandN, eps, cnt_shrink):
        # Too far from the guard, shrink the step size
        dt_shrink = dt_int * dt_shrinkrate
        dW_new = np.sqrt(dt_shrink) * RandN
        
        xt_shrinked = stochastic_integration_euler_SLIP(current_mode, xt, u, dt_shrink, eps, dW_new)
        
        new_condition = not (guard_slip_21(t, xt_shrinked) < 0 or cnt_shrink == 10)
        cnt_shrink += 1
        
        return xt_shrinked, dt_shrink, cnt_shrink, new_condition
    
    cnt_shrink = 0
    can_continue = True
    xt_shrinked = xt_next
    
    # Implementing the loop with Python's while
    while can_continue:
        xt_shrinked, dt_int, cnt_shrink, can_continue = while_loop_body(
            xt_current, u, t, dt_int, dt_shrinkrate, RandN, eps, cnt_shrink
        )
    
    # Execute the reset map after the loop
    xt_next, next_mode, new_reset_arg = reset_map_slip_21(t, xt_shrinked, current_mode, reset_arg)
    dW_new = np.sqrt(dt_int) * RandN
    
    return xt_next, next_mode, dW_new, new_reset_arg


def guard_false_func_slip(args):
    (_, current_mode, _, _, xt_next, dt_int, _, RandN, _, reset_arg) = args
    dW = np.sqrt(dt_int)*RandN
    return xt_next, current_mode, dW, reset_arg


# ===============================================================================================================
#                                       // End of SLIP guard condition handling //
# ===============================================================================================================

from functools import partial

reaction_mode_mismatch_slip = partial(reaction_mode_mismatch, cond_early_arrival=cond_early_arrival_slip)
hybrid_stochastic_integration_slip = partial(hybrid_stochastic_integration_euler, 
                                                stochastic_integration_euler_func = stochastic_integration_euler_SLIP, 
                                                guard_condition_func = guard_condition_slip, 
                                                guard_condition_true_fun = guard_true_func_slip, 
                                                guard_condition_false_fun = guard_false_func_slip)


hybrid_stochastic_feedback_rollout_discrete_slip = partial(hybrid_stochastic_feedback_rollout_discrete, 
                                                           cond_mode_mismatch_func=cond_mode_mismatch_slip,
                                                            reaction_mode_mismatch_func=reaction_mode_mismatch_slip,
                                                            hybrid_stochastic_integration_func=hybrid_stochastic_integration_slip)    
    
    
    # show_mismatch = False
    # if show_mismatch:
    #     # ======== Show mode mismatch ======== 
    #     fig2, axes = plt.subplots(1,2, figsize=(9, 6))
    #     ax5, ax6 = axes.flatten()
    #     ax5.grid(True)
    #     ax6.grid(True)
        
    #     ax5.plot(xt_trj[:,0], xt_trj[:,1],color='b',linewidth=1.5,label='Rollout')
    #     ax5.plot(xt_ref[:,0], xt_ref[:,1],color='k',linewidth=2.5,label='Reference')
    #     ax5.plot(xt_ref_actual[:,0], xt_ref_actual[:,1],color='r',linewidth=1.5,linestyle='--', label='Modified Reference')
        
    #     ax5.set_xlabel(r"z", fontsize=14)
    #     ax5.set_ylabel(r"$\dot z$", fontsize=14)
    #     ax5.legend(loc='upper right')
    #     plt.tight_layout()
        
    #     ax6.plot(xt_trj[:,0], xt_trj[:,1],color='b',linewidth=1.5,label='Rollout')
    #     ax6.plot(xt_ref[:,0], xt_ref[:,1],color='k',linewidth=2.5,label='Reference')
    #     ax6.plot(xt_ref_actual[:,0], xt_ref_actual[:,1],color='r',linewidth=1.5,linestyle='--',label='Modified Reference')
    #     ax6.set_xlabel(r"z", fontsize=14)
    #     ax6.set_ylabel(r"$\dot z$", fontsize=14)
    #     ax6.legend(loc='upper right')
    #     plt.tight_layout()
        
    #     plt.show()
    
    
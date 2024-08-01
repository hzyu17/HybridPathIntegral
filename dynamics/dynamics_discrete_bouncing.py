import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

import numpy as np
from dynamics.dynamics_bouncing import *
from dynamics.dynamics_discrete import *


def stochastic_integration_euler_bouncing(mode, x0, u, dt, eps, dW):
    B = np.array([[0],[1.0]], dtype=np.float64)
    xt_next = x0 + np.array([x0[1], u[0]-9.81], dtype=np.float64) * dt + np.sqrt(eps) * B@dW
    return xt_next


# ===============================================================================================================
#                                        SLIP Guard condition handling 
# ===============================================================================================================
def guard_condition_bouncing(xt, xt_next, current_mode):
    # assume time invariant guard for now
    return (current_mode==0) and (guard_bouncing_12(0.0,xt)>0) and (guard_bouncing_12(0.0,xt_next)<=0)

def guard_true_func_bouncing(args):
    (xt_current, current_mode, u_current, t, xt_next, dt_int, dt_shrinkrate, RandN, eps, reset_arg) = args
    
    def while_loop_body(xt, u, t, dt_int, dt_shrinkrate, RandN, eps, cnt_shrink):
        # Too far from the guard, shrink the step size
        dt_shrink = dt_int * dt_shrinkrate
        dW_new = np.sqrt(dt_shrink) * RandN
        
        xt_shrinked = stochastic_integration_euler_bouncing(current_mode, xt, u, dt_shrink, eps, dW_new)
        
        new_condition = ((guard_bouncing_12(0.0,xt_shrinked)<=0) or (cnt_shrink < 10))
        cnt_shrink += 1
        
        return xt_shrinked, dt_shrink, cnt_shrink, new_condition
    
    cnt_shrink = 0
    can_continue = True
    xt_shrinked = xt_next
    
    # Implementing the loop with Python's while
    while can_continue:
        xt_shrinked, dt_int, cnt_shrink, can_continue = while_loop_body(
            xt_current, u_current, t, dt_int, dt_shrinkrate, RandN, eps, cnt_shrink
        )
    
    # Execute the reset map after the loop
    xt_next, next_mode, new_reset_arg = reset_map_bouncing_12(t, xt_shrinked, current_mode, reset_arg)
    dW_new = np.sqrt(dt_int) * RandN
    
    return xt_next, next_mode, dW_new, new_reset_arg


def guard_false_func_bouncing(args):
    (_, current_mode, _, _, xt_next, dt_int, _, RandN, _, reset_arg) = args
    dW = np.sqrt(dt_int)*RandN
    return xt_next, current_mode, dW, reset_arg


# ===============================================================================================================
#                                       // End of SLIP guard condition handling //
# ===============================================================================================================

from functools import partial

reaction_mode_mismatch_bouncing = partial(reaction_mode_mismatch, cond_early_arrival=cond_early_arrival_bouncing)
hybrid_stochastic_integration_bouncing = partial(hybrid_stochastic_integration_euler, 
                                                stochastic_integration_euler_func = stochastic_integration_euler_bouncing, 
                                                guard_condition_func = guard_condition_bouncing, 
                                                guard_condition_true_fun = guard_true_func_bouncing, 
                                                guard_condition_false_fun = guard_false_func_bouncing)


hybrid_stochastic_feedback_rollout_discrete_bouncing = partial(hybrid_stochastic_feedback_rollout_discrete, 
                                                           cond_mode_mismatch_func=cond_mode_mismatch_bouncing,
                                                            reaction_mode_mismatch_func=reaction_mode_mismatch_bouncing,
                                                            hybrid_stochastic_integration_func=hybrid_stochastic_integration_bouncing)    
    
    
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
    
    
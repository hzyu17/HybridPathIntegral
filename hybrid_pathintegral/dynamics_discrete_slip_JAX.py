import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

import jax
import jax.numpy as jnp
from dynamics.ode_solver.dynamics_slip import *
from dynamics.dynamics_discrete import *


def reset_map_slip_21_padding(t, x_event, current_mode, args_reset):
    print("t:", t)
    print("Reset map from mode 2 to mode 1, with padding.")
    
    xp = args_reset
    r0 = 1
    theta, theta_dot, r, r_dot = x_event[0], x_event[1], x_event[2], x_event[3]
    
    px_reset = xp + r0*jnp.cos(theta)
    vx_reset = r_dot*jnp.cos(theta) - r*theta_dot*jnp.sin(theta)
    pz_reset = r0*jnp.sin(theta)
    vz_reset = r0*theta_dot*jnp.cos(theta) + r_dot*jnp.sin(theta)
    theta_reset = theta

    x_reset = jnp.array([px_reset, vx_reset, pz_reset, vz_reset, theta_reset])

    # return x_reset, 0, jnp.array([args_reset[0]])
    return x_reset, 0, args_reset

# @jax.jit
def stoch_integr_euler_SLIP_padding(mode, x0, u, dt, eps, dW):   
    def mode0_dynamics_true_func(args):
        (x0, u, dW, eps) = args
        # flight mode
        # [x, x_dot, z, z_dot, theta] = x0
        
        # debug: two inputs
        B = jnp.array([[0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]], dtype=jnp.float64)
        
        return x0 + jnp.array([x0[1], u[0], x0[3], u[1]-9.81, u[2]], dtype=jnp.float64) * dt + jnp.sqrt(eps) * B@dW
        
        # B = jnp.array([[0.0, 0.0],
        #                 [0.0, 0.0],
        #                 [0.0, 0.0],
        #                 [0.0, 0.0],
        #                 [1.0, 0.0]], dtype=jnp.float64)
        
        # return x0 + jnp.array([x0[1], 0, x0[3], -9.81, u[0]], dtype=jnp.float64) * dt + jnp.sqrt(eps) * B@dW
    
    def mode0_dynamics_false_func(args):
        (x0, u, dW, eps) = args
        # stance mode
        # [theta, theta_dot, r, r_dot] = x0
        
        g = 9.81
        k = 25.0
        m = 0.5
        r0 = 1
        
        theta, theta_dot, r, r_dot = x0[0], x0[1], x0[2], x0[3]
        
        # Defining the stance dynamics of the system
        B = jnp.array([[0.0, 0.0, 0.0], 
                        [0.0, 0.0, 0.0], 
                        [0.0, 1/m/r/r, 0.0], 
                        [k/m, 0.0, 0.0],
                        [0.0, 0.0, 0.0]], dtype=jnp.float64)
        
        # # debug: linear dynamics
        # xt_next = x0 + jnp.array([theta_dot, 
        #                             0.0, 
        #                             r_dot + u[1]/m/r/r, 
        #                             k*u[0]/m,
        #                             0.0], dtype=jnp.float64) * dt + jnp.sqrt(eps) * B@dW
    
        xt_next = x0 + jnp.array([theta_dot, 
                                -2*theta_dot*r_dot/r-g*jnp.cos(theta)/r, 
                                r_dot + u[1]/m/r/r, 
                                k/m*(r0-r) - g*jnp.sin(theta) + theta_dot*theta_dot*r + k*u[0]/m,
                                0.0], dtype=jnp.float64) * dt + jnp.sqrt(eps) * B@dW
        
        return xt_next
    
    cond_mode0 = (mode == 0)
    args_choose_dynamics = (x0, u, dW, eps)
    xt_next = jax.lax.cond(cond_mode0, 
                           mode0_dynamics_true_func, 
                           mode0_dynamics_false_func, args_choose_dynamics)

    return xt_next


# ===============================================================================================================
#                                        SLIP Guard condition handling 
# ===============================================================================================================

# -------------------------------- From mode 1 (flight) to mode 2 (stance) --------------------------------
def guard_cond_slip_12(xt, xt_next, current_mode):
    return jnp.logical_and(current_mode==0, jnp.logical_and(guard_slip_12(0.0,xt)<0, guard_slip_12(0.0,xt_next)>0))

def guard_true_slip_12_padding(args):
    (xt_current, current_mode, u, t, xt_next, dt_int, _, RandN, eps, reset_arg) = args
    
    # -----------------
    #    Bi-section 
    # -----------------    
    def bisection_while_body_12(carry):
        t_left, t_mid, t_right, x_left, x_mid, x_right, u_current, tol, randN, continue_cond = carry
        
        # Update t_mid, dt_new, dW_new, and x_mid
        t_mid = (t_left + t_right) / 2.0
        dt_new = t_mid - t_left
        dW_new = jnp.sqrt(dt_new) * randN
        
        # Assuming `stoch_integr_euler_SLIP_padding` is a user-defined function
        x_mid = stoch_integr_euler_SLIP_padding(current_mode, x_left, u_current, dt_new, eps, dW_new)
        
        # Define guards
        guard_left = guard_slip_12(t_left, x_left)
        guard_mid = guard_slip_12(t_mid, x_mid)
        
        # Compute continuation condition
        continue_cond = jnp.logical_and(guard_mid != 0, (t_right - t_left) / 2.0 >= tol)
        
        # Branch based on the condition
        def update_left(carry):
            t_left, t_mid, t_right, x_left, x_mid, x_right, u_current, tol, randN, continue_cond = carry
            t_left = t_mid
            x_left = x_mid
            return t_left, t_mid, t_right, x_left, x_mid, x_right, u_current, tol, randN, continue_cond
        
        def update_right(carry):
            t_left, t_mid, t_right, x_left, x_mid, x_right, u_current, tol, randN, continue_cond = carry
            t_right = t_mid
            x_right = x_mid
            return t_left, t_mid, t_right, x_left, x_mid, x_right, u_current, tol, randN, continue_cond
        
        # Use lax.cond for conditional updates
        carry = jax.lax.cond(guard_left * guard_mid < 0, update_right, update_left, carry)
        
        return carry

    # Initial setup
    tol = 1e-10
    t_left = t
    t_right = t + dt_int
    t_mid = t
    x_left = xt_current
    x_mid = xt_next
    x_right = xt_next
    continue_bisection = True

    # Define the condition function (whether to continue looping)
    def cond_fun(carry):
        t_left, t_mid, t_right, x_left, x_mid, x_right, u, tol, RandN, continue_bisection = carry
        return continue_bisection  # Continue while continue_bisection is True

    # Define the body of the while loop
    def body_fun(carry):
        print("While loop guard 12")
        t_left, t_mid, t_right, x_left, x_mid, x_right, u, tol, RandN, continue_bisection = carry
        carry = bisection_while_body_12((t_left, t_mid, t_right, x_left, x_mid, x_right, u, tol, RandN, continue_bisection))
        return carry

    # Set up the initial carry tuple
    carry = (t_left, t_mid, t_right, x_left, x_mid, x_right, u, tol, RandN, continue_bisection)

    # Use lax.while_loop to perform the bisection
    final_carry = jax.lax.while_loop(cond_fun, body_fun, carry)

    # Unpack the final carry values
    t_left, t_mid, t_right, x_left, x_mid, x_right, u, tol, RandN, continue_bisection = final_carry

    
    xt_next, next_mode, new_reset_arg = reset_map_slip_12_padding(t_left, x_left, current_mode, reset_arg)
    dt_final = t_left - t
    dW_new = jnp.sqrt(dt_final) * RandN
        
    return t_left, x_left, xt_next, next_mode, dW_new, new_reset_arg


def guard_false_slip_12(args):
    # (xt_current, current_mode, u, t, xt_next, dt_int, dt_shrinkrate, RandN, eps, reset_arg) = args
    (xt_current, current_mode, _, t, xt_next, dt_int, _, RandN, _, reset_arg) = args
    dW = jnp.sqrt(dt_int)*RandN
    return t, xt_current, xt_next, current_mode, dW, reset_arg



# --------------------------------------------------  
#       From mode 2 (stance) to mode 1 (flight)
# --------------------------------------------------
def guard_cond_slip_21(xt, xt_next, current_mode):
    return jnp.logical_and(current_mode==1, jnp.logical_and(guard_slip_21(0.0,xt)<=0, guard_slip_21(0.0,xt_next)>0))

def guard_true_slip_21_padding(args):
    (xt_current, current_mode, u, t, xt_next, dt_int, _, RandN, eps, reset_arg) = args
    
    # -----------------
    #    Bi-section 
    # -----------------    
    def bisection_while_body_21(carry):
        t_left, t_mid, t_right, x_left, x_mid, x_right, u_current, tol, randN, continue_cond = carry
        
        t_mid = (t_left + t_right) / 2.0
        dt_new = t_mid - t_left
        dW_new = jnp.sqrt(dt_new)*randN
        
        x_mid = stoch_integr_euler_SLIP_padding(current_mode, x_left, u_current, dt_new, eps, dW_new)

        # Define conditions
        guard_left = guard_slip_21(t_left, x_left)
        guard_mid = guard_slip_21(t_mid, x_mid)
        
        continue_cond = jnp.logical_and(guard_mid != 0, (t_right - t_left) / 2.0 >= tol)

        # Define the branches for lax.cond
        def update_right(carry):
            t_left, t_mid, t_right, x_left, x_mid, x_right, u_current, tol, randN, continue_cond = carry
            t_right = t_mid
            x_right = x_mid
            return t_left, t_mid, t_right, x_left, x_mid, x_right, u_current, tol, randN, continue_cond

        def update_left(carry):
            t_left, t_mid, t_right, x_left, x_mid, x_right, u_current, tol, randN, continue_cond = carry
            t_left = t_mid
            x_left = x_mid
            return t_left, t_mid, t_right, x_left, x_mid, x_right, u_current, tol, randN, continue_cond

        # Use lax.cond to conditionally update t_left, t_right, x_left, and x_right
        carry = jax.lax.cond(guard_left * guard_mid < 0, update_right, update_left, carry)
            
        return carry

    # Initial setup
    tol = 1e-10
    t_left = t
    t_right = t + dt_int
    t_mid = t
    x_left = xt_current
    x_mid = xt_next
    x_right = xt_next
    continue_bisection = True

    # Define the condition function (whether to continue looping)
    def cond_fun(carry):
        t_left, t_mid, t_right, x_left, x_mid, x_right, u, tol, RandN, continue_bisection = carry
        return continue_bisection  # Continue while continue_bisection is True

    # Define the body of the while loop
    def body_fun(carry):
        print("While loop guard 21")
        t_left, t_mid, t_right, x_left, x_mid, x_right, u, tol, RandN, continue_bisection = carry
        carry = bisection_while_body_21((t_left, t_mid, t_right, x_left, x_mid, x_right, u, tol, RandN, continue_bisection))
        return carry

    # Set up the initial carry tuple
    carry = (t_left, t_mid, t_right, x_left, x_mid, x_right, u, tol, RandN, continue_bisection)

    # Use lax.while_loop to perform the bisection
    final_carry = jax.lax.while_loop(cond_fun, body_fun, carry)

    # Unpack the final carry values
    t_left, t_mid, t_right, x_left, x_mid, x_right, u, tol, RandN, continue_bisection = final_carry
    
    # Debug statement
    # print(f"guard left: {guard_slip_21(t_left, x_left)}, guard right: {guard_slip_21(t_right, x_right)}, guard mid: {guard_slip_21(t_mid, x_mid)}")  
    
    xt_next, next_mode, new_reset_arg = reset_map_slip_21_padding(t_left, x_left, current_mode, reset_arg)
    dt_final = t_left - t
    dW_new = jnp.sqrt(dt_final) * RandN
    
    # # --------------------
    # #   Direct reset map 
    # # --------------------
    # xt_next, next_mode, new_reset_arg = reset_map_slip_21(t, xt_next, current_mode, reset_arg)
    # dW_new = np.sqrt(dt_int) * RandN
    
    return t_left, x_left, xt_next, next_mode, dW_new, new_reset_arg


def guard_false_slip_21(args):
    # (xt_current, current_mode, u, t, xt_next, dt_int, _, RandN, eps, reset_arg) = args
    (xt_current, current_mode, _, t, xt_next, dt_int, _, RandN, _, reset_arg) = args
    dW = jnp.sqrt(dt_int)*RandN
    return t, xt_current, xt_next, current_mode, dW, reset_arg


# ===============================================================================================================
#                                       // End of SLIP guard condition handling //
# ===============================================================================================================


# # ===============================================================================================================
# #                                        SLIP Guard condition handling 
# # ===============================================================================================================
# @jax.jit
# def guard_condition_slip_padding(xt, xt_next, current_mode):
#     return jnp.logical_and(current_mode==1, jnp.logical_and(guard_slip_21(0.0,xt)<0, guard_slip_21(0.0,xt_next)>=0))

# @jax.jit
# def guard_true_func_slip_padding(args):
#     print("slip_cond: True")
#     (xt_current, current_mode, u, t, xt_next, dt_int, dt_shrinkrate, RandN, eps, reset_arg) = args
    
#     def while_cond(vars):
#         (_, _, _, _, _, _, _, _, _, can_continue) = vars
#         return can_continue
    
#     def while_loop_body(vars):
#         (xt, _, u, t, dt_int, dt_shrinkrate, RandN, eps, cnt_shrink, _) = vars
        
#         # Too far from the guard, shrink the step size.
#         dt_shrink = dt_int * dt_shrinkrate
#         dW_new = jnp.sqrt(dt_shrink)*RandN
        
#         xt_shrinked = stoch_integr_euler_SLIP_padding(current_mode, xt, u, dt_shrink, eps, dW_new)
        
#         new_condition = jnp.logical_and(guard_slip_21(t, xt_shrinked) >= 0, cnt_shrink<50)
#         cnt_shrink += 1
        
#         new_vars = (xt, xt_shrinked, u, t, dt_shrink, dt_shrinkrate, RandN, eps, cnt_shrink, new_condition)
        
#         return new_vars
    
#     # --------------------------
#     #       Shrinking dt
#     # --------------------------
#     # init_vars = (xt_current, xt_next, u, t, dt_int, dt_shrinkrate, RandN, eps, 0, True)
#     # final_vars = jax.lax.while_loop(while_cond, while_loop_body, init_val=init_vars)
    
#     # # (_, xt_shrinked, _, _, dt_shrinked, _, RandN, _, _, _) = final_vars
#     # t_event = t + dt_shrinked
#     # x_event = xt_shrinked
#     # xt_reset, next_mode, new_reset_arg = reset_map_slip_21_padding(t_event, xt_shrinked, current_mode, reset_arg)
#     # dW_new = jnp.sqrt(dt_shrinked)*RandN
    
#     # --------------------------
#     #     Direct reset map
#     # --------------------------
#     t_event = t + dt_int
#     x_event = xt_next
#     xt_reset, next_mode, new_reset_arg = reset_map_slip_21_padding(t, xt_next, current_mode, reset_arg)
#     dW_new = jnp.sqrt(dt_int)*RandN
    
#     return t_event, x_event, xt_reset, next_mode, dW_new, new_reset_arg


# @jax.jit
# def guard_false_func_slip_padding(args):
#     (_, current_mode, _, t, xt_next, dt_int, _, RandN, eps, reset_arg) = args
    
#     t_next = t + dt_int
#     dW = jnp.sqrt(dt_int)*RandN
    
#     return t_next, xt_next, xt_next, current_mode, dW, reset_arg

# # ===============================================================================================================
# #                                       // End of SLIP guard condition handling //
# # ===============================================================================================================

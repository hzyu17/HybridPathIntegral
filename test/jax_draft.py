import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


def cond_mode0_true(args):
    ref_trj_0, ref_trj_1 = args
    return ref_trj_0

def cond_mode0_false(args):
    ref_trj_0, ref_trj_1 = args
    return ref_trj_1

def cond_true_func(args):
    ref_trj_0, ref_trj_1 = args
    return ref_trj_1

def cond_false_func(args):
    ref_trj_0, ref_trj_1 = args
    return ref_trj_0

def dynamics(current_mode, xt, ut):
    xt_next = xt + ut
    
    def change_mode_true(ut):
        next_mode = 0
        return next_mode
    
    def change_mode_false(ut):
        next_mode = 1
        return next_mode
    
    cond_ut = (ut>2)
    args = (ut, )
    next_mode = jax.lax.cond(cond_ut, change_mode_true, change_mode_false, args)
    
    return xt_next, next_mode
    

def scan_func(carry, inputs, ref_trj_0, ref_trj_1):
    current_mode, xt = carry
    current_ut, ref_mode = inputs
    
    cond_mode0 = (current_mode==0)
    args_mode0 = (ref_trj_0, ref_trj_1)
    ref_trj = jax.lax.cond(cond_mode0, cond_mode0_true, cond_mode0_false, args_mode0)
    
    next_mode = current_mode
    
    mode_mismatch = (current_mode!=ref_mode)
    cond_args = (ref_trj_0, ref_trj_1)
    ref_trj = jax.lax.cond(mode_mismatch, cond_true_func, cond_false_func, cond_args)
    
    xt_next, next_mode = dynamics(current_mode, xt, current_ut)
    
    return (next_mode, xt_next), (current_mode, xt, current_ut, ref_trj)




if __name__ == '__main__':
    ref_mode = jnp.array([1,1,1,1,0,0,0])
    ut = jnp.array([1,2,3,4,5,6,7])
    mode_t0 = 1
    x0 = 0
    
    xt_trj_0 = np.array([0,0,0])
    xt_trj_1 = np.array([1,1,1])
    
    scan_func_partial = partial(scan_func, ref_trj_0=xt_trj_0, ref_trj_1=xt_trj_1)
    
    init_carry = (mode_t0, x0)
    inputs = (ut, ref_mode)
    
    (mode, xt), (modes_trj, xt_trj, ut_trj, v_ref_trj) = jax.lax.scan(scan_func_partial, init_carry, inputs)
    
    print("ref_modes: ", ref_mode)
    print("modes_trj: ", modes_trj)
    print("xt_trj: ", xt_trj)
    print("ut_trj: ", ut_trj)
    print("v_ref_trj: ")
    print(v_ref_trj)
    
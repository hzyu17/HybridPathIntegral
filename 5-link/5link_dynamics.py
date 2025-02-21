import jax.numpy as jnp
import jax

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from matplotlib.lines import Line2D

from scipy.integrate import solve_ivp
from functools import partial

# param_list = {'g','p(0)';
# 	'L_torso','p(1)'; 'L_fem','p(2)'; 'L_tib','p(3)';
# 	'M_torso','p(4)'; 'M_fem','p(5)'; 'M_tib','p(6)';
# 	'MY_torso','p(7)'; 'MZ_torso','p(8)'; 'MZ_fem','p(9)'; 'MZ_tib','p(10)';
# 	'XX_torso','p(11)'; 'XX_fem','p(12)'; 'XX_tib','p(13)'};


def C_matrix(q, dq, params):
    """
    Compute the 5x5 matrix C for the biped model.
    
    Parameters:
      q      : a length-5 JAX array 
      dq     : a length-5 JAX array of velocities
      params : a JAX array containing the model parameters.
    
    Returns:
      C : a 5x5 JAX array.
    """
    p = params 
    
    # --- Row 1 ---
    C11 = p[3] * jnp.sin(q[2] - 2*q[0] + q[4]) * (dq[4] - 2*dq[0] + dq[2]) * (2*p[2]*p[5] - p[9] + p[2]*p[6] + p[2]*p[4])
    
    term1 = p[10] * dq[3] * ( p[3]*jnp.sin(q[0]-q[1]-q[2]+q[3]) + p[2]*jnp.sin(q[0]+q[1]-q[3]-q[4]) )
    term2 = dq[1] * ( p[2]*p[9]*jnp.sin(q[0]-q[1]) 
                      + p[3]*p[10]*jnp.sin(q[0]-q[1]-q[2]+q[3]) 
                      + p[2]*p[10]*jnp.sin(q[0]+q[1]-q[3]-q[4])
                      + p[3]*p[9]*jnp.sin(q[0]+q[1]-q[2]-q[4])
                      + (p[2]**2)*p[6]*jnp.sin(q[0]-q[1])
                      + p[2]*p[3]*p[6]*jnp.sin(q[0]+q[1]-q[2]-q[4]) )
    term3 = dq[4] * ( p[3]*jnp.sin(q[0]+q[1]-q[2]-q[4]) + p[2]*jnp.sin(q[0]-q[1]) ) * ( p[9] + p[2]*p[6] )
    C12 = term1 - term2 + term3
    
    C13 = p[3] * jnp.sin(q[2] - 2*q[0] + q[4]) * (dq[0] - dq[2]) * (2*p[2]*p[5] - p[9] + p[2]*p[6] + p[2]*p[4])
    
    C14 = p[10] * ( p[3]*jnp.sin(q[0]-q[1]-q[2]+q[3]) + p[2]*jnp.sin(q[0]+q[1]-q[3]-q[4]) ) * (dq[1] - dq[3])
    
    C15 = ( p[2]*p[9]*dq[1]*jnp.sin(q[0]-q[1])
           - p[2]*p[9]*dq[4]*jnp.sin(q[0]-q[1])
           - p[2]*p[7]*dq[4]*jnp.cos(q[0]-2*q[4])
           - p[2]*p[8]*dq[4]*jnp.sin(q[0]-2*q[4])
           - p[3]*p[9]*dq[4]*jnp.sin(q[0]+q[1]-q[2]-q[4])
           + p[3]*p[9]*dq[1]*jnp.sin(q[0]+q[1]-q[2]-q[4])
           - (p[2]**2)*p[6]*dq[4]*jnp.sin(q[0]-q[1])
           + (p[2]**2)*p[6]*dq[1]*jnp.sin(q[0]-q[1])
           + p[3]*p[7]*dq[4]*jnp.cos(q[0]-q[2]+q[4])
           + p[3]*p[9]*dq[4]*jnp.sin(q[2]-2*q[0]+q[4])
           - p[3]*p[9]*dq[0]*jnp.sin(q[2]-2*q[0]+q[4])
           - p[3]*p[8]*dq[4]*jnp.sin(q[0]-q[2]+q[4])
           - 2*p[2]*p[3]*p[5]*dq[4]*jnp.sin(q[2]-2*q[0]+q[4])
           + 2*p[2]*p[3]*p[5]*dq[0]*jnp.sin(q[2]-2*q[0]+q[4])
           - p[2]*p[3]*p[6]*dq[4]*jnp.sin(q[2]-2*q[0]+q[4])
           + p[2]*p[3]*p[6]*dq[0]*jnp.sin(q[2]-2*q[0]+q[4])
           - p[2]*p[3]*p[5]*dq[4]*jnp.sin(q[0]+q[1]-q[2]-q[4])
           + p[2]*p[3]*p[5]*dq[1]*jnp.sin(q[0]+q[1]-q[2]-q[4]) )
    
    # --- Row 2 ---
    C21 = ( p[2]*p[9]*dq[0]*jnp.sin(q[0]-q[1])
            - p[2]*p[9]*dq[4]*jnp.sin(q[0]-q[1])
            + p[3]*p[10]*dq[0]*jnp.sin(q[0]-q[1]-q[2]+q[3])
            - p[3]*p[10]*dq[2]*jnp.sin(q[0]-q[1]-q[2]+q[3])
            + p[2]*p[10]*dq[4]*jnp.sin(q[0]+q[1]-q[3]-q[4])
            - p[2]*p[10]*dq[0]*jnp.sin(q[0]+q[1]-q[3]-q[4])
            - p[3]*p[9]*dq[0]*jnp.sin(q[0]+q[1]-q[2]-q[4])
            + p[3]*p[9]*dq[2]*jnp.sin(q[0]+q[1]-q[2]-q[4])
            - (p[2]**2)*p[6]*dq[4]*jnp.sin(q[0]-q[1])
            + (p[2]**2)*p[6]*dq[0]*jnp.sin(q[0]-q[1])
            - p[2]*p[3]*p[6]*dq[0]*jnp.sin(q[0]+q[1]-q[2]-q[4])
            + p[2]*p[3]*p[6]*dq[2]*jnp.sin(q[0]+q[1]-q[2]-q[4]) )
    
    C22 = p[2]*p[10]*jnp.sin(q[3] - 2*q[1] + q[4]) * (dq[4] - 2*dq[1] + dq[3])
    
    C23 = p[3]*(dq[0]-dq[2]) * ( p[9]*jnp.sin(q[0]+q[1]-q[2]-q[4])
                                  - p[10]*jnp.sin(q[0]-q[1]-q[2]+q[3])
                                  + p[2]*p[6]*jnp.sin(q[0]+q[1]-q[2]-q[4]) )
    
    C24 = p[2]*p[10]*jnp.sin(q[3]-2*q[1]+q[4]) * (dq[1]-dq[3])
    
    C25 = ( p[2]*p[9]*dq[4]*jnp.sin(q[0]-q[1])
            - p[2]*p[9]*dq[0]*jnp.sin(q[0]-q[1])
            - p[2]*p[10]*dq[4]*jnp.sin(q[0]+q[1]-q[3]-q[4])
            + p[2]*p[10]*dq[0]*jnp.sin(q[0]+q[1]-q[3]-q[4])
            + (p[2]**2)*p[6]*dq[4]*jnp.sin(q[0]-q[1])
            - (p[2]**2)*p[6]*dq[0]*jnp.sin(q[0]-q[1])
            - p[2]*p[10]*dq[4]*jnp.sin(q[3]-2*q[1]+q[4])
            + p[2]*p[10]*dq[1]*jnp.sin(q[3]-2*q[1]+q[4]) )
    
    # --- Row 3 ---
    C31 = - p[3] * jnp.sin(q[2]-2*q[0]+q[4]) * (dq[4]-dq[0]) * (2*p[2]*p[5]-p[9]+p[2]*p[6]+p[2]*p[4])
    
    C32 = ( p[3]*dq[1]*( p[10]*jnp.sin(q[0]-q[1]-q[2]+q[3]) + p[9]*jnp.sin(q[0]+q[1]-q[2]-q[4]) + p[2]*p[6]*jnp.sin(q[0]+q[1]-q[2]-q[4]) )
            - p[3]*dq[4]*jnp.sin(q[0]+q[1]-q[2]-q[4])*(p[9]+p[2]*p[6])
            - p[3]*p[10]*dq[3]*jnp.sin(q[0]-q[1]-q[2]+q[3]) )
    
    C33 = 0.0
    
    C34 = - p[3]*p[10]*jnp.sin(q[0]-q[1]-q[2]+q[3]) * (dq[1]-dq[3])
    
    C35 = ( p[3]*dq[4]*( p[9]*jnp.sin(q[0]+q[1]-q[2]-q[4])
                         - p[7]*jnp.cos(q[0]-q[2]+q[4])
                         - p[9]*jnp.sin(q[2]-2*q[0]+q[4])
                         + p[8]*jnp.sin(q[0]-q[2]+q[4])
                         + p[2]*p[6]*jnp.sin(q[0]+q[1]-q[2]-q[4])
                         + 2*p[2]*p[5]*jnp.sin(q[2]-2*q[0]+q[4])
                         + p[2]*p[6]*jnp.sin(q[2]-2*q[0]+q[4])
                         + p[2]*p[4]*jnp.sin(q[2]-2*q[0]+q[4]) )
            - p[3]*dq[1]*jnp.sin(q[0]+q[1]-q[2]-q[4])*(p[9]+p[2]*p[6])
            - p[3]*dq[0]*jnp.sin(q[2]-2*q[0]+q[4])*(2*p[2]*p[5]-p[9]+p[2]*p[6]+p[2]*p[4]) )
    
    # --- Row 4 ---
    C41 = ( p[3]*p[10]*dq[2]*jnp.sin(q[0]-q[1]-q[2]+q[3])
            - p[3]*p[10]*dq[0]*jnp.sin(q[0]-q[1]-q[2]+q[3])
            - p[2]*p[10]*dq[4]*jnp.sin(q[0]+q[1]-q[3]-q[4])
            + p[2]*p[10]*dq[0]*jnp.sin(q[0]+q[1]-q[3]-q[4]) )
    
    C42 = - p[2]*p[10]*jnp.sin(q[3]-2*q[1]+q[4]) * (dq[4]-dq[1])
    
    C43 = p[3]*p[10]*jnp.sin(q[0]-q[1]-q[2]+q[3]) * (dq[0]-dq[2])
    
    C44 = 0.0
    
    C45 = p[2]*p[10]*( dq[4]*jnp.sin(q[0]+q[1]-q[3]-q[4])
                        - dq[0]*jnp.sin(q[0]+q[1]-q[3]-q[4])
                        + dq[4]*jnp.sin(q[3]-2*q[1]+q[4])
                        - dq[1]*jnp.sin(q[3]-2*q[1]+q[4]) )
    
    # --- Row 5 ---
    C51 = ( p[2]*p[7]*dq[0]*jnp.cos(q[0]-2*q[4])
            - p[2]*p[7]*dq[4]*jnp.cos(q[0]-2*q[4])
            + p[2]*p[9]*dq[4]*jnp.sin(q[0]-q[1])
            - p[2]*p[9]*dq[0]*jnp.sin(q[0]-q[1])
            - p[2]*p[8]*dq[4]*jnp.sin(q[0]-2*q[4])
            + p[2]*p[8]*dq[0]*jnp.sin(q[0]-2*q[4])
            + p[3]*p[9]*dq[0]*jnp.sin(q[0]+q[1]-q[2]-q[4])
            - p[3]*p[9]*dq[2]*jnp.sin(q[0]+q[1]-q[2]-q[4])
            + (p[2]**2)*p[6]*dq[4]*jnp.sin(q[0]-q[1])
            - (p[2]**2)*p[6]*dq[0]*jnp.sin(q[0]-q[1])
            + p[3]*p[7]*dq[0]*jnp.cos(q[0]-q[2]+q[4])
            - p[3]*p[7]*dq[2]*jnp.cos(q[0]-q[2]+q[4])
            - p[3]*p[9]*dq[0]*jnp.sin(q[2]-2*q[0]+q[4])
            + p[3]*p[9]*dq[2]*jnp.sin(q[2]-2*q[0]+q[4])
            - p[3]*p[8]*dq[0]*jnp.sin(q[0]-q[2]+q[4])
            + p[3]*p[8]*dq[2]*jnp.sin(q[0]-q[2]+q[4])
            - 2*p[2]*p[3]*p[5]*dq[0]*jnp.sin(q[2]-2*q[0]+q[4])
            + 2*p[2]*p[3]*p[5]*dq[2]*jnp.sin(q[2]-2*q[0]+q[4])
            - p[2]*p[3]*p[6]*dq[0]*jnp.sin(q[2]-2*q[0]+q[4])
            + p[2]*p[3]*p[6]*dq[2]*jnp.sin(q[2]-2*q[0]+q[4])
            - p[2]*p[3]*p[5]*dq[0]*jnp.sin(q[0]+q[1]-q[2]-q[4])
            + p[2]*p[3]*p[5]*dq[2]*jnp.sin(q[0]+q[1]-q[2]-q[4]) )
    
    C52 = ( p[2]*p[9]*dq[1]*jnp.sin(q[0]-q[1])
            - p[2]*p[9]*dq[4]*jnp.sin(q[0]-q[1])
            + p[2]*p[10]*dq[1]*jnp.sin(q[0]+q[1]-q[3]-q[4])
            - p[2]*p[10]*dq[3]*jnp.sin(q[0]+q[1]-q[3]-q[4])
            - (p[2]**2)*p[6]*dq[4]*jnp.sin(q[0]-q[1])
            + (p[2]**2)*p[6]*dq[1]*jnp.sin(q[0]-q[1])
            + p[2]*p[10]*dq[1]*jnp.sin(q[3]-2*q[1]+q[4])
            - p[2]*p[10]*dq[3]*jnp.sin(q[3]-2*q[1]+q[4]) )
    
    C53 = - p[3]*(dq[0]-dq[2]) * ( p[9]*jnp.sin(q[0]+q[1]-q[2]-q[4])
                                   + p[7]*jnp.cos(q[0]-q[2]+q[4])
                                   - p[9]*jnp.sin(q[2]-2*q[0]+q[4])
                                   - p[8]*jnp.sin(q[0]-q[2]+q[4])
                                   + p[2]*p[6]*jnp.sin(q[0]+q[1]-q[2]-q[4])
                                   + 2*p[2]*p[5]*jnp.sin(q[2]-2*q[0]+q[4])
                                   + p[2]*p[6]*jnp.sin(q[2]-2*q[0]+q[4])
                                   + p[2]*p[4]*jnp.sin(q[2]-2*q[0]+q[4]) )
    
    C54 = - p[2]*p[10]*( jnp.sin(q[0]+q[1]-q[3]-q[4]) + jnp.sin(q[3]-2*q[1]+q[4]) ) * (dq[1]-dq[3])
    
    C55 = ( 2*p[2]*dq[4]*( p[7]*jnp.cos(q[0]-2*q[4]) + p[8]*jnp.sin(q[0]-2*q[4]) )
            - p[2]*dq[0]*( p[7]*jnp.cos(q[0]-2*q[4]) - p[9]*jnp.sin(q[0]-q[1])
                          + p[8]*jnp.sin(q[0]-2*q[4]) - p[2]*p[6]*jnp.sin(q[0]-q[1]) )
            - p[2]*dq[1]*jnp.sin(q[0]-q[1])*( p[9] + p[2]*p[6] ) )
    
    # Assemble the 5x5 matrix
    C = jnp.array([[C11, C12, C13, C14, C15],
                   [C21, C22, C23, C24, C25],
                   [C31, C32, C33, C34, C35],
                   [C41, C42, C43, C44, C45],
                   [C51, C52, C53, C54, C55]])
    
    return C

def B_matrix():
    B = jnp.array([
        [ 1,  0, -2,  0],
        [ 0,  1,  0, -2],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1],
        [-2, -2,  1,  1]
    ])
    return B

def D_matrix(q, params):
    """
    Compute the 5x5 mass-inertia matrix D(q) for the biped model.

    Inputs:
      q      : a length-5 JAX array
      params : a JAX array of parameters.
    Returns:
      D : a 5x5 JAX array.
    """
    p = params  # shorthand

    # --- D(1,1) ---
    D11 = ( p[12] + p[13]
            + 2*(p[2]**2)*p[5] + (p[2]**2)*p[6]
            + 2*(p[3]**2)*p[5] + (p[2]**2)*p[4]
            + 2*(p[3]**2)*p[6] + (p[3]**2)*p[4]
            - 2*p[2]*p[9] - 2*p[3]*p[10]
            + 2*p[3]*p[9]*jnp.cos(q[2] - 2*q[0] + q[4])
            - 4*p[2]*p[3]*p[5]*jnp.cos(q[2] - 2*q[0] + q[4])
            - 2*p[2]*p[3]*p[6]*jnp.cos(q[2] - 2*q[0] + q[4])
            - 2*p[2]*p[3]*p[4]*jnp.cos(q[2] - 2*q[0] + q[4])
          )

    # --- D(1,2) ---
    D12 = ( p[2]*p[10]*jnp.cos(q[0]+q[1]-q[3]-q[4])
            - p[3]*p[10]*jnp.cos(q[0]-q[1]-q[2]+q[3])
            + p[3]*p[9]*jnp.cos(q[0]+q[1]-q[2]-q[4])
            - (p[2]**2)*p[6]*jnp.cos(q[0]-q[1])
            - p[2]*p[9]*jnp.cos(q[0]-q[1])
            + p[2]*p[3]*p[6]*jnp.cos(q[0]+q[1]-q[2]-q[4])
          )

    # --- D(1,3) ---
    D13 = ( 2*p[3]*p[10]
            - 2*(p[3]**2)*p[5]
            - 2*(p[3]**2)*p[6]
            - (p[3]**2)*p[4]
            - p[13]
            - p[3]*p[9]*jnp.cos(q[2]-2*q[0]+q[4])
            + 2*p[2]*p[3]*p[5]*jnp.cos(q[2]-2*q[0]+q[4])
            + p[2]*p[3]*p[6]*jnp.cos(q[2]-2*q[0]+q[4])
            + p[2]*p[3]*p[4]*jnp.cos(q[2]-2*q[0]+q[4])
          )

    # --- D(1,4) ---
    D14 = ( p[3]*p[10]*jnp.cos(q[0]-q[1]-q[2]+q[3])
            - p[2]*p[10]*jnp.cos(q[0]+q[1]-q[3]-q[4])
          )

    # --- D(1,5) ---
    D15 = ( 2*p[2]*p[9]
            - 2*(p[2]**2)*p[5]
            - (p[2]**2)*p[6]
            - (p[2]**2)*p[4]
            - p[12]
            + p[2]*p[7]*jnp.sin(q[0]-2*q[4])
            - p[3]*p[9]*jnp.cos(q[0]+q[1]-q[2]-q[4])
            + (p[2]**2)*p[6]*jnp.cos(q[0]-q[1])
            - p[3]*p[9]*jnp.cos(q[2]-2*q[0]+q[4])
            + p[3]*p[8]*jnp.cos(q[0]-q[2]+q[4])
            + p[3]*p[7]*jnp.sin(q[0]-q[2]+q[4])
            + p[2]*p[9]*jnp.cos(q[0]-q[1])
            - p[2]*p[8]*jnp.cos(q[0]-2*q[4])
            - p[2]*p[3]*p[6]*jnp.cos(q[0]+q[1]-q[2]-q[4])
            + 2*p[2]*p[3]*p[5]*jnp.cos(q[2]-2*q[0]+q[4])
            + p[2]*p[3]*p[6]*jnp.cos(q[2]-2*q[0]+q[4])
            + p[2]*p[3]*p[4]*jnp.cos(q[2]-2*q[0]+q[4])
          )

    # --- D(2,1) --- (by symmetry, D(2,1)=D(1,2))
    D21 = D12

    # --- D(2,2) ---
    D22 = ( p[12] + p[13]
            + (p[2]**2)*p[6]
            - 2*p[2]*p[10]*jnp.cos(q[3]-2*q[1]+q[4])
          )

    # --- D(2,3) ---
    D23 = ( p[3]*p[10]*jnp.cos(q[0]-q[1]-q[2]+q[3])
            - p[3]*p[9]*jnp.cos(q[0]+q[1]-q[2]-q[4])
            - p[2]*p[3]*p[6]*jnp.cos(q[0]+q[1]-q[2]-q[4])
          )

    # --- D(2,4) ---
    D24 = ( p[2]*p[10]*jnp.cos(q[3]-2*q[1]+q[4])
            - p[13]
          )

    # --- D(2,5) ---
    D25 = ( (p[2]**2)*p[6]*jnp.cos(q[0]-q[1])
            - (p[2]**2)*p[6]
            - p[2]*p[10]*jnp.cos(q[0]+q[1]-q[3]-q[4])
            - p[12]
            + p[2]*p[10]*jnp.cos(q[3]-2*q[1]+q[4])
            + p[2]*p[9]*jnp.cos(q[0]-q[1])
          )

    # --- D(3,1) --- (symmetric with D(1,3))
    D31 = D13

    # --- D(3,2) ---
    D32 = D23

    # --- D(3,3) ---
    D33 = ( p[13]
            + 2*(p[3]**2)*p[5] + 2*(p[3]**2)*p[6] + (p[3]**2)*p[4]
            - 2*p[3]*p[10]
          )

    # --- D(3,4) ---
    D34 = - p[3]*p[10]*jnp.cos(q[0]-q[1]-q[2]+q[3])

    # --- D(3,5) ---
    D35 = ( p[3]*p[9]*jnp.cos(q[0]+q[1]-q[2]-q[4])
            + p[3]*p[9]*jnp.cos(q[2]-2*q[0]+q[4])
            - p[3]*p[8]*jnp.cos(q[0]-q[2]+q[4])
            - p[3]*p[7]*jnp.sin(q[0]-q[2]+q[4])
            + p[2]*p[3]*p[6]*jnp.cos(q[0]+q[1]-q[2]-q[4])
            - 2*p[2]*p[3]*p[5]*jnp.cos(q[2]-2*q[0]+q[4])
            - p[2]*p[3]*p[6]*jnp.cos(q[2]-2*q[0]+q[4])
            - p[2]*p[3]*p[4]*jnp.cos(q[2]-2*q[0]+q[4])
          )

    # --- D(4,1) ---
    D41 = p[10]*( p[3]*jnp.cos(q[0]-q[1]-q[2]+q[3])
                  - p[2]*jnp.cos(q[0]+q[1]-q[3]-q[4]) )

    # --- D(4,2) ---
    D42 = p[2]*p[10]*jnp.cos(q[3]-2*q[1]+q[4]) - p[13]

    # --- D(4,3) ---
    D43 = - p[3]*p[10]*jnp.cos(q[0]-q[1]-q[2]+q[3])

    # --- D(4,4) ---
    D44 = p[13]

    # --- D(4,5) ---
    D45 = p[2]*p[10]*( jnp.cos(q[0]+q[1]-q[3]-q[4])
                       - jnp.cos(q[3]-2*q[1]+q[4]) )

    # --- D(5,1) ---
    D51 = D15

    # --- D(5,2) ---
    D52 = ( (p[2]**2)*p[6]*jnp.cos(q[0]-q[1])
            - (p[2]**2)*p[6]
            - p[2]*p[10]*jnp.cos(q[0]+q[1]-q[3]-q[4])
            - p[12]
            + p[2]*p[10]*jnp.cos(q[3]-2*q[1]+q[4])
            + p[2]*p[9]*jnp.cos(q[0]-q[1])
          )

    # --- D(5,3) ---
    D53 = D35

    # --- D(5,4) ---
    D54 = p[2]*p[10]*( jnp.cos(q[0]+q[1]-q[3]-q[4])
                       - jnp.cos(q[3]-2*q[1]+q[4]) )

    # --- D(5,5) ---
    D55 = ( 2*p[12] + p[11]
            + 2*(p[2]**2)*p[5] + 2*(p[2]**2)*p[6] + (p[2]**2)*p[4]
            - 2*p[2]*p[9]
            - 2*p[2]*p[7]*jnp.sin(q[0]-2*q[4])
            - 2*(p[2]**2)*p[6]*jnp.cos(q[0]-q[1])
            - 2*p[2]*p[9]*jnp.cos(q[0]-q[1])
            + 2*p[2]*p[8]*jnp.cos(q[0]-2*q[4])
          )

    # Assemble the matrix D row by row.
    D = jnp.array([[D11, D12, D13, D14, D15],
                   [D21, D22, D23, D24, D25],
                   [D31, D32, D33, D34, D35],
                   [D41, D42, D43, D44, D45],
                   [D51, D52, D53, D54, D55]])
    
    return D


def G_vector(q, params):
    """
    Compute the 5x1 gravitational vector G for the biped model.

    Inputs:
      q      : a length-5 JAX array, where:
               q[0] corresponds to MATLAB q(1),
               q[1] to q(2), ..., q[4] to q(5).
      params : a JAX array containing the parameters.
               
    Returns:
      G : a 5x1 JAX array representing the gravitational vector.
    """
    # For convenience, let p = params.
    p = params

    # G(1,1)
    G1 = p[0]*(2*p[3]*p[5]*jnp.sin(q[0]-q[2])
               - p[9]*jnp.sin(q[0]-q[4])
               - p[10]*jnp.sin(q[0]-q[2])
               + 2*p[3]*p[6]*jnp.sin(q[0]-q[2])
               + p[3]*p[4]*jnp.sin(q[0]-q[2])
               + 2*p[2]*p[5]*jnp.sin(q[0]-q[4])
               + p[2]*p[6]*jnp.sin(q[0]-q[4])
               + p[2]*p[4]*jnp.sin(q[0]-q[4])
              )

    # G(2,1)
    G2 = -p[0]*( p[10]*jnp.sin(q[1]-q[3])
                + p[9]*jnp.sin(q[1]-q[4])
                + p[2]*p[6]*jnp.sin(q[1]-q[4])
              )

    # G(3,1)
    G3 = -p[0]*jnp.sin(q[0]-q[2])*(2*p[3]*p[5] - p[10] + 2*p[3]*p[6] + p[3]*p[4])

    # G(4,1)
    G4 = p[10]*p[0]*jnp.sin(q[1]-q[3])

    # G(5,1)
    G5 = p[0]*( p[7]*jnp.cos(q[4])
               - p[8]*jnp.sin(q[4])
               + p[9]*jnp.sin(q[0]-q[4])
               + p[9]*jnp.sin(q[1]-q[4])
               - 2*p[2]*p[5]*jnp.sin(q[0]-q[4])
               - p[2]*p[6]*jnp.sin(q[0]-q[4])
               + p[2]*p[6]*jnp.sin(q[1]-q[4])
               - p[2]*p[4]*jnp.sin(q[0]-q[4])
             )

    # Assemble into a column vector (5x1)
    G = jnp.array([[G1],
                   [G2],
                   [G3],
                   [G4],
                   [G5]])
    return G


def De_matrix(q, params):
    """
    Compute the 7x7 matrix De given q and params.
    
    Inputs:
      q      : a length-5 JAX array, RELATIVE joint angles of the 5 links.
      params : a JAX array of parameters.
               
    Returns:
      De : a 7x7 JAX array.
    """
    p = params  # shorthand
        
    # Row 1:
    De11 = ( p[12] + p[13]
             + 2*(p[2]**2)*p[5] + (p[2]**2)*p[6]
             + 2*(p[3]**2)*p[5] + (p[2]**2)*p[4]
             + 2*(p[3]**2)*p[6] + (p[3]**2)*p[4]
             - 2*p[2]*p[9] - 2*p[3]*p[10]
             + 2*p[3]*p[9]*jnp.cos(q[2] - 2*q[0] + q[4])
             - 4*p[2]*p[3]*p[5]*jnp.cos(q[2] - 2*q[0] + q[4])
             - 2*p[2]*p[3]*p[6]*jnp.cos(q[2] - 2*q[0] + q[4])
             - 2*p[2]*p[3]*p[4]*jnp.cos(q[2] - 2*q[0] + q[4])
           )
    
    De12 = ( p[2]*p[10]*jnp.cos(q[0]+q[1]-q[3]-q[4])
             - p[3]*p[10]*jnp.cos(q[0]-q[1]-q[2]+q[3])
             + p[3]*p[9]*jnp.cos(q[0]+q[1]-q[2]-q[4])
             - (p[2]**2)*p[6]*jnp.cos(q[0]-q[1])
             - p[2]*p[9]*jnp.cos(q[0]-q[1])
             + p[2]*p[3]*p[6]*jnp.cos(q[0]+q[1]-q[2]-q[4])
           )
    
    De13 = ( 2*p[3]*p[10]
             - 2*(p[3]**2)*p[5]
             - 2*(p[3]**2)*p[6]
             - (p[3]**2)*p[4]
             - p[13]
             - p[3]*p[9]*jnp.cos(q[2]-2*q[0]+q[4])
             + 2*p[2]*p[3]*p[5]*jnp.cos(q[2]-2*q[0]+q[4])
             + p[2]*p[3]*p[6]*jnp.cos(q[2]-2*q[0]+q[4])
             + p[2]*p[3]*p[4]*jnp.cos(q[2]-2*q[0]+q[4])
           )
    
    De14 = p[10]*( p[3]*jnp.cos(q[0]-q[1]-q[2]+q[3])
                   - p[2]*jnp.cos(q[0]+q[1]-q[3]-q[4]) )
    
    De15 = ( 2*p[2]*p[9]
             - 2*(p[2]**2)*p[5]
             - (p[2]**2)*p[6]
             - (p[2]**2)*p[4]
             - p[12]
             + p[2]*p[7]*jnp.sin(q[0]-2*q[4])
             - p[3]*p[9]*jnp.cos(q[0]+q[1]-q[2]-q[4])
             + (p[2]**2)*p[6]*jnp.cos(q[0]-q[1])
             - p[3]*p[9]*jnp.cos(q[2]-2*q[0]+q[4])
             + p[3]*p[8]*jnp.cos(q[0]-q[2]+q[4])
             + p[3]*p[7]*jnp.sin(q[0]-q[2]+q[4])
             + p[2]*p[9]*jnp.cos(q[0]-q[1])
             - p[2]*p[8]*jnp.cos(q[0]-2*q[4])
             - p[2]*p[3]*p[6]*jnp.cos(q[0]+q[1]-q[2]-q[4])
             + 2*p[2]*p[3]*p[5]*jnp.cos(q[2]-2*q[0]+q[4])
             + p[2]*p[3]*p[6]*jnp.cos(q[2]-2*q[0]+q[4])
             + p[2]*p[3]*p[4]*jnp.cos(q[2]-2*q[0]+q[4])
           )
    
    De16 = ( p[10]*jnp.cos(q[0]-q[2])
             - p[9]*jnp.cos(q[0]-q[4])
             - 2*p[3]*p[5]*jnp.cos(q[0]-q[2])
             - 2*p[3]*p[6]*jnp.cos(q[0]-q[2])
             - p[3]*p[4]*jnp.cos(q[0]-q[2])
             + 2*p[2]*p[5]*jnp.cos(q[0]-q[4])
             + p[2]*p[6]*jnp.cos(q[0]-q[4])
             + p[2]*p[4]*jnp.cos(q[0]-q[4])
           )
    
    De17 = ( 2*p[3]*p[5]*jnp.sin(q[0]-q[2])
             - p[9]*jnp.sin(q[0]-q[4])
             - p[10]*jnp.sin(q[0]-q[2])
             + 2*p[3]*p[6]*jnp.sin(q[0]-q[2])
             + p[3]*p[4]*jnp.sin(q[0]-q[2])
             + 2*p[2]*p[5]*jnp.sin(q[0]-q[4])
             + p[2]*p[6]*jnp.sin(q[0]-q[4])
             + p[2]*p[4]*jnp.sin(q[0]-q[4])
           )
    
    # Row 2:
    De21 = De12
    De22 = p[12] + p[13] + (p[2]**2)*p[6] - 2*p[2]*p[10]*jnp.cos(q[3]-2*q[1]+q[4])
    De23 = ( p[3]*p[10]*jnp.cos(q[0]-q[1]-q[2]+q[3])
             - p[3]*p[9]*jnp.cos(q[0]+q[1]-q[2]-q[4])
             - p[2]*p[3]*p[6]*jnp.cos(q[0]+q[1]-q[2]-q[4])
           )
    De24 = p[2]*p[10]*jnp.cos(q[3]-2*q[1]+q[4]) - p[13]
    De25 = ( (p[2]**2)*p[6]*jnp.cos(q[0]-q[1])
             - (p[2]**2)*p[6]
             - p[2]*p[10]*jnp.cos(q[0]+q[1]-q[3]-q[4])
             - p[12]
             + p[2]*p[10]*jnp.cos(q[3]-2*q[1]+q[4])
             + p[2]*p[9]*jnp.cos(q[0]-q[1])
           )
    De26 = p[10]*jnp.cos(q[1]-q[3]) - p[9]*jnp.cos(q[1]-q[4]) - p[2]*p[6]*jnp.cos(q[1]-q[4])
    De27 = - p[10]*jnp.sin(q[1]-q[3]) - p[9]*jnp.sin(q[1]-q[4]) - p[2]*p[6]*jnp.sin(q[1]-q[4])
    
    # Row 3:
    De31 = De13
    De32 = De23
    De33 = p[13] + 2*(p[3]**2)*p[5] + 2*(p[3]**2)*p[6] + (p[3]**2)*p[4] - 2*p[3]*p[10]
    De34 = - p[3]*p[10]*jnp.cos(q[0]-q[1]-q[2]+q[3])
    De35 = ( p[3]*p[9]*jnp.cos(q[0]+q[1]-q[2]-q[4])
             + p[3]*p[9]*jnp.cos(q[2]-2*q[0]+q[4])
             - p[3]*p[8]*jnp.cos(q[0]-q[2]+q[4])
             - p[3]*p[7]*jnp.sin(q[0]-q[2]+q[4])
             + p[2]*p[3]*p[6]*jnp.cos(q[0]+q[1]-q[2]-q[4])
             - 2*p[2]*p[3]*p[5]*jnp.cos(q[2]-2*q[0]+q[4])
             - p[2]*p[3]*p[6]*jnp.cos(q[2]-2*q[0]+q[4])
             - p[2]*p[3]*p[4]*jnp.cos(q[2]-2*q[0]+q[4])
           )
    De36 = jnp.cos(q[0]-q[2])*(2*p[3]*p[5] - p[10] + 2*p[3]*p[6] + p[3]*p[4])
    De37 = - jnp.sin(q[0]-q[2])*(2*p[3]*p[5] - p[10] + 2*p[3]*p[6] + p[3]*p[4])
    
    # Row 4:
    De41 = p[10]*( p[3]*jnp.cos(q[0]-q[1]-q[2]+q[3]) - p[2]*jnp.cos(q[0]+q[1]-q[3]-q[4]) )
    De42 = p[2]*p[10]*jnp.cos(q[3]-2*q[1]+q[4]) - p[13]
    De43 = - p[3]*p[10]*jnp.cos(q[0]-q[1]-q[2]+q[3])
    De44 = p[13]
    De45 = p[2]*p[10]*( jnp.cos(q[0]+q[1]-q[3]-q[4]) - jnp.cos(q[3]-2*q[1]+q[4]) )
    De46 = - p[10]*jnp.cos(q[1]-q[3])
    De47 = p[10]*jnp.sin(q[1]-q[3])
    
    # Row 5:
    De51 = De15
    De52 = ( (p[2]**2)*p[6]*jnp.cos(q[0]-q[1])
             - (p[2]**2)*p[6]
             - p[2]*p[10]*jnp.cos(q[0]+q[1]-q[3]-q[4])
             - p[12]
             + p[2]*p[10]*jnp.cos(q[3]-2*q[1]+q[4])
             + p[2]*p[9]*jnp.cos(q[0]-q[1])
           )
    De53 = ( p[3]*p[9]*jnp.cos(q[0]+q[1]-q[2]-q[4])
             + p[3]*p[9]*jnp.cos(q[2]-2*q[0]+q[4])
             - p[3]*p[8]*jnp.cos(q[0]-q[2]+q[4])
             - p[3]*p[7]*jnp.sin(q[0]-q[2]+q[4])
             + p[2]*p[3]*p[6]*jnp.cos(q[0]+q[1]-q[2]-q[4])
             - 2*p[2]*p[3]*p[5]*jnp.cos(q[2]-2*q[0]+q[4])
             - p[2]*p[3]*p[6]*jnp.cos(q[2]-2*q[0]+q[4])
             - p[2]*p[3]*p[4]*jnp.cos(q[2]-2*q[0]+q[4])
           )
    De54 = p[2]*p[10]*( jnp.cos(q[0]+q[1]-q[3]-q[4]) - jnp.cos(q[3]-2*q[1]+q[4]) )
    De55 = ( 2*p[12] + p[11]
             + 2*(p[2]**2)*p[5] + 2*(p[2]**2)*p[6] + (p[2]**2)*p[4]
             - 2*p[2]*p[9] - 2*p[2]*p[7]*jnp.sin(q[0]-2*q[4])
             - 2*(p[2]**2)*p[6]*jnp.cos(q[0]-q[1]) - 2*p[2]*p[9]*jnp.cos(q[0]-q[1])
             + 2*p[2]*p[8]*jnp.cos(q[0]-2*q[4])
           )
    De56 = ( p[9]*jnp.cos(q[0]-q[4])
             - p[8]*jnp.cos(q[4])
             - p[7]*jnp.sin(q[4])
             - 0.5*p[6]*( p[2]*jnp.cos(q[0]-q[4]) - p[2]*jnp.cos(q[1]-q[4]) )
             + p[9]*jnp.cos(q[1]-q[4])
             - 2*p[2]*p[5]*jnp.cos(q[0]-q[4])
             - p[2]*p[4]*jnp.cos(q[0]-q[4])
           )
    De57 = ( p[7]*jnp.cos(q[4])
             - p[6]*( p[2]*jnp.sin(q[0]-q[4]) - p[2]*jnp.sin(q[1]-q[4]) )
             - p[8]*jnp.sin(q[4])
             + p[9]*jnp.sin(q[0]-q[4])
             + p[9]*jnp.sin(q[1]-q[4])
             - 2*p[2]*p[5]*jnp.sin(q[0]-q[4])
             - p[2]*p[4]*jnp.sin(q[0]-q[4])
           )
    
    # Row 6:
    De61 = De16  # symmetric with (1,6)
    De62 = p[10]*jnp.cos(q[1]-q[3]) - p[9]*jnp.cos(q[1]-q[4]) - p[2]*p[6]*jnp.cos(q[1]-q[4])
    De63 = jnp.cos(q[0]-q[2])*(2*p[3]*p[5] - p[10] + 2*p[3]*p[6] + p[3]*p[4])
    De64 = - p[10]*jnp.cos(q[1]-q[3])
    De65 = ( p[9]*jnp.cos(q[0]-q[4])
             - p[8]*jnp.cos(q[4])
             - p[7]*jnp.sin(q[4])
             - 0.5*p[6]*( p[2]*jnp.cos(q[0]-q[4]) - p[2]*jnp.cos(q[1]-q[4]) )
             + p[9]*jnp.cos(q[1]-q[4])
             - 2*p[2]*p[5]*jnp.cos(q[0]-q[4])
             - p[2]*p[4]*jnp.cos(q[0]-q[4])
           )
    De66 = 2*p[5] + 2*p[6] + p[4]
    De67 = 0.0
    
    # Row 7:
    De71 = De17
    De72 = - p[10]*jnp.sin(q[1]-q[3]) - p[9]*jnp.sin(q[1]-q[4]) - p[2]*p[6]*jnp.sin(q[1]-q[4])
    De73 = - jnp.sin(q[0]-q[2])*(2*p[3]*p[5] - p[10] + 2*p[3]*p[6] + p[3]*p[4])
    De74 = p[10]*jnp.sin(q[1]-q[3])
    De75 = ( p[7]*jnp.cos(q[4])
             - 0.5*p[6]*( p[2]*jnp.sin(q[0]-q[4]) - p[2]*jnp.sin(q[1]-q[4]) )
             - p[8]*jnp.sin(q[4])
             + p[9]*jnp.sin(q[0]-q[4])
             + p[9]*jnp.sin(q[1]-q[4])
             - 2*p[2]*p[5]*jnp.sin(q[0]-q[4])
             - p[2]*p[4]*jnp.sin(q[0]-q[4])
           )
    De76 = 0.0
    De77 = 2*p[5] + 2*p[6] + p[4]
    
    # Assemble the 7x7 matrix row by row.
    De = jnp.array([
        [De11, De12, De13, De14, De15, De16, De17],
        [De21, De22, De23, De24, De25, De26, De27],
        [De31, De32, De33, De34, De35, De36, De37],
        [De41, De42, De43, De44, De45, De46, De47],
        [De51, De52, De53, De54, De55, De56, De57],
        [De61, De62, De63, De64, De65, De66, De67],
        [De71, De72, De73, De74, De75, De76, De77]
    ])
    
    return De


def pos_hip(q, params):
    """
    Compute the position of the hip joint in the global frame.

    Inputs:
      q      : a length-5 JAX array. RELATIVE joint angles.
      params : a JAX array containing the parameters.
      [g, L_torso, L_fem, L_tib, M_torso, M_fem, M_tib,
       MY_torso, MZ_torso, MZ_fem, MZ_tib, XX_torso, XX_fem, XX_tib]
               
    Returns:
      p_hip : a 2x1 JAX array representing the position of the hip joint.
    """
    
    T = jnp.array([
        [1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1]
    ])
    
    q_abs = T@q
    
    q_fem1  = q_abs[0]
    q_tib1  = q_abs[2];
    
    L_fem, L_tib = params[2], params[3]
    
    p_hip  = jnp.array([[L_fem*jnp.sin(q_fem1) + L_tib*jnp.sin(q_tib1)],
                     [-L_fem*jnp.cos(q_fem1) - L_tib*jnp.cos(q_tib1)]])
    
    return p_hip


def limb_velocities(x, params):
  
    q, dq = x[0:5], x[5:10]
    
    T = jnp.array([
        [1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1]
    ])
    
    q_abs = T@q
    dq_abs = T@dq
    
    q_fem1  = q_abs[0]
    q_fem2  = q_abs[1]
    q_tib1  = q_abs[2]
    q_tib2  = q_abs[3]
    q_torso = q_abs[4]  
    
    dq_fem1  = dq_abs[0]
    dq_fem2  = dq_abs[1]
    dq_tib1  = dq_abs[2]
    dq_tib2  = dq_abs[3]
    dq_torso = dq_abs[4]
       
    pos_hip_partial = partial(pos_hip, params=params)
    pos_knee1_partial = partial(pos_knee1, params=params) 
    pos_knee2_partial = partial(pos_knee2, params=params)
    v_hip = jax.jacrev(pos_hip_partial)(q) @ dq
    # v_hip = jax.jacobian(p_hip)(q) @ dq
    v_knee1 = jax.jacobian(pos_knee1_partial)(q) @ dq
    v_knee2 = jax.jacobian(pos_knee2_partial)(q) @ dq

    # Compute relative angular velocities:
    R_torso = jnp.array([[jnp.cos(q_torso), -jnp.sin(q_torso)],
                        [jnp.sin(q_torso),  jnp.cos(q_torso)]])
    v_torso = R_torso.T @ v_hip * dq_torso

    R_fem1 = jnp.array([[jnp.cos(q_fem1), -jnp.sin(q_fem1)],
                        [jnp.sin(q_fem1),  jnp.cos(q_fem1)]])
    v_fem1 = R_fem1.T @ v_hip * dq_fem1

    R_fem2 = jnp.array([[jnp.cos(q_fem2), -jnp.sin(q_fem2)],
                        [jnp.sin(q_fem2),  jnp.cos(q_fem2)]])
    v_fem2 = R_fem2.T @ v_hip * dq_fem2

    R_tib1 = jnp.array([[jnp.cos(q_tib1), -jnp.sin(q_tib1)],
                        [jnp.sin(q_tib1),  jnp.cos(q_tib1)]])
    v_tib1 = R_tib1.T @ v_knee1 * dq_tib1

    R_tib2 = jnp.array([[jnp.cos(q_tib2), -jnp.sin(q_tib2)],
                        [jnp.sin(q_tib2),  jnp.cos(q_tib2)]])
    v_tib2 = R_tib2.T @ v_knee2 * dq_tib2
    
    return v_hip.flatten(), v_knee1.flatten(), v_knee2.flatten(), v_tib1.flatten(), v_tib2.flatten()


def limb_positions(q, params):
    """
    Compute the position of the limbs in the global frame.

    Inputs:
      q = [q_fem1, q_fem2, q_tib1, q_tib2, q_torso], RELATIVE joint angles.
      
      params : a JAX array containing the parameters.
      params = [g, L_torso, L_fem, L_tib, M_torso, M_fem, M_tib,
       MY_torso, MZ_torso, MZ_fem, MZ_tib, XX_torso, XX_fem, XX_tib]
               
    Returns:
      pos_knee1: the positions of the knee1 joint.
    """
    
    T = jnp.array([
        [1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1]
    ])
    
    q_abs = T@q

    q_fem1  = q_abs[0]    
    q_fem2  = q_abs[1]  
    q_tib2  = q_abs[3]
    
    L_fem = params[2]
    L_tib = params[3]
    
    p_hip  = pos_hip(q, params)
    
    p_knee1 = p_hip + jnp.array([[-L_fem*jnp.sin(q_fem1)],
                                 [L_fem*jnp.cos(q_fem1)]])
    
    p_knee2 = p_hip + jnp.array([[-L_fem*jnp.sin(q_fem2)],
                                 [L_fem*jnp.cos(q_fem2)]])
    
    p_tib2 = p_knee2 + jnp.array([[-L_tib*jnp.sin(q_tib2)],
                                  [L_tib*jnp.cos(q_tib2)]])
    
    return p_hip.flatten(), p_knee1.flatten(), p_knee2.flatten(), p_tib2.flatten()


def pos_knee1(q, params):
    q_abs = T@q
    q_fem1  = q_abs[0]    
    
    L_fem = params[2]
    p_hip  = pos_hip(q, params)
    p_knee1 = p_hip + jnp.array([[-L_fem*jnp.sin(q_fem1)],
                                 [L_fem*jnp.cos(q_fem1)]])
    
    return p_knee1
  

def pos_knee2(q, params):
    q_abs = T@q
    q_fem2  = q_abs[1]    
    
    L_fem = params[2]
    p_hip  = pos_hip(q, params)
    p_knee2 = p_hip + jnp.array([[-L_fem*jnp.sin(q_fem2)],
                                 [L_fem*jnp.cos(q_fem2)]])
    
    return p_knee2
  
  
def pos_tib2(q, params):
    q_abs = T@q
    q_tib2  = q_abs[3]    
    
    L_fem = params[2]
    L_tib = params[3]
    p_hip  = pos_hip(q, params)
    p_knee2 = p_hip + jnp.array([[-L_fem*jnp.sin(q[1])],
                                 [L_fem*jnp.cos(q[1])]])
    p_tib2 = p_knee2 + jnp.array([[-L_tib*jnp.sin(q_tib2)],
                                  [L_tib*jnp.cos(q_tib2)]])
    
    return p_tib2.flatten()
  

def vel_hip(x, params):
    q, dq = x[0:5], x[5:10]
    p_hip = pos_hip(q, params)
    
    v_hip = jax.jacobian(p_hip, q) @ dq

    return v_hip


def draw_5link(x, params, ax=None, legend=True):
    """Draw the 5-link biped model in the given configuration.
       Assuming here the stance foot is at the origin.
    Args:
        x (jax.array): x=[q,dq]
        params (list): parameters of the model
    """
    
    if ax is None:
      fig, ax = plt.subplots()
    
    q, dq = x[0:5], x[5:10]
    
    # Draw ground
    buffer = 5
    ground = Line2D([-buffer, buffer], [0, 0], color='k', linewidth=2)
    ax.add_line(ground)
    
    p_hip, p_knee1, p_knee2, p_tib2 = limb_positions(q, params)
    
    # Draw fem 1
    fem1, = ax.plot([p_knee1[0], p_hip[0]], [p_knee1[1], p_hip[1]], color='r', linewidth=2, label='femur 1')

    # Draw fem 2
    fem2, = ax.plot([p_knee2[0], p_hip[0]], [p_knee2[1], p_hip[1]], color='g', linewidth=2, label='femur 2')

    # Draw tib 1
    tib1, = ax.plot([float(p_knee1[0]), 0.0], [float(p_knee1[1]), 0.0], color='b', linewidth=2, label='tib 1')
    
    # Draw tib 2
    tib2, = ax.plot([p_knee2[0], p_tib2[0]], [p_knee2[1], p_tib2[1]], color='c', linewidth=2, label='tib 2')
    
    # Draw torso
    L_torso = params[1]
    q_torso = q[4]
    p_torso_tip = p_hip + jnp.array([[-L_torso*jnp.sin(q_torso)],
                                     [L_torso*jnp.cos(q_torso)]]).flatten()
    torso, = ax.plot([p_hip[0], p_torso_tip[0]], [p_hip[1], p_torso_tip[1]], color='k', linewidth=2, label='torso')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.5, 2)
    
    if legend:
      ax.legend()
    
    plt.draw()
    # plt.show()
    

def fxgu_5link(t, x, u, params):
    q = x[0:5]
    dq = x[5:10]
    B = B_matrix()
    C = C_matrix(q, dq, params)
    D = D_matrix(q, params)
    G = G_vector(q, params)
    
    # Compute Fx and Gx
    Fx = np.linalg.solve(D, -C @ x[5:10] - G.flatten())
    Gx = np.linalg.solve(D, B)
    
    # Compute state derivatives
    dx = np.zeros_like(x)
    dx[:5] = x[5:10]
    dx[5:10] = Fx + Gx @ u
    
    return dx


def foot_touching_events(t, x, params):
    _, _, _, p_tib2 = limb_positions(x[0:5], params)
    
    return p_tib2[1]
foot_touching_events.terminal = True 
foot_touching_events.direction = -1


def E_matrix(q_event, params):
  # Compute E matrix
  pos_tib2_partial = partial(pos_tib2, params=params)
  first_part = jax.jacrev(pos_tib2_partial)(q_event)
  second_part = jnp.eye(2)
  
  E = jnp.hstack([first_part, second_part])
  
  return E


def resetmap_5link(x_event, params):
    q_event = x_event[0:5]
    
    E2 = E_matrix(q_event, params)
    De = De_matrix(q_event, params)
    
    # Solve for the transition using the equation from Grizzle's paper
    A = jnp.block([[De, -E2.T], [E2, jnp.zeros((2, 2))]])
    b = jnp.hstack([De @ jnp.hstack([x_event[5:10], [0, 0]]), [0, 0]])
    tmp_vec = jnp.linalg.solve(A, b)

    # Update state vector after impact
    x_new = jnp.zeros(10)
    
    x_new = x_new.at[0].set(x_event[1])
    x_new = x_new.at[1].set(x_event[0])
    x_new = x_new.at[2].set(x_event[3])
    x_new = x_new.at[3].set(x_event[2])
    x_new = x_new.at[4].set(x_event[4])
    
    x_new = x_new.at[5].set(tmp_vec[1])
    x_new = x_new.at[6].set(tmp_vec[0])
    x_new = x_new.at[7].set(tmp_vec[3])
    x_new = x_new.at[8].set(tmp_vec[2])
    x_new = x_new.at[9].set(tmp_vec[4])
    
    return x_new
  

def integrate_fxgu(x0, u, params):
    print("Integrating the 5-link model with the given initial state.")
    tstart = 0.0
    tfinal = 0.2
    
    num_t_eval = 100
    t_eval = np.linspace(tstart, tfinal, num_t_eval)
    t_span = [tstart, tfinal]
    
    event_func = partial(foot_touching_events, params=params)
    
    options = {
        'rtol': 1e-5,
        'atol': 1e-6,
        'events': event_func
    }
        
    # Solve until the first terminal event
    sol = solve_ivp(
       fun=lambda t, x: fxgu_5link(t, x, u, params),
        t_span=t_span,
        t_eval=t_eval,
        y0=x0,
        method='RK23', 
        rtol=options['rtol'],
        atol=options['atol'],
        events=options['events']
    )
    
    t = sol.t
    x_trj = sol.y.T
    
    if sol.t_events:
      te = sol.t_events[0][0]
      xe = sol.y_events[0][0]
      
      print("Swing foot guard time:")
      print(te)
      
      print("Swing foot guard state:")
      print(xe)
      
      print("Swing foot guard function value:")
      print(foot_touching_events(te, xe, params))

      # Create a mask for all integrated times strictly less than te.
      mask = t <= te

      # Select all times and states up until the event
      t = sol.t[mask]
      x_trj = x_trj[mask]
    
    return x_trj


def plot_states(x_trj):

    n_time_steps = x_trj.shape[0]
    
    p_hip = np.zeros((n_time_steps, 2))
    v_hip = np.zeros((n_time_steps, 2))
    
    p_fem1 = np.zeros((n_time_steps, 2))
    v_fem1 = np.zeros((n_time_steps, 2))
    
    p_fem2 = np.zeros((n_time_steps, 2))
    v_fem2 = np.zeros((n_time_steps, 2))
    
    p_tib1 = np.zeros((n_time_steps, 2))
    v_tib1 = np.zeros((n_time_steps, 2))
    
    p_tib2 = np.zeros((n_time_steps, 2))
    v_tib2 = np.zeros((n_time_steps, 2))
    
    for i_t in range(n_time_steps):
      p_hip[i_t], p_fem1[i_t], p_fem2[i_t], p_tib2[i_t] = limb_positions(x_trj[i_t, 0:5], params)  
      v_hip[i_t], v_fem1[i_t], v_fem2[i_t], v_tib1[i_t], v_tib2[i_t] = limb_velocities(x_trj[i_t], params)
      
    fig = plt.figure(figsize=(16, 9))
    ax1 = plt.subplot(5, 2, 1)
    ax1.grid(True)
    ax1.plot(np.arange(n_time_steps), p_hip[:, 0], label='p_hip_x')
    ax1.plot(np.arange(n_time_steps), p_hip[:, 1], label='p_hip_y')  
    
    ax2 = plt.subplot(5, 2, 2)
    ax2.grid(True)
    ax2.plot(np.arange(n_time_steps), v_hip[:, 0], label='v_hip_x')
    ax2.plot(np.arange(n_time_steps), v_hip[:, 1], label='v_hip_y')
    
    ax3 = plt.subplot(5, 2, 3)
    ax3.grid(True)
    ax3.plot(np.arange(n_time_steps), p_fem1[:, 0], label='p_fem1_x')
    ax3.plot(np.arange(n_time_steps), p_fem1[:, 1], label='p_fem1_y')
    
    ax4 = plt.subplot(5, 2, 4)  
    ax4.grid(True)
    ax4.plot(np.arange(n_time_steps), v_fem1[:, 0], label='v_fem1_x')
    ax4.plot(np.arange(n_time_steps), v_fem1[:, 1], label='v_fem1_y')
    
    ax5 = plt.subplot(5, 2, 5)
    ax5.grid(True)
    ax5.plot(np.arange(n_time_steps), p_fem2[:, 0], label='p_fem2_x')
    ax5.plot(np.arange(n_time_steps), p_fem2[:, 1], label='p_fem2_y')
    
    ax6 = plt.subplot(5, 2, 6)
    ax6.grid(True)
    ax6.plot(np.arange(n_time_steps), v_fem2[:, 0], label='v_fem2_x')
    ax6.plot(np.arange(n_time_steps), v_fem2[:, 1], label='v_fem2_y')
    
    ax7 = plt.subplot(5, 2, 7)
    ax7.grid(True)
    ax7.plot(np.arange(n_time_steps), p_tib2[:, 0], label='p_tib2_x')
    ax7.plot(np.arange(n_time_steps), p_tib2[:, 1], label='p_tib2_y')
    
    ax8 = plt.subplot(5, 2, 8)
    ax8.grid(True)
    ax8.plot(np.arange(n_time_steps), v_tib2[:, 0], label='v_tib2_x')
    ax8.plot(np.arange(n_time_steps), v_tib2[:, 1], label='v_tib2_y')
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()
    ax7.legend()
    ax8.legend()
    
    plt.tight_layout()
    plt.show()
  

def animate(x_trj, step=1):
    """Animate a trajectory of states for the 5-link biped model.

    Args:
        x_trj (np.array): The ABSOLUTE angles of the joints. 
        step (int, the simulation frequency): Defaults to 1.
    """
    fig, ax = plt.subplots()
    
    plt.ion()
    
    n_time_steps = x_trj.shape[0]
    
    for i in range(0, n_time_steps, step):
        x_i = x_trj[i, :]
        draw_5link(x_i, params, ax, legend=False)
        plt.pause(0.0001)
        ax.clear()  
    plt.ioff()
    
    draw_5link(x_trj[-1], params, ax, legend=False)
    plt.show()


if __name__ == '__main__':
    g = 9.81
    L_torso = 0.63
    L_fem = 0.4
    L_tib = 0.4
    
    M_torso = 12.0
    M_fem = 6.8
    M_tib = 3.2
    
    MY_torso = 0.01
    MZ_torso = 0.2
    MZ_fem = 0.11
    MZ_tib = 0.24
    
    XX_torso = 1.33
    XX_fem = 0.47
    XX_tib = 0.2
    
    params = [g, L_torso, L_fem, L_tib, M_torso, M_fem, M_tib,
              MY_torso, MZ_torso, MZ_fem, MZ_tib, XX_torso, XX_fem, XX_tib]
    
    q_abs = jnp.array([200, 160, 170, 130, -10])/180*jnp.pi
    dq_abs = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    T = jnp.array([
        [1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1]
    ])
    
    q_rel = jnp.linalg.solve(T, q_abs)
    dq_rel = dq_abs
    
    x = jnp.concatenate([q_rel, dq_rel])    
    
#     draw_5link(x, params)

    # u0: The torque between hip and fem1
    # u1: The torque between hip and fem2
    # u2: The torque between fem1 and tib1
    # u3: The torque between fem2 and tib2
    
    u1 = np.array([-150.0, 100.0, 100.0, -25.0])

    x_trj_1 = integrate_fxgu(x0=x, u=u1, params=params)
    # plot_states(x_trj_1)
    
    # animate(x_trj)
    
    # Apply the reset map
    x_new = resetmap_5link(x_trj_1[-1], params=params)
    
    u2 = np.array([-500.0, 200.0, 100.0, 0.0])
    x_trj_2 = integrate_fxgu(x0=x_new, u=u2, params=params)
    # plot_states(x_trj_2)
    
    # animate(x_trj_2)
    
    x_trj = jnp.concatenate([x_trj_1, x_trj_2])
    
    plot_states(x_trj)
    animate(x_trj)
    
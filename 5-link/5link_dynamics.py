import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from matplotlib.lines import Line2D

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


def pos_hip(q, params):
    """
    Compute the position of the hip joint in the global frame.

    Inputs:
      q      : a length-5 JAX array.
      params : a JAX array containing the parameters.
      [g, L_torso, L_fem, L_tib, M_torso, M_fem, M_tib,
       MY_torso, MZ_torso, MZ_fem, MZ_tib, XX_torso, XX_fem, XX_tib]
               
    Returns:
      pos_hip : a 2x1 JAX array representing the position of the hip joint.
    """
    
    q_fem1  = q[0]
    q_tib1  = q[2];
    
    L_fem, L_tib = params[2], params[3]
    
    pos_hip  = jnp.array([[L_fem*jnp.sin(q_fem1) + L_tib*jnp.sin(q_tib1)],
                     [-L_fem*jnp.cos(q_fem1) - L_tib*jnp.cos(q_tib1)]])
    
    return pos_hip


def limb_positions(x, params):
    """
    Compute the position of the limbs in the global frame.

    Inputs:
      x      : a length-10 JAX array. x = [q,dq]
      q = [q_fem1, q_fem2, q_tib1, q_tib2, q_torso].
      
      params : a JAX array containing the parameters.
      params = [g, L_torso, L_fem, L_tib, M_torso, M_fem, M_tib,
       MY_torso, MZ_torso, MZ_fem, MZ_tib, XX_torso, XX_fem, XX_tib]
               
    Returns:
      pos_knee1: the positions of the knee1 joint.
    """
    q = x[0:5]
    
    q_fem1  = q[0]    
    q_fem2  = q[1]  
    q_tib1  = q[2]
    q_tib2  = q[3]
    q_torso = q[4]
    
    L_torso = params[1]
    L_fem = params[2]
    L_tib = params[3]
    
    p_hip  = pos_hip(q, params)
    
    p_knee1 = p_hip + jnp.array([[-L_fem*jnp.sin(q_fem1)],
                                 [L_fem*jnp.cos(q_fem1)]])
    
    p_knee2 = p_hip + jnp.array([[-L_fem*jnp.sin(q_fem2)],
                                 [L_fem*jnp.cos(q_fem2)]])
    
    p_tib2 = p_knee2 + jnp.array([[-L_tib*jnp.sin(q_tib2)],
                                  [L_tib*jnp.cos(q_tib2)]])
    
    return p_hip, p_knee1, p_knee2, p_tib2


def vel_hip(x, params):
    q, dq = x[0:5], x[5:10]
    p_hip = pos_hip(q, params)
    
    v_hip = jax.jacobian(p_hip, q) @ dq

    return v_hip


def draw_5link(x, params):
    """Draw the 5-link biped model in the given configuration.
       Assuming here the stance foot is at the origin.
    Args:
        x (jax.array): x=[q,dq]
        params (list): parameters of the model
    """
    fig, ax = plt.subplots()
    
    q, dq = x[0:5], x[5:10]
    
    # Draw ground
    buffer = 5
    ground = Line2D([-buffer, buffer], [0, 0], color='k', linewidth=2)
    ax.add_line(ground)
    
    p_hip, p_knee1, p_knee2, p_tib2 = limb_positions(x, params)
    
    # Draw fem 1
    fem1, = ax.plot([p_knee1[0], p_hip[0]], [p_knee1[1], p_hip[1]], color='r', linewidth=2, label='femur 1')

    # Draw fem 2
    fem2, = ax.plot([p_knee2[0], p_hip[0]], [p_knee2[1], p_hip[1]], color='g', linewidth=2, label='femur 2')

    # Draw tib 1
    tib1, = ax.plot([p_knee1[0], jnp.array([0])], [p_knee1[1], jnp.array([0])], color='b', linewidth=2, label='tib 1')
    
    # Draw tib 2
    tib2, = ax.plot([p_knee2[0], p_tib2[0]], [p_knee2[1], p_tib2[1]], color='c', linewidth=2, label='tib 2')
    
    # Draw torso
    L_torso = params[1]
    q_torso = q[4]
    p_torso_tip = p_hip + jnp.array([[-L_torso*jnp.sin(q_torso)],
                                     [L_torso*jnp.cos(q_torso)]])
    torso, = ax.plot([p_hip[0], p_torso_tip[0]], [p_hip[1], p_torso_tip[1]], color='k', linewidth=2, label='torso')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.5, 2)
    ax.legend()
    
    plt.draw()
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
    MZ_torso = 0.4
    MZ_fem = 0.5
    MZ_tib = 0.5
    
    XX_torso = 1.33
    XX_fem = 0.47
    XX_tib = 0.2
    
    params = [g, L_torso, L_fem, L_tib, M_torso, M_fem, M_tib,
              MY_torso, MZ_torso, MZ_fem, MZ_tib, XX_torso, XX_fem, XX_tib]
    
    q = jnp.array([200, 160, 170, 170, -30])/180*jnp.pi
    dq = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    x = jnp.concatenate([q, dq])    
    
    draw_5link(x, params)
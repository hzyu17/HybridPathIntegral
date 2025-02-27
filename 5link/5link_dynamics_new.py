import jax.numpy as jnp
import jax
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import numpy as np

def D_matrix(q):
    # Unpack the 7 generalized coordinates (MATLAB q(1) ... q(7))
    xbar   = q[0]
    zbar   = q[1]
    rotY   = q[2]
    q1R    = q[3]
    q2R    = q[4]
    q1L    = q[5]
    q2L    = q[6]

    # Construct the 7x7 mass-inertia matrix.
    # (Each MATLAB elementwise operation “.*” and “.^” has been replaced with
    # the corresponding Python jnp operations.)
    Mmat = jnp.array([
        [  # Row 1
          12*jnp.cos(rotY)**2 + 12*jnp.sin(rotY)**2 +
          (34/5)*((jnp.cos(rotY)*jnp.sin(q1L) + jnp.cos(q1L)*jnp.sin(rotY))**2) +
          (34/5)*((jnp.cos(rotY)*jnp.sin(q1R) + jnp.cos(q1R)*jnp.sin(rotY))**2) +
          (34/5)*((jnp.cos(q1L)*jnp.cos(rotY) - jnp.sin(q1L)*jnp.sin(rotY))**2) +
          (34/5)*((jnp.cos(q1R)*jnp.cos(rotY) - jnp.sin(q1R)*jnp.sin(rotY))**2) +
          (16/5)*((jnp.cos(rotY)*(jnp.cos(q1L)*jnp.cos(q2L) - jnp.sin(q1L)*jnp.sin(q2L)) +
                   (-jnp.cos(q2L)*jnp.sin(q1L) - jnp.cos(q1L)*jnp.sin(q2L))*jnp.sin(rotY))**2) +
          (16/5)*((jnp.cos(rotY)*(jnp.cos(q2L)*jnp.sin(q1L) + jnp.cos(q1L)*jnp.sin(q2L)) +
                   (jnp.cos(q1L)*jnp.cos(q2L) - jnp.sin(q1L)*jnp.sin(q2L))*jnp.sin(rotY))**2) +
          (16/5)*((jnp.cos(rotY)*(jnp.cos(q1R)*jnp.cos(q2R) - jnp.sin(q1R)*jnp.sin(q2R)) +
                   (-jnp.cos(q2R)*jnp.sin(q1R) - jnp.cos(q1R)*jnp.sin(q2R))*jnp.sin(rotY))**2) +
          (16/5)*((jnp.cos(rotY)*(jnp.cos(q2R)*jnp.sin(q1R) + jnp.cos(q1R)*jnp.sin(q2R)) +
                   (jnp.cos(q1R)*jnp.cos(q2R) - jnp.sin(q1R)*jnp.sin(q2R))*jnp.sin(rotY))**2)
        ],
        [  # Row 2
          (34/5)*(( -jnp.cos(rotY)*jnp.sin(q1L) - jnp.cos(q1L)*jnp.sin(rotY)) *
                   (jnp.cos(q1L)*jnp.cos(rotY) - jnp.sin(q1L)*jnp.sin(rotY))) +
          (34/5)*((  jnp.cos(rotY)*jnp.sin(q1L) + jnp.cos(q1L)*jnp.sin(rotY)) *
                   (jnp.cos(q1L)*jnp.cos(rotY) - jnp.sin(q1L)*jnp.sin(rotY))) +
          (34/5)*(( -jnp.cos(rotY)*jnp.sin(q1R) - jnp.cos(q1R)*jnp.sin(rotY)) *
                   (jnp.cos(q1R)*jnp.cos(rotY) - jnp.sin(q1R)*jnp.sin(rotY))) +
          (34/5)*((  jnp.cos(rotY)*jnp.sin(q1R) + jnp.cos(q1R)*jnp.sin(rotY)) *
                   (jnp.cos(q1R)*jnp.cos(rotY) - jnp.sin(q1R)*jnp.sin(rotY))) +
          (16/5)*(( jnp.cos(rotY)*(jnp.cos(q1L)*jnp.cos(q2L) - jnp.sin(q1L)*jnp.sin(q2L)) +
                    (-jnp.cos(q2L)*jnp.sin(q1L) - jnp.cos(q1L)*jnp.sin(q2L))*jnp.sin(rotY)) *
                   ( jnp.cos(rotY)*(-jnp.cos(q2L)*jnp.sin(q1L) - jnp.cos(q1L)*jnp.sin(q2L)) -
                    (jnp.cos(q1L)*jnp.cos(q2L) - jnp.sin(q1L)*jnp.sin(q2L))*jnp.sin(rotY))) +
          (16/5)*(( jnp.cos(rotY)*(jnp.cos(q1L)*jnp.cos(q2L) - jnp.sin(q1L)*jnp.sin(q2L)) +
                    (-jnp.cos(q2L)*jnp.sin(q1L) - jnp.cos(q1L)*jnp.sin(q2L))*jnp.sin(rotY)) *
                   ( jnp.cos(rotY)*(jnp.cos(q2L)*jnp.sin(q1L) + jnp.cos(q1L)*jnp.sin(q2L)) +
                    (jnp.cos(q1L)*jnp.cos(q2L) - jnp.sin(q1L)*jnp.sin(q2L))*jnp.sin(rotY))) +
          (16/5)*(( jnp.cos(rotY)*(jnp.cos(q1R)*jnp.cos(q2R) - jnp.sin(q1R)*jnp.sin(q2R)) +
                    (-jnp.cos(q2R)*jnp.sin(q1R) - jnp.cos(q1R)*jnp.sin(q2R))*jnp.sin(rotY)) *
                   ( jnp.cos(rotY)*(-jnp.cos(q2R)*jnp.sin(q1R) - jnp.cos(q1R)*jnp.sin(q2R)) -
                    (jnp.cos(q1R)*jnp.cos(q2R) - jnp.sin(q1R)*jnp.sin(q2R))*jnp.sin(rotY))) +
          (16/5)*(( jnp.cos(rotY)*(jnp.cos(q1R)*jnp.cos(q2R) - jnp.sin(q1R)*jnp.sin(q2R)) +
                    (-jnp.cos(q2R)*jnp.sin(q1R) - jnp.cos(q1R)*jnp.sin(q2R))*jnp.sin(rotY)) *
                   ( jnp.cos(rotY)*(jnp.cos(q2R)*jnp.sin(q1R) + jnp.cos(q1R)*jnp.sin(q2R)) +
                    (jnp.cos(q1R)*jnp.cos(q2R) - jnp.sin(q1R)*jnp.sin(q2R))*jnp.sin(rotY)))
        ],
        [  # Row 3
          (72/25)*jnp.cos(rotY) +
          (34/5)*((11/100)*jnp.cos(q1L)**2 + (11/100)*jnp.sin(q1L)**2)*
                  (jnp.cos(q1L)*jnp.cos(rotY) - jnp.sin(q1L)*jnp.sin(rotY)) +
          (34/5)*((11/100)*jnp.cos(q1R)**2 + (11/100)*jnp.sin(q1R)**2)*
                  (jnp.cos(q1R)*jnp.cos(rotY) - jnp.sin(q1R)*jnp.sin(rotY)) +
          (16/5)*(
              (((2/5)*(1 - jnp.cos(q2L)) + (16/25)*jnp.cos(q2L))*jnp.sin(q1L) +
               (6/25)*jnp.cos(q1L)*jnp.sin(q2L))*
              (jnp.cos(q2L)*jnp.sin(q1L) + jnp.cos(q1L)*jnp.sin(q2L)) +
              (jnp.cos(q1L)*jnp.cos(q2L) - jnp.sin(q1L)*jnp.sin(q2L))*
              (jnp.cos(q1L)*((2/5)*(1 - jnp.cos(q2L)) + (16/25)*jnp.cos(q2L)) -
               (6/25)*jnp.sin(q1L)*jnp.sin(q2L))
          )*(
              jnp.cos(rotY)*(jnp.cos(q1L)*jnp.cos(q2L) - jnp.sin(q1L)*jnp.sin(q2L)) +
              (-jnp.cos(q2L)*jnp.sin(q1L) - jnp.cos(q1L)*jnp.sin(q2L))*jnp.sin(rotY)
          ) +
          (16/5)*(
              (((2/5)*(1 - jnp.cos(q2R)) + (16/25)*jnp.cos(q2R))*jnp.sin(q1R) +
               (6/25)*jnp.cos(q1R)*jnp.sin(q2R))*
              (jnp.cos(q2R)*jnp.sin(q1R) + jnp.cos(q1R)*jnp.sin(q2R)) +
              (jnp.cos(q1R)*jnp.cos(q2R) - jnp.sin(q1R)*jnp.sin(q2R))*
              (jnp.cos(q1R)*((2/5)*(1 - jnp.cos(q2R)) + (16/25)*jnp.cos(q2R)) -
               (6/25)*jnp.sin(q1R)*jnp.sin(q2R))
          )*(
              jnp.cos(rotY)*(jnp.cos(q1R)*jnp.cos(q2R) - jnp.sin(q1R)*jnp.sin(q2R)) +
              (-jnp.cos(q2R)*jnp.sin(q1R) - jnp.cos(q1R)*jnp.sin(q2R))*jnp.sin(rotY)
          )
        ],
        [  # Row 4
          (187/250)*(jnp.cos(q1R)*jnp.cos(rotY) - jnp.sin(q1R)*jnp.sin(rotY)) +
          (16/5)*(((2/5)*(1 - jnp.cos(q2R)) + (16/25)*jnp.cos(q2R))*jnp.cos(q2R) +
                   (6/25)*jnp.sin(q2R)**2)*
          (jnp.cos(rotY)*(jnp.cos(q1R)*jnp.cos(q2R) - jnp.sin(q1R)*jnp.sin(q2R)) +
           (-jnp.cos(q2R)*jnp.sin(q1R) - jnp.cos(q1R)*jnp.sin(q2R))*jnp.sin(rotY)) +
          (16/5)*(((2/5)*(1 - jnp.cos(q2R)) + (16/25)*jnp.cos(q2R))*jnp.sin(q2R) +
                   (-6/25)*jnp.cos(q2R)*jnp.sin(q2R))*
          (jnp.cos(rotY)*(jnp.cos(q1R)*jnp.cos(q2R) - jnp.sin(q1R)*jnp.sin(q2R)) +
           (-jnp.cos(q2R)*jnp.sin(q1R) - jnp.cos(q1R)*jnp.sin(q2R))*jnp.sin(rotY)),
          
          (187/250)*((-jnp.cos(rotY)*jnp.sin(q1R)) - jnp.cos(q1R)*jnp.sin(rotY)) +
          (16/5)*(((2/5)*(1 - jnp.cos(q2R)) + (16/25)*jnp.cos(q2R))*jnp.sin(q2R) +
                   (-6/25)*jnp.cos(q2R)*jnp.sin(q2R))*
          (jnp.cos(rotY)*(jnp.cos(q1R)*jnp.cos(q2R) - jnp.sin(q1R)*jnp.sin(q2R)) +
           (-jnp.cos(q2R)*jnp.sin(q1R) - jnp.cos(q1R)*jnp.sin(q2R))*jnp.sin(rotY)) +
          (16/5)*(((2/5)*(1 - jnp.cos(q2R)) + (16/25)*jnp.cos(q2R))*jnp.cos(q2R) +
                   (6/25)*jnp.sin(q2R)**2)*
          (jnp.cos(rotY)*(-jnp.cos(q2R)*jnp.sin(q1R) - jnp.cos(q1R)*jnp.sin(q2R)) +
           (-jnp.cos(q1R)*jnp.cos(q2R) + jnp.sin(q1R)*jnp.sin(q2R))*jnp.sin(rotY)) +
          (96/125)*(jnp.cos(rotY)*(jnp.cos(q1R)*jnp.cos(q2R) - jnp.sin(q1R)*jnp.sin(q2R)) +
                    (-jnp.cos(q2R)*jnp.sin(q1R) - jnp.cos(q1R)*jnp.sin(q2R))*jnp.sin(rotY))
        ],
        [  # Row 5
          (187/250)*(jnp.cos(q1L)*jnp.cos(rotY) - jnp.sin(q1L)*jnp.sin(rotY)) +
          (16/5)*(((2/5)*(1 - jnp.cos(q2L)) + (16/25)*jnp.cos(q2L))*jnp.sin(q2L) +
                   (-6/25)*jnp.cos(q2L)*jnp.sin(q2L))*
          (jnp.cos(rotY)*(jnp.cos(q1L)*jnp.cos(q2L) - jnp.sin(q1L)*jnp.sin(q2L)) +
           (-jnp.cos(q2L)*jnp.sin(q1L) - jnp.cos(q1L)*jnp.sin(q2L))*jnp.sin(rotY)) +
          (16/5)*(((2/5)*(1 - jnp.cos(q2L)) + (16/25)*jnp.cos(q2L))*jnp.cos(q2L) +
                   (6/25)*jnp.sin(q2L)**2)*
          (jnp.cos(rotY)*(-jnp.cos(q2L)*jnp.sin(q1L) - jnp.cos(q1L)*jnp.sin(q2L)) +
           (-jnp.cos(q1L)*jnp.cos(q2L) + jnp.sin(q1L)*jnp.sin(q2L))*jnp.sin(rotY)) +
          (96/125)*(jnp.cos(rotY)*(jnp.cos(q1L)*jnp.cos(q2L) - jnp.sin(q1L)*jnp.sin(q2L)) +
                    (-jnp.cos(q2L)*jnp.sin(q1L) - jnp.cos(q1L)*jnp.sin(q2L))*jnp.sin(rotY))
        ],
        [  # Row 6
          (96/125)*(jnp.cos(rotY)*(jnp.cos(q1L)*jnp.cos(q2L) - jnp.sin(q1L)*jnp.sin(q2L)) +
                    (-jnp.cos(q2L)*jnp.sin(q1L) - jnp.cos(q1L)*jnp.sin(q2L))*jnp.sin(rotY)),
          (96/125)*(jnp.cos(rotY)*(-jnp.cos(q1L)*jnp.cos(q2L) + jnp.sin(q1L)*jnp.sin(q2L)) +
                    (-jnp.cos(q2L)*jnp.sin(q1L) + jnp.cos(q1L)*jnp.sin(q2L))*jnp.sin(rotY)),
          (1/5) + (96/125)*(
              (((2/5)*(1 - jnp.cos(q2L)) + (16/25)*jnp.cos(q2L))*jnp.sin(q1L) +
               (6/25)*jnp.cos(q1L)*jnp.sin(q2L))*(jnp.cos(q2L)*jnp.sin(q1L) +
                                                   jnp.cos(q1L)*jnp.sin(q2L)) +
              (jnp.cos(q1L)*jnp.cos(q2L) - jnp.sin(q1L)*jnp.sin(q2L))*
              (jnp.cos(q1L)*((2/5)*(1 - jnp.cos(q2L)) + (16/25)*jnp.cos(q2L)) -
               (6/25)*jnp.sin(q1L)*jnp.sin(q2L))
          ),
          (1/5) + (96/125)*(
              (((2/5)*(1 - jnp.cos(q2R)) + (16/25)*jnp.cos(q2R))*jnp.sin(q1R) +
               (6/25)*jnp.cos(q1R)*jnp.sin(q2R))*(jnp.cos(q2R)*jnp.sin(q1R) +
                                                   jnp.cos(q1R)*jnp.sin(q2R)) +
              (jnp.cos(q1R)*jnp.cos(q2R) - jnp.sin(q1R)*jnp.sin(q2R))*
              (jnp.cos(q1R)*((2/5)*(1 - jnp.cos(q2R)) + (16/25)*jnp.cos(q2R)) -
               (6/25)*jnp.sin(q1R)*jnp.sin(q2R))
          )
        ],
        [  # Row 7
          (187/250)*(jnp.cos(q1R)*jnp.cos(rotY) - jnp.sin(q1R)*jnp.sin(rotY)) +
          (16/5)*(((2/5)*(1 - jnp.cos(q2R)) + (16/25)*jnp.cos(q2R))*jnp.cos(q2R) +
                   (6/25)*jnp.sin(q2R)**2)*
          (jnp.cos(rotY)*(jnp.cos(q1R)*jnp.cos(q2R) - jnp.sin(q1R)*jnp.sin(q2R)) +
           (-jnp.cos(q2R)*jnp.sin(q1R) - jnp.cos(q1R)*jnp.sin(q2R))*jnp.sin(rotY)) +
          (16/5)*(((2/5)*(1 - jnp.cos(q2R)) + (16/25)*jnp.cos(q2R))*jnp.sin(q2R) +
                   (-6/25)*jnp.cos(q2R)*jnp.sin(q2R))*
          (jnp.cos(rotY)*(jnp.cos(q1R)*jnp.cos(q2R) - jnp.sin(q1R)*jnp.sin(q2R)) +
           (-jnp.cos(q2R)*jnp.sin(q1R) - jnp.cos(q1R)*jnp.sin(q2R))*jnp.sin(rotY)) +
          (187/250)*(( -jnp.cos(rotY)*jnp.sin(q1L)) - jnp.cos(q1L)*jnp.sin(rotY)) +
          (16/5)*(((2/5)*(1 - jnp.cos(q2L)) + (16/25)*jnp.cos(q2L))*jnp.sin(q2L) +
                   (-6/25)*jnp.cos(q2L)*jnp.sin(q2L))*
          (jnp.cos(rotY)*(jnp.cos(q1L)*jnp.cos(q2L) - jnp.sin(q1L)*jnp.sin(q2L)) +
           (-jnp.cos(q2L)*jnp.sin(q1L) - jnp.cos(q1L)*jnp.sin(q2L))*jnp.sin(rotY)) +
          (16/5)*(((2/5)*(1 - jnp.cos(q2L)) + (16/25)*jnp.cos(q2L))*jnp.cos(q2L) +
                   (6/25)*jnp.sin(q2L)**2)*
          (jnp.cos(rotY)*(-jnp.cos(q2L)*jnp.sin(q1L) - jnp.cos(q1L)*jnp.sin(q2L)) +
           (-jnp.cos(q1L)*jnp.cos(q2L) + jnp.sin(q1L)*jnp.sin(q2L))*jnp.sin(rotY))
        ]
    ])
    
    return Mmat


def G_vector(q):
    # Unpack the 7 generalized coordinates (MATLAB q(1) ... q(7))
    xbar  = q[0]
    zbar  = q[1]
    rotY  = q[2]
    q1R   = q[3]
    q2R   = q[4]
    q1L   = q[5]
    q2L   = q[6]
    
    # Row 1:
    G0 = 0.0

    # Row 2:
    G1 = -0.31392e3

    # Define common subexpressions
    # These expressions appear repeatedly in the formulas.
    expr_q1L   = -jnp.cos(rotY)*jnp.sin(q1L) - jnp.cos(q1L)*jnp.sin(rotY)
    expr_q1R   = -jnp.cos(rotY)*jnp.sin(q1R) - jnp.cos(q1R)*jnp.sin(rotY)
    expr_q1L_c = -jnp.cos(q1L)*jnp.cos(rotY) + jnp.sin(q1L)*jnp.sin(rotY)
    expr_q1R_c = -jnp.cos(q1R)*jnp.cos(rotY) + jnp.sin(q1R)*jnp.sin(rotY)
    
    # Row 3:
    # Note: 1+(-1)*cos(q2) is 1 - cos(q2)
    innerL = (2/5)*(1 - jnp.cos(q2L))*expr_q1L \
             - (2/5)*jnp.sin(q2L)*expr_q1L_c \
             + (16/25)*( jnp.cos(q2L)*expr_q1L + jnp.sin(q2L)*expr_q1L_c )
    innerR = (2/5)*(1 - jnp.cos(q2R))*expr_q1R \
             - (2/5)*jnp.sin(q2R)*expr_q1R_c \
             + (16/25)*( jnp.cos(q2R)*expr_q1R + jnp.sin(q2R)*expr_q1R_c )
    G2 = 0.282528e2 * jnp.sin(rotY) \
         - 0.733788e1 * expr_q1L \
         - 0.733788e1 * expr_q1R \
         - 0.31392e2 * innerL \
         - 0.31392e2 * innerR

    # Row 4:
    innerR_row4 = (2/5)*(1 - jnp.cos(q2R))*expr_q1R \
                  - (2/5)*jnp.sin(q2R)*expr_q1R_c \
                  + (16/25)*( jnp.cos(q2R)*expr_q1R + jnp.sin(q2R)*expr_q1R_c )
    G3 = -0.733788e1 * expr_q1R - 0.31392e2 * innerR_row4

    # Row 5:
    innerL_row5 = (2/5)*(1 - jnp.cos(q2L))*expr_q1L \
                  - (2/5)*jnp.sin(q2L)*expr_q1L_c \
                  + (16/25)*( jnp.cos(q2L)*expr_q1L + jnp.sin(q2L)*expr_q1L_c )
    G4 = -0.733788e1 * expr_q1L - 0.31392e2 * innerL_row5

    # Row 6:
    # Here the second term inside the bracket involves (cos(q1R)*cos(rotY) - sin(q1R)*sin(rotY))
    commonR = jnp.cos(q1R)*jnp.cos(rotY) - jnp.sin(q1R)*jnp.sin(rotY)
    G5 = -0.31392e2 * (
            (-2/5)*jnp.cos(q2R)*expr_q1R +
            (2/5)*jnp.sin(q2R)*commonR +
            (16/25)*( jnp.cos(q2R)*expr_q1R - jnp.sin(q2R)*commonR )
         )

    # Row 7:
    commonL = jnp.cos(q1L)*jnp.cos(rotY) - jnp.sin(q1L)*jnp.sin(rotY)
    G6 = -0.31392e2 * (
            (-2/5)*jnp.cos(q2L)*expr_q1L +
            (2/5)*jnp.sin(q2L)*commonL +
            (16/25)*( jnp.cos(q2L)*expr_q1L - jnp.sin(q2L)*commonL )
         )

    return jnp.array([G0, G1, G2, G3, G4, G5, G6])
  
  
def C_matrix(D_func, q, dq):
    """
    Compute the Coriolis matrix.
    
    Parameters:
      D_func : function
          A function that takes q as input and returns the (n x n) mass matrix D(q).
      q : jnp.ndarray
          The vector of generalized coordinates (shape (n,)).
      dq : jnp.ndarray
          The vector of generalized velocities (shape (n,)).
    
    Returns:
      C_mat : jnp.ndarray
          The (n x n) Coriolis matrix.
    """
    n = q.shape[0]
    C_mat = jnp.zeros((n, n))
    
    # Loop over each matrix element (i, j)
    for i in range(n):
        for j in range(n):
            temp = 0.0
            # Sum over k from 0 to n-1
            for k in range(n):
                # Each term is computed as (1/2)*dq[k]*(∂D[i,j]/∂q[k] + ∂D[i,k]/∂q[j] - ∂D[j,k]/∂q[i])
                dD_ij_dqk = jax.grad(lambda qq: D_func(qq)[i, j])(q)[k]
                dD_ik_dqj = jax.grad(lambda qq: D_func(qq)[i, k])(q)[j]
                dD_jk_dqi = jax.grad(lambda qq: D_func(qq)[j, k])(q)[i]
                temp += 0.5 * dq[k] * (dD_ij_dqk + dD_ik_dqj - dD_jk_dqi)
            # Update the (i, j) entry in C_mat
            C_mat = C_mat.at[i, j].set(temp)
    return C_mat


def hip_position(q):
    # Unpack the 7 generalized coordinates (MATLAB q(1) ... q(7) -> Python q[0] ... q[6])
    xbar  = q[0]
    zbar  = q[1]
    rotY  = q[2]
    q1R   = q[3]
    q2R   = q[4]
    q1L   = q[5]
    q2L   = q[6]

    # Compute the hip position
    posHip = jnp.array([xbar + (63/100) * jnp.sin(rotY),
                        zbar + (63/100) * jnp.cos(rotY)])
    return posHip


def left_swing_foot_position(q):
    # Unpack the 7 generalized coordinates (MATLAB: q(1) ... q(7))
    xbar  = q[0]
    zbar  = q[1]
    rotY  = q[2]
    q1R   = q[3]
    q2R   = q[4]
    q1L   = q[5]
    q2L   = q[6]

    # Compute the x-coordinate of the left swing foot position
    # (MATLAB:  (2/5).*(1+(-1).*cos(q2L)) becomes (2/5)*(1 - cos(q2L)) etc.)
    term1_x = (2/5) * (1 - jnp.cos(q2L)) * (jnp.cos(rotY)*jnp.sin(q1L) + jnp.cos(q1L)*jnp.sin(rotY))
    term2_x = (-2/5) * jnp.sin(q2L) * (jnp.cos(q1L)*jnp.cos(rotY) - jnp.sin(q1L)*jnp.sin(rotY))
    term3_x = (4/5) * (
                  jnp.cos(q2L)*(jnp.cos(rotY)*jnp.sin(q1L) + jnp.cos(q1L)*jnp.sin(rotY))
                + jnp.sin(q2L)*(jnp.cos(q1L)*jnp.cos(rotY) - jnp.sin(q1L)*jnp.sin(rotY))
              )
    pos_x = xbar + term1_x + term2_x + term3_x

    # Compute the z-coordinate of the left swing foot position
    # Note: The MATLAB expression with (-1).* is translated into negative signs in Python.
    term1_z = (2/5) * jnp.sin(q2L) * (jnp.cos(rotY)*jnp.sin(q1L) + jnp.cos(q1L)*jnp.sin(rotY))
    term2_z = (2/5) * (1 - jnp.cos(q2L)) * (jnp.cos(q1L)*jnp.cos(rotY) - jnp.sin(q1L)*jnp.sin(rotY))
    term3_z = (4/5) * (
                  - jnp.sin(q2L)*(jnp.cos(rotY)*jnp.sin(q1L) + jnp.cos(q1L)*jnp.sin(rotY))
                + jnp.cos(q2L)*(jnp.cos(q1L)*jnp.cos(rotY) - jnp.sin(q1L)*jnp.sin(rotY))
              )
    pos_z = zbar + term1_z + term2_z + term3_z

    # Combine the two coordinates into the position vector.
    pos_L_sw = jnp.array([pos_x, pos_z])
    return pos_L_sw


def right_stance_foot_position(q):
    # Unpack the 7 generalized coordinates (MATLAB: q(1)...q(7))
    xbar  = q[0]
    zbar  = q[1]
    rotY  = q[2]
    q1R   = q[3]
    q2R   = q[4]
    q1L   = q[5]  # not used in this function
    q2L   = q[6]  # not used in this function

    # --- Compute the x-coordinate of the right stance foot ---
    # MATLAB: (2/5).*(1+(-1).*cos(q2R)) becomes (2/5)*(1 - cos(q2R))
    term1 = (2/5) * (1 - jnp.cos(q2R)) * (jnp.cos(rotY)*jnp.sin(q1R) + jnp.cos(q1R)*jnp.sin(rotY))
    term2 = (-2/5) * jnp.sin(q2R) * (jnp.cos(q1R)*jnp.cos(rotY) - jnp.sin(q1R)*jnp.sin(rotY))
    term3 = (4/5) * (
                jnp.cos(q2R) * (jnp.cos(rotY)*jnp.sin(q1R) + jnp.cos(q1R)*jnp.sin(rotY))
              + jnp.sin(q2R) * (jnp.cos(q1R)*jnp.cos(rotY) - jnp.sin(q1R)*jnp.sin(rotY))
            )
    pos_x = xbar + term1 + term2 + term3

    # --- Compute the z-coordinate of the right stance foot ---
    # First, note that in MATLAB, (-1).*cos(rotY).*sin(q1R)+(-1).*cos(q1R).*sin(rotY)
    # becomes - (cos(rotY)*sin(q1R) + cos(q1R)*sin(rotY))
    term4 = (-2/5) * jnp.sin(q2R) * (-jnp.cos(rotY)*jnp.sin(q1R) - jnp.cos(q1R)*jnp.sin(rotY))
    # (1+(-1).*cos(q2R)) becomes (1 - cos(q2R))
    term5 = (2/5) * (1 - jnp.cos(q2R)) * (jnp.cos(q1R)*jnp.cos(rotY) - jnp.sin(q1R)*jnp.sin(rotY))
    term6 = (4/5) * (
                jnp.sin(q2R) * (-jnp.cos(rotY)*jnp.sin(q1R) - jnp.cos(q1R)*jnp.sin(rotY))
              + jnp.cos(q2R) * (jnp.cos(q1R)*jnp.cos(rotY) - jnp.sin(q1R)*jnp.sin(rotY))
            )
    pos_z = zbar + term4 + term5 + term6

    posR = jnp.array([pos_x, pos_z])
    return posR


def com_position(q):
    # Unpack the 7 generalized coordinates
    xbar  = q[0]
    zbar  = q[1]
    rotY  = q[2]
    q1R   = q[3]
    q2R   = q[4]
    q1L   = q[5]
    q2L   = q[6]
    
    # Precompute common trigonometric values
    sin_rotY = jnp.sin(rotY)
    cos_rotY = jnp.cos(rotY)
    
    sin_q1L = jnp.sin(q1L)
    cos_q1L = jnp.cos(q1L)
    
    sin_q1R = jnp.sin(q1R)
    cos_q1R = jnp.cos(q1R)
    
    # -----------------------------
    # Compute the x-coordinate of COM
    # -----------------------------
    # Term A: contribution from the trunk
    termA_x = 12 * (xbar + (6/25) * sin_rotY)
    
    # Term B: left leg, first term
    termB_x = (34/5) * (xbar + (11/100) * (cos_rotY * sin_q1L + cos_q1L * sin_rotY))
    
    # Term C: right leg, first term
    termC_x = (34/5) * (xbar + (11/100) * (cos_rotY * sin_q1R + cos_q1R * sin_rotY))
    
    # Term D: left leg swing contribution
    termD_x = (16/5) * (
        xbar +
        (2/5) * (1 - jnp.cos(q2L)) * (cos_rotY * sin_q1L + cos_q1L * sin_rotY) +
        (-2/5) * jnp.sin(q2L) * (cos_q1L * cos_rotY - sin_q1L * sin_rotY) +
        (16/25) * (
            jnp.cos(q2L) * (cos_rotY * sin_q1L + cos_q1L * sin_rotY) +
            jnp.sin(q2L) * (cos_q1L * cos_rotY - sin_q1L * sin_rotY)
        )
    )
    
    # Term E: right leg swing contribution
    termE_x = (16/5) * (
        xbar +
        (2/5) * (1 - jnp.cos(q2R)) * (cos_rotY * sin_q1R + cos_q1R * sin_rotY) +
        (-2/5) * jnp.sin(q2R) * (cos_q1R * cos_rotY - sin_q1R * sin_rotY) +
        (16/25) * (
            jnp.cos(q2R) * (cos_rotY * sin_q1R + cos_q1R * sin_rotY) +
            jnp.sin(q2R) * (cos_q1R * cos_rotY - sin_q1R * sin_rotY)
        )
    )
    
    posCOM_x = (1/32) * (termA_x + termB_x + termC_x + termD_x + termE_x)
    
    # -----------------------------
    # Compute the z-coordinate of COM
    # -----------------------------
    # Term A: contribution from the trunk
    termA_z = 12 * (zbar + (6/25) * cos_rotY)
    
    # Term B: left leg, first term
    termB_z = (34/5) * (zbar + (11/100) * (cos_q1L * cos_rotY - sin_q1L * sin_rotY))
    
    # Term C: right leg, first term
    termC_z = (34/5) * (zbar + (11/100) * (cos_q1R * cos_rotY - sin_q1R * sin_rotY))
    
    # For the swing contributions below note:
    #   (-1).*cos(rotY).*sin(q1) becomes -cos(rotY)*sin(q1)
    #   and (cos(q1).*cos(rotY)+(-1).*sin(q1).*sin(rotY)) becomes (cos(q1)*cos(rotY) - sin(q1)*sin(rotY))
    
    # Term D: left leg swing contribution for z
    termD_z = (16/5) * (
        zbar +
        (-2/5) * jnp.sin(q2L) * (- (cos_rotY * sin_q1L + cos_q1L * sin_rotY)) +
        (2/5) * (1 - jnp.cos(q2L)) * (cos_q1L * cos_rotY - sin_q1L * sin_rotY) +
        (16/25) * (
            jnp.sin(q2L) * (- (cos_rotY * sin_q1L + cos_q1L * sin_rotY)) +
            jnp.cos(q2L) * (cos_q1L * cos_rotY - sin_q1L * sin_rotY)
        )
    )
    # Simplify the negatives in termD_z:
    #   -2/5 * sin(q2L) * ( -(...) ) = (2/5)* sin(q2L)*(cos_rotY*sin_q1L+cos_q1L*sin_rotY)
    #   and similarly for the (16/25) term.
    termD_z = (16/5) * (
        zbar +
        (2/5) * jnp.sin(q2L) * (cos_rotY * sin_q1L + cos_q1L * sin_rotY) +
        (2/5) * (1 - jnp.cos(q2L)) * (cos_q1L * cos_rotY - sin_q1L * sin_rotY) +
        (16/25) * (
            - jnp.sin(q2L) * (cos_rotY * sin_q1L + cos_q1L * sin_rotY) +
            jnp.cos(q2L) * (cos_q1L * cos_rotY - sin_q1L * sin_rotY)
        )
    )
    
    # Term E: right leg swing contribution for z
    termE_z = (16/5) * (
        zbar +
        (-2/5) * jnp.sin(q2R) * (- (cos_rotY * sin_q1R + cos_q1R * sin_rotY)) +
        (2/5) * (1 - jnp.cos(q2R)) * (cos_q1R * cos_rotY - sin_q1R * sin_rotY) +
        (16/25) * (
            jnp.sin(q2R) * (- (cos_rotY * sin_q1R + cos_q1R * sin_rotY)) +
            jnp.cos(q2R) * (cos_q1R * cos_rotY - sin_q1R * sin_rotY)
        )
    )
    termE_z = (16/5) * (
        zbar +
        (2/5) * jnp.sin(q2R) * (cos_rotY * sin_q1R + cos_q1R * sin_rotY) +
        (2/5) * (1 - jnp.cos(q2R)) * (cos_q1R * cos_rotY - sin_q1R * sin_rotY) +
        (16/25) * (
            - jnp.sin(q2R) * (cos_rotY * sin_q1R + cos_q1R * sin_rotY) +
            jnp.cos(q2R) * (cos_q1R * cos_rotY - sin_q1R * sin_rotY)
        )
    )
    
    posCOM_z = (1/32) * (termA_z + termB_z + termC_z + termD_z + termE_z)
    
    # Return the COM position as a 2-element vector [x, z]
    return jnp.array([posCOM_x, posCOM_z])


def B_matrix():
    return jnp.vstack([jnp.zeros(3,4), jnp.eye(4)])
    

def fxgu_floating_base(t, x, u, params):
    q = x[0:7]
    dq = x[7:]
    B = B_matrix()
    C = C_matrix(q, dq, params)
    D = D_matrix(q, params)
    G = G_vector(q, params)
    
    # Compute Fx and Gx
    Fx = jnp.linalg.solve(D, -C @ x[5:10] - G.flatten())
    Gx = jnp.linalg.solve(D, B)
    
    # Compute state derivatives
    dx = jnp.zeros_like(x)
    dx[:5] = x[5:10]
    dx[5:10] = Fx + Gx @ u
    
    return dx


def foot_touching_event(t,x):
    q = x[0:7]
    return left_swing_foot_position(q)[1]


def integrate_fxgu(x0, u, params, event=True, tstart=0.0, tfinal=0.2):
    print("Integrating the 5-link model with the given initial state.")    
    num_t_eval = 300
    t_eval = np.linspace(tstart, tfinal, num_t_eval)
    t_span = [tstart, tfinal]
    
    if event:
      event_func = foot_touching_event
    else:
      event_func = None
    
    options = {
        'rtol': 1e-5,
        'atol': 1e-6,
        'events': event_func
    }
    
    # Solve until the first terminal event
    sol = solve_ivp(
       fun=lambda t, x: fxgu_floating_base(t, x, u, params),
        t_span=t_span,
        t_eval=t_eval,
        y0=x0,
        # method='RK45', 
        # rtol=options['rtol'],
        # atol=options['atol'],
        events=options['events']
    )
    
    t_trj = sol.t
    x_trj = sol.y.T
    
    if sol.t_events:
      te = sol.t_events[0][0]
      xe = sol.y_events[0][0]
      
      print("Swing foot guard time:")
      print(te)
      
      print("Swing foot guard state:")
      print(xe)
      
      print("Swing foot guard function value:")
      print(foot_touching_event(te, xe))

      # Create a mask for all integrated times strictly less than te.
      mask = t_trj <= te
      
      # Select all times and states up until the event
      t_trj = sol.t[mask]
      x_trj = x_trj[mask]
    
    return t_trj, x_trj


def animate(t_trj, x_trj, step=5):
    """Animate a trajectory of states for the 5-link biped model.

    Args:
        x_trj (np.array): The ABSOLUTE angles of the joints. 
        step (int, the simulation frequency): Defaults to 1.
    """
    fig, ax = plt.subplots()
    
    plt.ion()
    
    n_time_steps = x_trj.shape[0]
    
    p_hip = hip_position(x_trj[0, 0:5])
    dt_trj = t_trj[1:] - t_trj[:-1]
    
    for i in range(0, n_time_steps, step):
        x_i = x_trj[i, :]
        dt_i = dt_trj[i]
        v_hip = vel_hip(x_i, params)
        p_hip += v_hip*dt_i
        
        draw_5link(x_i, p_hip, params, ax, legend=False)
        plt.pause(0.001)
        ax.clear()  
    plt.ioff()
    
    draw_5link(x_trj[-1], p_hip, params, ax, legend=False)
    plt.show()
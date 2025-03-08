import jax.numpy as jnp
import jax


def D_matrix(q):
    # Unpack the 7 generalized coordinates (MATLAB q(1) \ q(7))
    rotY   = q[2]
    q1R    = q[3]
    q2R    = q[4]
    q1L    = q[5]
    q2L    = q[6]

    # Construct the 7x7 mass-inertia matrix.
    # (Each MATLAB elementwise operation “*” and “**” has been replaced with
    # the corresponding Python jnp operations.)
    
    M11 = 12*jnp.cos(rotY)**2+12*jnp.sin(rotY)**2+(34/5)*\
        (jnp.cos(rotY)*jnp.sin(q1L)+ jnp.cos(q1L)*jnp.sin(rotY))**2+\
        (34/5)*(jnp.cos(rotY)*jnp.sin(q1R)+jnp.cos(q1R)*jnp.sin(rotY))**2+\
        (34/5)*(jnp.cos(q1L)*jnp.cos(rotY)+(-1)*jnp.sin(q1L)*\
        jnp.sin(rotY))**2+(34/5)*(jnp.cos(q1R)*jnp.cos(rotY)+\
        (-1)*jnp.sin(q1R)*jnp.sin(rotY))**2+(16/5)*\
        (jnp.cos(rotY)*(jnp.cos(q1L)*jnp.cos(q2L)+(-1)*jnp.sin(q1L)*\
        jnp.sin(q2L))+((-1)*jnp.cos(q2L)*jnp.sin(q1L)+(-1)*jnp.cos(q1L)*\
        jnp.sin(q2L))*jnp.sin(rotY))**2+(16/5)*(jnp.cos(rotY)*(jnp.cos(q2L)*\
        jnp.sin(q1L)+jnp.cos(q1L)*jnp.sin(q2L))+(jnp.cos(q1L)*jnp.cos(q2L)+\
        (-1)*jnp.sin(q1L)*jnp.sin(q2L))*jnp.sin(rotY))**2+(16/5)*(jnp.cos(rotY)*\
        (jnp.cos(q1R)*jnp.cos(q2R)+(-1)*jnp.sin(q1R)*jnp.sin(q2R))+\
        ((-1)*jnp.cos(q2R)*jnp.sin(q1R)+(-1)*jnp.cos(q1R)*jnp.sin(q2R))*\
        jnp.sin(rotY))**2+(16/5)*(jnp.cos(rotY)*(jnp.cos(q2R)*jnp.sin(q1R)+\
        jnp.cos(q1R)*jnp.sin(q2R))+(jnp.cos(q1R)*jnp.cos(q2R)+(-1)*\
        jnp.sin(q1R)*jnp.sin(q2R))*jnp.sin(rotY))**2

    M12 = (34/5)*((-1)*jnp.cos(rotY)*jnp.sin(q1L)+(-1)*jnp.cos(q1L)* \
            jnp.sin(rotY))*(jnp.cos(q1L)*jnp.cos(rotY)+(-1)*jnp.sin(q1L)*jnp.sin(rotY))+( \
            34/5)*(jnp.cos(rotY)*jnp.sin(q1L)+jnp.cos(q1L)*jnp.sin(rotY))*(jnp.cos(q1L)* \
            jnp.cos(rotY)+(-1)*jnp.sin(q1L)*jnp.sin(rotY))+(34/5)*((-1)*jnp.cos(rotY)* \
            jnp.sin(q1R)+(-1)*jnp.cos(q1R)*jnp.sin(rotY))*(jnp.cos(q1R)*jnp.cos(rotY)+(-1)* \
            jnp.sin(q1R)*jnp.sin(rotY))+(34/5)*(jnp.cos(rotY)*jnp.sin(q1R)+jnp.cos(q1R)*jnp.sin( \
            rotY))*(jnp.cos(q1R)*jnp.cos(rotY)+(-1)*jnp.sin(q1R)*jnp.sin(rotY))+(16/5) \
            *(jnp.cos(rotY)*(jnp.cos(q1L)*jnp.cos(q2L)+(-1)*jnp.sin(q1L)*jnp.sin(q2L))+((-1) \
            *jnp.cos(q2L)*jnp.sin(q1L)+(-1)*jnp.cos(q1L)*jnp.sin(q2L))*jnp.sin(rotY))*(jnp.cos( \
            rotY)*((-1)*jnp.cos(q2L)*jnp.sin(q1L)+(-1)*jnp.cos(q1L)*jnp.sin(q2L))+(-1)* \
            (jnp.cos(q1L)*jnp.cos(q2L)+(-1)*jnp.sin(q1L)*jnp.sin(q2L))*jnp.sin(rotY))+(16/5) \
            *(jnp.cos(rotY)*(jnp.cos(q1L)*jnp.cos(q2L)+(-1)*jnp.sin(q1L)*jnp.sin(q2L))+(-1) \
            *(jnp.cos(q2L)*jnp.sin(q1L)+jnp.cos(q1L)*jnp.sin(q2L))*jnp.sin(rotY))*(jnp.cos( \
            rotY)*(jnp.cos(q2L)*jnp.sin(q1L)+jnp.cos(q1L)*jnp.sin(q2L))+(jnp.cos(q1L)*jnp.cos( \
            q2L)+(-1)*jnp.sin(q1L)*jnp.sin(q2L))*jnp.sin(rotY))+(16/5)*(jnp.cos(rotY)*( \
            jnp.cos(q1R)*jnp.cos(q2R)+(-1)*jnp.sin(q1R)*jnp.sin(q2R))+((-1)*jnp.cos(q2R)*jnp.sin( \
            q1R)+(-1)*jnp.cos(q1R)*jnp.sin(q2R))*jnp.sin(rotY))*(jnp.cos(rotY)*((-1)* \
            jnp.cos(q2R)*jnp.sin(q1R)+(-1)*jnp.cos(q1R)*jnp.sin(q2R))+(-1)*(jnp.cos(q1R)*jnp.cos( \
            q2R)+(-1)*jnp.sin(q1R)*jnp.sin(q2R))*jnp.sin(rotY))+(16/5)*(jnp.cos(rotY)*( \
            jnp.cos(q1R)*jnp.cos(q2R)+(-1)*jnp.sin(q1R)*jnp.sin(q2R))+(-1)*(jnp.cos(q2R)*jnp.sin( \
            q1R)+jnp.cos(q1R)*jnp.sin(q2R))*jnp.sin(rotY))*(jnp.cos(rotY)*(jnp.cos(q2R)* \
            jnp.sin(q1R)+jnp.cos(q1R)*jnp.sin(q2R))+(jnp.cos(q1R)*jnp.cos(q2R)+(-1)*jnp.sin(q1R)* \
            jnp.sin(q2R))*jnp.sin(rotY))
    
    # Finally, sum all the terms:
    M13 = (72/25)*jnp.cos(rotY)+(34/5)*((11/100)*jnp.cos( \
            q1L)**2+(11/100)*jnp.sin(q1L)**2)*(jnp.cos(q1L)*jnp.cos(rotY)+(-1)*jnp.sin( \
            q1L)*jnp.sin(rotY))+(34/5)*((11/100)*jnp.cos(q1R)**2+(11/100)*jnp.sin( \
            q1R)**2)*(jnp.cos(q1R)*jnp.cos(rotY)+(-1)*jnp.sin(q1R)*jnp.sin(rotY))+(16/5) \
            *((((2/5)*(1+(-1)*jnp.cos(q2L))+(16/25)*jnp.cos(q2L))*jnp.sin(q1L)+(6/25) \
            *jnp.cos(q1L)*jnp.sin(q2L))*(jnp.cos(q2L)*jnp.sin(q1L)+jnp.cos(q1L)*jnp.sin(q2L))+( \
            jnp.cos(q1L)*jnp.cos(q2L)+(-1)*jnp.sin(q1L)*jnp.sin(q2L))*(jnp.cos(q1L)*((2/5)*( \
            1+(-1)*jnp.cos(q2L))+(16/25)*jnp.cos(q2L))+(-6/25)*jnp.sin(q1L)*jnp.sin(q2L))) \
            *(jnp.cos(rotY)*(jnp.cos(q1L)*jnp.cos(q2L)+(-1)*jnp.sin(q1L)*jnp.sin(q2L))+((-1) \
            *jnp.cos(q2L)*jnp.sin(q1L)+(-1)*jnp.cos(q1L)*jnp.sin(q2L))*jnp.sin(rotY))+(16/5) \
            *((-1)*(((2/5)*(1+(-1)*jnp.cos(q2L))+(16/25)*jnp.cos(q2L))*jnp.sin(q1L)+ \
            (6/25)*jnp.cos(q1L)*jnp.sin(q2L))*(jnp.cos(q1L)*jnp.cos(q2L)+(-1)*jnp.sin(q1L)* \
            jnp.sin(q2L))+(-1)*((-1)*jnp.cos(q2L)*jnp.sin(q1L)+(-1)*jnp.cos(q1L)*jnp.sin(q2L) \
            )*(jnp.cos(q1L)*((2/5)*(1+(-1)*jnp.cos(q2L))+(16/25)*jnp.cos(q2L))+( \
            -6/25)*jnp.sin(q1L)*jnp.sin(q2L)))*(jnp.cos(rotY)*(jnp.cos(q2L)*jnp.sin(q1L)+ \
            jnp.cos(q1L)*jnp.sin(q2L))+(jnp.cos(q1L)*jnp.cos(q2L)+(-1)*jnp.sin(q1L)*jnp.sin(q2L)) \
            *jnp.sin(rotY))+(16/5)*((((2/5)*(1+(-1)*jnp.cos(q2R))+(16/25)*jnp.cos( \
            q2R))*jnp.sin(q1R)+(6/25)*jnp.cos(q1R)*jnp.sin(q2R))*(jnp.cos(q2R)*jnp.sin(q1R)+ \
            jnp.cos(q1R)*jnp.sin(q2R))+(jnp.cos(q1R)*jnp.cos(q2R)+(-1)*jnp.sin(q1R)*jnp.sin(q2R)) \
            *(jnp.cos(q1R)*((2/5)*(1+(-1)*jnp.cos(q2R))+(16/25)*jnp.cos(q2R))+(-6/25) \
            *jnp.sin(q1R)*jnp.sin(q2R)))*(jnp.cos(rotY)*(jnp.cos(q1R)*jnp.cos(q2R)+(-1)* \
            jnp.sin(q1R)*jnp.sin(q2R))+((-1)*jnp.cos(q2R)*jnp.sin(q1R)+(-1)*jnp.cos(q1R)*jnp.sin( \
            q2R))*jnp.sin(rotY))+(16/5)*((-1)*(((2/5)*(1+(-1)*jnp.cos(q2R))+( \
            16/25)*jnp.cos(q2R))*jnp.sin(q1R)+(6/25)*jnp.cos(q1R)*jnp.sin(q2R))*(jnp.cos(q1R) \
            *jnp.cos(q2R)+(-1)*jnp.sin(q1R)*jnp.sin(q2R))+(-1)*((-1)*jnp.cos(q2R)*jnp.sin( \
            q1R)+(-1)*jnp.cos(q1R)*jnp.sin(q2R))*(jnp.cos(q1R)*((2/5)*(1+(-1)*jnp.cos( \
            q2R))+(16/25)*jnp.cos(q2R))+(-6/25)*jnp.sin(q1R)*jnp.sin(q2R)))*(jnp.cos( \
            rotY)*(jnp.cos(q2R)*jnp.sin(q1R)+jnp.cos(q1R)*jnp.sin(q2R))+(jnp.cos(q1R)*jnp.cos( \
            q2R)+(-1)*jnp.sin(q1R)*jnp.sin(q2R))*jnp.sin(rotY))
    
    M14 = (187/250)*(jnp.cos(q1R)*jnp.cos(rotY)+(-1)*jnp.sin(q1R)*jnp.sin(rotY))+(16/5)*(((2/5)*(1+(-1)* \
            jnp.cos(q2R))+(16/25)*jnp.cos(q2R))*jnp.cos(q2R)+(6/25)*jnp.sin(q2R)**2)*(jnp.cos( \
            rotY)*(jnp.cos(q1R)*jnp.cos(q2R)+(-1)*jnp.sin(q1R)*jnp.sin(q2R))+((-1)*jnp.cos( \
            q2R)*jnp.sin(q1R)+(-1)*jnp.cos(q1R)*jnp.sin(q2R))*jnp.sin(rotY))+(16/5)*((( \
            2/5)*(1+(-1)*jnp.cos(q2R))+(16/25)*jnp.cos(q2R))*jnp.sin(q2R)+(-6/25)* \
            jnp.cos(q2R)*jnp.sin(q2R))*(jnp.cos(rotY)*(jnp.cos(q2R)*jnp.sin(q1R)+jnp.cos(q1R)* \
            jnp.sin(q2R))+(jnp.cos(q1R)*jnp.cos(q2R)+(-1)*jnp.sin(q1R)*jnp.sin(q2R))*jnp.sin(rotY))
            
    M15 = (96/125) *(jnp.cos(rotY) *(jnp.cos(q1R) *jnp.cos(q2R)+(-1) *jnp.sin(q1R) \
            *jnp.sin(q2R))+((-1) *jnp.cos(q2R) *jnp.sin(q1R)+(-1) *jnp.cos(q1R) *jnp.sin(q2R)) *jnp.sin(rotY))
    
    M16 = (187/250) *(jnp.cos(q1L) *jnp.cos(rotY)+(-1) *jnp.sin(q1L) *jnp.sin(rotY))+(16/5) * \
        (((2/5) *(1+(-1) *jnp.cos(q2L))+(16/25) *jnp.cos(q2L)) *jnp.cos(q2L)+(6/25) *jnp.sin(q2L) **2) * \
        (jnp.cos(rotY) *(jnp.cos(q1L) *jnp.cos(q2L)+(-1) *jnp.sin(q1L) *jnp.sin(q2L))+((-1) *jnp.cos(q2L) *jnp.sin(q1L)+(-1) *jnp.cos(q1L)* \
        jnp.sin(q2L)) *jnp.sin(rotY))+(16/5) *(((2/5) *(1+(-1) *jnp.cos(q2L))+( \
        16/25) *jnp.cos(q2L)) *jnp.sin(q2L)+(-6/25) *jnp.cos(q2L) *jnp.sin(q2L)) *(jnp.cos( \
        rotY) *(jnp.cos(q2L) *jnp.sin(q1L)+jnp.cos(q1L) *jnp.sin(q2L))+(jnp.cos(q1L) *jnp.cos( \
        q2L)+(-1) *jnp.sin(q1L) *jnp.sin(q2L)) *jnp.sin(rotY))
        
    M17 = (96/125)*(jnp.cos(rotY)*(jnp.cos(q1L)*jnp.cos(q2L)+(-1)*jnp.sin(q1L)*jnp.sin(q2L))+((-1)* \
        jnp.cos(q2L)* jnp.sin(q1L)+(-1)*jnp.cos(q1L)*jnp.sin(q2L))*jnp.sin(rotY))
    
    M21 = M12
        
    M22 = 12 * jnp.cos(rotY)**2+12 * jnp.sin(rotY)**2+(34/5) *((-1) * jnp.cos(rotY) * jnp.sin( \
        q1L)+(-1) * jnp.cos(q1L) * jnp.sin(rotY))**2+(34/5) *((-1) * jnp.cos(rotY) * \
         jnp.sin(q1R)+(-1) * jnp.cos(q1R) * jnp.sin(rotY))**2+(34/5) *( jnp.cos(q1L) * jnp.cos( \
        rotY)+(-1) * jnp.sin(q1L) * jnp.sin(rotY))**2+(34/5) *( jnp.cos(q1R) * jnp.cos( \
        rotY)+(-1) * jnp.sin(q1R) * jnp.sin(rotY))**2+(16/5) *( jnp.cos(rotY) *( jnp.cos( \
        q1L) * jnp.cos(q2L)+(-1) * jnp.sin(q1L) * jnp.sin(q2L))+(-1) *( jnp.cos(q2L) * jnp.sin(q1L) \
        + jnp.cos(q1L) * jnp.sin(q2L)) * jnp.sin(rotY))**2+(16/5) *( jnp.cos(rotY) *((-1) * \
         jnp.cos(q2L) * jnp.sin(q1L)+(-1) * jnp.cos(q1L) * jnp.sin(q2L))+(-1) *( jnp.cos(q1L) * jnp.cos( \
        q2L)+(-1) * jnp.sin(q1L) * jnp.sin(q2L)) * jnp.sin(rotY))**2+(16/5) *( jnp.cos(rotY) \
         *( jnp.cos(q1R) * jnp.cos(q2R)+(-1) * jnp.sin(q1R) * jnp.sin(q2R))+(-1) *( jnp.cos(q2R) * \
         jnp.sin(q1R)+ jnp.cos(q1R) * jnp.sin(q2R)) * jnp.sin(rotY))**2+(16/5) *( jnp.cos(rotY) * \
        ((-1) * jnp.cos(q2R) * jnp.sin(q1R)+(-1) * jnp.cos(q1R) * jnp.sin(q2R))+(-1) *( jnp.cos( \
        q1R) * jnp.cos(q2R)+(-1) * jnp.sin(q1R) * jnp.sin(q2R)) * jnp.sin(rotY))**2
         
    M23 = (-72/25) *jnp.sin(rotY)+(34/5) *((11/100) *jnp.cos(q1L) **2+(11/100) *jnp.sin(q1L) **2) \
             *((-1) *jnp.cos(rotY) *jnp.sin(q1L)+(-1) *jnp.cos(q1L) *jnp.sin(rotY))+(34/5) * \
            ((11/100) *jnp.cos(q1R) **2+(11/100) *jnp.sin(q1R) **2) *((-1) *jnp.cos(rotY) * \
            jnp.sin(q1R)+(-1) *jnp.cos(q1R) *jnp.sin(rotY))+(16/5) *((-1) *(((2/5) *(1+( \
            -1) *jnp.cos(q2L))+(16/25) *jnp.cos(q2L)) *jnp.sin(q1L)+(6/25) *jnp.cos(q1L) *jnp.sin( \
            q2L)) *(jnp.cos(q1L) *jnp.cos(q2L)+(-1) *jnp.sin(q1L) *jnp.sin(q2L))+(-1) *((-1) * \
            jnp.cos(q2L) *jnp.sin(q1L)+(-1) *jnp.cos(q1L) *jnp.sin(q2L)) *(jnp.cos(q1L) *((2/5) *( \
            1+(-1) *jnp.cos(q2L))+(16/25) *jnp.cos(q2L))+(-6/25) *jnp.sin(q1L) *jnp.sin(q2L))) \
             *(jnp.cos(rotY) *(jnp.cos(q1L) *jnp.cos(q2L)+(-1) *jnp.sin(q1L) *jnp.sin(q2L))+(-1) \
             *(jnp.cos(q2L) *jnp.sin(q1L)+jnp.cos(q1L) *jnp.sin(q2L)) *jnp.sin(rotY))+(16/5) *((( \
            (2/5) *(1+(-1) *jnp.cos(q2L))+(16/25) *jnp.cos(q2L)) *jnp.sin(q1L)+(6/25) * \
            jnp.cos(q1L) *jnp.sin(q2L)) *(jnp.cos(q2L) *jnp.sin(q1L)+jnp.cos(q1L) *jnp.sin(q2L))+(jnp.cos( \
            q1L) *jnp.cos(q2L)+(-1) *jnp.sin(q1L) *jnp.sin(q2L)) *(jnp.cos(q1L) *((2/5) *(1+( \
            -1) *jnp.cos(q2L))+(16/25) *jnp.cos(q2L))+(-6/25) *jnp.sin(q1L) *jnp.sin(q2L))) *( \
            jnp.cos(rotY) *((-1) *jnp.cos(q2L) *jnp.sin(q1L)+(-1) *jnp.cos(q1L) *jnp.sin(q2L))+( \
            -1) *(jnp.cos(q1L) *jnp.cos(q2L)+(-1) *jnp.sin(q1L) *jnp.sin(q2L)) *jnp.sin(rotY))+( \
            16/5) *((-1) *(((2/5) *(1+(-1) *jnp.cos(q2R))+(16/25) *jnp.cos(q2R)) *jnp.sin( \
            q1R)+(6/25) *jnp.cos(q1R) *jnp.sin(q2R)) *(jnp.cos(q1R) *jnp.cos(q2R)+(-1) *jnp.sin( \
            q1R) *jnp.sin(q2R))+(-1) *((-1) *jnp.cos(q2R) *jnp.sin(q1R)+(-1) *jnp.cos(q1R) * \
            jnp.sin(q2R)) *(jnp.cos(q1R) *((2/5) *(1+(-1) *jnp.cos(q2R))+(16/25) *jnp.cos(q2R) \
            )+(-6/25) *jnp.sin(q1R) *jnp.sin(q2R))) *(jnp.cos(rotY) *(jnp.cos(q1R) *jnp.cos(q2R)+ \
            (-1) *jnp.sin(q1R) *jnp.sin(q2R))+(-1) *(jnp.cos(q2R) *jnp.sin(q1R)+jnp.cos(q1R) *jnp.sin( \
            q2R)) *jnp.sin(rotY))+(16/5) *((((2/5) *(1+(-1) *jnp.cos(q2R))+(16/25) * \
            jnp.cos(q2R)) *jnp.sin(q1R)+(6/25) *jnp.cos(q1R) *jnp.sin(q2R)) *(jnp.cos(q2R) *jnp.sin( \
            q1R)+jnp.cos(q1R) *jnp.sin(q2R))+(jnp.cos(q1R) *jnp.cos(q2R)+(-1) *jnp.sin(q1R) *jnp.sin( \
            q2R)) *(jnp.cos(q1R) *((2/5) *(1+(-1) *jnp.cos(q2R))+(16/25) *jnp.cos(q2R))+( \
            -6/25) *jnp.sin(q1R) *jnp.sin(q2R))) *(jnp.cos(rotY) *((-1) *jnp.cos(q2R) *jnp.sin( \
            q1R)+(-1) *jnp.cos(q1R) *jnp.sin(q2R))+(-1) *(jnp.cos(q1R) *jnp.cos(q2R)+(-1) * \
            jnp.sin(q1R) *jnp.sin(q2R)) *jnp.sin(rotY))
            
    M24 = (187/250) *((-1) * jnp.cos(rotY) * jnp.sin(q1R)+(-1) * jnp.cos(q1R) * jnp.sin(rotY))+(16/5) * \
        (((2/5) *(1+(-1) * jnp.cos(q2R))+(16/25) * jnp.cos(q2R)) * jnp.sin(q2R)+(-6/25) * jnp.cos(q2R) * jnp.sin(q2R)) *( jnp.cos(rotY) *( jnp.cos(q1R) * jnp.cos(q2R)+(-1) * \
        jnp.sin(q1R) * jnp.sin(q2R))+(-1)*( jnp.cos(q2R) * jnp.sin(q1R)+ jnp.cos(q1R) * \
        jnp.sin(q2R)) * jnp.sin(rotY))+(16/5) *(((2/5) *(1+(-1) * jnp.cos(q2R))+(16/25) * \
        jnp.cos(q2R)) * jnp.cos(q2R)+(6/25) * jnp.sin(q2R)**2) *( jnp.cos(rotY) *((-1) * \
        jnp.cos(q2R) * jnp.sin(q1R)+(-1) * jnp.cos(q1R) *jnp.sin(q2R))+(-1) *(jnp.cos(q1R) * \
        jnp.cos(q2R)+(-1) * jnp.sin(q1R) * jnp.sin(q2R)) * jnp.sin(rotY))
        
    M25 = (96/125)*(jnp.cos(rotY)*((-1)*jnp.cos(q2R)*jnp.sin(q1R)+(-1)*jnp.cos(q1R)*jnp.sin(q2R))+(-1)* \
        (jnp.cos(q1R)*jnp.cos(q2R)+(-1)*jnp.sin(q1R)* jnp.sin(q2R))*jnp.sin(rotY))
    
    M26 = (187/250) *((-1) *jnp.cos(rotY) *jnp.sin(q1L)+(-1) *jnp.cos(q1L) * \
        jnp.sin(rotY))+(16/5) *(((2/5) * (1+(-1) *jnp.cos(q2L))+ \
        ( 16/25) *jnp.cos(q2L)) *jnp.sin(q2L)+(-6/25) *jnp.cos(q2L) * \
        jnp.sin(q2L)) *(jnp.cos(rotY) *(jnp.cos(q1L) * \
        jnp.cos(q2L)+(-1) *jnp.sin(q1L) *jnp.sin(q2L))+(-1) *(jnp.cos( \
        q2L) *jnp.sin(q1L)+jnp.cos(q1L) *jnp.sin(q2L)) *jnp.sin(rotY))+(16/5) *(((2/5) *( \
        1+(-1) *jnp.cos(q2L))+(16/25) *jnp.cos(q2L)) *jnp.cos(q2L)+(6/25) *jnp.sin(q2L) \
        **2) *(jnp.cos(rotY) *((-1) *jnp.cos(q2L) *jnp.sin(q1L)+(-1) *jnp.cos(q1L) *jnp.sin( \
        q2L))+(-1) *(jnp.cos(q1L) *jnp.cos(q2L)+(-1) *jnp.sin(q1L) *jnp.sin(q2L)) *jnp.sin(rotY))
        
    M27 = (96/125)*(jnp.cos(rotY)*((-1)*jnp.cos(q2L)*jnp.sin(q1L)+(-1)*jnp.cos(q1L)*jnp.sin(q2L))+(-1)*(jnp.cos(q1L)*jnp.cos(q2L)+(-1)*jnp.sin(q1L)*jnp.sin(q2L))*jnp.sin(rotY))
    
    M31 = M13
    M32 = M23
    M33 = (8403/2500)+(34/5)*((11/100)*jnp.cos(q1L)**2+(11/100)*jnp.sin(q1L)**2)**2+ \
        (34/5)*((11/100)*jnp.cos(q1R)**2+(11/100)*jnp.sin(q1R)**2)**2+(16/5)*((-1)*(((2/5)*(1+(-1)* \
        jnp.cos(q2L))+(16/25)*jnp.cos(q2L))*jnp.sin(q1L)+(6/25)*jnp.cos(q1L)*jnp.sin(q2L)) \
        *(jnp.cos(q1L)*jnp.cos(q2L)+(-1)*jnp.sin(q1L)*jnp.sin(q2L))+(-1)*((-1)*jnp.cos( \
        q2L)*jnp.sin(q1L)+(-1)*jnp.cos(q1L)*jnp.sin(q2L))*(jnp.cos(q1L)*((2/5)*(1+( \
        -1)*jnp.cos(q2L))+(16/25)*jnp.cos(q2L))+(-6/25)*jnp.sin(q1L)*jnp.sin(q2L))) \
        **2+(16/5)*((((2/5)*(1+(-1)*jnp.cos(q2L))+(16/25)*jnp.cos(q2L))*jnp.sin( \
        q1L)+(6/25)*jnp.cos(q1L)*jnp.sin(q2L))*(jnp.cos(q2L)*jnp.sin(q1L)+jnp.cos(q1L)* \
        jnp.sin(q2L))+(jnp.cos(q1L)*jnp.cos(q2L)+(-1)*jnp.sin(q1L)*jnp.sin(q2L))*(jnp.cos(q1L) \
        *((2/5)*(1+(-1)*jnp.cos(q2L))+(16/25)*jnp.cos(q2L))+(-6/25)*jnp.sin(q1L) \
        *jnp.sin(q2L)))**2+(16/5)*((-1)*(((2/5)*(1+(-1)*jnp.cos(q2R))+(16/25) \
        *jnp.cos(q2R))*jnp.sin(q1R)+(6/25)*jnp.cos(q1R)*jnp.sin(q2R))*(jnp.cos(q1R)*jnp.cos( \
        q2R)+(-1)*jnp.sin(q1R)*jnp.sin(q2R))+(-1)*((-1)*jnp.cos(q2R)*jnp.sin(q1R)+( \
        -1)*jnp.cos(q1R)*jnp.sin(q2R))*(jnp.cos(q1R)*((2/5)*(1+(-1)*jnp.cos(q2R))+( \
        16/25)*jnp.cos(q2R))+(-6/25)*jnp.sin(q1R)*jnp.sin(q2R)))**2+(16/5)*(((( \
        2/5)*(1+(-1)*jnp.cos(q2R))+(16/25)*jnp.cos(q2R))*jnp.sin(q1R)+(6/25)*jnp.cos( \
        q1R)*jnp.sin(q2R))*(jnp.cos(q2R)*jnp.sin(q1R)+jnp.cos(q1R)*jnp.sin(q2R))+(jnp.cos(q1R) \
        *jnp.cos(q2R)+(-1)*jnp.sin(q1R)*jnp.sin(q2R))*(jnp.cos(q1R)*((2/5)*(1+(-1)* \
        jnp.cos(q2R))+(16/25)*jnp.cos(q2R))+(-6/25)*jnp.sin(q1R)*jnp.sin(q2R)))**2
    
    M34 = (67/100)+(187/250)*((11/100)*jnp.cos(q1R)**2+(11/100)*jnp.sin(q1R)**2)+( \
        16/5)*(((2/5)*(1+(-1)*jnp.cos(q2R))+(16/25)*jnp.cos(q2R))*jnp.sin(q2R)+( \
        -6/25)*jnp.cos(q2R)*jnp.sin(q2R))*((-1)*(((2/5)*(1+(-1)*jnp.cos(q2R))+( \
        16/25)*jnp.cos(q2R))*jnp.sin(q1R)+(6/25)*jnp.cos(q1R)*jnp.sin(q2R))*(jnp.cos(q1R) \
        *jnp.cos(q2R)+(-1)*jnp.sin(q1R)*jnp.sin(q2R))+(-1)*((-1)*jnp.cos(q2R)*jnp.sin( \
        q1R)+(-1)*jnp.cos(q1R)*jnp.sin(q2R))*(jnp.cos(q1R)*((2/5)*(1+(-1)*jnp.cos( \
        q2R))+(16/25)*jnp.cos(q2R))+(-6/25)*jnp.sin(q1R)*jnp.sin(q2R)))+(16/5)*((( \
        2/5)*(1+(-1)*jnp.cos(q2R))+(16/25)*jnp.cos(q2R))*jnp.cos(q2R)+(6/25)*jnp.sin( \
        q2R)**2)*((((2/5)*(1+(-1)*jnp.cos(q2R))+(16/25)*jnp.cos(q2R))*jnp.sin( \
        q1R)+(6/25)*jnp.cos(q1R)*jnp.sin(q2R))*(jnp.cos(q2R)*jnp.sin(q1R)+jnp.cos(q1R)* \
        jnp.sin(q2R))+(jnp.cos(q1R)*jnp.cos(q2R)+(-1)*jnp.sin(q1R)*jnp.sin(q2R))*(jnp.cos(q1R) \
        *((2/5)*(1+(-1)*jnp.cos(q2R))+(16/25)*jnp.cos(q2R))+(-6/25)*jnp.sin(q1R) \
        *jnp.sin(q2R)))
        
    M35 = (1/5)+(96/125)*((((2/5)*(1+(-1)*jnp.cos(q2R))+(16/25)*jnp.cos(q2R))*jnp.sin(q1R)+(6/25)*jnp.cos(q1R)*jnp.sin(q2R))*(jnp.cos(q2R)*jnp.sin( \
        q1R)+jnp.cos(q1R)*jnp.sin(q2R))+(jnp.cos(q1R)*jnp.cos(q2R)+(-1)*jnp.sin(q1R)*jnp.sin( \
        q2R))*(jnp.cos(q1R)*((2/5)*(1+(-1)*jnp.cos(q2R))+(16/25)*jnp.cos(q2R))+( \
        -6/25)*jnp.sin(q1R)*jnp.sin(q2R)))
        
    M36 = (67/100)+(187/250)*((11/100)*jnp.cos(q1L)**2+(11/100)*jnp.sin(q1L)**2)+(16/5)*(((2/5)*(1+(-1)*jnp.cos(q2L)) \
        +(16/25)*jnp.cos(q2L))*jnp.sin(q2L)+(-6/25)*jnp.cos(q2L)*jnp.sin(q2L))*((-1) \
        *(((2/5)*(1+(-1)*jnp.cos(q2L))+(16/25)*jnp.cos(q2L))*jnp.sin(q1L)+(6/25) \
        *jnp.cos(q1L)*jnp.sin(q2L))*(jnp.cos(q1L)*jnp.cos(q2L)+(-1)*jnp.sin(q1L)*jnp.sin( \
        q2L))+(-1)*((-1)*jnp.cos(q2L)*jnp.sin(q1L)+(-1)*jnp.cos(q1L)*jnp.sin(q2L))*( \
        jnp.cos(q1L)*((2/5)*(1+(-1)*jnp.cos(q2L))+(16/25)*jnp.cos(q2L))+(-6/25)* \
        jnp.sin(q1L)*jnp.sin(q2L)))+(16/5)*(((2/5)*(1+(-1)*jnp.cos(q2L))+(16/25)* \
        jnp.cos(q2L))*jnp.cos(q2L)+(6/25)*jnp.sin(q2L)**2)*((((2/5)*(1+(-1)*jnp.cos( \
        q2L))+(16/25)*jnp.cos(q2L))*jnp.sin(q1L)+(6/25)*jnp.cos(q1L)*jnp.sin(q2L))*( \
        jnp.cos(q2L)*jnp.sin(q1L)+jnp.cos(q1L)*jnp.sin(q2L))+(jnp.cos(q1L)*jnp.cos(q2L)+(-1)* \
        jnp.sin(q1L)*jnp.sin(q2L))*(jnp.cos(q1L)*((2/5)*(1+(-1)*jnp.cos(q2L))+(16/25) \
        *jnp.cos(q2L))+(-6/25)*jnp.sin(q1L)*jnp.sin(q2L)))
        
    M37 = (1/5)+(96/125)*((((2/5)*(1+(-1)*jnp.cos(q2L))+(16/25)*jnp.cos(q2L))*jnp.sin(q1L)+ \
        (6/25)*jnp.cos(q1L)*jnp.sin(q2L))*(jnp.cos(q2L)*jnp.sin(q1L)+jnp.cos(q1L)*jnp.sin(q2L))+(jnp.cos(q1L)*jnp.cos(q2L)+(-1)*jnp.sin(q1L)*jnp.sin(q2L))*(jnp.cos(q1L)*((2/5)*(1+(-1)* \
        jnp.cos(q2L))+(16/25)*jnp.cos(q2L))+(-6/25)*jnp.sin(q1L)*jnp.sin(q2L)))
    
    
    M41 = M14
    M42 = M24
    M43 = M34
    M44 = (18807/25000)+(16/5)*(((2/5)*(1+(-1)*jnp.cos(q2R))+(16/25)*jnp.cos(q2R))*jnp.sin(q2R)+(-6/25)*jnp.cos(q2R)*jnp.sin(q2R))**2+(16/5)*(((2/5)*(1+(-1)*jnp.cos(q2R))+(16/25)*jnp.cos(q2R))*jnp.cos(q2R)+(6/25)*jnp.sin(q2R)**2)**2
    
    M45 = (1/5)+(96/125)*(((2/5)*(1+(-1)*jnp.cos(q2R))+(16/25)*jnp.cos(q2R))*jnp.cos(q2R)+(6/25)*jnp.sin(q2R)**2)
    
    M46 = 0.0
    M47 = 0.0
    
    M51 = M15
    M52 = M25
    M53 = M35
    M54 = M45
    M55 = (1201/3125)
    M56 = 0.0
    M57 = 0.0
    
    M61 = M16
    M62 = M26
    M63 = M36
    M64 = M46
    M65 = M56
    M66 = (18807/25000)+(16/5)*(((2/5)*(1+(-1)*jnp.cos(q2L))+(16/25)*jnp.cos(q2L))*jnp.sin(q2L)+(-6/25)* \
        jnp.cos(q2L)*jnp.sin(q2L))**2+(16/5)*(((2/5)*(1+(-1)*jnp.cos(q2L))+(16/25) \
        *jnp.cos(q2L))*jnp.cos(q2L)+(6/25)*jnp.sin(q2L)**2)**2
        
    M67 = (1/5)+(96/125)*(((2/5)*(1+(-1)*jnp.cos(q2L))+(16/25)*jnp.cos(q2L))*jnp.cos(q2L)+(6/25)*jnp.sin(q2L)**2)
    
    M71 = M17
    M72 = M27
    M73 = M37
    M74 = M47
    M75 = M57
    M76 = M67
    M77 = 1201/3125
    
    Mmat = jnp.array([
        [ M11, M12, M13, M14, M15, M16, M17 ],
        [ M21, M22, M23, M24, M25, M26, M27 ],
        [ M31, M32, M33, M34, M35, M36, M37 ],
        [ M41, M42, M43, M44, M45, M46, M47 ],
        [ M51, M52, M53, M54, M55, M56, M57 ],
        [ M61, M62, M63, M64, M65, M66, M67 ],
        [ M71, M72, M73, M74, M75, M76, M77]
    ])
    
    return Mmat


def G_vector(q):
    # Unpack the 7 generalized coordinates (MATLAB q(1) \ q(7))
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
  
  
def C_matrix(q, dq):
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
                dD_ij_dqk = jax.grad(lambda qq: D_matrix(qq)[i, j])(q)[k]
                dD_ik_dqj = jax.grad(lambda qq: D_matrix(qq)[i, k])(q)[j]
                dD_jk_dqi = jax.grad(lambda qq: D_matrix(qq)[j, k])(q)[i]
                temp += 0.5 * dq[k] * (dD_ij_dqk + dD_ik_dqj - dD_jk_dqi)
            # Update the (i, j) entry in C_mat
            C_mat = C_mat.at[i, j].set(temp)
    return C_mat


def B_matrix():
    return jnp.vstack([jnp.zeros(3,4), jnp.eye(4)])

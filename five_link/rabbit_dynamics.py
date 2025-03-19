import jax.numpy as jnp
import jax

@jax.jit
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

@jax.jit
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
    # Here the second term inside the bracket involves (cos(q1R)*cos(rotY) - sin(q1R)*sin(rotY))
    commonR = jnp.cos(q1R)*jnp.cos(rotY) - jnp.sin(q1R)*jnp.sin(rotY)
    G4 = -0.31392e2 * (
            (-2/5)*jnp.cos(q2R)*expr_q1R +
            (2/5)*jnp.sin(q2R)*commonR +
            (16/25)*( jnp.cos(q2R)*expr_q1R - jnp.sin(q2R)*commonR )
         )

    # Row 6:
    innerL_row5 = (2/5)*(1 - jnp.cos(q2L))*expr_q1L \
                  - (2/5)*jnp.sin(q2L)*expr_q1L_c \
                  + (16/25)*( jnp.cos(q2L)*expr_q1L + jnp.sin(q2L)*expr_q1L_c )
    G5 = -0.733788e1 * expr_q1L - 0.31392e2 * innerL_row5

    # Row 7:
    commonL = jnp.cos(q1L)*jnp.cos(rotY) - jnp.sin(q1L)*jnp.sin(rotY)
    G6 = -0.31392e2 * (
            (-2/5)*jnp.cos(q2L)*expr_q1L +
            (2/5)*jnp.sin(q2L)*commonL +
            (16/25)*( jnp.cos(q2L)*expr_q1L - jnp.sin(q2L)*commonL )
         )

    return jnp.array([G0, G1, G2, G3, G4, G5, G6])
  
@jax.jit
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

@jax.jit
def B_matrix():
    return jnp.vstack([jnp.zeros((3,4)), jnp.eye(4)])

# =========================================
#             Left Swing Foot 
# =========================================
@jax.jit
def Left_Swing_Foot_Position(q):
    xbar,zbar,rotY,q1R,q2R,q1L,q2L = q[0],q[1],q[2],q[3],q[4],q[5],q[6]

    pos_L_sw = jnp.array([xbar+(2/5)*(1+(-1)*jnp.cos(q2L))*(jnp.cos(rotY)*jnp.sin(q1L)+jnp.cos(q1L)* \
      jnp.sin(rotY))+(-2/5)*jnp.sin(q2L)*(jnp.cos(q1L)*jnp.cos(rotY)+(-1)*jnp.sin(q1L) \
      *jnp.sin(rotY))+(4/5)*(jnp.cos(q2L)*(jnp.cos(rotY)*jnp.sin(q1L)+jnp.cos(q1L)* \
      jnp.sin(rotY))+jnp.sin(q2L)*(jnp.cos(q1L)*jnp.cos(rotY)+(-1)*jnp.sin(q1L)*jnp.sin( \
      rotY))),zbar+(-2/5)*jnp.sin(q2L)*((-1)*jnp.cos(rotY)*jnp.sin(q1L)+(-1)* \
      jnp.cos(q1L)*jnp.sin(rotY))+(2/5)*(1+(-1)*jnp.cos(q2L))*(jnp.cos(q1L)*jnp.cos( \
      rotY)+(-1)*jnp.sin(q1L)*jnp.sin(rotY))+(4/5)*(jnp.sin(q2L)*((-1)*jnp.cos( \
      rotY)*jnp.sin(q1L)+(-1)*jnp.cos(q1L)*jnp.sin(rotY))+jnp.cos(q2L)*(jnp.cos(q1L)* \
      jnp.cos(rotY)+(-1)*jnp.sin(q1L)*jnp.sin(rotY)))])
    return pos_L_sw
J_swing_foot = jax.jacrev(Left_Swing_Foot_Position)

@jax.jit
def vel_left_foot(q, dotq):
    return J_swing_foot(q)@dotq
Jdot_swing_foot = jax.jacrev(vel_left_foot, argnums=0)
  
  
# =========================================
#             Right Stance Foot 
# =========================================
@jax.jit
def Right_Stance_Foot_Position(q):
  xbar,zbar,rotY,q1R,q2R,q1L,q2L = q[0],q[1],q[2],q[3],q[4],q[5],q[6]

  posR = jnp.array([xbar+(2/5)*(1+(-1)*jnp.cos(q2R))*(jnp.cos(rotY)*jnp.sin(q1R)+jnp.cos(q1R)* \
    jnp.sin(rotY))+(-2/5)*jnp.sin(q2R)*(jnp.cos(q1R)*jnp.cos(rotY)+(-1)*jnp.sin(q1R) \
    *jnp.sin(rotY))+(4/5)*(jnp.cos(q2R)*(jnp.cos(rotY)*jnp.sin(q1R)+jnp.cos(q1R)* \
    jnp.sin(rotY))+jnp.sin(q2R)*(jnp.cos(q1R)*jnp.cos(rotY)+(-1)*jnp.sin(q1R)*jnp.sin( \
    rotY))),
    0,
    zbar+(-2/5)*jnp.sin(q2R)*((-1)*jnp.cos(rotY)*jnp.sin(q1R)+(-1)* \
    jnp.cos(q1R)*jnp.sin(rotY))+(2/5)*(1+(-1)*jnp.cos(q2R))*(jnp.cos(q1R)*jnp.cos( \
    rotY)+(-1)*jnp.sin(q1R)*jnp.sin(rotY))+(4/5)*(jnp.sin(q2R)*((-1)*jnp.cos( \
    rotY)*jnp.sin(q1R)+(-1)*jnp.cos(q1R)*jnp.sin(rotY))+jnp.cos(q2R)*(jnp.cos(q1R)* \
    jnp.cos(rotY)+(-1)*jnp.sin(q1R)*jnp.sin(rotY)))])

  posR = jnp.array([posR[0], posR[2]]);
  return posR
J_stance_foot = jax.jacrev(Right_Stance_Foot_Position)

@jax.jit
def vel_right_foot(q, dotq):
    return J_stance_foot(q)@dotq
Jdot_stance_foot = jax.jacrev(vel_right_foot, argnums=0)

@jax.jit
def Hip_Position(q):
    xbar,zbar,rotY,q1R,q2R,q1L,q2L = q[0],q[1],q[2],q[3],q[4],q[5],q[6]

    posHip = jnp.array([xbar+(63/100)*jnp.sin(rotY), zbar+(63/100)*jnp.cos(rotY)])

    return posHip
J_hip = jax.jacrev(Hip_Position)

@jax.jit
def COM_Position(q):
    xbar,zbar,rotY,q1R,q2R,q1L,q2L = q[0],q[1],q[2],q[3],q[4],q[5],q[6]

    posCOM = jnp.array([(1/32)*(12*(xbar+(6/25)*jnp.sin(rotY))+(34/5)*(xbar+(11/100)*(jnp.cos( \
      rotY)*jnp.sin(q1L)+jnp.cos(q1L)*jnp.sin(rotY)))+(34/5)*(xbar+(11/100)*( \
      jnp.cos(rotY)*jnp.sin(q1R)+jnp.cos(q1R)*jnp.sin(rotY)))+(16/5)*(xbar+(2/5)*(1+ \
      (-1)*jnp.cos(q2L))*(jnp.cos(rotY)*jnp.sin(q1L)+jnp.cos(q1L)*jnp.sin(rotY))+( \
      -2/5)*jnp.sin(q2L)*(jnp.cos(q1L)*jnp.cos(rotY)+(-1)*jnp.sin(q1L)*jnp.sin(rotY)) \
      +(16/25)*(jnp.cos(q2L)*(jnp.cos(rotY)*jnp.sin(q1L)+jnp.cos(q1L)*jnp.sin(rotY))+ \
      jnp.sin(q2L)*(jnp.cos(q1L)*jnp.cos(rotY)+(-1)*jnp.sin(q1L)*jnp.sin(rotY))))+( \
      16/5)*(xbar+(2/5)*(1+(-1)*jnp.cos(q2R))*(jnp.cos(rotY)*jnp.sin(q1R)+jnp.cos( \
      q1R)*jnp.sin(rotY))+(-2/5)*jnp.sin(q2R)*(jnp.cos(q1R)*jnp.cos(rotY)+(-1)* \
      jnp.sin(q1R)*jnp.sin(rotY))+(16/25)*(jnp.cos(q2R)*(jnp.cos(rotY)*jnp.sin(q1R)+ \
      jnp.cos(q1R)*jnp.sin(rotY))+jnp.sin(q2R)*(jnp.cos(q1R)*jnp.cos(rotY)+(-1)*jnp.sin( \
      q1R)*jnp.sin(rotY))))), \
      (1/32)*(12*(zbar+(6/25)*jnp.cos(rotY))+(34/5)* \
      (zbar+(11/100)*(jnp.cos(q1L)*jnp.cos(rotY)+(-1)*jnp.sin(q1L)*jnp.sin(rotY)))+( \
      34/5)*(zbar+(11/100)*(jnp.cos(q1R)*jnp.cos(rotY)+(-1)*jnp.sin(q1R)*jnp.sin( \
      rotY)))+(16/5)*(zbar+(-2/5)*jnp.sin(q2L)*((-1)*jnp.cos(rotY)*jnp.sin(q1L) \
      +(-1)*jnp.cos(q1L)*jnp.sin(rotY))+(2/5)*(1+(-1)*jnp.cos(q2L))*(jnp.cos(q1L) \
      *jnp.cos(rotY)+(-1)*jnp.sin(q1L)*jnp.sin(rotY))+(16/25)*(jnp.sin(q2L)*((-1) \
      *jnp.cos(rotY)*jnp.sin(q1L)+(-1)*jnp.cos(q1L)*jnp.sin(rotY))+jnp.cos(q2L)*(jnp.cos( \
      q1L)*jnp.cos(rotY)+(-1)*jnp.sin(q1L)*jnp.sin(rotY))))+(16/5)*(zbar+(-2/5) \
      *jnp.sin(q2R)*((-1)*jnp.cos(rotY)*jnp.sin(q1R)+(-1)*jnp.cos(q1R)*jnp.sin(rotY) \
      )+(2/5)*(1+(-1)*jnp.cos(q2R))*(jnp.cos(q1R)*jnp.cos(rotY)+(-1)*jnp.sin(q1R) \
      *jnp.sin(rotY))+(16/25)*(jnp.sin(q2R)*((-1)*jnp.cos(rotY)*jnp.sin(q1R)+(-1) \
      *jnp.cos(q1R)*jnp.sin(rotY))+jnp.cos(q2R)*(jnp.cos(q1R)*jnp.cos(rotY)+(-1)*jnp.sin( \
      q1R)*jnp.sin(rotY)))))])

    return posCOM
J_com_world = jax.jacrev(COM_Position)

@jax.jit
def vel_com_world(q, dotq):
    return J_com_world(q) @ dotq
  
@jax.jit
def pos_com_right_foot(q):
  return COM_Position(q) - Right_Stance_Foot_Position(q)
J_com_stance_foot = jax.jacrev(pos_com_right_foot)

@jax.jit
def vel_com_right_foot(q, dotq):
    return J_com_stance_foot(q)@dotq


# ==================================== 
#       Floating base dynamics
# ==================================== 
@jax.jit
def f_qddot_wrench(x, u):
    n_q = 7
    n_w = 2
    n_u = 4
    q = x[0:n_q]
    dq = x[n_q:]
    B = 50.0*B_matrix() # Multiply by 50 b/c of gear reduction
    # C = C_matrix(q, dq)
    C = jnp.zeros((n_q, n_q)) # omit Coriolis for now
    D = D_matrix(q)
    G = -G_vector(q).flatten()
    
    J_right_foot = J_stance_foot(q)
    Jdot_right_foot = Jdot_stance_foot(q, dq)
    
    De = jnp.vstack([jnp.hstack([D, -J_right_foot.T]), 
                     jnp.hstack([J_right_foot, jnp.zeros((n_w,n_w))])])
    
    Ce = jnp.vstack([C, Jdot_right_foot])
    Ge = jnp.concatenate([G, jnp.zeros(n_w).flatten()]).flatten()
    Be = jnp.vstack([B, jnp.zeros((n_w,n_u))])
    
    # Compute ddot_q and wrench
    qddot_wrench = jnp.linalg.solve(De, -Ce @ dq - Ge.flatten() + Be@u)
    
    return qddot_wrench


@jax.jit
def f_NL_fivelink(t, x, *args):
    
    n_q = 7
    n_u = 4
    dq = x[n_q:]
    
    if len(args) == 0:
        u = jnp.zeros(n_u)
    else:
        u = args[0]
    qddot_wrench = f_qddot_wrench(x, u)
    qddot = qddot_wrench[:n_q]
    xdot = jnp.concatenate([dq, qddot]).flatten()
    
    return xdot


@jax.jit
def wrench_st(x, u):
    n_q = 7
    qddot_wrench = f_qddot_wrench(x, u)
    w_sym = qddot_wrench[n_q:]
    return w_sym


@jax.jit
def f_euler_fivelink(x,u,dt):
    return x + f_NL_fivelink(0.0,x,u)*dt

@jax.jit
def f_rk4_fivelink(x, u, dt):
    # Runge Kutta
    M = 4
    dt_rk = dt / M
    x_rk4 = x
    for j in range(M):
        k1 = f_NL_fivelink(0.0, x_rk4, u);
        k2 = f_NL_fivelink(0.0, x_rk4 + (dt_rk/2)*k1,u)
        k3 = f_NL_fivelink(0.0, x_rk4 + (dt_rk/2)*k2,u)
        k4 = f_NL_fivelink(0.0, x_rk4 + dt_rk*k3,u)
        x_rk4 = x_rk4 + (dt_rk/6) * (k1 + 2*k2 + 2*k3 + k4)
    return x_rk4



# ---------------------------------
#       Uncontrolled Dynamics
# ---------------------------------
@jax.jit
def f_qddot_wrench_uncontrolled(x):
    n_q = 7
    n_w = 2

    q = x[0:n_q]
    dq = x[n_q:]

    C = jnp.zeros((n_q, n_q)) # omit Coriolis for now
    D = D_matrix(q)
    G = -G_vector(q).flatten()
    
    J_right_foot = J_stance_foot(q)
    Jdot_right_foot = Jdot_stance_foot(q, dq)
    
    De = jnp.vstack([jnp.hstack([D, -J_right_foot.T]), 
                     jnp.hstack([J_right_foot, jnp.zeros((n_w,n_w))])])
    
    Ce = jnp.vstack([C, Jdot_right_foot])
    Ge = jnp.concatenate([G, jnp.zeros(n_w).flatten()]).flatten()
    
    # Compute ddot_q and wrench
    qddot_wrench = jnp.linalg.solve(De, -Ce @ dq - Ge.flatten())
    
    return qddot_wrench

@jax.jit
def f_NL_fivelink_uncontrolled(t, x):
    n_q = 7
    dq = x[n_q:]
    
    qddot_wrench = f_qddot_wrench_uncontrolled(x)
    qddot = qddot_wrench[:n_q]
    xdot = jnp.concatenate([dq, qddot]).flatten()
    
    return xdot

# ============================
#          Impact Map
# ============================
# ----------------------------------
#   Reset Map from mode 1 to mode 2
# ----------------------------------
@jax.jit
def impact_map(x):
    q = x[0:7]
    qdot = x[7:]
    
    # Compute impact contact force and post-impact velocities
    D = D_matrix(q)
    J_left_foot = J_swing_foot(q)
    
    De = jnp.vstack([jnp.hstack([D, -J_left_foot.T]), 
                     jnp.hstack([J_left_foot, jnp.zeros((2,2))])])
    RHS = jnp.concatenate([(D@qdot).flatten(), jnp.zeros(2)])
    resetmap_5link_12 = jnp.linalg.solve(De, RHS)
    
    x_impact = jnp.concatenate([q, resetmap_5link_12[0:7]])
    
    # Relabel the links to switch the swing foot and the stance foot
    # Original order: [xbar, zbar, rotY, q1R, q2R, q1L, q2L]
    EYE = jnp.eye(14)
    new_order = jnp.array([0, 1, 2, 5, 6, 3, 4, 7, 8, 9, 12, 13, 10, 11])
    Rotation = EYE[new_order, :]
    x_new = Rotation@x_impact
    
    return x_new.flatten()

def resetmap_5link_12(x_event, current_mode, reset_args):
    next_mode = 1
    other_output = None
    return impact_map(x_event), next_mode, other_output

Rx_5link_12 = jax.jacrev(impact_map)
def Rt_5link_12(x):
    return 0.0

@jax.jit
def impact_wrench(x):
    q = x[0:7]
    qdot = x[7:]
    
    # Compute impact contact force and post-impact velocities
    D = D_matrix(q)
    J_left_foot = J_swing_foot(q)
    
    De = jnp.vstack([jnp.hstack([D, -J_left_foot.T]), 
                     jnp.hstack([J_left_foot, jnp.zeros((2,2))])])
    RHS = jnp.concatenate([(D@qdot).flatten(), jnp.zeros(2)])
    resetmap_5link_12 = jnp.linalg.solve(De, RHS)
    
    return resetmap_5link_12[7:]

# ----------------------------------
#   Reset Map from mode 2 to mode 1
# ----------------------------------
def resetmap_5link_21(x_event, current_mode, reset_args):
    return x_event, 0, None

def Rx_5link_21(x):
    nx = x.shape[0]
    return jnp.eye(nx)

def Rt_5link_21(x):
    return 0.0

# -------------------------------
#          Guard Functions
# -------------------------------
# Mode 0: swing foot descending
# Mode 1: Swing foot ascending

# Event from mode 0 to mode 1
def sw_foot_ground_touching_event(x):
    below_ground = Left_Swing_Foot_Position(x[0:7])[1] <= 0.0
    negative_vel = vel_left_foot(x[0:7], x[7:14])[1] < 0.0
    
    return (below_ground and negative_vel)

# guard function from mode 0 to mode 1
def guard_12_5link(t, x):
    return Left_Swing_Foot_Position(x)[1]
guard_12_5link.direction = -1

gx_12_5link = jax.jacrev(guard_12_5link, argnums=[1])

def gt_12_5link(t, x):
    return 0.0

# Event for from mode 1 to mode 0
def sw_foot_descending_event(x):
    above_ground = Left_Swing_Foot_Position(x[0:7])[1] > 0.0
    negative_vel = vel_left_foot(x[0:7], x[7:14])[1] < 0.0
    
    return (above_ground and negative_vel)

# guard function from mode 1 to mode 0
def guard_21_5link(t, x):
    return vel_left_foot(x[0:7], x[7:14])[1]
guard_21_5link.direction = -1

gx_21_5link = jax.jacrev(guard_21_5link, argnums=1)

def gt_21_5link(t, x):
    return 0.0


# -------------------------------- From mode 1 to mode 2  --------------------------------
def guard_cond_fivelink_12(xt, xt_next, current_mode):
    # assume time invariant guard for now
    return (current_mode==0) and (guard_12_5link(0.0,xt)>=0) and (guard_12_5link(0.0,xt_next)<0)

# -------------------------------- From mode 2 to mode 1  --------------------------------
def guard_cond_fivelink_21(xt, xt_next, current_mode):
    # assume time invariant guard for now
    return (current_mode==1) and (guard_21_5link(0.0,xt)>=0) and (guard_21_5link(0.0,xt_next)<0)

# ============================
#       Cost Functions
# ============================
def com_moving_cost(x, u, target_com_vel_x= 1.0):
    # com_world_velocity_x = vel_com_world(x[0:7], x[7:])[0]
    # return 0.01*jnp.linalg.norm(com_world_velocity_x-target_com_vel_x) + u.T@u/2
    return u.T@u/2

def deltx_norm_cost(x, x_tar):
    return jnp.linalg.norm(x-x_tar)

# ====================================
#       Event Detect Function
# ====================================
from dynamics.dynamics_discrete import event_detect_onestep_discrete

def detect_fivelink(current_mode, 
                    x0, u, 
                    t0, dt, 
                    reset_args, 
                    detect=True, 
                    backwards=False):
    
    smoothdyn_5link = {0:f_NL_fivelink, 1:f_NL_fivelink}
    
    Rxs_5link = {0:Rx_5link_12, 1:Rx_5link_21}
    Rts_5link = {0:Rt_5link_12, 1:Rt_5link_21}
    
    gxs_5link = {0:gx_12_5link, 1:gx_21_5link}
    gts_5link = {0:gt_12_5link, 1:gt_21_5link}
    
    guards_5link = {0:guard_12_5link, 1: guard_21_5link}
    resetmaps_5link = {0:resetmap_5link_12, 1:resetmap_5link_21}
    
    return event_detect_onestep_discrete(x0, u, 
                                         t0, dt, 
                                         current_mode, 
                                         smoothdyn_5link, 
                                         guards_5link, 
                                         gxs_5link, gts_5link, 
                                         resetmaps_5link, 
                                         Rxs_5link, Rts_5link, 
                                         reset_args, 
                                         guard_cond_func_0=guard_cond_fivelink_12, 
                                         guard_cond_func_1=guard_cond_fivelink_21, 
                                         detection=True, 
                                         backwards=False)
from five_link.rabbit_kinematics import *
from five_link.rabbit_dynamics import *

# ============================
#       Cost Functions
# ============================
def com_moving_cost(x, u, target_com_vel_x= 0.1):
    com_world_pos_z = COM_Position(x[0:7])[1]
    com_world_velocity_x = vel_com_world(x[0:7], x[7:])[0]
    hip_z = Hip_Position(x[0:7])[2]
    # return jnp.linalg.norm(com_world_velocity_x-target_com_vel_x) - 0.1*hip_z + 0.01*u.T@u/2
    return 5.0*jnp.linalg.norm(com_world_velocity_x-target_com_vel_x) + 10.0*jnp.linalg.norm(com_world_pos_z-0.6) + 1.0*(x[2]-jnp.pi/6)**2 + 0.01*u.T@u/2
    # return u.T@u/2

@jax.jit
def deltx_norm_cost_fivelink(x, x_tar):
    return 0.2*jnp.linalg.norm(x[2:8]-x_tar[2:8]) + 0.2*jnp.linalg.norm(x[10:]-x_tar[10:])
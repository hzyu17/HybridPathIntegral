# covariance steering for 1D bouncing ball dynamics
# Hongzhe Yu, 02/26/2024

from control.hybrid_riccati import *
from simulation.bouncing_ball_1D import *


def covsteering_bouncing_1D(x0, xT, Sig0, SigT, t0, tf):
    
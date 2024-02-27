import scipy
import matplotlib.pyplot as plt

import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from tools.plot_ellipsoid import *
from dynamics.constant_flow import *
from simulation_2D import *


def ode_constant_flow(z0, vz0, t0, tf, nt, Modes, n_events=None, return_saltation=False, Sig0=None):
    return simulation_2d(z0, vz0, t0, tf, nt, Modes, 
                         guard_ctflow, resetmap_ctflow, mode_jump_ctflow, 
                         Rx_ctflow, Rt_ctflow, gx_ctflow, gt_ctflow, linearization_ctflow, 
                         n_events, return_saltation, Sig0, guard_direction=1)

## A collection of 1D bouncing balls which are sampled from a Gaussian distribution
#  N(m0, Sig0), m0=[x1, x2]^T.

def constflow_samples(x1, x2, Sig0, t0, tf, nt, n_samples, Modes, n_events=None):
    """
    The bouncing ball solution for a set of sampled trajectories.
    """
    t_collections = []
    trj_collections = []
    tevent_collections = []
    xevent_collections = []
    xreset_collections = []
    
    m0 = np.array([x1, x2], dtype=np.float64)
    initial_distribution = np.zeros([n_samples, 2], dtype=np.float64)
    
    for i_s in range(n_samples):
        x0 = m0 + scipy.linalg.sqrtm(Sig0)@np.random.randn(2)
        
        t_i, trj_i, tevents_i, xevents_i, xresets_i, _ = ode_constant_flow(x0[0], x0[1], t0, tf, nt, Modes, n_events, return_saltation=False, Sig0=None)
        
        t_collections.append(t_i)
        trj_collections.append(trj_i)
        tevent_collections.append(tevents_i)
        xevent_collections.append(xevents_i)
        xreset_collections.append(xresets_i)
        initial_distribution[i_s] = x0
             
    return t_collections, trj_collections, tevent_collections, xevent_collections, xreset_collections, initial_distribution


if __name__ == '__main__':
    
    x1 = 0.0
    x2 = 0.0
    t0 = 0.0
    tf = 20.0
    nt = 300
    n_events = 2
    
    # -------------------------- Plot samples --------------------------
    sig0 = 0.1
    Sig0 = sig0*np.eye(2)
    
    Modes = [dyn_f1, dyn_f2]
        
    # ========================== Plot collection movements ========================== 
    t_mean, xX_mean, t_event, x_event, x_reset, saltations = ode_constant_flow(x1, x2, t0, tf, nt, Modes, n_events=n_events, return_saltation=True, Sig0=Sig0)
    x_mean = xX_mean[0:2, :]
    
    n_samples = 500
    t_collections, trj_collections, tevent_collections, xevent_collections, xreset_collections, initial_distribution = constflow_samples(x1, x2, Sig0, t0, tf, nt, n_samples, Modes, n_events=n_events)

    fig1, axs = plt.subplots(2, 2)
    ax1, ax2, ax3, ax4 = axs.flatten()
    
    # ========================== ax1: Plot z-t ========================== 
    # -------------------------- Plot samples --------------------------
    for t_i, trj_i in zip(t_collections, trj_collections):
        ax1.plot(t_i, trj_i[0,:].T, '-.', alpha=0.1)
        
    # -------------------------- Plot mean --------------------------
    ax1.plot(t_mean, x_mean[0,:].T, 'r')
                
    ax1.grid(True)
    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$z$')
    ax1.set_title('1D Bouncing ball')
    
    # ========================== ax2: z - vz plot ========================== 
    # -------------------------- Plot samples --------------------------
    for t_i, trj_i in zip(t_collections, trj_collections):
        ax2.plot(trj_i[0,:].T, trj_i[1,:].T, '-.', alpha=0.1)
        
    # -------------------------- Plot mean --------------------------
    ax2.plot(x_mean[0,:].T, x_mean[1,:].T, 'r')
    
    # -------------------------- Plot covariance --------------------------   
    for i in range(0, n_events*nt, 20):
        _, ax2 = plot_2d_ellipsoid_boundary(xX_mean[0:2,i], xX_mean[2:6,i].reshape([2,2]), ax2, 'k')
    
    ax2.grid(True)
    ax2.set_xlabel('z')
    ax2.set_ylabel(r'$\dot z$')
    
    # ========================= plot the pre-contact samples and post-contact samples =========================    
    # ======================= ax3: pre-contact samples =======================
    ## ------ find the earliest contact time -----
    tevent_min = np.min(np.array(tevent_collections), axis=0)
    
    # ------ find the timestamp for other samples that is close to the earliest contact time ------    
    pre_contact_index = np.argmax(np.array(t_collections) > tevent_min[0], axis=1)
    
    # ------ plot the distribution of the samples right before the ealiest contact -----
    for sample_i in range(n_samples):
        ax3.scatter(trj_collections[sample_i][0, pre_contact_index[sample_i]], 
                    trj_collections[sample_i][1, pre_contact_index[sample_i]], s=1.2, c='b', alpha=0.5)

    # ----- plot the covariance ellipsoid -----
    # find the mean state at the first pre-contact time
    pre_contact_index_mean = np.argmax(t_mean > tevent_min[0])
    pre_contact_mean = xX_mean[0:2, pre_contact_index_mean]
    pre_contact_cov = xX_mean[2:6, pre_contact_index_mean].reshape([2,2])
    
    _, ax3 = plot_2d_ellipsoid_boundary(pre_contact_mean, pre_contact_cov, ax3, 'r')
    
    # plot the pre-contact time covariance in the covariance trajectory
    _, ax2 = plot_2d_ellipsoid_boundary(pre_contact_mean, pre_contact_cov, ax2, 'r')
    
    # ======================= ax4: post-contact samples =======================
    ## ------ find the latest contact time -----
    tevent_max = np.max(np.array(tevent_collections), axis=0)
        
    # ------ find the timestamp for other samples that is close to the earliest contact time ------      
    post_contact_index = np.argmax(np.array(t_collections) > tevent_max[0] + 0.1, axis=1)
    
    # ------ plot the distribution of the samples right before the ealiest contact -----
    for sample_i in range(n_samples):
        ax4.scatter(trj_collections[sample_i][0, post_contact_index[sample_i]], 
                    trj_collections[sample_i][1, post_contact_index[sample_i]], s=1.2, c='b', alpha=0.5)
        
    # ----- plot the covariance ellipsoid -----
    # find the mean state at the last post-contact time
    post_times = t_mean > tevent_max[0] + 0.1
    
    post_contact_index_mean = np.argmax(t_mean > tevent_max[0] + 0.1)
    post_contact_mean = xX_mean[0:2, post_contact_index_mean]
    post_contact_cov = xX_mean[2:6, post_contact_index_mean].reshape([2,2])
    
    _, ax4 = plot_2d_ellipsoid_boundary(post_contact_mean, post_contact_cov, ax4, 'g')
    
    # plot the post-time covariance in the covariance trajectory
    _, ax2 = plot_2d_ellipsoid_boundary(post_contact_mean, post_contact_cov, ax2, 'g')
    
    ax3.set_title('Pre-contact')
    ax4.set_title('Post-contact')
    
    ax3.grid(True)
    ax4.grid(True)
    
    # ax3.axis('equal')
    ax3.set_xlabel(r'$z$')
    ax3.set_ylabel(r'$\dot z$')
    
    # ax4.axis('equal')
    ax4.set_xlabel(r'$z$')
    ax4.set_ylabel(r'$\dot z$')
    
    plt.show()
    
    # fig1, axs = plt.subplots(2, 2)
    # ax1, ax2, ax3, ax4 = axs.flatten()
    
    # for t_i, trj_i in zip(t_collections, trj_collections):
    #     ax1.plot(t_i, trj_i[0,:].T, '-.', alpha=0.1)
        
    # # -------------------------- Plot mean --------------------------
    # ax1.plot(t_mean, x_mean[0,:].T, 'r')
    
    # ax1.grid(True)
    # ax1.set_xlabel(r'$t$')
    # ax1.set_ylabel(r'$z$')
    # ax1.set_title('Constant flow')
    
    # # ========================== z - vz plot ==========================     
    # # -------------------------- Plot samples --------------------------
    # for t_i, trj_i in zip(t_collections, trj_collections):
    #     ax2.plot(trj_i[0,:].T, trj_i[1,:].T, '-.', alpha=0.1)
        
    # # -------------------------- Plot mean --------------------------
    # ax2.plot(x_mean[0,:].T, x_mean[1,:].T, 'r')
    # ax2.grid(True)
    # ax2.set_xlabel(r'$z$')
    # ax2.set_ylabel(r'$\dot z$')
    # ax2.axis('equal')
    
    # # ========================= plot the pre-contact samples and post-contact samples =========================    
    # # ======================= pre-contact samples =======================
    # ## ------ find the earliest contact time -----
    # tevent_min = np.min(np.array(tevent_collections))
    # print("tevent_min")
    # print(tevent_min)
    
    # # ------ find the timestamp for other samples that is close to the earliest contact time ------ 
    # less_than_tevent_min = np.array(t_collections) <= tevent_min
    # pre_contact_index = np.max(np.where(less_than_tevent_min, np.arange(less_than_tevent_min.shape[1]), -1), axis=1)

    # # ------ plot the distribution of the samples right before the ealiest contact -----
    # for sample_i in range(n_samples):
    #     ax3.scatter(trj_collections[sample_i][0, pre_contact_index[sample_i]], 
    #                 trj_collections[sample_i][1, pre_contact_index[sample_i]], s=1.2, c='b', alpha=0.5)
        
    #     ax3.scatter(initial_distribution[sample_i, 0], initial_distribution[sample_i, 1], s=1.2, c='b', alpha=0.5)
    
    # # ----- plot the covariance ellipsoid -----
    # # find the mean state at the first pre-contact time
    # pre_contact_index_mean = np.max(np.where(t_mean <= tevent_min, np.arange(len(t_mean)), -1))
    # pre_contact_mean = x_mean[:, pre_contact_index_mean]
    # ellipse_boundary, ax3 = plot_2d_ellipsoid_boundary(pre_contact_mean, Sig0, ax3, 'r')
    
    # # ======================= post-contact samples =======================
    # ## ------ find the latest contact time -----
    # tevent_max = np.max(np.array(tevent_collections))
        
    # # ------ find the timestamp for other samples that is close to the latest contact time ------      
    # post_contact_index = np.argmax(np.array(t_collections) >= tevent_max, axis=1)
    
    # # ------ plot the distribution of the samples right after the latest contact -----
    # for sample_i in range(n_samples):
    #     ax4.scatter(trj_collections[sample_i][0, post_contact_index[sample_i]], 
    #                 trj_collections[sample_i][1, post_contact_index[sample_i]], s=1.2, c='b', alpha=0.5)
        
    # # ----- plot the covariance ellipsoid -----
    # # find the mean state at the last post-contact time
    # post_contact_index_mean = np.argmax(t_mean > tevent_max)
    # post_contact_mean = x_mean[:, post_contact_index_mean]
    
    # print("saltations[0]")
    # print(saltations[0])
    
    # Cov_plus = saltations[0] @ Sig0 @ saltations[0].transpose()
    
    # print("Cov_plus")
    # print(Cov_plus)
    
    # _, ax4 = plot_2d_ellipsoid_boundary(post_contact_mean, Cov_plus, ax4, 'g')
    
    # ax3.set_title('Pre-contact')
    # ax4.set_title('Post-contact')
    
    # ax3.grid(True)
    # ax4.grid(True)
    
    # ax3.axis('equal')
    # ax3.set_xlabel(r'$z$')
    # ax3.set_ylabel(r'$\dot z$')
    
    # ax4.axis('equal')
    # ax4.set_xlabel(r'$z$')
    # ax4.set_ylabel(r'$\dot z$')
    
    # plt.show()
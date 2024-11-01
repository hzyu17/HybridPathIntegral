import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    
    
    x0 = 2.0
    dt = 0.001
    nt = 2000
    n_samples = 50
    xt_trj = np.zeros((n_samples, nt))
    xt_trj[:,0] = x0
    eps = 0.1
    
    for i_s in range(n_samples):
        xt_trj_i = np.zeros(nt)
        xt_trj_i[0] = x0
        xt = x0
        for i in range(nt-1):
            xt = xt + np.sqrt(dt*eps) * np.random.randn(1)
            xt_trj_i[i+1] = xt
        xt_trj[i_s] = xt_trj_i
        
    fig, ax = plt.subplots()
    for i_s in range(n_samples):
        ax.plot(np.arange(nt), xt_trj[i_s], linewidth=0.5)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig("experiments/sde_samples.pdf", dpi=2000)
    plt.show()
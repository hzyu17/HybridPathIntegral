# Abalation study on the impact of \epsilon to the improvement of the E[cost]

import os

if __name__ == '__main__':
    # for n_samples in [10, 50, 100, 200, 500, 1000, 2000, 5000]:
    #     run_command = f'python3.9 /home/hyu419/git/HybridPathIntegral/experiments/example_bouncingball_jax.py --epsilon=2.0 --nsamples={n_samples}'
    #     os.system(run_command)
    
    for eps in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]:
        run_command = f'python3 /home/hzyu/git/HybridPathIntegral/experiments/example_slip_jax_threading.py --epsilon={eps} --nsamples={5000}'
        os.system(run_command)
    
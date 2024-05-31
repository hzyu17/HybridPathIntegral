# Abalation study on the impact of \epsilon to the improvement of the E[cost]

import os


if __name__ == '__main__':
    for n_samples in [10, 50, 100, 200, 500, 1000, 2000, 5000]:
        run_command = f'/usr/bin/python3 /home/hzyu/git/HybridPathIntegral/experiments/example_bouncingball_jax.py --epsilon=2.0 --nsamples={n_samples}'
        os.system(run_command)
    
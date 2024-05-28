# Abalation study on the impact of \epsilon to the improvement of the E[cost]

import os


if __name__ == '__main__':
    for n_spls in range(10, 5010, 500):
        run_command = f'/usr/bin/python3 /home/hzyu/git/HybridPathIntegral/experiments/example_bouncingball_jax.py --epsilon=2 --nsamples={n_spls}'
        os.system(run_command)
    
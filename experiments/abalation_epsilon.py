# Abalation study on the impact of \epsilon to the improvement of the E[cost]

import os


if __name__ == '__main__':
    for eps in [10, 20, 50]:
        run_command = f'/usr/bin/python3 /home/hzyu/git/HybridPathIntegral/experiments/example_bouncingball_jax.py --epsilon={eps} --nsamples=10000'
        os.system(run_command)
    
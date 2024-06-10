# Abalation study on the impact of \epsilon to the improvement of the E[cost]

import os


if __name__ == '__main__':
    # for n_samples in [10, 50, 100, 200, 500, 1000, 2000, 5000]:
    #     run_command = f'python3.9 /home/hyu419/git/HybridPathIntegral/experiments/example_bouncingball_jax.py --epsilon=2.0 --nsamples={n_samples}'
    #     os.system(run_command)
    
    for eps in [0.1, 0.5, 1.0, 1.5,
                2.0, 2.5, 3.0, 3.5, 
                4.0, 4.5, 5.0, 5.5,
                6.0, 6.5, 7.0, 7.5, 
                8.0, 8.5, 9.0, 9.5,
                10.0, 10.5, 11.0, 11.5, 
                12.0, 12.5, 13.0, 13.5,
                14.0, 14.5, 15.0, 15.5, 
                16.0, 16.5, 17.0, 17.5, 
                18.0, 18.5, 19.0, 19.5, 20.0]:
        run_command = f'python3.9 /home/hyu419/git/HybridPathIntegral/experiments/example_bouncingball_jax_threading.py --epsilon={eps} --nsamples={5000}'
        os.system(run_command)
    
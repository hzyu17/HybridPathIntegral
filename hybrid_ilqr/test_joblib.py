from joblib import Parallel, delayed
import time
import numpy as np

import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from dynamics.symbolic_bouncing_1D import *

nt = 1000
nu = 1
nx = 2
ut = np.zeros((nt, nu), dtype=np.float64)
epsilon = 0.1
t0 = 0.0
tf = 3.0
x0 = np.array([5.0, 1.0])
n_samples = 50

# sequential updates
start_time = time.time()
trj_samples = np.zeros((n_samples, nt, nx), dtype=np.float64)
for i in range(n_samples):
    print("Sample trajectory: ", i)
    trj_samples[i] = rollout_bouncing_stochastic(x0, ut, t0, tf, epsilon)
end_time = time.time()  # End time
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"Elapsed time: {elapsed_time} seconds")
print(trj_samples[9])

# parallel updates
start_time = time.time()
trj_samples = np.zeros((n_samples, nt, nx), dtype=np.float64)
def process(results_i, i):
    print("Sample trajectory: ", i)
    results_i = rollout_bouncing_stochastic(x0, ut, t0, tf, epsilon)
    return results_i, i

results = Parallel(n_jobs=-1)(delayed(process)(trj_samples[i,:,:], i) for i in range(trj_samples.shape[0]))

for updated_value, index in results:
    trj_samples[index] = updated_value

end_time = time.time()  # End time
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"Elapsed time: {elapsed_time} seconds")

print(trj_samples[9])

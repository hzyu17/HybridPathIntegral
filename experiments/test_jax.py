import os
import jax
import jax.numpy as jnp
from jax.lib import xla_bridge

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Print out current preallocation setting
preallocate = os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'Not Set')
print(f"XLA_PYTHON_CLIENT_PREALLOCATE: {preallocate}")

# Check JAX backend and available GPUs
print(f"JAX default backend: {jax.default_backend()}")
print(f"Available GPUs: {jax.devices('gpu')}")
print(f"XLA backend platform: {xla_bridge.get_backend().platform}")


import multiprocessing as mp
import jax
import jax.numpy as jnp

import gc


def compute_on_gpu(inputs):
    gc.collect()
    # Perform some computation using JAX
    
    # Run a simple JAX computation
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1000, 1000))
    result = jnp.dot(x, x.T)
    print("JAX computation result:", result)

    y = jnp.sin(inputs)
    return y

if __name__ == "__main__":
    # Set the start method to 'spawn'
    mp.set_start_method('spawn', force=True)

    # Create a pool of workers
    with mp.Pool(processes=4) as pool:
        # Sample input data
        inputs = [jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0])]

        # Map the function to the inputs
        results = pool.map(compute_on_gpu, inputs)

        # Print the results
        for result in results:
            print(result)
import os
import jax
import jax.numpy as jnp
from jax.lib import xla_bridge

# Print out current preallocation setting
preallocate = os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'Not Set')
print(f"XLA_PYTHON_CLIENT_PREALLOCATE: {preallocate}")

# Check JAX backend and available GPUs
print(f"JAX default backend: {jax.default_backend()}")
print(f"Available GPUs: {jax.devices('gpu')}")
print(f"XLA backend platform: {xla_bridge.get_backend().platform}")

# Run a simple JAX computation
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1000, 1000))
result = jnp.dot(x, x.T)
print("JAX computation result:", result)
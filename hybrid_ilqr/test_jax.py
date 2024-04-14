import jax
jax.devices()

import numpy as np
import jax.numpy as jnp

x = np.arange(5)
w = np.array([2., 3., 4.])

def convolve(x, w):
  output = []
  for i in range(1, len(x)-1):
    output.append(jnp.dot(x[i-1:i+2], w))
  return jnp.array(output)

print("convolve(x, w): ", convolve(x, w))

n_devices = jax.local_device_count() 
print("n_devices: ", n_devices)

xs = np.arange(5 * n_devices).reshape(-1, 5)
ws = np.stack([w] * n_devices)

print("xs: ", xs)
print("ws: ", ws)

jax.vmap(convolve)(xs, ws)
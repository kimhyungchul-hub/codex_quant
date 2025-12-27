import jax
import os
print('JAX_PLATFORM_NAME env:', os.environ.get('JAX_PLATFORM_NAME'))
print('default backend:', jax.default_backend())
print('devices:', jax.devices())

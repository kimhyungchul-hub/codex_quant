import os
import time

# Ensure we don't accidentally pick CPU due to a lower-case env value.
os.environ.setdefault("JAX_PLATFORM_NAME", "METAL")
if str(os.environ.get("JAX_PLATFORM_NAME", "")).strip().lower() == "metal":
    os.environ["JAX_PLATFORM_NAME"] = "METAL"

os.environ.setdefault("JAX_PLATFORMS", "METAL")
if str(os.environ.get("JAX_PLATFORMS", "")).strip().lower() == "metal":
    os.environ["JAX_PLATFORMS"] = "METAL"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxlib  # noqa: E402

try:
    # Some jax-metal builds expose the backend as "METAL" (uppercase).
    jax.config.update("jax_platform_name", "METAL")  # type: ignore[attr-defined]
except Exception:
    pass

print("jax:", jax.__version__)
print("jaxlib:", jaxlib.__version__)
print("backend:", jax.default_backend())
print("devices:", jax.devices())
print("JAX_PLATFORM_NAME:", os.environ.get("JAX_PLATFORM_NAME"))


@jax.jit
def compute(a, b):
    # Metal에서 SVD/eigh는 미지원일 수 있으니 matmul 기반으로 테스트
    return jnp.sum(jnp.tanh(a @ b))


a = jnp.ones((1024, 1024), dtype=jnp.float32)
b = jnp.ones((1024, 1024), dtype=jnp.float32)

print("start")
t0 = time.time()
y = compute(a, b).block_until_ready()
print("done:", time.time() - t0)
print("device:", y.device)

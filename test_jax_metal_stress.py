import jax
import jax.numpy as jnp
import numpy as np
import time

def test_jax_metal_stress():
    print(f"JAX Version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    
    # Core MC simulation stress
    n_symbols = 18
    n_paths = 1024
    n_steps = 3600
    
    keys = jax.random.PRNGKey(42)
    keys = jax.random.split(keys, n_symbols)
    
    s0s = jnp.ones(n_symbols, dtype=jnp.float32) * 100.0
    mus = jnp.linspace(0.01, 1.0, n_symbols, dtype=jnp.float32)
    sigmas = jnp.ones(n_symbols, dtype=jnp.float32) * 0.2
    
    @jax.jit
    def core_sim(key, s0, mu, sigma):
        z = jax.random.normal(key, shape=(n_paths, n_steps), dtype=jnp.float32)
        drift = (mu - 0.5 * sigma**2) * (1.0/31536000.0)
        diffusion = sigma * jnp.sqrt(1.0/31536000.0)
        logret = jnp.cumsum(drift + diffusion * z, axis=1)
        return s0 * jnp.exp(logret)

    # Vmap it
    batch_sim = jax.vmap(core_sim)
    
    print("Starting batch simulation...")
    try:
        t0 = time.perf_counter()
        res = batch_sim(keys, s0s, mus, sigmas)
        res.block_until_ready()
        t1 = time.perf_counter()
        print(f"Batch simulation successful In {t1-t0:.4f}s. Shape: {res.shape}")
    except Exception as e:
        print(f"Batch simulation FAILED: {e}")

    # CVaR stress (jnp.sort)
    @jax.jit
    def cvar_test(returns):
        n = returns.shape[0]
        sorted_rets = jnp.sort(returns)
        k = int(n * 0.05)
        return jnp.mean(sorted_rets[:k])

    batch_cvar = jax.vmap(cvar_test)
    
    print("Starting CVaR test...")
    try:
        rets = jax.random.normal(jax.random.PRNGKey(0), shape=(n_symbols, n_paths))
        t0 = time.perf_counter()
        cvars = batch_cvar(rets)
        cvars.block_until_ready()
        t1 = time.perf_counter()
        print(f"CVaR test successful in {t1-t0:.4f}s.")
    except Exception as e:
        print(f"CVaR test FAILED: {e}")

if __name__ == "__main__":
    test_jax_metal_stress()

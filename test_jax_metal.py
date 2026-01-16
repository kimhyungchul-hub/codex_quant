#!/usr/bin/env python3
"""JAX Metal GPU test suite for Mac M4 Pro."""
import os
os.environ.pop("JAX_PLATFORMS", None)  # Remove CPU forcing

import jax
import jax.numpy as jnp
import jax.random as jrand

print("="*70)
print("JAX Metal GPU Test Suite")
print("="*70)

# Test 1: Basic Configuration
print(f"\n✅ Test 1: Basic Configuration")
print(f"   JAX version: {jax.__version__}")
print(f"   Default backend: {jax.default_backend()}")
print(f"   Available devices: {jax.devices()}")

# Test 2: Simple Array Operation
print(f"\n✅ Test 2: Simple Array Creation")
try:
    x = jnp.array([1.0, 2.0, 3.0])
    print(f"   ✓ Array created: {x}")
    print(f"   ✓ Device: {x.device()}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

# Test 3: Matrix Multiplication
print(f"\n✅ Test 3: Matrix Multiplication")
try:
    A = jnp.ones((100, 100))
    B = jnp.ones((100, 100))
    C = jnp.dot(A, B)
    print(f"   ✓ Result shape: {C.shape}")
    print(f"   ✓ Device: {C.device()}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

# Test 4: Random Number Generation with PRNGKey
print(f"\n✅ Test 4: Random Number Generation")
try:
    key = jrand.PRNGKey(0)
    print(f"   ✓ PRNGKey created: {key}")
    
    # Correct way: split key before each random operation
    key, subkey = jrand.split(key)
    random_array = jrand.normal(subkey, shape=(1000,))
    print(f"   ✓ Random array shape: {random_array.shape}")
    print(f"   ✓ Random array mean: {jnp.mean(random_array):.4f}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

# Test 5: Student-t Distribution (for MC simulation)
print(f"\n✅ Test 5: Student-t Distribution")
try:
    key = jrand.PRNGKey(42)
    key, subkey = jrand.split(key)
    
    df = 6.0
    t_samples = jrand.t(subkey, df=df, shape=(1000,))
    
    # Normalize for unit variance
    if df > 2:
        t_samples = t_samples / jnp.sqrt(df / (df - 2.0))
    
    print(f"   ✓ t-distribution samples shape: {t_samples.shape}")
    print(f"   ✓ t-distribution mean: {jnp.mean(t_samples):.4f}")
    print(f"   ✓ t-distribution std: {jnp.std(t_samples):.4f}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

# Test 6: JIT Compilation
print(f"\n✅ Test 6: JIT Compilation")
try:
    @jax.jit
    def fast_function(x):
        return jnp.sum(x ** 2)
    
    x = jnp.arange(1000)
    result = fast_function(x)
    print(f"   ✓ JIT function result: {result}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

# Test 7: Cumulative Sum (used in MC path simulation)
print(f"\n✅ Test 7: Cumulative Sum (MC Path Simulation)")
try:
    key = jrand.PRNGKey(123)
    key, subkey = jrand.split(key)
    
    # Simulate price path
    n_paths = 1000
    n_steps = 100
    
    z = jrand.normal(subkey, shape=(n_paths, n_steps))
    drift = 0.001
    diffusion = 0.02
    
    logret = jnp.cumsum(drift + diffusion * z, axis=1)
    prices = 100.0 * jnp.exp(logret)
    
    print(f"   ✓ Price paths shape: {prices.shape}")
    print(f"   ✓ Final price mean: {jnp.mean(prices[:, -1]):.2f}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

print("\n" + "="*70)
print("Test Suite Complete!")
print("="*70)

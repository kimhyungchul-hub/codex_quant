"""
Benchmark script for NAPV calculation: NumPy vs JAX Metal
"""

import numpy as np
import time
from core.napv_engine_jax import NAPVEngineJAX, NAPVConfig

def benchmark_napv():
    print("=" * 60)
    print("NAPV Performance Benchmark: NumPy (CPU) vs JAX (Metal GPU)")
    print("=" * 60)
    
    # Setup
    n_symbols = 18
    n_horizons = 4
    
    # Generate test data
    np.random.seed(42)
    horizons = np.array([300, 600, 1800, 3600], dtype=np.float32)
    
    horizons_batch = np.tile(horizons, (n_symbols, 1))
    ev_rates_batch = np.random.randn(n_symbols, n_horizons).astype(np.float32) * 0.01
    costs = np.full(n_symbols, 0.0003, dtype=np.float32)
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    
    rho = 0.00001
    r_f = 0.000001
    
    # Initialize JAX engine (compiles kernels)
    print("\n[1] Initializing JAX Metal engine...")
    engine = NAPVEngineJAX(NAPVConfig())
    
    # Warmup (JIT compilation)
    print("[2] Warming up JIT compiler...")
    _ = engine.calculate_batch(symbols, horizons_batch, ev_rates_batch, rho, r_f, costs)
    
    # Benchmark: Single symbol (old way)
    print("\n[3] Benchmarking SINGLE symbol calculation...")
    n_iter = 100
    
    start = time.perf_counter()
    for _ in range(n_iter):
        for i in range(n_symbols):
            napv, t_star = engine.calculate_single(
                horizons, ev_rates_batch[i], rho, r_f, "full"
            )
    single_time = (time.perf_counter() - start) / n_iter
    
    print(f"   Single-symbol (sequential): {single_time*1000:.2f} ms per cycle")
    print(f"   ({n_symbols} symbols × {n_iter} iterations)")
    
    # Benchmark: Batch (new way)
    print("\n[4] Benchmarking BATCH calculation (GPU)...")
    
    start = time.perf_counter()
    for _ in range(n_iter):
        results = engine.calculate_batch(
            symbols, horizons_batch, ev_rates_batch, rho, r_f, costs
        )
    batch_time = (time.perf_counter() - start) / n_iter
    
    print(f"   Batch (parallel GPU): {batch_time*1000:.2f} ms per cycle")
    print(f"   ({n_symbols} symbols in 1 GPU call)")
    
    # Results
    speedup = single_time / batch_time
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Single-symbol time: {single_time*1000:.2f} ms")
    print(f"Batch time:         {batch_time*1000:.2f} ms")
    print(f"Speedup:            {speedup:.1f}x FASTER 🚀")
    print(f"GPU:                Apple M4 Pro (Metal)")
    print("=" * 60)
    
    # Verify correctness
    print("\n[5] Verifying correctness...")
    single_results = {}
    for i, sym in enumerate(symbols):
        napv, t_star = engine.calculate_single(horizons, ev_rates_batch[i], rho, r_f, "full")
        single_results[sym] = napv
    
    batch_results = engine.calculate_batch(symbols, horizons_batch, ev_rates_batch, rho, r_f, costs)
    batch_napv = {sym: napv for sym, (napv, _) in batch_results.items()}
    
    all_close = True
    for sym in symbols:
        diff = abs(single_results[sym] - batch_napv[sym])
        if diff > 1e-5:
            print(f"   ❌ Mismatch for {sym}: {diff:.8f}")
            all_close = False
    
    if all_close:
        print("   ✅ All results match (within tolerance)")
    
    print("\n✨ Benchmark complete!\n")

if __name__ == "__main__":
    benchmark_napv()

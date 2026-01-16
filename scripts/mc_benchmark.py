#!/usr/bin/env python3
import os
import sys
import time
from types import SimpleNamespace

# Ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Optionally override env for verbose prints
os.environ.setdefault('MC_VERBOSE_PRINT', '1')
# Keep n_paths moderate for quicker runs (can be overridden externally)
os.environ.setdefault('MC_N_PATHS_LIVE', os.environ.get('MC_N_PATHS_LIVE', '256'))

from engines.mc.monte_carlo_engine import MonteCarloEngine


def make_task(symbol: str, seed: int, price: float = 100.0, mu_alpha: float = 1.0):
    ctx = {
        'symbol': symbol,
        'price': float(price),
        'mu_sim': 0.0,
        'mu_base': 0.0,
        'sigma': 0.02,
        'mu_alpha': float(mu_alpha),
        'leverage': 1.0,
        'regime': 'bull',
    }
    params = SimpleNamespace(use_historical_drift=False)
    return {'ctx': ctx, 'params': params, 'seed': int(seed)}


def run_once(engine, tasks):
    t0 = time.perf_counter()
    out = engine.evaluate_entry_metrics_batch(tasks)
    t1 = time.perf_counter()
    print(f"[MICRO_BENCH] call elapsed={(t1-t0):.3f}s num_tasks={len(tasks)}")
    for i, o in enumerate(out):
        perf = o.get('meta', {}).get('perf', {})
        print(f" task {i} ev={o.get('ev'):.6f} gen1={perf.get('gen1'):.3f}s sum={perf.get('sum'):.3f}s exit_jax={perf.get('exit_jax'):.3f}s")
    return out


def main():
    engine = MonteCarloEngine()
    # Build a small batch (4 symbols)
    tasks = [make_task(f'TEST{i}/USDT', seed=1000 + i, price=100.0 + i, mu_alpha=1.0 + i*0.5) for i in range(4)]

    # Warmup runs (JAX + device compile)
    print("[MICRO_BENCH] Warmup run (may include JAX compilation)")
    run_once(engine, tasks)

    # Measured runs
    runs = 3
    times = []
    for r in range(runs):
        t0 = time.perf_counter()
        out = run_once(engine, tasks)
        t1 = time.perf_counter()
        times.append(t1-t0)

    print(f"[MICRO_BENCH] runs={runs} median={(sorted(times)[len(times)//2]):.3f}s mean={(sum(times)/len(times)):.3f}s")


if __name__ == '__main__':
    main()

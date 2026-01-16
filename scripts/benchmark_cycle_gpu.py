#!/usr/bin/env python3
"""
GPU/CPU cycle-time benchmark for GlobalBatchEvaluator.

Measures steady-state (post-JIT) time for one "decision cycle"
across N symbols with N paths and multiple horizons.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List
import config
from utils.helpers import _load_env_file

import numpy as np


def _parse_horizons(text: str) -> List[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [int(p) for p in parts]


def _generate_paths(n_symbols: int, n_paths: int, n_steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    price_paths = np.empty((n_symbols, n_paths, n_steps + 1), dtype=np.float32)
    for i in range(n_symbols):
        s0 = float(rng.uniform(50.0, 200.0))
        drift = float(rng.uniform(-0.2, 0.2))
        vol = float(rng.uniform(0.15, 0.35))

        z = rng.standard_normal((n_paths, n_steps), dtype=np.float32)
        log_returns = (drift - 0.5 * vol * vol) + vol * z
        log_prices = np.log(s0) + np.cumsum(log_returns, axis=1)
        # Prevent overflow in exp for long horizons.
        log_prices = np.clip(log_prices, -20.0, 20.0)
        prices = np.exp(log_prices)

        price_paths[i, :, 0] = s0
        price_paths[i, :, 1:] = prices

    return price_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark GlobalBatchEvaluator cycle time.")
    parser.add_argument("--symbols", type=int, default=18)
    # prefer central `config.MC_N_PATHS_LIVE` when available, otherwise fall back to a smaller default
    parser.add_argument(
        "--paths",
        type=int,
        default=int(getattr(config, "MC_N_PATHS_LIVE", 1024)),
    )
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--horizons", type=str, default="300,600,1800,3600")
    parser.add_argument("--horizon-step", type=int, default=0)
    parser.add_argument("--horizon-min", type=int, default=0)
    parser.add_argument("--horizon-max", type=int, default=0)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--platform", type=str, default="")
    args = parser.parse_args()

    if args.platform:
        plat = str(args.platform)
        if plat.lower() == "metal":
            os.environ.pop("JAX_PLATFORM_NAME", None)
            os.environ.pop("JAX_PLATFORMS", None)
        else:
            os.environ["JAX_PLATFORM_NAME"] = plat

    # Load local .env (developer convenience) so config values reflect `.env` when present
    _load_env_file(str(config.BASE_DIR / ".env"))

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    import jax
    import jax.numpy as jnp
    from jax import tree_util

    from engines.mc.entry_evaluation_vmap import GlobalBatchEvaluator

    if args.horizon_step and args.horizon_step > 0:
        h_min = int(args.horizon_min or args.horizon_step)
        h_max = int(args.horizon_max or 3600)
        if h_min < args.horizon_step:
            h_min = int(args.horizon_step)
        if h_max < h_min:
            h_max = h_min
        horizons = list(range(h_min, h_max + 1, int(args.horizon_step)))
    else:
        horizons = _parse_horizons(args.horizons)

    print("=" * 72)
    print("GlobalBatchEvaluator Cycle Benchmark")
    print("=" * 72)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"Symbols={args.symbols} Paths={args.paths} Steps={args.steps} Horizons={horizons}")
    print(f"Runs={args.runs} Warmup={args.warmup}")

    print("\n[1] Generating synthetic price paths...")
    price_paths_np = _generate_paths(args.symbols, args.paths, args.steps, args.seed)

    evaluator = GlobalBatchEvaluator()
    horizons_jax = jnp.array(horizons, dtype=jnp.int32)
    leverages_jax = jnp.ones(args.symbols, dtype=jnp.float32) * 10.0
    fees_jax = jnp.ones(args.symbols, dtype=jnp.float32) * 0.0015
    tp_targets_jax = jnp.ones((args.symbols, len(horizons)), dtype=jnp.float32) * 0.01
    sl_targets_jax = jnp.ones((args.symbols, len(horizons)), dtype=jnp.float32) * -0.005

    print("[2] Moving data to device...")
    price_paths_jax = jax.device_put(jnp.array(price_paths_np))

    print("[3] JIT warmup...")
    for i in range(max(1, args.warmup)):
        out = evaluator.jit_compute(
            price_paths_jax,
            horizons_jax,
            leverages_jax,
            fees_jax,
            tp_targets_jax,
            sl_targets_jax,
        )
        tree_util.tree_map(lambda x: x.block_until_ready(), out)
        print(f"  warmup {i + 1}/{max(1, args.warmup)} done")

    print("[4] Timing steady-state runs...")
    times_ms = []
    for _ in range(args.runs):
        jax.block_until_ready(price_paths_jax)
        t0 = time.perf_counter()
        out = evaluator.jit_compute(
            price_paths_jax,
            horizons_jax,
            leverages_jax,
            fees_jax,
            tp_targets_jax,
            sl_targets_jax,
        )
        tree_util.tree_map(lambda x: x.block_until_ready(), out)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    if len(times_ms) > 1:
        times_ms = times_ms[1:]

    avg = float(np.mean(times_ms)) if times_ms else 0.0
    std = float(np.std(times_ms)) if times_ms else 0.0
    per_symbol = avg / max(1, args.symbols)

    print("\nResults:")
    print(f"  Runs(ms): {[f'{t:.2f}' for t in times_ms]}")
    print(f"  Avg: {avg:.2f} ms  Std: {std:.2f} ms")
    print(f"  Per symbol: {per_symbol:.2f} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
PROPER Benchmark: M4 Pro Unified Memory 활용
==============================================

핵심 수정사항:
1. jax.device_put()으로 데이터를 미리 GPU 메모리에 배치
2. 컴파일 시간 완전 분리 (여러 번 워밍업)
3. block_until_ready()로 정확한 GPU 동기화
4. 매 실행마다 데이터 재생성 방지
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
sys.path.insert(0, '/Users/jeonghwakim/codex_quant')

from engines.mc.entry_evaluation_vmap import GlobalBatchEvaluator

def generate_test_data(n_symbols=18, n_paths=4096, n_steps=3600):
    """테스트 데이터 생성 (한 번만)"""
    np.random.seed(42)
    
    price_paths = []
    for i in range(n_symbols):
        s0 = np.random.uniform(50, 200)
        drift = np.random.uniform(-0.2, 0.2)
        vol = np.random.uniform(0.15, 0.35)
        
        dW = np.random.randn(n_paths, n_steps) * np.sqrt(1.0)
        log_returns = (drift - 0.5 * vol**2) + vol * dW
        log_prices = np.log(s0) + np.cumsum(log_returns, axis=1)
        prices = np.exp(log_prices)
        
        prices_with_s0 = np.concatenate([
            np.ones((n_paths, 1)) * s0,
            prices
        ], axis=1)
        
        price_paths.append(prices_with_s0)
    
    return np.array(price_paths)


def benchmark_cpu_numpy(price_paths_np, horizons, n_runs=10):
    """
    CPU Numpy 기준선 (데이터 재생성 없음)
    """
    print("\n" + "="*70)
    print("CPU Numpy Baseline")
    print("="*70)
    
    n_symbols = price_paths_np.shape[0]
    leverage = 10.0
    fee = 0.0015
    
    times = []
    
    for run in range(n_runs):
        t0 = time.perf_counter()
        
        for sym_idx in range(n_symbols):
            paths = price_paths_np[sym_idx]
            
            for h_idx in horizons:
                s0 = paths[:, 0]
                st = paths[:, h_idx]
                
                gross_ret = (st - s0) / np.maximum(s0, 1e-12)
                
                # Long/Short
                for direction in [1, -1]:
                    net = direction * gross_ret * leverage - fee * leverage
                    
                    # 통계
                    ev = np.mean(net)
                    p_pos = np.mean(net > 0)
                    
                    # TP/SL
                    tp_target = 0.01 * leverage
                    sl_target = -0.005 * leverage
                    p_tp = np.mean(net >= tp_target)
                    p_sl = np.mean(net <= sl_target)
                    
                    # CVaR
                    sorted_net = np.sort(net)
                    cutoff = int(0.05 * len(sorted_net))
                    cvar = np.mean(sorted_net[:cutoff]) if cutoff > 0 else sorted_net[0]
        
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    # 첫 실행 제외 (캐시 워밍업)
    times_clean = times[1:]
    avg = np.mean(times_clean)
    std = np.std(times_clean)
    
    print(f"  Runs: {times_clean}")
    print(f"  Average: {avg:.2f}ms ± {std:.2f}ms")
    print(f"  Per symbol: {avg/n_symbols:.2f}ms")
    
    return avg


def benchmark_gpu_jax_proper(price_paths_np, horizons, n_runs=10):
    """
    GPU JAX with Unified Memory (제대로 된 측정)
    """
    print("\n" + "="*70)
    print("GPU JAX with Unified Memory (PROPER)")
    print("="*70)
    
    n_symbols = price_paths_np.shape[0]
    
    # 1. Evaluator 생성
    evaluator = GlobalBatchEvaluator()
    
    # 2. 파라미터 준비 및 GPU에 미리 배치 (Unified Memory 활용)
    print("  [Step 1] Moving data to GPU memory (Unified Memory)...")
    
    # JAX array로 변환하고 GPU에 배치
    price_paths_jax = jax.device_put(jnp.array(price_paths_np))
    horizons_jax = jax.device_put(jnp.array(horizons, dtype=jnp.int32))
    leverages_jax = jax.device_put(jnp.ones(n_symbols) * 10.0)
    fees_jax = jax.device_put(jnp.ones(n_symbols) * 0.0015)
    tp_targets_jax = jax.device_put(jnp.ones((n_symbols, len(horizons))) * 0.01)
    sl_targets_jax = jax.device_put(jnp.ones((n_symbols, len(horizons))) * -0.005)
    
    print("  [Step 2] JIT Compilation (Warm-up)...")
    
    # 3. 컴파일 워밍업 (여러 번 실행해서 확실히 컴파일)
    for i in range(3):
        result = evaluator.jit_compute(
            price_paths_jax, horizons_jax, leverages_jax,
            fees_jax, tp_targets_jax, sl_targets_jax
        )
        # 모든 결과를 block_until_ready
        jax.tree_map(lambda x: x.block_until_ready(), result)
        print(f"    Warmup {i+1}/3 complete")
    
    print("  [Step 3] Measuring actual execution time...")
    
    # 4. 실제 시간 측정 (데이터는 이미 GPU에 있음)
    times = []
    
    for run in range(n_runs):
        # GPU 동기화 후 시작
        jax.block_until_ready(price_paths_jax)
        
        t0 = time.perf_counter()
        
        # 단일 함수 호출
        result = evaluator.jit_compute(
            price_paths_jax, horizons_jax, leverages_jax,
            fees_jax, tp_targets_jax, sl_targets_jax
        )
        
        # GPU 완료 대기 (중요!)
        jax.tree_map(lambda x: x.block_until_ready(), result)
        
        t1 = time.perf_counter()
        
        elapsed = (t1 - t0) * 1000
        times.append(elapsed)
    
    # 첫 실행 제외
    times_clean = times[1:]
    avg = np.mean(times_clean)
    std = np.std(times_clean)
    
    print(f"  Runs: {[f'{t:.2f}' for t in times_clean]}")
    print(f"  Average: {avg:.2f}ms ± {std:.2f}ms")
    print(f"  Per symbol: {avg/n_symbols:.2f}ms")
    
    return avg


def main():
    print("\n" + "="*70)
    print("PROPER BENCHMARK: M4 Pro Unified Memory")
    print("="*70)
    
    n_symbols = 18
    n_paths = 4096
    horizons = [300, 600, 1800, 3600]
    n_runs = 10
    
    print(f"\nConfiguration:")
    print(f"  - Symbols: {n_symbols}")
    print(f"  - Paths per symbol: {n_paths}")
    print(f"  - Total parallel work: {n_symbols * n_paths} = {n_symbols * n_paths:,}")
    print(f"  - Horizons: {horizons}")
    print(f"  - Runs: {n_runs}")
    print(f"\nKey improvements:")
    print(f"  ✅ Data pre-loaded to GPU (jax.device_put)")
    print(f"  ✅ JIT compilation separated (3x warmup)")
    print(f"  ✅ Proper GPU sync (block_until_ready)")
    print(f"  ✅ No data regeneration per run")
    
    # 데이터 생성 (한 번만)
    print("\n[SETUP] Generating test data...")
    price_paths = generate_test_data(n_symbols, n_paths, 3600)
    print(f"[SETUP] Data shape: {price_paths.shape}")
    print(f"[SETUP] Data size: {price_paths.nbytes / 1024 / 1024:.2f} MB")
    
    # CPU Baseline
    time_cpu = benchmark_cpu_numpy(price_paths, horizons, n_runs)
    
    # GPU JAX (Proper)
    time_gpu = benchmark_gpu_jax_proper(price_paths, horizons, n_runs)
    
    # 결과
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    speedup = time_cpu / time_gpu
    
    print(f"\n  CPU Numpy:           {time_cpu:.2f}ms")
    print(f"  GPU JAX (Proper):    {time_gpu:.2f}ms")
    print(f"\n  Speedup: {speedup:.2f}x")
    
    if speedup > 1.0:
        print(f"\n  ✅ GPU is {speedup:.2f}x FASTER!")
        print(f"  ✅ Time saved: {time_cpu - time_gpu:.2f}ms per batch")
        print(f"  ✅ Unified Memory + JIT = Win!")
        
        # 처리량 계산
        total_evals = n_symbols * len(horizons) * 2  # long + short
        throughput_cpu = total_evals / (time_cpu / 1000)
        throughput_gpu = total_evals / (time_gpu / 1000)
        
        print(f"\n  Throughput:")
        print(f"    CPU: {throughput_cpu:.0f} evaluations/sec")
        print(f"    GPU: {throughput_gpu:.0f} evaluations/sec")
        
    elif speedup > 0.8:
        print(f"\n  ≈ GPU and CPU are roughly equal")
        print(f"  → Hybrid approach recommended")
    else:
        print(f"\n  ⚠️  GPU is {1/speedup:.2f}x slower")
        print(f"  → Need further optimization or CPU is better")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

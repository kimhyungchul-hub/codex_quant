"""
Performance Benchmark: 기존 방식 vs Global Batching
====================================================

18개 심볼에 대한 Monte Carlo 평가 성능 비교
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
sys.path.insert(0, '/Users/jeonghwakim/codex_quant')

from engines.mc.entry_evaluation_vmap import GlobalBatchEvaluator

# ============================================================================
# 테스트 데이터 생성
# ============================================================================

def generate_test_data(n_symbols=18, n_paths=4096, n_steps=3600):
    """테스트용 가격 경로 생성"""
    print(f"[SETUP] Generating test data: {n_symbols} symbols, {n_paths} paths, {n_steps} steps")
    
    # 랜덤 시드 고정 (재현성)
    np.random.seed(42)
    
    # GBM으로 가격 경로 생성
    dt = 1.0
    drifts = np.random.uniform(-0.1, 0.1, n_symbols)
    vols = np.random.uniform(0.1, 0.3, n_symbols)
    
    price_paths = []
    for i in range(n_symbols):
        s0 = 100.0
        drift = drifts[i]
        vol = vols[i]
        
        # Brownian motion
        dW = np.random.randn(n_paths, n_steps) * np.sqrt(dt)
        
        # Log returns
        log_returns = (drift - 0.5 * vol**2) * dt + vol * dW
        
        # Price path
        log_prices = np.log(s0) + np.cumsum(log_returns, axis=1)
        prices = np.exp(log_prices)
        
        # s0 추가
        prices_with_s0 = np.concatenate([
            np.ones((n_paths, 1)) * s0,
            prices
        ], axis=1)
        
        price_paths.append(prices_with_s0)
    
    price_paths = np.array(price_paths)  # (n_symbols, n_paths, n_steps+1)
    
    print(f"[SETUP] Price paths shape: {price_paths.shape}")
    return price_paths, drifts, vols


# ============================================================================
# Benchmark 1: 기존 방식 (순차 처리 시뮬레이션)
# ============================================================================

def benchmark_sequential(price_paths, horizons, n_runs=5):
    """
    기존 방식 시뮬레이션: 각 심볼을 순차적으로 처리
    (실제 코드는 더 복잡하지만, 핵심은 순차 처리)
    """
    print("\n" + "="*70)
    print("BENCHMARK 1: Sequential Processing (기존 방식)")
    print("="*70)
    
    n_symbols = price_paths.shape[0]
    n_paths = price_paths.shape[1]
    
    times = []
    
    for run in range(n_runs):
        t0 = time.perf_counter()
        
        # 각 심볼을 순차 처리
        for sym_idx in range(n_symbols):
            paths = price_paths[sym_idx]  # (n_paths, n_steps+1)
            
            # 각 horizon을 순차 처리
            for h_idx in horizons:
                # CPU에서 Numpy 연산 (CPUSum)
                s0 = paths[:, 0]
                st = paths[:, h_idx]
                
                gross_ret = (st - s0) / np.maximum(s0, 1e-12)
                net_long = gross_ret * 10.0 - 0.0015 * 10.0
                net_short = -gross_ret * 10.0 - 0.0015 * 10.0
                
                # 통계 계산 (CPU)
                ev_long = np.mean(net_long)
                ev_short = np.mean(net_short)
                p_pos_long = np.mean(net_long > 0)
                p_pos_short = np.mean(net_short > 0)
                
                # CVaR 계산 (CPU 정렬)
                sorted_long = np.sort(net_long)
                sorted_short = np.sort(net_short)
                cutoff = int(0.05 * len(sorted_long))
                cvar_long = np.mean(sorted_long[:cutoff]) if cutoff > 0 else sorted_long[0]
                cvar_short = np.mean(sorted_short[:cutoff]) if cutoff > 0 else sorted_short[0]
        
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        
        print(f"  Run {run+1}/{n_runs}: {elapsed*1000:.2f}ms")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n  Average: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    print(f"  Per symbol: {avg_time/n_symbols*1000:.2f}ms")
    
    return avg_time


# ============================================================================
# Benchmark 2: Global Batching (새 방식)
# ============================================================================

def benchmark_global_batching(price_paths, horizons, n_runs=5):
    """
    새 방식: Global Batching with JAX vmap
    """
    print("\n" + "="*70)
    print("BENCHMARK 2: Global Batching (새 방식)")
    print("="*70)
    
    n_symbols = price_paths.shape[0]
    
    # Evaluator 생성 및 워밍업
    print("  [Warmup] JIT compiling...")
    evaluator = GlobalBatchEvaluator()
    
    # 파라미터 준비
    leverages = np.ones(n_symbols) * 10.0
    fees = np.ones(n_symbols) * 0.0015
    tp_targets = np.ones((n_symbols, len(horizons))) * 0.01
    sl_targets = np.ones((n_symbols, len(horizons))) * -0.005
    
    # 워밍업 (컴파일 시간 제외)
    _ = evaluator.evaluate_batch(
        price_paths, horizons, leverages, fees,
        tp_targets, sl_targets
    )
    print("  [Warmup] Complete!")
    
    times = []
    
    for run in range(n_runs):
        t0 = time.perf_counter()
        
        # 단일 함수 호출로 모든 심볼 처리
        results = evaluator.evaluate_batch(
            price_paths, horizons, leverages, fees,
            tp_targets, sl_targets
        )
        
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        
        print(f"  Run {run+1}/{n_runs}: {elapsed*1000:.2f}ms")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n  Average: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    print(f"  Per symbol: {avg_time/n_symbols*1000:.2f}ms")
    
    return avg_time


# ============================================================================
# Main Benchmark
# ============================================================================

def main():
    print("\n" + "="*70)
    print("Monte Carlo Evaluation Performance Benchmark")
    print("="*70)
    
    # 테스트 설정
    n_symbols = 18
    n_paths = 4096
    n_steps = 3600
    horizons = [300, 600, 1800, 3600]
    n_runs = 5
    
    print(f"\nConfiguration:")
    print(f"  - Symbols: {n_symbols}")
    print(f"  - Paths per symbol: {n_paths}")
    print(f"  - Steps: {n_steps}")
    print(f"  - Horizons: {horizons}")
    print(f"  - Runs per benchmark: {n_runs}")
    
    # 데이터 생성
    price_paths, drifts, vols = generate_test_data(n_symbols, n_paths, n_steps)
    
    # Benchmark 1: 기존 방식
    time_sequential = benchmark_sequential(price_paths, horizons, n_runs)
    
    # Benchmark 2: Global Batching
    time_batching = benchmark_global_batching(price_paths, horizons, n_runs)
    
    # 결과 요약
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    speedup = time_sequential / time_batching
    
    print(f"\n  Sequential (기존):     {time_sequential*1000:.2f}ms")
    print(f"  Global Batching (새):  {time_batching*1000:.2f}ms")
    print(f"\n  ⚡ Speedup: {speedup:.2f}x faster")
    print(f"  ⚡ Time saved: {(time_sequential - time_batching)*1000:.2f}ms ({(1-time_batching/time_sequential)*100:.1f}% reduction)")
    
    # 처리량 계산
    total_evaluations = n_symbols * len(horizons) * 2  # long + short
    throughput_seq = total_evaluations / time_sequential
    throughput_batch = total_evaluations / time_batching
    
    print(f"\n  Throughput:")
    print(f"    Sequential:      {throughput_seq:.1f} evaluations/sec")
    print(f"    Global Batching: {throughput_batch:.1f} evaluations/sec")
    
    print("\n" + "="*70)
    print("✅ Benchmark Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

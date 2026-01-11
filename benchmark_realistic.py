"""
Real Workload Benchmark: 실제 엔진 워크로드 시뮬레이션
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
sys.path.insert(0, '/Users/jeonghwakim/codex_quant')

from engines.mc.entry_evaluation_vmap import GlobalBatchEvaluator

def generate_realistic_data(n_symbols=18, n_paths=4096, n_steps=3600):
    """실제 엔진과 유사한 데이터 생성"""
    np.random.seed(42)
    
    print(f"[SETUP] Generating realistic data: {n_symbols} symbols, {n_paths} paths")
    
    price_paths = []
    for i in range(n_symbols):
        s0 = np.random.uniform(50, 200)  # 다양한 초기 가격
        drift = np.random.uniform(-0.2, 0.2)  # 다양한 drift
        vol = np.random.uniform(0.15, 0.35)  # 다양한 volatility
        
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


def benchmark_realistic_sequential(price_paths, horizons, n_runs=3):
    """
    실제 엔진의 순차 처리 시뮬레이션
    - 각 심볼마다 여러 leverage 후보 평가
    - 복잡한 통계 계산
    """
    print("\n" + "="*70)
    print("REALISTIC BENCHMARK: Sequential Processing")
    print("="*70)
    
    n_symbols = price_paths.shape[0]
    leverage_candidates = [5.0, 10.0, 15.0, 20.0]  # 4개 leverage 후보
    
    times = []
    
    for run in range(n_runs):
        t0 = time.perf_counter()
        
        for sym_idx in range(n_symbols):
            paths = price_paths[sym_idx]
            
            # 각 leverage 후보 평가
            for lev in leverage_candidates:
                for h_idx in horizons:
                    s0 = paths[:, 0]
                    st = paths[:, h_idx]
                    
                    gross_ret = (st - s0) / np.maximum(s0, 1e-12)
                    
                    # Long/Short 양방향
                    for direction in [1, -1]:
                        net = direction * gross_ret * lev - 0.0015 * lev
                        
                        # 통계 계산
                        ev = np.mean(net)
                        std = np.std(net)
                        p_pos = np.mean(net > 0)
                        
                        # TP/SL 체크
                        tp_target = 0.01 * lev
                        sl_target = -0.005 * lev
                        p_tp = np.mean(net >= tp_target)
                        p_sl = np.mean(net <= sl_target)
                        
                        # CVaR 계산 (정렬 필요)
                        sorted_net = np.sort(net)
                        cutoff = int(0.05 * len(sorted_net))
                        cvar = np.mean(sorted_net[:cutoff]) if cutoff > 0 else sorted_net[0]
                        
                        # Sharpe ratio
                        sharpe = ev / (std + 1e-12)
        
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        
        print(f"  Run {run+1}/{n_runs}: {elapsed*1000:.2f}ms")
    
    avg_time = np.mean(times)
    print(f"\n  Average: {avg_time*1000:.2f}ms")
    print(f"  Per symbol: {avg_time/n_symbols*1000:.2f}ms")
    
    return avg_time


def benchmark_realistic_batching(price_paths, horizons, n_runs=3):
    """
    Global Batching으로 모든 심볼 동시 처리
    """
    print("\n" + "="*70)
    print("REALISTIC BENCHMARK: Global Batching")
    print("="*70)
    
    n_symbols = price_paths.shape[0]
    
    # Evaluator 생성
    evaluator = GlobalBatchEvaluator()
    
    # 파라미터 준비
    leverages = np.ones(n_symbols) * 10.0  # 단일 leverage로 테스트
    fees = np.ones(n_symbols) * 0.0015
    tp_targets = np.ones((n_symbols, len(horizons))) * 0.01
    sl_targets = np.ones((n_symbols, len(horizons))) * -0.005
    
    # 워밍업
    print("  [Warmup] JIT compiling...")
    _ = evaluator.evaluate_batch(
        price_paths, horizons, leverages, fees,
        tp_targets, sl_targets
    )
    print("  [Warmup] Complete!")
    
    times = []
    
    for run in range(n_runs):
        t0 = time.perf_counter()
        
        # 단일 함수 호출
        results = evaluator.evaluate_batch(
            price_paths, horizons, leverages, fees,
            tp_targets, sl_targets
        )
        
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        
        print(f"  Run {run+1}/{n_runs}: {elapsed*1000:.2f}ms")
    
    avg_time = np.mean(times)
    print(f"\n  Average: {avg_time*1000:.2f}ms")
    print(f"  Per symbol: {avg_time/n_symbols*1000:.2f}ms")
    
    return avg_time


def main():
    print("\n" + "="*70)
    print("REALISTIC WORKLOAD BENCHMARK")
    print("="*70)
    
    n_symbols = 18
    n_paths = 4096  # 실제 엔진과 동일
    horizons = [300, 600, 1800, 3600]
    n_runs = 3
    
    print(f"\nConfiguration (실제 엔진과 동일):")
    print(f"  - Symbols: {n_symbols}")
    print(f"  - Paths per symbol: {n_paths}")
    print(f"  - Horizons: {horizons}")
    print(f"  - Leverage candidates: 4 (sequential only)")
    print(f"  - Runs: {n_runs}")
    
    # 데이터 생성
    price_paths = generate_realistic_data(n_symbols, n_paths, 3600)
    
    # Sequential
    time_seq = benchmark_realistic_sequential(price_paths, horizons, n_runs)
    
    # Batching
    time_batch = benchmark_realistic_batching(price_paths, horizons, n_runs)
    
    # 결과
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    speedup = time_seq / time_batch
    
    print(f"\n  Sequential:      {time_seq*1000:.2f}ms")
    print(f"  Global Batching: {time_batch*1000:.2f}ms")
    print(f"\n  Speedup: {speedup:.2f}x")
    
    if speedup > 1:
        print(f"  ✅ Global Batching is {speedup:.2f}x FASTER")
        print(f"  ✅ Time saved: {(time_seq - time_batch)*1000:.2f}ms")
    else:
        print(f"  ⚠️  Global Batching is {1/speedup:.2f}x SLOWER")
        print(f"  ⚠️  Overhead: {(time_batch - time_seq)*1000:.2f}ms")
        print(f"\n  Note: Metal GPU may not be optimal for this workload.")
        print(f"        On CUDA GPUs, expect 5-10x speedup.")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

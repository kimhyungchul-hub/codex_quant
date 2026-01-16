"""
Scalability Test: 다양한 배치 크기에서 성능 비교
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
sys.path.insert(0, '/Users/jeonghwakim/codex_quant')

from engines.mc.entry_evaluation_vmap import GlobalBatchEvaluator

def generate_test_data(n_symbols, n_paths=1024, n_steps=3600):
    """테스트용 가격 경로 생성"""
    np.random.seed(42)
    
    price_paths = []
    for i in range(n_symbols):
        s0 = 100.0
        drift = np.random.uniform(-0.1, 0.1)
        vol = np.random.uniform(0.1, 0.3)
        
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


def benchmark_sequential(price_paths, horizons):
    """순차 처리"""
    n_symbols = price_paths.shape[0]
    
    t0 = time.perf_counter()
    
    for sym_idx in range(n_symbols):
        paths = price_paths[sym_idx]
        
        for h_idx in horizons:
            s0 = paths[:, 0]
            st = paths[:, h_idx]
            
            gross_ret = (st - s0) / np.maximum(s0, 1e-12)
            net_long = gross_ret * 10.0 - 0.0015 * 10.0
            
            ev_long = np.mean(net_long)
            sorted_long = np.sort(net_long)
            cutoff = int(0.05 * len(sorted_long))
            cvar_long = np.mean(sorted_long[:cutoff]) if cutoff > 0 else sorted_long[0]
    
    t1 = time.perf_counter()
    return (t1 - t0) * 1000


def benchmark_batching(price_paths, horizons, evaluator):
    """Global Batching"""
    n_symbols = price_paths.shape[0]
    
    leverages = np.ones(n_symbols) * 10.0
    fees = np.ones(n_symbols) * 0.0015
    tp_targets = np.ones((n_symbols, len(horizons))) * 0.01
    sl_targets = np.ones((n_symbols, len(horizons))) * -0.005
    
    t0 = time.perf_counter()
    
    results = evaluator.evaluate_batch(
        price_paths, horizons, leverages, fees,
        tp_targets, sl_targets
    )
    
    t1 = time.perf_counter()
    return (t1 - t0) * 1000


def main():
    print("\n" + "="*70)
    print("Scalability Test: Performance vs Batch Size")
    print("="*70)
    
    horizons = [300, 600, 1800, 3600]
    batch_sizes = [1, 5, 10, 18, 50, 100]
    n_paths = 1024  # 경로 수를 줄여서 빠르게 테스트
    
    print(f"\nConfiguration:")
    print(f"  - Paths per symbol: {n_paths}")
    print(f"  - Horizons: {horizons}")
    print(f"  - Batch sizes: {batch_sizes}")
    
    # Evaluator 생성 및 워밍업
    print("\n[Warmup] Initializing evaluator...")
    evaluator = GlobalBatchEvaluator()
    
    # 작은 데이터로 워밍업
    dummy_data = generate_test_data(2, n_paths, 3600)
    leverages = np.ones(2) * 10.0
    fees = np.ones(2) * 0.0015
    tp_targets = np.ones((2, len(horizons))) * 0.01
    sl_targets = np.ones((2, len(horizons))) * -0.005
    _ = evaluator.evaluate_batch(dummy_data, horizons, leverages, fees, tp_targets, sl_targets)
    print("[Warmup] Complete!\n")
    
    results = []
    
    print(f"{'Batch Size':<12} {'Sequential':<15} {'Batching':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for batch_size in batch_sizes:
        # 데이터 생성
        price_paths = generate_test_data(batch_size, n_paths, 3600)
        
        # Sequential
        time_seq = benchmark_sequential(price_paths, horizons)
        
        # Batching
        time_batch = benchmark_batching(price_paths, horizons, evaluator)
        
        speedup = time_seq / time_batch
        
        print(f"{batch_size:<12} {time_seq:<15.2f} {time_batch:<15.2f} {speedup:<10.2f}x")
        
        results.append({
            'batch_size': batch_size,
            'sequential': time_seq,
            'batching': time_batch,
            'speedup': speedup
        })
    
    print("\n" + "="*70)
    print("Analysis")
    print("="*70)
    
    # 최고 성능 찾기
    best = max(results, key=lambda x: x['speedup'])
    worst = min(results, key=lambda x: x['speedup'])
    
    print(f"\nBest speedup: {best['speedup']:.2f}x at batch size {best['batch_size']}")
    print(f"Worst speedup: {worst['speedup']:.2f}x at batch size {worst['batch_size']}")
    
    # Break-even point
    break_even = [r for r in results if 0.9 <= r['speedup'] <= 1.1]
    if break_even:
        print(f"\nBreak-even point: ~{break_even[0]['batch_size']} symbols")
    else:
        print(f"\nNo break-even point found in tested range")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

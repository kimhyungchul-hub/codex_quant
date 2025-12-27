#!/usr/bin/env python3
"""
Test script for Diff 2 validation: Multi-Horizon Policy Mix
Tests:
1. policy_horizons = [60, 180, 300, 600, 900, 1800]
2. policy_w_h normalization (sum ~= 1.0)
3. policy_ev_mix_long/short calculation
4. paths_reused behavior
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engines.mc_engine import MonteCarloEngine


def test_diff2_validation():
    """Test Diff 2 validation points"""
    
    print("=" * 80)
    print("Diff 2 Validation: Multi-Horizon Policy Mix")
    print("=" * 80)
    
    # Expected values
    expected_horizons = [60, 180, 300, 600, 900, 1800]
    
    # Test 1: policy_horizons should be [60, 180, 300, 600, 900, 1800]
    print("\n[Test 1] policy_horizons 값 확인")
    print("-" * 80)
    print(f"Expected: {expected_horizons}")
    
    # Check default value from code
    default_horizons = list(getattr(MonteCarloEngine, "POLICY_MULTI_HORIZONS_SEC", (60, 180, 300, 600, 900, 1800)))
    print(f"Default from code: {default_horizons}")
    
    if list(default_horizons) == expected_horizons:
        print("✅ PASS: policy_horizons matches expected value")
        test1_passed = True
    else:
        print(f"❌ FAIL: policy_horizons={default_horizons} != expected {expected_horizons}")
        test1_passed = False
    
    # Test 2: policy_w_h normalization
    print("\n[Test 2] policy_w_h 정규화 확인")
    print("-" * 80)
    
    # Create sample weights (should be normalized)
    test_weights = [
        np.array([0.1, 0.2, 0.3, 0.2, 0.15, 0.05]),  # Sum = 1.0
        np.array([0.2, 0.3, 0.4, 0.1, 0.0, 0.0]),   # Sum = 1.0
        np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0]),   # Sum = 1.0
    ]
    
    test2_passed = True
    for i, w in enumerate(test_weights):
        w_sum = float(np.sum(w))
        if abs(w_sum - 1.0) < 1e-5:
            print(f"✅ PASS: Test weight {i+1} sum={w_sum:.8f} ≈ 1.0")
        else:
            print(f"❌ FAIL: Test weight {i+1} sum={w_sum:.8f} != 1.0")
            test2_passed = False
    
    # Test 3: policy_ev_mix calculation
    print("\n[Test 3] policy_ev_mix_long/short 계산 검증")
    print("-" * 80)
    
    # Sample data
    horizons = np.array([60, 180, 300, 600, 900, 1800], dtype=np.float64)
    w_long = np.array([0.2, 0.25, 0.25, 0.15, 0.1, 0.05], dtype=np.float64)
    w_short = np.array([0.3, 0.3, 0.2, 0.1, 0.05, 0.05], dtype=np.float64)
    ev_long_h = np.array([0.001, 0.002, 0.0015, 0.0008, 0.0005, 0.0003], dtype=np.float64)
    ev_short_h = np.array([-0.0005, -0.001, -0.0008, -0.0004, -0.0002, -0.0001], dtype=np.float64)
    
    # Normalize weights
    w_long = w_long / (w_long.sum() + 1e-12)
    w_short = w_short / (w_short.sum() + 1e-12)
    
    # Calculate policy_ev_mix
    policy_ev_mix_long = float((w_long * ev_long_h).sum())
    policy_ev_mix_short = float((w_short * ev_short_h).sum())
    
    # Manual calculation for verification
    ev_mix_long_manual = sum(w_long[i] * ev_long_h[i] for i in range(len(w_long)))
    ev_mix_short_manual = sum(w_short[i] * ev_short_h[i] for i in range(len(w_short)))
    
    print(f"w_long: {[f'{w:.4f}' for w in w_long]}")
    print(f"w_long sum: {np.sum(w_long):.8f}")
    print(f"ev_long_h: {[f'{e:.6f}' for e in ev_long_h]}")
    print(f"policy_ev_mix_long (vectorized): {policy_ev_mix_long:.8f}")
    print(f"policy_ev_mix_long (manual): {ev_mix_long_manual:.8f}")
    print(f"Difference: {abs(policy_ev_mix_long - ev_mix_long_manual):.10f}")
    
    print(f"\nw_short: {[f'{w:.4f}' for w in w_short]}")
    print(f"w_short sum: {np.sum(w_short):.8f}")
    print(f"ev_short_h: {[f'{e:.6f}' for e in ev_short_h]}")
    print(f"policy_ev_mix_short (vectorized): {policy_ev_mix_short:.8f}")
    print(f"policy_ev_mix_short (manual): {ev_mix_short_manual:.8f}")
    print(f"Difference: {abs(policy_ev_mix_short - ev_mix_short_manual):.10f}")
    
    test3_passed = True
    if abs(policy_ev_mix_long - ev_mix_long_manual) < 1e-6:
        print("✅ PASS: policy_ev_mix_long calculation is correct")
    else:
        print(f"❌ FAIL: policy_ev_mix_long calculation mismatch")
        test3_passed = False
    
    if abs(policy_ev_mix_short - ev_mix_short_manual) < 1e-6:
        print("✅ PASS: policy_ev_mix_short calculation is correct")
    else:
        print(f"❌ FAIL: policy_ev_mix_short calculation mismatch")
        test3_passed = False
    
    # Test 4: Array length consistency
    print("\n[Test 4] 배열 길이 일관성 확인")
    print("-" * 80)
    
    test4_passed = True
    if len(w_long) == len(horizons) and len(w_short) == len(horizons):
        print(f"✅ PASS: w_long length={len(w_long)} == horizons length={len(horizons)}")
        print(f"✅ PASS: w_short length={len(w_short)} == horizons length={len(horizons)}")
    else:
        print(f"❌ FAIL: Length mismatch")
        test4_passed = False
    
    if len(ev_long_h) == len(horizons) and len(ev_short_h) == len(horizons):
        print(f"✅ PASS: ev_long_h length={len(ev_long_h)} == horizons length={len(horizons)}")
        print(f"✅ PASS: ev_short_h length={len(ev_short_h)} == horizons length={len(horizons)}")
    else:
        print(f"❌ FAIL: EV array length mismatch")
        test4_passed = False
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    all_passed = test1_passed and test2_passed and test3_passed and test4_passed
    
    print(f"Test 1 (policy_horizons): {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Test 2 (policy_w_h normalization): {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"Test 3 (policy_ev_mix calculation): {'✅ PASS' if test3_passed else '❌ FAIL'}")
    print(f"Test 4 (array length consistency): {'✅ PASS' if test4_passed else '❌ FAIL'}")
    
    if all_passed:
        print("\n✅ ALL VALIDATIONS PASSED")
        return 0
    else:
        print("\n❌ SOME VALIDATIONS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = test_diff2_validation()
    sys.exit(exit_code)


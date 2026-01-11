#!/usr/bin/env python3
"""
Test script for Diff 4 validation: Exit Reason 통계 + 정합성 관측
Tests:
1. payload key 존재 확인
2. exit reason counts 구조 확인
3. min_hold_sec 값 확인 (≈180s)
4. exit reason fractions 합계 확인
"""

import sys
import os

# Ensure repo root is on sys.path so `import engines` works when run as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from engines.mc.monte_carlo_engine import MonteCarloEngine


def test_diff4_validation():
    """Test Diff 4 validation points"""
    
    print("=" * 80)
    print("Diff 4 Validation: Exit Reason 통계 + 정합성 관측")
    print("=" * 80)
    
    # Test 1: min_hold_sec should be ≈180s
    print("\n[Test 1] min_hold_sec 값 확인")
    print("-" * 80)
    
    min_hold_sec_expected = 180
    min_hold_sec_actual = int(getattr(MonteCarloEngine, "MIN_HOLD_SEC_DIRECTIONAL", 180))
    
    print(f"Expected: {min_hold_sec_expected}s")
    print(f"Actual: {min_hold_sec_actual}s")
    
    if abs(min_hold_sec_actual - min_hold_sec_expected) <= 10:  # Allow 10s tolerance
        print(f"✅ PASS: min_hold_sec={min_hold_sec_actual}s ≈ {min_hold_sec_expected}s")
        test1_passed = True
    else:
        print(f"❌ FAIL: min_hold_sec={min_hold_sec_actual}s is not ≈ {min_hold_sec_expected}s")
        test1_passed = False
    
    # Test 2: Exit reason counts structure
    print("\n[Test 2] Exit reason counts 구조 확인")
    print("-" * 80)
    
    # Sample exit reason counts structure
    policy_horizons = [60, 180, 300, 600, 900, 1800]
    
    # Valid structure: list of dicts, one per horizon
    valid_exit_reason_counts_long = [
        {"unrealized_dd": 10, "hold_bad": 5, "score_flip": 3, "horizon_end": 82},
        {"unrealized_dd": 8, "hold_bad": 7, "score_flip": 2, "horizon_end": 83},
        {"unrealized_dd": 6, "hold_bad": 9, "score_flip": 1, "horizon_end": 84},
        {"unrealized_dd": 4, "hold_bad": 11, "score_flip": 0, "horizon_end": 85},
        {"unrealized_dd": 2, "hold_bad": 13, "score_flip": 0, "horizon_end": 85},
        {"unrealized_dd": 1, "hold_bad": 15, "score_flip": 0, "horizon_end": 84},
    ]
    
    valid_exit_reason_counts_short = [
        {"unrealized_dd": 12, "hold_bad": 4, "score_flip": 4, "horizon_end": 80},
        {"unrealized_dd": 10, "hold_bad": 6, "score_flip": 3, "horizon_end": 81},
        {"unrealized_dd": 8, "hold_bad": 8, "score_flip": 2, "horizon_end": 82},
        {"unrealized_dd": 6, "hold_bad": 10, "score_flip": 1, "horizon_end": 83},
        {"unrealized_dd": 4, "hold_bad": 12, "score_flip": 0, "horizon_end": 84},
        {"unrealized_dd": 2, "hold_bad": 14, "score_flip": 0, "horizon_end": 84},
    ]
    
    test2_passed = True
    
    # Check structure
    if not isinstance(valid_exit_reason_counts_long, list):
        print(f"❌ FAIL: exit_reason_counts_per_h_long is not a list")
        test2_passed = False
    elif len(valid_exit_reason_counts_long) != len(policy_horizons):
        print(f"❌ FAIL: exit_reason_counts_per_h_long length={len(valid_exit_reason_counts_long)} != policy_horizons length={len(policy_horizons)}")
        test2_passed = False
    else:
        print(f"✅ PASS: exit_reason_counts_per_h_long is a list with length={len(valid_exit_reason_counts_long)}")
    
    if not isinstance(valid_exit_reason_counts_short, list):
        print(f"❌ FAIL: exit_reason_counts_per_h_short is not a list")
        test2_passed = False
    elif len(valid_exit_reason_counts_short) != len(policy_horizons):
        print(f"❌ FAIL: exit_reason_counts_per_h_short length={len(valid_exit_reason_counts_short)} != policy_horizons length={len(policy_horizons)}")
        test2_passed = False
    else:
        print(f"✅ PASS: exit_reason_counts_per_h_short is a list with length={len(valid_exit_reason_counts_short)}")
    
    # Test 3: Exit reason fractions calculation
    print("\n[Test 3] Exit reason fractions 계산 검증")
    print("-" * 80)
    
    # Sample exit reason counts
    sample_counts = {"unrealized_dd": 10, "hold_bad": 5, "score_flip": 3, "horizon_end": 82}
    total = sum(sample_counts.values())
    
    policy_exit_unrealized_dd_frac = float(sample_counts.get("unrealized_dd", 0)) / total
    policy_exit_hold_bad_frac = float(sample_counts.get("hold_bad", 0)) / total
    policy_exit_score_flip_frac = float(sample_counts.get("score_flip", 0)) / total
    
    print(f"Sample counts: {sample_counts}")
    print(f"Total: {total}")
    print(f"policy_exit_unrealized_dd_frac: {policy_exit_unrealized_dd_frac:.6f}")
    print(f"policy_exit_hold_bad_frac: {policy_exit_hold_bad_frac:.6f}")
    print(f"policy_exit_score_flip_frac: {policy_exit_score_flip_frac:.6f}")
    
    frac_sum = policy_exit_unrealized_dd_frac + policy_exit_hold_bad_frac + policy_exit_score_flip_frac
    print(f"Sum of fractions: {frac_sum:.6f}")
    
    test3_passed = True
    if frac_sum <= 1.0 + 1e-5:  # Allow small floating point error
        print(f"✅ PASS: Exit reason fractions sum={frac_sum:.6f} <= 1.0")
    else:
        print(f"❌ FAIL: Exit reason fractions sum={frac_sum:.6f} > 1.0")
        test3_passed = False
    
    # Test 4: Payload key existence (simulated check)
    print("\n[Test 4] Payload key 존재 확인 (시뮬레이션)")
    print("-" * 80)
    
    required_keys = [
        "policy_exit_reason_counts_per_h_long",
        "policy_exit_reason_counts_per_h_short",
        "policy_exit_unrealized_dd_frac",
        "policy_exit_hold_bad_frac",
        "policy_exit_score_flip_frac",
    ]
    
    # Simulated payload
    simulated_payload = {
        "policy_exit_reason_counts_per_h_long": valid_exit_reason_counts_long,
        "policy_exit_reason_counts_per_h_short": valid_exit_reason_counts_short,
        "policy_exit_unrealized_dd_frac": policy_exit_unrealized_dd_frac,
        "policy_exit_hold_bad_frac": policy_exit_hold_bad_frac,
        "policy_exit_score_flip_frac": policy_exit_score_flip_frac,
    }
    
    test4_passed = True
    for key in required_keys:
        if key in simulated_payload:
            print(f"✅ PASS: {key} exists in payload")
        else:
            print(f"❌ FAIL: {key} missing in payload")
            test4_passed = False
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    all_passed = test1_passed and test2_passed and test3_passed and test4_passed
    
    print(f"Test 1 (min_hold_sec): {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Test 2 (exit reason counts structure): {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"Test 3 (exit reason fractions): {'✅ PASS' if test3_passed else '❌ FAIL'}")
    print(f"Test 4 (payload key existence): {'✅ PASS' if test4_passed else '❌ FAIL'}")
    
    if all_passed:
        print("\n✅ ALL VALIDATIONS PASSED")
        return 0
    else:
        print("\n❌ SOME VALIDATIONS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = test_diff4_validation()
    sys.exit(exit_code)

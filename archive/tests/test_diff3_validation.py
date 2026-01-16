#!/usr/bin/env python3
"""
Test script for Diff 3 validation: Rule-based dynamic weights
Tests:
1. signal_strength ↑ → policy_h_eff_sec ↓
2. policy_w_short_sum is in [0, 1]
"""

import numpy as np
import sys
import os

# Ensure repo root is on sys.path so `import engines` works when run as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from engines.mc.monte_carlo_engine import MonteCarloEngine


def test_weights_for_horizons():
    """Test _weights_for_horizons method with various signal_strength values"""
    
    # Typical horizons used in the system
    policy_horizons = [60, 180, 300, 600, 1200, 1800, 3600]
    h_arr = np.asarray(policy_horizons, dtype=np.float64)
    
    # Test cases: signal_strength values
    test_cases = [
        (0.0, "Minimum signal strength"),
        (1.0, "Low signal strength"),
        (2.0, "Medium signal strength"),
        (3.0, "High signal strength"),
        (4.0, "Maximum signal strength (clipped)"),
        (5.0, "Above maximum (should be clipped to 4.0)"),
    ]
    
    print("=" * 80)
    print("Diff 3 Validation: Rule-based Dynamic Weights")
    print("=" * 80)
    print(f"Horizons: {policy_horizons}")
    print()
    
    results = []
    
    for signal_strength, description in test_cases:
        # Calculate weights using static method
        w_prior = MonteCarloEngine._weights_for_horizons(policy_horizons, signal_strength)
        
        # Calculate half_life
        s_clip = float(np.clip(signal_strength, 0.0, 4.0))
        half_life = 1800.0 / (1.0 + s_clip)
        
        # Calculate policy_h_eff_sec_prior (from w_prior only)
        policy_h_eff_sec_prior = float(np.sum(w_prior * h_arr)) if w_prior.size > 0 else 0.0
        
        # Calculate policy_w_short_sum (≤300s weights)
        policy_w_short_sum = float(np.sum(w_prior[h_arr <= 300.0])) if w_prior.size > 0 else 0.0
        
        results.append({
            'signal_strength': signal_strength,
            's_clip': s_clip,
            'half_life': half_life,
            'policy_h_eff_sec_prior': policy_h_eff_sec_prior,
            'policy_w_short_sum': policy_w_short_sum,
            'w_prior': w_prior,
        })
        
        print(f"Signal Strength: {signal_strength:.2f} ({description})")
        print(f"  s_clip: {s_clip:.2f}")
        print(f"  half_life: {half_life:.2f} sec")
        print(f"  policy_h_eff_sec_prior: {policy_h_eff_sec_prior:.2f} sec")
        print(f"  policy_w_short_sum: {policy_w_short_sum:.6f}")
        print(f"  Weights: {[f'{w:.4f}' for w in w_prior]}")
        print(f"  Weight sum: {np.sum(w_prior):.6f} (should be ~1.0)")
        print()
    
    # Validation 1: signal_strength ↑ → policy_h_eff_sec_prior ↓
    print("=" * 80)
    print("Validation 1: signal_strength ↑ → policy_h_eff_sec_prior ↓")
    print("=" * 80)
    
    prev_h_eff = None
    prev_s_clip = None
    validation1_passed = True
    
    for i, result in enumerate(results):
        signal_strength = result['signal_strength']
        s_clip = result['s_clip']
        h_eff = result['policy_h_eff_sec_prior']
        
        if prev_h_eff is not None:
            # If s_clip increased, h_eff should decrease
            # If s_clip is same (due to clipping), h_eff should be same (not an error)
            if s_clip > prev_s_clip:
                if h_eff >= prev_h_eff:
                    print(f"❌ FAIL: signal_strength={signal_strength:.2f} (s_clip={s_clip:.2f}), h_eff={h_eff:.2f} >= previous {prev_h_eff:.2f}")
                    validation1_passed = False
                else:
                    print(f"✅ PASS: signal_strength={signal_strength:.2f} (s_clip={s_clip:.2f}), h_eff={h_eff:.2f} < previous {prev_h_eff:.2f}")
            elif s_clip == prev_s_clip:
                if h_eff == prev_h_eff:
                    print(f"✅ PASS: signal_strength={signal_strength:.2f} (s_clip={s_clip:.2f}, clipped), h_eff={h_eff:.2f} == previous {prev_h_eff:.2f} (expected)")
                else:
                    print(f"⚠️  WARN: signal_strength={signal_strength:.2f} (s_clip={s_clip:.2f}, clipped), h_eff={h_eff:.2f} != previous {prev_h_eff:.2f} (unexpected)")
            else:
                print(f"⚠️  WARN: signal_strength={signal_strength:.2f} (s_clip={s_clip:.2f}), s_clip decreased from {prev_s_clip:.2f}")
        else:
            print(f"   INIT: signal_strength={signal_strength:.2f} (s_clip={s_clip:.2f}), h_eff={h_eff:.2f}")
        
        prev_h_eff = h_eff
        prev_s_clip = s_clip
    
    print()
    if validation1_passed:
        print("✅ Validation 1 PASSED: policy_h_eff_sec_prior decreases as signal_strength increases")
    else:
        print("❌ Validation 1 FAILED: policy_h_eff_sec_prior does not decrease as signal_strength increases")
    print()
    
    # Validation 2: policy_w_short_sum in [0, 1]
    print("=" * 80)
    print("Validation 2: policy_w_short_sum in [0, 1]")
    print("=" * 80)
    
    validation2_passed = True
    
    for result in results:
        signal_strength = result['signal_strength']
        w_short_sum = result['policy_w_short_sum']
        
        if w_short_sum < 0.0 or w_short_sum > 1.0:
            print(f"❌ FAIL: signal_strength={signal_strength:.2f}, policy_w_short_sum={w_short_sum:.6f} is out of range [0, 1]")
            validation2_passed = False
        else:
            print(f"✅ PASS: signal_strength={signal_strength:.2f}, policy_w_short_sum={w_short_sum:.6f} is in range [0, 1]")
    
    print()
    if validation2_passed:
        print("✅ Validation 2 PASSED: All policy_w_short_sum values are in [0, 1]")
    else:
        print("❌ Validation 2 FAILED: Some policy_w_short_sum values are out of range [0, 1]")
    print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    if validation1_passed and validation2_passed:
        print("✅ ALL VALIDATIONS PASSED")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        if not validation1_passed:
            print("  - Validation 1: signal_strength ↑ → policy_h_eff_sec_prior ↓")
        if not validation2_passed:
            print("  - Validation 2: policy_w_short_sum in [0, 1]")
        return 1


if __name__ == "__main__":
    exit_code = test_weights_for_horizons()
    sys.exit(exit_code)

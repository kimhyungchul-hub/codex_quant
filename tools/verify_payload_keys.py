#!/usr/bin/env python3
"""
Payload 검증 스크립트: _row 메서드의 return 딕셔너리에 모든 검증 포인트가 포함되는지 확인
"""

import sys
import os
import inspect

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_engine_mc_v2_final import LiveOrchestrator


def verify_payload_keys():
    """_row 메서드의 return 딕셔너리에 모든 검증 포인트가 포함되는지 확인"""
    
    print("=" * 80)
    print("Payload Keys 검증: _row 메서드 return 딕셔너리 확인")
    print("=" * 80)
    
    # _row 메서드의 소스 코드를 읽어서 return 딕셔너리 확인
    source = inspect.getsource(LiveOrchestrator._row)
    
    # 검증 포인트 정의
    validation_points = {
        "Diff 2": [
            "policy_horizons",
            "policy_w_h",
            "policy_h_eff_sec",
            "policy_ev_mix_long",
            "policy_ev_mix_short",
            "paths_reused",
        ],
        "Diff 3": [
            "policy_signal_strength",
            "policy_h_eff_sec",
            "policy_h_eff_sec_prior",
            "policy_w_short_sum",
        ],
        "Diff 4": [
            "policy_exit_reason_counts_per_h_long",
            "policy_exit_reason_counts_per_h_short",
            "policy_exit_unrealized_dd_frac",
            "policy_exit_hold_bad_frac",
            "policy_exit_score_flip_frac",
        ],
        "Diff 5": [
            "exec_mode",
            "pmaker_entry",
            "fee_roundtrip_fee_mix",
        ],
    }
    
    all_passed = True
    
    for diff_name, keys in validation_points.items():
        print(f"\n[{diff_name}] 검증 포인트 확인")
        print("-" * 80)
        
        for key in keys:
            # return 딕셔너리에 key가 포함되어 있는지 확인
            # "key": 또는 '"key":' 패턴으로 검색
            patterns = [
                f'"{key}":',
                f"'{key}':",
                f'"{key}"',
                f"'{key}'",
            ]
            
            found = any(pattern in source for pattern in patterns)
            
            if found:
                print(f"✅ PASS: {key} found in return dict")
            else:
                print(f"❌ FAIL: {key} NOT found in return dict")
                all_passed = False
    
    # 추가 검증: fee_roundtrip_fee_taker도 확인 (Diff 5 검증에 필요)
    print(f"\n[추가 검증] fee_roundtrip_fee_taker 확인")
    print("-" * 80)
    if '"fee_roundtrip_fee_taker":' in source or "'fee_roundtrip_fee_taker':" in source:
        print("✅ PASS: fee_roundtrip_fee_taker found in return dict")
    else:
        print("⚠️  WARN: fee_roundtrip_fee_taker not found (may be in nested structure)")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    if all_passed:
        print("✅ ALL VALIDATION POINTS ARE IN PAYLOAD")
        print("\n참고: 실제 런타임에서 값이 None이거나 0일 수 있지만,")
        print("     key 자체는 payload에 포함되어 있어야 합니다.")
        return 0
    else:
        print("❌ SOME VALIDATION POINTS ARE MISSING FROM PAYLOAD")
        return 1


if __name__ == "__main__":
    exit_code = verify_payload_keys()
    sys.exit(exit_code)


#!/usr/bin/env python3
"""
모든 Diff (2-7) 검증 포인트가 payload에 포함되어 있는지 확인하는 스크립트
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_engine_mc_v2_final import LiveOrchestrator

# 모든 Diff 검증 포인트 정의
DIFF_KEYS = {
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
        "fee_roundtrip_fee_taker",
        "fee_roundtrip_fee_maker",
    ],
    "Diff 6": [
        "pmaker_entry",
        "pmaker_entry_delay_sec",
        "pmaker",  # pmaker status (dict with model_loaded, params_total)
    ],
    "Diff 7": [
        "pmaker_entry_delay_penalty_r",
        "policy_entry_shift_steps",
    ],
}

def check_row_method():
    """_row 메서드의 반환값에 모든 키가 포함되어 있는지 확인"""
    
    print("=" * 80)
    print("모든 Diff 검증 포인트 Payload 포함 여부 확인")
    print("=" * 80)
    
    # _row 메서드 소스 코드 읽기
    import inspect
    from main_engine_mc_v2_final import LiveOrchestrator
    
    row_source = inspect.getsource(LiveOrchestrator._row)
    
    all_passed = True
    missing_keys = {}
    
    for diff_name, keys in DIFF_KEYS.items():
        print(f"\n[{diff_name}] 검증 포인트 확인:")
        print("-" * 80)
        
        diff_missing = []
        for key in keys:
            # _row 메서드의 return 딕셔너리에 키가 포함되어 있는지 확인
            # return 문 이후의 딕셔너리 키를 찾음
            if f'"{key}"' in row_source or f"'{key}'" in row_source:
                print(f"  ✅ {key}: 포함됨")
            else:
                print(f"  ❌ {key}: 누락됨")
                diff_missing.append(key)
                all_passed = False
        
        if diff_missing:
            missing_keys[diff_name] = diff_missing
    
    # pmaker status 내부 키 확인 (Diff 6)
    print(f"\n[Diff 6 추가] pmaker status 내부 키 확인:")
    print("-" * 80)
    
    pmaker_status_keys = ["model_loaded", "params_total"]
    pmaker_source = inspect.getsource(LiveOrchestrator._pmaker_status)
    
    for key in pmaker_status_keys:
        if f'"{key}"' in pmaker_source or f"'{key}'" in pmaker_source:
            print(f"  ✅ pmaker.{key}: 포함됨")
        else:
            print(f"  ❌ pmaker.{key}: 누락됨")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    if all_passed:
        print("✅ 모든 Diff 검증 포인트가 payload에 포함되어 있습니다.")
        return 0
    else:
        print("❌ 일부 Diff 검증 포인트가 누락되었습니다:")
        for diff_name, keys in missing_keys.items():
            print(f"  {diff_name}: {', '.join(keys)}")
        return 1

if __name__ == "__main__":
    exit_code = check_row_method()
    sys.exit(exit_code)


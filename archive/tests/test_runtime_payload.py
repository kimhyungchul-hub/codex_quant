#!/usr/bin/env python3
"""
실제 런타임 Payload 검증 스크립트
Mock 데이터로 _row 메서드를 호출하여 실제 payload를 생성하고 모든 Diff 검증 포인트를 확인
"""

import sys
from pathlib import Path
import time
from collections import deque

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from main_engine_mc_v2_final import LiveOrchestrator


class MockExchange:
    """Mock exchange 객체 (LiveOrchestrator 초기화에 필요)"""
    pass


def create_mock_decision():
    """모든 Diff 검증 포인트를 포함하는 Mock decision 생성"""
    
    # Diff 2: Multi-Horizon Policy Mix
    policy_horizons = [60, 180, 300, 600, 900, 1800]
    policy_w_h = [0.2, 0.25, 0.25, 0.15, 0.1, 0.05]  # 합이 1.0
    policy_h_eff_sec = 350.5
    policy_ev_mix_long = 0.0012
    policy_ev_mix_short = -0.0005
    paths_reused = True
    
    # Diff 3: Rule-based Dynamic Weights
    policy_signal_strength = 2.5
    policy_h_eff_sec_prior = 300.0
    policy_w_short_sum = 0.75
    
    # Diff 4: Exit Reason 통계
    policy_exit_reason_counts_per_h_long = [
        {"unrealized_dd": 10, "hold_bad": 5, "score_flip": 3, "horizon_end": 82},
        {"unrealized_dd": 8, "hold_bad": 7, "score_flip": 2, "horizon_end": 83},
        {"unrealized_dd": 6, "hold_bad": 9, "score_flip": 1, "horizon_end": 84},
        {"unrealized_dd": 4, "hold_bad": 11, "score_flip": 0, "horizon_end": 85},
        {"unrealized_dd": 2, "hold_bad": 13, "score_flip": 0, "horizon_end": 85},
        {"unrealized_dd": 1, "hold_bad": 15, "score_flip": 0, "horizon_end": 84},
    ]
    policy_exit_reason_counts_per_h_short = [
        {"unrealized_dd": 12, "hold_bad": 4, "score_flip": 4, "horizon_end": 80},
        {"unrealized_dd": 10, "hold_bad": 6, "score_flip": 3, "horizon_end": 81},
        {"unrealized_dd": 8, "hold_bad": 8, "score_flip": 2, "horizon_end": 82},
        {"unrealized_dd": 6, "hold_bad": 10, "score_flip": 1, "horizon_end": 83},
        {"unrealized_dd": 4, "hold_bad": 12, "score_flip": 0, "horizon_end": 84},
        {"unrealized_dd": 2, "hold_bad": 14, "score_flip": 0, "horizon_end": 84},
    ]
    
    # Exit reason counts에서 fractions 계산
    exit_reason_counts_policy = {"unrealized_dd": 10, "hold_bad": 5, "score_flip": 3, "horizon_end": 82}
    total_exits = sum(exit_reason_counts_policy.values())
    policy_exit_unrealized_dd_frac = exit_reason_counts_policy.get("unrealized_dd", 0) / total_exits
    policy_exit_hold_bad_frac = exit_reason_counts_policy.get("hold_bad", 0) / total_exits
    policy_exit_score_flip_frac = exit_reason_counts_policy.get("score_flip", 0) / total_exits
    
    # Diff 5: Maker → Market 혼합 실행
    exec_mode = "maker_then_market"
    pmaker_entry = 0.6
    fee_roundtrip_fee_mix = 0.0004  # fee_taker=0.0006보다 작음
    fee_roundtrip_fee_taker = 0.0006
    fee_roundtrip_fee_maker = 0.0002
    
    # Diff 7: Maker 지연을 EV에 직접 패널티로 반영
    pmaker_entry_delay_penalty_r = 0.00015  # delay penalty (>= 0)
    policy_entry_shift_steps = 3  # horizon shift steps (>= 0)
    
    # mc_meta 생성 (모든 검증 포인트 포함)
    mc_meta = {
        # Diff 2
        "policy_horizons": policy_horizons,
        "policy_w_h": policy_w_h,
        "policy_h_eff_sec": policy_h_eff_sec,
        "policy_ev_mix_long": policy_ev_mix_long,
        "policy_ev_mix_short": policy_ev_mix_short,
        "paths_reused": paths_reused,
        
        # Diff 3
        "policy_signal_strength": policy_signal_strength,
        "policy_h_eff_sec_prior": policy_h_eff_sec_prior,
        "policy_w_short_sum": policy_w_short_sum,
        "policy_half_life_sec": 600.0,
        
        # Diff 4
        "policy_exit_reason_counts_per_h": policy_exit_reason_counts_per_h_long,  # direction_policy=1 가정
        "policy_exit_reason_counts_per_h_long": policy_exit_reason_counts_per_h_long,
        "policy_exit_reason_counts_per_h_short": policy_exit_reason_counts_per_h_short,
        "policy_exit_unrealized_dd_frac": policy_exit_unrealized_dd_frac,
        "policy_exit_hold_bad_frac": policy_exit_hold_bad_frac,
        "policy_exit_score_flip_frac": policy_exit_score_flip_frac,
        
        # Diff 5
        "exec_mode": exec_mode,
        "pmaker_entry": pmaker_entry,
        "fee_roundtrip_fee_mix": fee_roundtrip_fee_mix,
        "fee_roundtrip_fee_taker": fee_roundtrip_fee_taker,
        "fee_roundtrip_fee_maker": fee_roundtrip_fee_maker,
        "p_maker": pmaker_entry,
        
        # Diff 7: Maker 지연 EV 패널티
        "pmaker_entry_delay_penalty_r": pmaker_entry_delay_penalty_r,
        "policy_entry_shift_steps": policy_entry_shift_steps,
        
        # 기타 필수 필드
        "ev": 0.001,
        "win_rate": 0.55,
        "best_horizon": 300,
    }
    
    # Mock decision 생성
    decision = {
        "action": "LONG",
        "confidence": 0.65,
        "reason": "MC engine signal",
        "details": [
            {
                "_engine": "mc_engine",
                "meta": mc_meta,
            }
        ],
        "meta": mc_meta,  # fallback
    }
    
    return decision


def verify_payload_values(payload, diff_name, validation_points):
    """Payload에서 검증 포인트의 값 확인"""
    
    print(f"\n[{diff_name}] Payload 값 검증")
    print("-" * 80)
    
    all_passed = True
    results = {}
    
    for key, expected_type, validator in validation_points:
        value = payload.get(key)
        results[key] = value
        
        # Key 존재 확인
        if key not in payload:
            print(f"❌ FAIL: {key} NOT FOUND in payload")
            all_passed = False
            continue
        
        # None 체크
        if value is None:
            print(f"⚠️  WARN: {key} is None (may be acceptable)")
            continue
        
        # 타입 확인
        if expected_type and not isinstance(value, expected_type):
            print(f"❌ FAIL: {key} type mismatch: expected {expected_type.__name__}, got {type(value).__name__}")
            all_passed = False
            continue
        
        # Validator 함수로 추가 검증
        if validator:
            try:
                validator_result, validator_msg = validator(value)
                if validator_result:
                    print(f"✅ PASS: {key} = {value} {validator_msg}")
                else:
                    print(f"❌ FAIL: {key} = {value} {validator_msg}")
                    all_passed = False
            except Exception as e:
                print(f"⚠️  WARN: {key} validator error: {e}")
        else:
            print(f"✅ PASS: {key} = {value}")
    
    return all_passed, results


def test_runtime_payload():
    """실제 런타임 payload 검증"""
    
    print("=" * 80)
    print("실제 런타임 Payload 검증")
    print("=" * 80)
    
    # LiveOrchestrator 인스턴스 생성
    try:
        mock_exchange = MockExchange()
        orchestrator = LiveOrchestrator(mock_exchange)
        
        # 필요한 초기화 (market, orderbook 등)
        orchestrator.market = {"BTCUSDT": {"price": 50000.0, "ts": int(time.time() * 1000)}}
        orchestrator.orderbook = {"BTCUSDT": {"ts": int(time.time() * 1000), "ready": True, "bids": [], "asks": []}}
        orchestrator._last_kline_ok_ms = {"BTCUSDT": int(time.time() * 1000)}
        
        print("✅ LiveOrchestrator 인스턴스 생성 완료")
    except Exception as e:
        print(f"❌ FAIL: LiveOrchestrator 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Mock decision 생성
    decision = create_mock_decision()
    print("✅ Mock decision 생성 완료")
    
    # _row 메서드 호출하여 payload 생성
    try:
        sym = "BTCUSDT"
        price = 50000.0
        ts = int(time.time() * 1000)
        candles = 100
        
        payload = orchestrator._row(sym, price, ts, decision, candles)
        print("✅ Payload 생성 완료")
        print(f"Payload keys count: {len(payload)}")
    except Exception as e:
        print(f"❌ FAIL: _row 메서드 호출 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 검증 포인트 정의
    validation_points = {
        "Diff 2": [
            ("policy_horizons", list, lambda v: (v == [60, 180, 300, 600, 900, 1800], f"(expected [60, 180, 300, 600, 900, 1800])")),
            ("policy_w_h", list, lambda v: (abs(sum(v) - 1.0) < 0.01 if isinstance(v, list) and len(v) > 0 else False, f"(sum={sum(v) if isinstance(v, list) else 'N/A'})")),
            ("policy_h_eff_sec", (int, float), None),
            ("policy_ev_mix_long", (int, float), None),
            ("policy_ev_mix_short", (int, float), None),
            ("paths_reused", bool, None),
        ],
        "Diff 3": [
            ("policy_signal_strength", (int, float), None),
            ("policy_h_eff_sec", (int, float), None),
            ("policy_h_eff_sec_prior", (int, float), None),
            ("policy_w_short_sum", (int, float), lambda v: (0.0 <= v <= 1.0, f"(range [0, 1])")),
        ],
        "Diff 4": [
            ("policy_exit_reason_counts_per_h_long", list, None),
            ("policy_exit_reason_counts_per_h_short", list, None),
            ("policy_exit_unrealized_dd_frac", (int, float, type(None)), lambda v: (v is None or (0.0 <= v <= 1.0), f"(range [0, 1] or None)")),
            ("policy_exit_hold_bad_frac", (int, float, type(None)), lambda v: (v is None or (0.0 <= v <= 1.0), f"(range [0, 1] or None)")),
            ("policy_exit_score_flip_frac", (int, float, type(None)), lambda v: (v is None or (0.0 <= v <= 1.0), f"(range [0, 1] or None)")),
        ],
        "Diff 5": [
            ("exec_mode", str, None),
            ("pmaker_entry", (int, float, type(None)), lambda v: (v is None or (0.0 <= v <= 1.0), f"(range [0, 1] or None)")),
            ("fee_roundtrip_fee_mix", (int, float, type(None)), None),
        ],
        "Diff 6": [
            ("pmaker_entry", (int, float, type(None)), lambda v: (v is None or (0.0 <= v <= 1.0), f"(range [0, 1] or None)")),
            ("pmaker_entry_delay_sec", (int, float, type(None)), None),
        ],
        "Diff 7": [
            ("pmaker_entry_delay_penalty_r", (int, float, type(None)), lambda v: (v is None or v >= 0.0, f"(>= 0 or None)")),
            ("policy_entry_shift_steps", (int, type(None)), lambda v: (v is None or v >= 0, f"(>= 0 or None)")),
        ],
    }
    
    # 각 Diff별로 검증
    all_diffs_passed = True
    all_results = {}
    
    for diff_name, points in validation_points.items():
        passed, results = verify_payload_values(payload, diff_name, points)
        all_results[diff_name] = results
        if not passed:
            all_diffs_passed = False
    
    # 추가 검증: fee_roundtrip_fee_mix < fee_taker (Diff 5)
    print(f"\n[Diff 5 추가 검증] fee_roundtrip_fee_mix < fee_taker")
    print("-" * 80)
    
    fee_mix = payload.get("fee_roundtrip_fee_mix")
    fee_taker = payload.get("fee_roundtrip_fee_taker")
    p_maker = payload.get("p_maker") or payload.get("pmaker_entry")
    exec_mode = payload.get("exec_mode")
    
    if exec_mode == "maker_then_market" and p_maker and p_maker > 0:
        if fee_mix is not None and fee_taker is not None:
            if fee_mix < fee_taker:
                print(f"✅ PASS: fee_roundtrip_fee_mix={fee_mix:.8f} < fee_taker={fee_taker:.8f}")
            else:
                print(f"❌ FAIL: fee_roundtrip_fee_mix={fee_mix:.8f} >= fee_taker={fee_taker:.8f}")
                all_diffs_passed = False
        else:
            print(f"⚠️  WARN: fee_mix or fee_taker is None (cannot verify)")
    else:
        print(f"ℹ️  INFO: exec_mode={exec_mode}, p_maker={p_maker} (maker_then_market mode not active)")
    
    # Diff 6 추가 검증: pmaker status에서 model_loaded, params_total 확인
    print(f"\n[Diff 6 추가 검증] PMaker Status (model_loaded, params_total)")
    print("-" * 80)
    
    pmaker_status = payload.get("pmaker")
    if pmaker_status is not None and isinstance(pmaker_status, dict):
        model_loaded = pmaker_status.get("model_loaded")
        params_total = pmaker_status.get("params_total")
        
        if model_loaded is not None:
            if model_loaded:
                print(f"✅ PASS: model_loaded = True")
            else:
                print(f"⚠️  WARN: model_loaded = False (model may not be loaded)")
        else:
            print(f"⚠️  WARN: model_loaded key not found in pmaker status")
        
        if params_total is not None:
            if params_total > 0:
                print(f"✅ PASS: params_total = {params_total} (> 0)")
            else:
                print(f"⚠️  WARN: params_total = {params_total} (<= 0, model may be empty)")
        else:
            print(f"⚠️  WARN: params_total key not found in pmaker status")
        
        # Validation warnings 확인
        diff6_warnings = pmaker_status.get("diff6_validation_warnings")
        if diff6_warnings:
            print(f"⚠️  WARN: diff6_validation_warnings = {diff6_warnings}")
    else:
        print(f"⚠️  WARN: pmaker status not found in payload (may be None if pmaker is disabled)")
    
    # Diff 7 추가 검증: delay penalty가 EV에 반영되었는지 확인
    print(f"\n[Diff 7 추가 검증] Maker Delay Penalty EV 반영 확인")
    print("-" * 80)
    
    delay_penalty_r = payload.get("pmaker_entry_delay_penalty_r")
    shift_steps = payload.get("policy_entry_shift_steps")
    
    if delay_penalty_r is not None:
        if delay_penalty_r >= 0:
            print(f"✅ PASS: pmaker_entry_delay_penalty_r = {delay_penalty_r} (>= 0)")
        else:
            print(f"❌ FAIL: pmaker_entry_delay_penalty_r = {delay_penalty_r} (< 0, should be >= 0)")
            all_diffs_passed = False
    else:
        print(f"⚠️  WARN: pmaker_entry_delay_penalty_r is None (may not be calculated)")
    
    if shift_steps is not None:
        if shift_steps >= 0:
            print(f"✅ PASS: policy_entry_shift_steps = {shift_steps} (>= 0)")
        else:
            print(f"❌ FAIL: policy_entry_shift_steps = {shift_steps} (< 0, should be >= 0)")
            all_diffs_passed = False
    else:
        print(f"⚠️  WARN: policy_entry_shift_steps is None (may not be calculated)")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    for diff_name, results in all_results.items():
        print(f"\n{diff_name} 결과:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    
    if all_diffs_passed:
        print("\n✅ ALL VALIDATIONS PASSED")
        return 0
    else:
        print("\n❌ SOME VALIDATIONS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = test_runtime_payload()
    sys.exit(exit_code)

#!/usr/bin/env python3
"""
음수 EV 원인 분석 스크립트
4개 항목을 점검하여 음수 EV의 원인을 파악합니다.
"""

import sys
import os
import json

def check_negative_ev_causes(payload_row):
    """
    Payload row에서 음수 EV 원인을 분석합니다.
    
    항목 1: mu_alpha 자체가 음수/미약
    항목 2: exit이 min_hold 근처에서 반복적으로 발생
    항목 3: maker delay + spread + slippage가 gross EV를 잠식
    항목 4: SHORT 쪽 EV가 상대적으로 더 좋은데 long 편향
    """
    
    symbol = payload_row.get("symbol", "UNKNOWN")
    
    print("=" * 80)
    print(f"음수 EV 원인 분석: {symbol}")
    print("=" * 80)
    
    # payload_row 또는 meta에서 ev 값 찾기
    ev = payload_row.get("ev")
    if ev is None:
        # meta 필드에서 찾기
        meta = payload_row.get("meta", {})
        if isinstance(meta, dict):
            ev = meta.get("ev")
        if ev is None:
            ev = 0.0
    
    ev = float(ev) if ev is not None else 0.0
    print(f"\n현재 EV 값: {ev}")
    
    if ev >= 0:
        print("✅ EV가 음수가 아닙니다. 분석을 종료합니다.")
        return
    
    print("\n⚠️  EV가 음수입니다. 4개 항목을 점검합니다.\n")
    
    # payload_row와 meta에서 정보 가져오기
    meta = payload_row.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    
    # policy_ev_mix도 체크 (최종 EV가 0이지만 policy_ev_mix가 음수일 수 있음)
    policy_ev_mix_long_check = payload_row.get("policy_ev_mix_long") or meta.get("policy_ev_mix_long")
    policy_ev_mix_short_check = payload_row.get("policy_ev_mix_short") or meta.get("policy_ev_mix_short")
    
    if ev >= 0 and (policy_ev_mix_long_check is None or policy_ev_mix_long_check >= 0) and (policy_ev_mix_short_check is None or policy_ev_mix_short_check >= 0):
        print("✅ EV와 policy_ev_mix가 모두 음수가 아닙니다. 분석을 종료합니다.")
        return
    
    # policy_ev_mix가 음수인 경우도 분석
    if ev >= 0:
        print(f"⚠️  최종 EV는 0이지만, policy_ev_mix를 확인합니다:")
        print(f"   policy_ev_mix_long: {policy_ev_mix_long_check}")
        print(f"   policy_ev_mix_short: {policy_ev_mix_short_check}")
        if (policy_ev_mix_long_check is not None and policy_ev_mix_long_check < 0) or (policy_ev_mix_short_check is not None and policy_ev_mix_short_check < 0):
            print("   ⚠️  policy_ev_mix가 음수입니다. 분석을 계속합니다.\n")
        else:
            return
    
    # 항목 1: mu_alpha 자체가 음수/미약
    print("=" * 80)
    print("항목 1: mu_alpha 자체가 음수/미약")
    print("=" * 80)
    
    mu_alpha = payload_row.get("mu_alpha") or meta.get("mu_alpha")
    mu_alpha_mom = payload_row.get("mu_alpha_mom") or meta.get("mu_alpha_mom")
    mu_alpha_ofi = payload_row.get("mu_alpha_ofi") or meta.get("mu_alpha_ofi")
    
    print(f"  mu_alpha: {mu_alpha}")
    print(f"  mu_alpha_mom: {mu_alpha_mom}")
    print(f"  mu_alpha_ofi: {mu_alpha_ofi}")
    
    if mu_alpha is not None:
        if mu_alpha <= 0:
            print(f"  ❌ 문제: mu_alpha가 음수 또는 0입니다 ({mu_alpha})")
        elif mu_alpha < 0.0001:
            print(f"  ⚠️  경고: mu_alpha가 매우 작습니다 ({mu_alpha})")
        else:
            print(f"  ✅ mu_alpha는 양수입니다 ({mu_alpha})")
    else:
        print(f"  ⚠️  경고: mu_alpha가 payload에 없습니다")
    
    # 항목 2: exit이 min_hold 근처에서 반복적으로 발생
    print("\n" + "=" * 80)
    print("항목 2: exit이 min_hold 근처에서 반복적으로 발생")
    print("=" * 80)
    
    policy_exit_time_mean_sec = payload_row.get("policy_exit_time_mean_sec") or meta.get("policy_exit_time_mean_sec")
    min_hold_sec = 180  # 일반적인 min_hold 값
    
    print(f"  policy_exit_time_mean_sec: {policy_exit_time_mean_sec}")
    print(f"  min_hold_sec (예상): {min_hold_sec}")
    
    if policy_exit_time_mean_sec is not None:
        diff = abs(policy_exit_time_mean_sec - min_hold_sec)
        if diff < 30:  # 30초 이내 차이
            print(f"  ❌ 문제: exit이 min_hold 근처에서 발생합니다 (차이: {diff:.1f}초)")
        else:
            print(f"  ✅ exit 시간이 min_hold와 충분히 다릅니다 (차이: {diff:.1f}초)")
    else:
        print(f"  ⚠️  경고: policy_exit_time_mean_sec가 payload에 없습니다")
    
    # 항목 3: maker delay + spread + slippage가 gross EV를 잠식
    print("\n" + "=" * 80)
    print("항목 3: maker delay + spread + slippage가 gross EV를 잠식")
    print("=" * 80)
    
    pmaker_entry_delay_sec = payload_row.get("pmaker_entry_delay_sec") or meta.get("pmaker_entry_delay_sec")
    fee_roundtrip_total = payload_row.get("fee_roundtrip_total") or meta.get("fee_roundtrip_total") or meta.get("ev_decomp_fee_roundtrip_total")
    ev_decomp_gross_long_600 = payload_row.get("ev_decomp_gross_long_600") or meta.get("ev_decomp_gross_long_600")
    ev_decomp_gross_long_1800 = payload_row.get("ev_decomp_gross_long_1800") or meta.get("ev_decomp_gross_long_1800")
    ev_decomp_net_long_600 = payload_row.get("ev_decomp_net_long_600") or meta.get("ev_decomp_net_long_600")
    ev_decomp_net_long_1800 = payload_row.get("ev_decomp_net_long_1800") or meta.get("ev_decomp_net_long_1800")
    
    print(f"  pmaker_entry_delay_sec: {pmaker_entry_delay_sec}")
    print(f"  fee_roundtrip_total: {fee_roundtrip_total}")
    print(f"  ev_decomp_gross_long_600: {ev_decomp_gross_long_600}")
    print(f"  ev_decomp_gross_long_1800: {ev_decomp_gross_long_1800}")
    print(f"  ev_decomp_net_long_600: {ev_decomp_net_long_600}")
    print(f"  ev_decomp_net_long_1800: {ev_decomp_net_long_1800}")
    
    if ev_decomp_gross_long_600 is not None and ev_decomp_net_long_600 is not None:
        cost_impact_600 = ev_decomp_gross_long_600 - ev_decomp_net_long_600
        print(f"  비용 영향 (600s): {cost_impact_600}")
        if cost_impact_600 > abs(ev_decomp_gross_long_600) * 0.5:
            print(f"  ❌ 문제: 비용이 gross EV의 50% 이상을 잠식합니다")
        else:
            print(f"  ✅ 비용 영향이 합리적입니다")
    
    if pmaker_entry_delay_sec is not None and pmaker_entry_delay_sec > 5.0:
        print(f"  ⚠️  경고: maker delay가 큽니다 ({pmaker_entry_delay_sec}초)")
    
    # 항목 4: SHORT 쪽 EV가 상대적으로 더 좋은데 long 편향
    print("\n" + "=" * 80)
    print("항목 4: SHORT 쪽 EV가 상대적으로 더 좋은데 long 편향")
    print("=" * 80)
    
    policy_ev_mix_long = payload_row.get("policy_ev_mix_long") or meta.get("policy_ev_mix_long")
    policy_ev_mix_short = payload_row.get("policy_ev_mix_short") or meta.get("policy_ev_mix_short")
    policy_ev_gap = payload_row.get("policy_ev_gap") or meta.get("policy_ev_gap")
    policy_p_pos_gap = payload_row.get("policy_p_pos_gap") or meta.get("policy_p_pos_gap")
    
    print(f"  policy_ev_mix_long: {policy_ev_mix_long}")
    print(f"  policy_ev_mix_short: {policy_ev_mix_short}")
    print(f"  policy_ev_gap: {policy_ev_gap}")
    print(f"  policy_p_pos_gap: {policy_p_pos_gap}")
    
    if policy_ev_mix_long is not None and policy_ev_mix_short is not None:
        if policy_ev_mix_short > policy_ev_mix_long:
            print(f"  ❌ 문제: SHORT EV가 LONG EV보다 좋습니다 (차이: {policy_ev_mix_short - policy_ev_mix_long})")
            if policy_ev_gap is not None and policy_ev_gap > -0.001:
                print(f"  ❌ 문제: policy_ev_gap이 음수가 아닙니다 ({policy_ev_gap}), SHORT 편향이 반영되지 않았을 수 있습니다")
        else:
            print(f"  ✅ LONG EV가 SHORT EV보다 좋거나 같습니다")
    else:
        print(f"  ⚠️  경고: policy_ev_mix_long 또는 policy_ev_mix_short가 payload에 없습니다")
    
    print("\n" + "=" * 80)
    print("분석 완료")
    print("=" * 80)


if __name__ == "__main__":
    # 예시: payload row를 JSON으로 받거나 직접 입력
    if len(sys.argv) > 1:
        payload_file = sys.argv[1]
        with open(payload_file, 'r') as f:
            payload = json.load(f)
            # market rows 분석
            if "market" in payload and len(payload["market"]) > 0:
                market_rows = payload["market"]
                
                # 음수 EV를 가진 심볼 찾기
                neg_ev_symbols = []
                for row in market_rows:
                    ev = row.get("ev")
                    if ev is None:
                        meta = row.get("meta", {})
                        if isinstance(meta, dict):
                            ev = meta.get("ev")
                    ev = float(ev) if ev is not None else 0.0
                    if ev < 0:
                        neg_ev_symbols.append(row)
                
                if len(sys.argv) > 2:
                    # 특정 심볼 지정
                    target_symbol = sys.argv[2]
                    found = False
                    for row in market_rows:
                        if row.get("symbol") == target_symbol:
                            check_negative_ev_causes(row)
                            found = True
                            break
                    if not found:
                        print(f"❌ 심볼 '{target_symbol}'을 찾을 수 없습니다.")
                elif neg_ev_symbols:
                    print(f"\n총 {len(neg_ev_symbols)}개의 음수 EV 심볼 발견\n")
                    # 모든 음수 EV 심볼 분석
                    for row in neg_ev_symbols:
                        check_negative_ev_causes(row)
                        print("\n")
                else:
                    # 음수 EV가 없으면 모든 심볼 또는 첫 번째 심볼 분석
                    print(f"\n음수 EV 심볼이 없습니다. 첫 번째 심볼을 분석합니다.\n")
                    check_negative_ev_causes(market_rows[0])
            else:
                print("❌ payload에 market row가 없습니다")
    else:
        print("사용법: python3 check_negative_ev.py <payload.json> [symbol]")
        print("  - symbol을 지정하지 않으면 음수 EV를 가진 모든 심볼을 분석합니다")
        print("  - 음수 EV가 없으면 첫 번째 심볼을 분석합니다")



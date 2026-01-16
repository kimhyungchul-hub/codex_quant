#!/usr/bin/env python3
"""
로그 파일에서 음수 EV 원인 분석
"""

import re
import sys

def extract_symbol_data(log_file, symbol):
    """로그 파일에서 특정 심볼의 데이터 추출"""
    data = {
        'symbol': symbol,
        'ev_long_h': None,
        'ev_short_h': None,
        'policy_ev_mix_long': None,
        'policy_ev_mix_short': None,
        'mu_alpha': None,
        'mu_alpha_mom': None,
        'mu_alpha_ofi': None,
        'fee_roundtrip_total': None,
        'execution_cost': None,
        'ev_decomp_gross_long_600': None,
        'ev_decomp_net_long_600': None,
        'policy_exit_time_mean_sec': None,
    }
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if symbol not in line:
            continue
        
        # policy_ev_mix 추출
        if "policy_ev_mix_long=" in line and data['policy_ev_mix_long'] is None:
            match = re.search(r'policy_ev_mix_long=(-?\d+\.?\d*)', line)
            if match:
                data['policy_ev_mix_long'] = float(match.group(1))
        
        if "policy_ev_mix_short=" in line and data['policy_ev_mix_short'] is None:
            match = re.search(r'policy_ev_mix_short=(-?\d+\.?\d*)', line)
            if match:
                data['policy_ev_mix_short'] = float(match.group(1))
        
        # ev_long_h, ev_short_h 추출
        if "Before mix: ev_long_h=" in line and data['ev_long_h'] is None:
            match = re.search(r'ev_long_h=\[(.*?)\]', line)
            if match:
                ev_str = match.group(1)
                try:
                    data['ev_long_h'] = [float(x.strip()) for x in ev_str.split(',')]
                except:
                    pass
        
        if "Before mix: ev_short_h=" in line and data['ev_short_h'] is None:
            match = re.search(r'ev_short_h=\[(.*?)\]', line)
            if match:
                ev_str = match.group(1)
                try:
                    data['ev_short_h'] = [float(x.strip()) for x in ev_str.split(',')]
                except:
                    pass
        
        # mu_alpha 추출
        if "EV_VALIDATION_1" in line and "mu_alpha=" in line:
            match = re.search(r'mu_alpha=(-?\d+\.?\d*)', line)
            if match and data['mu_alpha'] is None:
                data['mu_alpha'] = float(match.group(1))
            match = re.search(r'mu_mom=(-?\d+\.?\d*)', line)
            if match and data['mu_alpha_mom'] is None:
                data['mu_alpha_mom'] = float(match.group(1))
            match = re.search(r'mu_ofi=(-?\d+\.?\d*)', line)
            if match and data['mu_alpha_ofi'] is None:
                data['mu_alpha_ofi'] = float(match.group(1))
        
        # execution_cost 추출
        if "execution_cost=" in line:
            match = re.search(r'execution_cost=(-?\d+\.?\d*)', line)
            if match and data['execution_cost'] is None:
                data['execution_cost'] = float(match.group(1))
        
        # fee_roundtrip_total 추출
        if "fee_roundtrip_total" in line:
            match = re.search(r'fee_roundtrip_total[=:](\d+\.?\d*)', line)
            if match and data['fee_roundtrip_total'] is None:
                data['fee_roundtrip_total'] = float(match.group(1))
    
    return data

def analyze_negative_ev(data):
    """추출된 데이터로 음수 EV 원인 분석"""
    symbol = data['symbol']
    
    print("=" * 80)
    print(f"음수 EV 원인 분석: {symbol}")
    print("=" * 80)
    
    policy_ev_mix_long = data.get('policy_ev_mix_long')
    policy_ev_mix_short = data.get('policy_ev_mix_short')
    
    print(f"\npolicy_ev_mix_long: {policy_ev_mix_long}")
    print(f"policy_ev_mix_short: {policy_ev_mix_short}")
    
    if policy_ev_mix_long is None and policy_ev_mix_short is None:
        print("⚠️  policy_ev_mix 데이터를 찾을 수 없습니다.")
        return
    
    if (policy_ev_mix_long is not None and policy_ev_mix_long >= 0) and \
       (policy_ev_mix_short is not None and policy_ev_mix_short >= 0):
        print("✅ policy_ev_mix가 모두 양수입니다.")
        return
    
    print("\n⚠️  policy_ev_mix가 음수입니다. 4개 항목을 점검합니다.\n")
    
    # 항목 1: mu_alpha 자체가 음수/미약
    print("=" * 80)
    print("항목 1: mu_alpha 자체가 음수/미약")
    print("=" * 80)
    
    mu_alpha = data.get('mu_alpha')
    mu_alpha_mom = data.get('mu_alpha_mom')
    mu_alpha_ofi = data.get('mu_alpha_ofi')
    
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
        print(f"  ⚠️  경고: mu_alpha를 찾을 수 없습니다")
    
    # 항목 2: 모든 horizon에서 EV가 음수
    print("\n" + "=" * 80)
    print("항목 2: 모든 horizon에서 EV가 음수인지 확인")
    print("=" * 80)
    
    ev_long_h = data.get('ev_long_h')
    ev_short_h = data.get('ev_short_h')
    
    print(f"  ev_long_h: {ev_long_h}")
    print(f"  ev_short_h: {ev_short_h}")
    
    if ev_long_h:
        all_neg_long = all(ev < 0 for ev in ev_long_h)
        print(f"  Long: 모든 horizon이 음수? {all_neg_long}")
        if all_neg_long:
            print(f"  ❌ 문제: 모든 long horizon에서 EV가 음수입니다")
    
    if ev_short_h:
        all_neg_short = all(ev < 0 for ev in ev_short_h)
        print(f"  Short: 모든 horizon이 음수? {all_neg_short}")
        if all_neg_short:
            print(f"  ❌ 문제: 모든 short horizon에서 EV가 음수입니다")
    
    # 항목 3: execution cost가 gross EV를 잠식
    print("\n" + "=" * 80)
    print("항목 3: execution cost가 gross EV를 잠식")
    print("=" * 80)
    
    execution_cost = data.get('execution_cost')
    fee_roundtrip_total = data.get('fee_roundtrip_total')
    
    print(f"  execution_cost: {execution_cost}")
    print(f"  fee_roundtrip_total: {fee_roundtrip_total}")
    
    if execution_cost is not None:
        print(f"  execution_cost = {execution_cost:.6f} (~{execution_cost*100:.2f}%)")
        if execution_cost > 0.002:  # 0.2% 이상
            print(f"  ⚠️  경고: execution_cost가 큽니다 ({execution_cost:.6f})")
    
    # 항목 4: SHORT 쪽 EV가 상대적으로 더 좋은데 long 편향
    print("\n" + "=" * 80)
    print("항목 4: SHORT 쪽 EV가 상대적으로 더 좋은데 long 편향")
    print("=" * 80)
    
    if policy_ev_mix_long is not None and policy_ev_mix_short is not None:
        ev_gap = policy_ev_mix_long - policy_ev_mix_short
        print(f"  policy_ev_mix_long: {policy_ev_mix_long:.6f}")
        print(f"  policy_ev_mix_short: {policy_ev_mix_short:.6f}")
        print(f"  ev_gap (long - short): {ev_gap:.6f}")
        
        if policy_ev_mix_short > policy_ev_mix_long:
            print(f"  ❌ 문제: SHORT EV가 LONG EV보다 좋습니다 (차이: {policy_ev_mix_short - policy_ev_mix_long:.6f})")
        else:
            print(f"  ✅ LONG EV가 SHORT EV보다 좋거나 같습니다")
    
    print("\n" + "=" * 80)
    print("분석 완료")
    print("=" * 80)

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "state/ev_verification_20251223_110300.log"
    symbol = sys.argv[2] if len(sys.argv) > 2 else "ETH/USDT"
    
    data = extract_symbol_data(log_file, symbol)
    analyze_negative_ev(data)


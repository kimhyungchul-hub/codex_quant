#!/usr/bin/env python3
"""
PMaker가 mu_alpha와 EV에 미치는 영향 분석
"""

import sys
import re
import json

def analyze_pmaker_impact(log_file, symbol):
    """로그에서 PMaker 영향 분석"""
    print("=" * 80)
    print(f"PMaker 영향 분석: {symbol}")
    print("=" * 80)
    
    pmaker_data = {}
    ev_data = {}
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if symbol not in line:
            continue
        
        # PMaker prediction 정보
        if '_pmaker_predict_entry_ctx returned' in line:
            match = re.search(r'_pmaker_predict_entry_ctx returned: ({.*?})', line)
            if match:
                try:
                    data = eval(match.group(1))
                    pmaker_data = data
                except:
                    pass
        
        # ctx에 pmaker 전달 확인
        if 'ctx pmaker_entry=' in line or 'evaluate_entry_metrics: pmaker_entry=' in line:
            match = re.search(r'pmaker_entry=([\d.]+)', line)
            if match:
                pmaker_data['ctx_pmaker_entry'] = float(match.group(1))
            match = re.search(r'delay=([\d.]+)', line)
            if match:
                pmaker_data['ctx_delay'] = float(match.group(1))
        
        # policy_ev_mix 정보
        if 'policy_ev_mix_long=' in line:
            match_long = re.search(r'policy_ev_mix_long=(-?\d+\.?\d*)', line)
            match_short = re.search(r'policy_ev_mix_short=(-?\d+\.?\d*)', line)
            if match_long and match_short:
                ev_data['policy_ev_mix_long'] = float(match_long.group(1))
                ev_data['policy_ev_mix_short'] = float(match_short.group(1))
        
        # delay penalty 정보
        if 'delay_penalty_r=' in line:
            match = re.search(r'delay_penalty_r=([\d.e+-]+)', line)
            if match:
                ev_data['delay_penalty_r'] = float(match.group(1))
        
        # alpha decay 정보 찾기
        if 'ALPHA_DELAY_DECAY' in line or 'decay' in line.lower():
            # 다음 몇 줄 확인
            for j in range(i, min(i+5, len(lines))):
                if 'decay' in lines[j].lower() and symbol in lines[j]:
                    print(f"  Alpha decay 관련: {lines[j].strip()}")
    
    print("\n1. PMaker 예측 정보:")
    print("-" * 80)
    if pmaker_data:
        print(f"  pmaker_entry: {pmaker_data.get('pmaker_entry', 'N/A')}")
        print(f"  pmaker_entry_delay_sec: {pmaker_data.get('pmaker_entry_delay_sec', 'N/A')}")
        print(f"  pmaker_p_buy: {pmaker_data.get('pmaker_p_buy', 'N/A')}")
        print(f"  ctx_pmaker_entry: {pmaker_data.get('ctx_pmaker_entry', 'N/A')}")
        print(f"  ctx_delay: {pmaker_data.get('ctx_delay', 'N/A')}")
    else:
        print("  ⚠️  PMaker 데이터를 찾을 수 없습니다")
    
    print("\n2. EV 정보:")
    print("-" * 80)
    if ev_data:
        print(f"  policy_ev_mix_long: {ev_data.get('policy_ev_mix_long', 'N/A')}")
        print(f"  policy_ev_mix_short: {ev_data.get('policy_ev_mix_short', 'N/A')}")
        if 'delay_penalty_r' in ev_data:
            print(f"  delay_penalty_r: {ev_data['delay_penalty_r']}")
    else:
        print("  ⚠️  EV 데이터를 찾을 수 없습니다")
    
    print("\n3. PMaker 반영 여부 확인:")
    print("-" * 80)
    if pmaker_data.get('ctx_pmaker_entry') is None:
        print("  ❌ 문제: PMaker가 ctx에 전달되지 않았습니다")
        print("  → evaluate_entry_metrics에서 pmaker_entry를 찾을 수 없음")
        print("  → delay penalty가 적용되지 않음")
        print("  → mu_alpha decay가 적용되지 않음")
    else:
        print("  ✅ PMaker가 ctx에 전달되었습니다")
        delay = pmaker_data.get('pmaker_entry_delay_sec', 0)
        if delay > 3.0:
            print(f"  ⚠️  경고: delay가 큽니다 ({delay:.2f}초)")
            print(f"  → mu_alpha decay: exp(-{delay:.2f}/tau) = exp(-{delay:.2f}/30.0) = {exp(-delay/30.0):.4f}")
            print(f"  → mu_alpha가 {100*(1-exp(-delay/30.0)):.1f}% 감소")
    
    print("\n4. mu_alpha 개선 메커니즘:")
    print("-" * 80)
    print("  현재 구현:")
    print("    - PMaker delay는 mu_adj를 decay시킴 (exp(-delay/tau))")
    print("    - delay penalty는 EV에서 직접 차감 (extra_entry_delay_penalty_r)")
    print("    - entry delay는 시뮬레이션 시작을 늦춤 (start_shift_steps)")
    print()
    print("  ⚠️  문제:")
    print("    - mu_alpha 자체를 개선하는 로직이 없음")
    print("    - PMaker는 delay 예측만 하고, mu_alpha 개선에는 사용되지 않음")
    print("    - mu_alpha는 여전히 mu_mom + mu_ofi 기반으로만 계산됨")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "state/ev_verification_20251223_110300.log"
    symbol = sys.argv[2] if len(sys.argv) > 2 else "ETH/USDT:USDT"
    
    import math
    exp = math.exp
    
    analyze_pmaker_impact(log_file, symbol)


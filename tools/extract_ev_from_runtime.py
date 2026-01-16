#!/usr/bin/env python3
"""
실행 중인 런타임에서 EV 값을 추출하고 분석
"""

import sys
import os
import json
import re
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_negative_ev import check_negative_ev_causes

def extract_ev_from_logs():
    """로그 파일에서 EV 값 추출"""
    log_files = []
    
    # 최근 로그 파일 찾기
    import glob
    log_files.extend(glob.glob("state/ev_verification_*.log"))
    log_files.extend(glob.glob("state/engine_run.log"))
    
    ev_values = []
    
    for log_file in sorted(log_files, key=os.path.getmtime, reverse=True)[:3]:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
                # [EV_DEBUG] 태그에서 EV 값 추출
                ev_debug_matches = re.findall(r'\[EV_DEBUG\].*?ev=([-\d.]+)', content)
                for ev_str in ev_debug_matches:
                    try:
                        ev_values.append(float(ev_str))
                    except:
                        pass
                
                # hub.decide returned ev= 패턴
                hub_matches = re.findall(r'hub\.decide returned ev=([-\d.]+)', content)
                for ev_str in hub_matches:
                    try:
                        ev_values.append(float(ev_str))
                    except:
                        pass
        except Exception as e:
            pass
    
    return ev_values

def create_mock_payload_from_ev(ev_value):
    """EV 값으로부터 mock payload 생성 (분석용)"""
    # 실제 payload 구조를 모방하되, EV 값만 실제 값 사용
    return {
        "ev": ev_value,
        "mu_alpha": None,  # 실제 값은 payload에서 가져와야 함
        "mu_alpha_mom": None,
        "mu_alpha_ofi": None,
        "policy_exit_time_mean_sec": None,
        "pmaker_entry_delay_sec": None,
        "fee_roundtrip_total": None,
        "ev_decomp_gross_long_600": None,
        "ev_decomp_net_long_600": None,
        "policy_ev_mix_long": None,
        "policy_ev_mix_short": None,
        "policy_ev_gap": None,
        "policy_p_pos_gap": None,
    }

def main():
    print("=" * 80)
    print("실행 중인 런타임에서 EV 값 확인 및 분석")
    print("=" * 80)
    print("")
    
    # 방법 1: 로그에서 EV 값 추출
    print("[1] 로그 파일에서 EV 값 추출 중...")
    ev_values = extract_ev_from_logs()
    
    if ev_values:
        print(f"✅ {len(ev_values)}개의 EV 값을 찾았습니다.")
        print(f"   최근 EV 값들: {ev_values[-10:]}")
        print("")
        
        # 가장 최근 EV 값 확인
        latest_ev = ev_values[-1] if ev_values else 0.0
        print(f"가장 최근 EV 값: {latest_ev}")
        print("")
        
        # EV가 음수이거나 작은 양수인 경우
        if latest_ev < 0 or (latest_ev > 0 and latest_ev < 0.0005):
            print("⚠️  EV가 음수이거나 작은 양수입니다.")
            print("")
            print("원인 분석을 위해서는 실제 payload가 필요합니다.")
            print("다음 방법 중 하나를 사용하세요:")
            print("")
            print("  1. Dashboard (http://localhost:9999)에서 payload를 JSON으로 저장")
            print("  2. 저장한 파일로 분석:")
            print("     python3 check_negative_ev.py <payload.json>")
            print("")
            print("또는 명령줄 인자로 payload 파일을 제공하세요:")
            print("  python3 extract_ev_from_runtime.py <payload.json>")
        else:
            print(f"✅ EV가 합리적인 범위입니다 ({latest_ev})")
    else:
        print("⚠️  로그에서 EV 값을 찾을 수 없습니다.")
        print("")
        print("다음 방법을 시도하세요:")
        print("  1. Dashboard (http://localhost:9999)에서 payload를 JSON으로 저장")
        print("  2. 저장한 파일로 분석:")
        print("     python3 check_negative_ev.py <payload.json>")
    
    # 명령줄 인자로 payload 파일이 제공된 경우
    if len(sys.argv) > 1:
        payload_file = sys.argv[1]
        print("")
        print("=" * 80)
        print(f"제공된 payload 파일 분석: {payload_file}")
        print("=" * 80)
        print("")
        
        try:
            with open(payload_file, 'r') as f:
                payload = json.load(f)
                
            if "market" in payload and len(payload["market"]) > 0:
                print(f"✅ {len(payload['market'])}개의 market row를 찾았습니다.")
                print("")
                
                for idx, row in enumerate(payload["market"]):
                    symbol = row.get("symbol", f"row_{idx}")
                    ev = row.get("ev", 0.0)
                    
                    print(f"\n[{symbol}] EV 값: {ev}")
                    
                    # EV가 음수이거나 작은 양수인 경우 분석
                    if ev < 0 or (ev > 0 and ev < 0.0005):
                        print(f"⚠️  EV가 음수이거나 작은 양수입니다. 원인 분석을 시작합니다...")
                        print("")
                        check_negative_ev_causes(row)
                        print("")
                    else:
                        print(f"✅ EV가 합리적인 범위입니다 ({ev})")
            else:
                print("❌ payload에 market row가 없습니다")
        except Exception as e:
            print(f"❌ 파일 읽기 실패: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()



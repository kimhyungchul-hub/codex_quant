#!/usr/bin/env python3
"""
실행 중인 프로세스에서 EV 값을 확인하고, 음수이거나 작은 양수면 원인 분석
"""

import sys
import os
import json
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_negative_ev import check_negative_ev_causes

def fetch_payload_from_dashboard():
    """Dashboard에서 payload 가져오기 시도"""
    try:
        import websocket
        import threading
        
        payload_received = {"data": None}
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if "market" in data and len(data.get("market", [])) > 0:
                    payload_received["data"] = data
                    ws.close()
            except Exception:
                pass
        
        def on_error(ws, error):
            pass
        
        def on_close(ws, close_status_code, close_msg):
            pass
        
        def on_open(ws):
            pass
        
        ws = websocket.WebSocketApp(
            "ws://localhost:9999/ws",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # 5초 타임아웃
        ws.run_forever(timeout=5)
        
        return payload_received["data"]
    except ImportError:
        print("⚠️  websocket-client가 설치되지 않았습니다. pip install websocket-client")
        return None
    except Exception as e:
        print(f"⚠️  WebSocket 접속 실패: {e}")
        return None

def create_mock_payload_from_log():
    """로그에서 EV 값 추정하여 mock payload 생성"""
    # 실제로는 로그를 파싱하거나, 사용자가 제공한 payload를 사용
    return None

def check_ev_and_analyze():
    """EV 값을 확인하고 필요시 원인 분석"""
    
    print("=" * 80)
    print("EV 값 확인 및 원인 분석")
    print("=" * 80)
    print("")
    
    # 방법 1: Dashboard WebSocket에서 payload 가져오기
    print("[1] Dashboard WebSocket에서 payload 가져오기 시도...")
    payload = fetch_payload_from_dashboard()
    
    if payload and "market" in payload and len(payload["market"]) > 0:
        print("✅ Payload를 성공적으로 가져왔습니다.")
        print("")
        
        # 각 market row의 EV 확인
        for idx, row in enumerate(payload["market"]):
            symbol = row.get("symbol", f"row_{idx}")
            ev = row.get("ev", 0.0)
            
            print(f"\n[{symbol}] EV 값: {ev}")
            
            # EV가 음수이거나 작은 양수인 경우 분석
            if ev < 0 or (ev > 0 and ev < 0.0005):  # 0.05% 미만은 작은 양수로 간주
                print(f"⚠️  EV가 음수이거나 작은 양수입니다. 원인 분석을 시작합니다...")
                print("")
                check_negative_ev_causes(row)
                print("")
            else:
                print(f"✅ EV가 합리적인 범위입니다 ({ev})")
    else:
        print("⚠️  Dashboard에서 payload를 가져올 수 없습니다.")
        print("")
        print("대안 방법:")
        print("  1. Dashboard (http://localhost:9999)에서 payload를 JSON으로 저장")
        print("  2. 저장한 파일로 분석:")
        print("     python3 check_negative_ev.py <payload.json>")
        print("")
        print("또는 명령줄 인자로 payload 파일을 제공하세요:")
        print("  python3 fetch_and_check_ev.py <payload.json>")
        
        # 명령줄 인자로 payload 파일이 제공된 경우
        if len(sys.argv) > 1:
            payload_file = sys.argv[1]
            try:
                with open(payload_file, 'r') as f:
                    payload = json.load(f)
                    if "market" in payload and len(payload["market"]) > 0:
                        print(f"\n✅ 파일에서 payload를 로드했습니다: {payload_file}")
                        for idx, row in enumerate(payload["market"]):
                            symbol = row.get("symbol", f"row_{idx}")
                            ev = row.get("ev", 0.0)
                            print(f"\n[{symbol}] EV 값: {ev}")
                            if ev < 0 or (ev > 0 and ev < 0.0005):
                                print(f"⚠️  EV가 음수이거나 작은 양수입니다. 원인 분석을 시작합니다...")
                                check_negative_ev_causes(row)
                            else:
                                print(f"✅ EV가 합리적인 범위입니다 ({ev})")
                    else:
                        print("❌ payload에 market row가 없습니다")
            except Exception as e:
                print(f"❌ 파일 읽기 실패: {e}")

if __name__ == "__main__":
    check_ev_and_analyze()



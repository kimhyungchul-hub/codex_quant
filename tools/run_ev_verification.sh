#!/usr/bin/env bash
# EV 값 생성 확인 및 음수 EV 원인 분석을 위한 런타임 실행 스크립트

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Testnet 모드로 실행 (안전)
export BYBIT_TESTNET=1
export EXEC_MODE=maker_then_market
export PMAKER_ENABLE=1
export PMAKER_DEBUG=1
export ENABLE_LIVE_ORDERS=0  # 주문 비활성화 (검증만)
export PYTHONUNBUFFERED=1
export LOG_STDOUT=1

# 짧은 시간만 실행 (검증용) - 90초로 설정하여 충분한 데이터 수집
export PMAKER_PROBE_MAX_RUN_SEC=90

# EV 디버그를 위한 설정
export PMAKER_DELAY_PENALTY_K=1.0
export PMAKER_ENTRY_DELAY_SHIFT=1

echo "=========================================="
echo "EV 값 생성 확인 및 음수 EV 원인 분석"
echo "=========================================="
echo "주의: ENABLE_LIVE_ORDERS=0으로 설정되어 있어 실제 주문은 발생하지 않습니다."
echo ""
echo "확인 사항:"
echo "  1. EV 값이 정상적으로 생성되는지 확인"
echo "  2. 로그에서 [EV_DEBUG] 태그로 EV 계산 과정 추적"
echo "  3. EV가 음수인 경우 check_negative_ev.py로 원인 분석"
echo ""
echo "로그 필터링:"
echo "  grep '[EV_DEBUG]' 로그파일"
echo "  grep 'hub.decide returned ev=' 로그파일"
echo ""

# 로그 파일 설정
LOG_FILE="state/ev_verification_$(date +%Y%m%d_%H%M%S).log"
echo "로그 파일: $LOG_FILE"
echo ""

# 실제 런타임 실행 (로그 파일로도 저장)
exec python3 -u main_engine_mc_v2_final.py 2>&1 | tee "$LOG_FILE"



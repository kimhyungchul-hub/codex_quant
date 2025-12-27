#!/usr/bin/env bash
# 실제 런타임에서 모든 Diff (2-7) 검증을 위한 테스트 스크립트
# 주의: 실제 거래가 발생할 수 있으므로 testnet 모드로 실행 권장

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

# 짧은 시간만 실행 (검증용) - 60초로 설정하여 충분한 데이터 수집
export PMAKER_PROBE_MAX_RUN_SEC=60

# Diff 7 검증을 위한 설정
export PMAKER_DELAY_PENALTY_K=1.0
export PMAKER_ENTRY_DELAY_SHIFT=1  # entry delay shift 활성화

echo "=========================================="
echo "실제 런타임 모든 Diff (2-7) 검증 시작"
echo "=========================================="
echo "주의: ENABLE_LIVE_ORDERS=0으로 설정되어 있어 실제 주문은 발생하지 않습니다."
echo ""
echo "검증 포인트:"
echo "  - Diff 2: Multi-Horizon Policy Mix"
echo "  - Diff 3: Rule-based Dynamic Weights"
echo "  - Diff 4: Exit Reason 통계"
echo "  - Diff 5: Maker → Market 혼합 실행"
echo "  - Diff 6: PMaker Survival 모델"
echo "  - Diff 7: Maker 지연 EV 패널티"
echo ""
echo "로그에서 다음 키워드를 확인하세요:"
echo "  - pmaker_entry_delay_penalty_r"
echo "  - policy_entry_shift_steps"
echo "  - policy_horizons, policy_w_h"
echo "  - policy_exit_reason_counts_per_h"
echo "  - model_loaded, params_total"
echo ""

# 실제 런타임 실행
exec python3 -u main_engine_mc_v2_final.py


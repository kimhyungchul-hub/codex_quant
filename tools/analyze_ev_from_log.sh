#!/usr/bin/env bash
# 로그 파일에서 EV 값 추출 및 분석

LOG_FILE="${1:-state/engine_run.log}"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ 로그 파일을 찾을 수 없습니다: $LOG_FILE"
    echo "사용법: $0 <로그파일>"
    exit 1
fi

echo "=========================================="
echo "EV 값 분석 (로그 파일: $LOG_FILE)"
echo "=========================================="
echo ""

echo "[1] EV 계산 과정 추적"
echo "----------------------------------------"
grep -E "\[EV_DEBUG\]" "$LOG_FILE" | tail -20

echo ""
echo "[2] hub.decide 반환값"
echo "----------------------------------------"
grep -E "hub\.decide returned ev=" "$LOG_FILE" | tail -10

echo ""
echo "[3] 각 engine의 EV 값"
echo "----------------------------------------"
grep -E "EngineHub\.decide: result\[.*\] engine=" "$LOG_FILE" | tail -10

echo ""
echo "[4] evaluate_entry_metrics 결과"
echo "----------------------------------------"
grep -E "evaluate_entry_metrics: symbol=.* ev=" "$LOG_FILE" | tail -10

echo ""
echo "[5] 최종 decision의 EV 값"
echo "----------------------------------------"
grep -E "_compute_decision_task: hub\.decide returned ev=" "$LOG_FILE" | tail -10

echo ""
echo "[6] 음수 EV 발생 횟수"
echo "----------------------------------------"
NEGATIVE_COUNT=$(grep -E "hub\.decide returned ev=" "$LOG_FILE" | grep -E "ev=-[0-9]" | wc -l | tr -d ' ')
echo "음수 EV 발생: $NEGATIVE_COUNT 회"

if [ "$NEGATIVE_COUNT" -gt 0 ]; then
    echo ""
    echo "⚠️  음수 EV가 발생했습니다. check_negative_ev.py를 사용하여 원인을 분석하세요."
    echo "   python3 check_negative_ev.py <payload.json>"
fi



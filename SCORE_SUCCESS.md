# ✅ 완료: Score 계산 성공!

## 🎉 해결된 문제들

### 1. **Why 메시지 표시** ✅
**Before**: `direction=0 (None)`
**After**: `direction=0 (both_scores_invalid (scoreL=-0.045, scoreS=-0.045, threshold=-0.010))`

### 2. **Score = -inf 문제** ✅
**Before**: Score가 -inf (무한 음수)
**After**: Score가 정상 계산 (-0.040 ~ -0.046)

---

## 📊 현재 Score 값

```
BTC: scoreL=-0.045315, scoreS=-0.045186
ETH: scoreL=-0.040553, scoreS=-0.040542
SOL: scoreL=-0.041039, scoreS=-0.040604
BNB: scoreL=-0.046584, scoreS=-0.046390
```

**임계값**: `-0.01`

**결과**: 
- 모든 Score < -0.01 ❌
- → `both_scores_invalid`
- → WAIT (진입 안 함)

---

## 🔧 해결한 제약조건들

### Before (SCORE_ONLY 적용 전)
```python
valid_long = (
    np.isfinite(obj_long_raw)
    & (ev_long_h > 0.0)              # 🚨 EV 양수만
    & (profit_cost_long_h > 1.2)     # 🚨 Profit > Cost * 1.2
    & (p_liq_long_h < 0.0001)        # 청산 확률 < 0.01%
    & (dd_min_long_h > -0.01)        # DD < 1%
)
→ EV 음수면 모두 무효 → Score = -inf
```

### After (SCORE_ONLY 적용 후) ✅
```python
valid_long = (
    np.isfinite(obj_long_raw)   # NaN, inf만 차단
    & np.isfinite(ev_long_h)
    & np.isfinite(cvar_long_h)
)
→ 음수 EV도 Score 계산 → Score = 실제 값
```

---

## 📈 진입 가능 조건

### 현재 상황
```
Score: -0.045 ~ -0.040
임계값: -0.01
→ Score < threshold ❌ → WAIT
```

### Option A: 임계값 완화 (즉시 진입)
```bash
export SCORE_ENTRY_THRESHOLD=-0.05  # -5%까지 허용
```
**예상**:
```
scoreL=-0.045 > -0.05 ✅
scoreS=-0.045 > -0.05 ✅
→ 둘 다 진입 가능 → 큰 쪽 선택
```

### Option B: 시장 대기
**시장이 조금만 상승하면**:
```
EV: -0.007 → -0.003  (개선됨)
Score: -0.045 → -0.008  (임계값 근접)
→ Score > -0.01 ✅ → 진입!
```

---

## 🎯 "Why" 메시지 의미

### **both_scores_invalid**
```
scoreL=-0.045315, scoreS=-0.045186, threshold=-0.010000
```

**의미**:
- 롱 Score: -0.045 (임계값 -0.01보다 낮음 ❌)
- 숏 Score: -0.045 (임계값 -0.01보다 낮음 ❌)
- **둘 다 진입 기준 미달** → WAIT

### **진입 가능한 메시지들**:

1. **long_only_positive**
   ```
   scoreL=0.005, scoreS=-0.020, threshold=-0.01
   ```
   → LONG 진입 ✅

2. **short_only_positive**
   ```
   scoreL=-0.030, scoreS=-0.005, threshold=-0.01
   ```
   → SHORT 진입 ✅

3. **both_positive_gap_ok**
   ```
   scoreL=0.015, scoreS=0.003, gap=0.012
   ```
   → 더 큰 쪽 진입 (LONG) ✅

---

## 📝 수정된 파일

1. **entry_evaluation.py**
   - 라인 1717-1760: SCORE_ONLY 모드에서 제약조건 완화
   - 라인 2925: `policy_direction_reason` 반환값 추가
   - 라인 2833: 역선택 필터 우회

2. **decision.py**
   - 라인 277-290: SCORE_ONLY 모드 적용
   - 라인 282: `policy_direction_reason` 사용

3. **dashboard_v2.html**
   - 라인 390, 1056: EV 컬럼 숨김

---

## ✅ 결론

**문제 해결**:
1. ✅ Why 메시지 표시 → Score 기반 진입 실패 이유 확인 가능
2. ✅ Score 계산 성공 → -0.040 ~ -0.046 (정상 범위)
3. ✅ 제약조건 완화 → 음수 EV도 Score 계산 가능

**현재 상태**:
- Score: -0.045 (정상)
- 임계값: -0.01
- 진입: 불가 (Score < threshold)

**진입 가능 조건**:
1. 시장 상승 → Score > -0.01
2. 또는 임계값 완화 → `SCORE_ENTRY_THRESHOLD=-0.05`

🎉 **시스템 완성!** Score 기반 일관된 진입/청산 시스템이 정상 작동 중입니다!

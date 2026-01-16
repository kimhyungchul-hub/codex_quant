# Score 임계값 현실성 검토

## 현재 임계값

```python
# entry_evaluation.py 라인 1889-1890
score_long_valid = math.isfinite(float(score_long)) and float(score_long) > 0.0
score_short_valid = math.isfinite(float(score_short)) and float(score_short) > 0.0
```

**임계값**: `score > 0.0` (양수만 허용)

---

## 현실성 평가

### ❌ 너무 타이트함!

**문제점**:
1. **현재 EV = -0.68% ~ -0.75%** (모두 음수)
2. **비용 = 0.072% ~ 0.078%** (roundtrip)
3. **Score 공식**:
   ```
   J = EV / (|CVaR| + 2*StdDev) * (1/sqrt(T))
   
   EV = -0.007
   CVaR ≈ -0.009
   StdDev ≈ 0.01
   T = 60s
   
   J ≈ -0.007 / (0.009 + 0.02) * 0.129
     ≈ -0.007 / 0.029 * 0.129
     ≈ -0.031
   
   → Score ≈ -0.031 (음수!)
   ```

4. **score > 0 조건**: 현재 시장에서 **절대 진입 불가**

---

## 권장 임계값

### Option A: 완화 (추천)
```python
# 작은 음수까지 허용
SCORE_THRESHOLD = -0.01  # -1% 까지 허용

score_long_valid = (
    math.isfinite(float(score_long)) and 
    float(score_long) > SCORE_THRESHOLD
)
```

**근거**:
- 현재 Score ≈ -0.03
- -0.01로 완화하면 Score가 -0.01 ~ +0.00 범위에 들면 진입
- 예상: 시장이 약간이라도 움직이면 진입 가능

### Option B: 비용 기반 (보수적)
```python
# 비용보다 나은 Score
cost_threshold = execution_cost * leverage  # ~0.0007
SCORE_THRESHOLD = -cost_threshold

score_long_valid = (
    math.isfinite(float(score_long)) and 
    float(score_long) > -cost_threshold  # ~-0.0007
)
```

### Option C: 적응형 (최적)
```python
# Regime별 동적 임계값
SCORE_THRESHOLDS = {
    "bull": -0.005,    # 상승장: 조금 완화
    "bear": -0.005,    # 하락장: 조금 완화
    "chop": -0.015,    # 횡보장: 많이 완화
    "volatile": -0.020 # 변동성장: 가장 완화
}

threshold = SCORE_THRESHOLDS.get(regime, -0.01)
score_long_valid = float(score_long) > threshold
```

---

## 테스트 시나리오

### 현재 (score > 0.0)
```
score_long = -0.031
score_short = -0.033
→ 둘 다 진입 불가 ❌
```

### 완화 후 (score > -0.01)
```
score_long = -0.005  (일부 호라이즌에서)
score_short = -0.033
→ LONG 진입 가능! ✅
```

### 더 완화 (score > -0.05)
```
score_long = -0.031
score_short = -0.033
→ 둘 다 진입 가능 ✅ (너무 공격적!)
```

---

## 추천 설정

### 환경변수로 조정 가능하게
```bash
# 임시 테스트
export SCORE_ENTRY_THRESHOLD=-0.01

# 보수적
export SCORE_ENTRY_THRESHOLD=-0.005

# 공격적
export SCORE_ENTRY_THRESHOLD=-0.02
```

### 코드 수정
```python
# entry_evaluation.py
try:
    score_threshold = float(os.environ.get("SCORE_ENTRY_THRESHOLD", "0.0"))
except:
    score_threshold = 0.0

score_long_valid = (
    math.isfinite(float(score_long)) and 
    float(score_long) > score_threshold
)
score_short_valid = (
    math.isfinite(float(score_short)) and 
    float(score_short) > score_threshold
)
```

---

## ✅ 결론

**현재 임계값 `> 0.0`은 너무 보수적!**

**추천**:
1. **단기 테스트**: `-0.01` (1% 손실까지 허용)
2. **실전**: `-0.005` (0.5% 손실까지)
3. **장기**: 적응형 임계값 (Regime별)

**다음 단계**:
1. 환경변수 `SCORE_ENTRY_THRESHOLD=-0.01` 설정
2. 재시작
3. 진입 여부 모니터링

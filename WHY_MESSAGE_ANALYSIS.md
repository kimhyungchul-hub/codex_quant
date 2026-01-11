# 🔍 "Why" 메시지 분석

## 현재 메시지

```
Why: SCORE_ONLY: direction=0 (both_scores_invalid (scoreL=-inf, scoreS=-inf, threshold=-0.010000))
```

---

## 📊 메시지 해석

### 구조
```
SCORE_ONLY: direction=0 (상세 이유)
```

**의미**:
- `SCORE_ONLY`: Score 기반 진입 모드
- `direction=0`: 진입하지 않음 (롱도 숏도 아님 → WAIT)
- `상세 이유`: 왜 direction=0인지 설명

---

### 상세 이유 종류

#### 1. **both_scores_invalid** (현재 상황)
```
both_scores_invalid (scoreL=-inf, scoreS=-inf, threshold=-0.01)
```

**의미**:
- 롱 Score: `-inf` (무한 음수)
- 숏 Score: `-inf` (무한 음수)
- 임계값: `-0.01`
- **둘 다 임계값보다 낮음** → WAIT

**원인**: 
- Score 계산 중 에러 발생
- Objective가 제약조건 실패로 `-inf`로 설정됨

#### 2. **long_only_positive**
```
long_only_positive (scoreL=0.005, scoreS=-0.020, threshold=-0.01)
```

**의미**:
- 롱 Score만 임계값 통과
- → **LONG 진입**

#### 3. **short_only_positive**
```
short_only_positive (scoreL=-0.030, scoreS=-0.005, threshold=-0.01)
```

**의미**:
- 숏 Score만 임계값 통과
- → **SHORT 진입**

#### 4. **both_positive_gap_ok**
```
both_positive_gap_ok (scoreL=0.015, scoreS=0.003, gap=0.012)
```

**의미**:
- 둘 다 임계값 통과
- gap이 충분히 큼
- → 더 큰 쪽 진입 (LONG)

#### 5. **both_positive_small_gap**
```
both_positive_small_gap (scoreL=0.002, scoreS=0.001, gap=0.001<0.01)
```

**의미**:
- 둘 다 임계값 통과
- 하지만 gap이 작음
- → 더 큰 쪽 진입 (LONG)

---

## 🚨 현재 문제: Score = -inf

### 원인 분석

**Score = -inf**가 나오는 경우:
```python
# entry_evaluation.py 라인 1720-1721
obj_long_h = np.where(valid_long, obj_long_raw, -np.inf)
obj_short_h = np.where(valid_short, obj_short_raw, -np.inf)
```

**제약조건** (라인 1700-1719):
```python
valid_long = (
    np.isfinite(obj_long_raw)
    & np.isfinite(p_liq_long_h)
    & np.isfinite(dd_min_long_h)
    & np.isfinite(profit_cost_long_h)
    & (ev_long_h > 0.0)  # 🚨 EV가 양수여야 함!
    & (p_liq_long_h < max_p_liq)
    & (profit_cost_long_h > min_profit_cost)
    & ((-dd_min_long_h) <= max_dd_abs)
)
```

**문제**:
- `ev_long_h > 0.0` 조건 때문에
- EV가 음수면 **모든 horizon에서 무효화**
- → `valid_long = False`
- → `obj_long_h = -inf`
- → `score_long = -inf`

---

## ✅ 해결 방법

### Option A: EV > 0 조건 제거 (추천)
```python
# 라인 1705, 1715
# & (ev_long_h > 0.0)  # 제거
# & (ev_short_h > 0.0)  # 제거
```

**이유**:
- Score 기반 진입에서는 음수 EV도 허용
- 제약조건은 극단적인 리스크만 차단
- EV 필터는 Score 임계값으로 처리

### Option B: EV 임계값 완화
```python
ev_threshold = -0.005  # -0.5%까지 허용

valid_long = (
    ...
    & (ev_long_h > ev_threshold)  # 약간 완화
    ...
)
```

### Option C: 제약조건 환경변수화
```python
min_ev_for_valid = float(os.environ.get("MIN_EV_FOR_VALID", "-0.01"))

valid_long = (
    ...
    & (ev_long_h > min_ev_for_valid)
    ...
)
```

---

## 📝 메시지 의미 요약

| 메시지 | Score L | Score S | 결과 |
|--------|---------|---------|------|
| both_scores_invalid | < threshold | < threshold | WAIT |
| long_only_positive | >= threshold | < threshold | LONG |
| short_only_positive | < threshold | >= threshold | SHORT |
| both_positive_gap_ok | >= threshold | >= threshold | 큰 쪽 |
| both_positive_small_gap | >= threshold | >= threshold | 큰 쪽 |

**현재 상황**:
```
scoreL = -inf < -0.01 ❌
scoreS = -inf < -0.01 ❌
→ both_scores_invalid
→ WAIT
```

---

## 🎯 다음 단계

1. **EV > 0 제약조건 제거** (긴급)
   - 음수 EV도 Score 계산 가능하게
   
2. **Score 재계산 확인**
   - 제약조건 완화 후 Score 값 확인
   
3. **메시지 모니터링**
   - `long_only_positive` 또는 `short_only_positive` 나오는지 확인

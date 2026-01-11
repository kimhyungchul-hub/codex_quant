# Score 기반 진입/청산 로직 분석

## 📊 현재 Score 계산 구조

### 1. Objective Function (목표 함수)
**환경변수**: `POLICY_OBJECTIVE_MODE` (기본값: `new_objective`)

```python
# 라인 1657-1671: New Objective (현재 활성화)
# J = (EV_net) / (CVaR + 2.0 * StdDev) * (1 / sqrt(T))

j_new_long = (ev_long_h / denominator_long) * time_w
# where:
#   denominator_long = |CVaR| + (2.0 * StdDev)
#   time_w = 1 / sqrt(horizon_seconds)
```

**다른 모드들**:
- `ratio`: `EV / |CVaR|`
- `ratio_time`: `(EV / |CVaR|) * (1/sqrt(T))`
- `ev_var`: `EV - λ * Variance`

### 2. Score 계산 (Neighbor 보정 포함)

```python
# 라인 1847-1848
score_long = best_obj_long + neighbor_bonus_long - neighbor_penalty_long
score_short = best_obj_short + neighbor_bonus_short - neighbor_penalty_short

# Neighbor Bonus (인접 horizon이 같은 방향 지지)
neighbor_bonus_long = 0.25 * sum(adjacent_obj) if consistent
neighbor_bonus_short = 0.25 * sum(adjacent_obj) if consistent

# Neighbor Penalty (인접 horizon이 반대 방향 지지)
neighbor_penalty_long = 0.25 * sum(opposite_obj) if conflicting
neighbor_penalty_short = 0.25 * sum(opposite_obj) if conflicting
```

### 3. 진입 결정

```python
# 라인 1868-1876
if max(score_long, score_short) <= 0.0:
    direction = 0  # WAIT (둘 다 음수)
elif abs(score_long - score_short) < min_gap:
    direction = 0  # WAIT (차이가 너무 작음)
else:
    direction = 1 if (score_long > score_short) else -1
```

---

## 🎯 Score 기반 단순화 전략

### 현재 문제: Multi-Layer Filters

**Layer 1**: Entry Gate (entry_evaluation.py 라인 2615-2663)
- ❌ `ev > profit_target` (0.06~0.07%)
- ❌ `win >= min_win` (52~53%)
- ❌ `cvar > -cost*3`

**Layer 2**: Funnel Filter (decision.py 라인 272-382)
- ❌ `ev > 0.0`
- ❌ `win >= 48~50%` (비활성화됨)
- ❌ `cvar >= -9~12%`

**Layer 3**: Score Gate (entry_evaluation.py 라인 1868-1876)
- ✅ `max(score) > 0.0`
- ✅ `abs(score_gap) >= min_gap`

---

## ✅ 단순화 방안

### Option A: Score만으로 진입 (가장 단순)

```python
# 진입 조건:
if max(score_long, score_short) > SCORE_THRESHOLD:
    direction = sign(score_long - score_short)
else:
    direction = 0  # WAIT
```

### Option B: Score + 최소 EV (보수적)

```python
# 진입 조건:
if (max(score_long, score_short) > SCORE_THRESHOLD) and (best_ev > 0.0):
    direction = sign(score_long - score_short)
else:
    direction = 0  # WAIT
```

### Option C: Score + Gap (현재 로직 유지, 다른 필터 제거)

```python
# 진입 조건:
if max(score_long, score_short) > 0.0 and abs(ev_gap) >= min_gap:
    direction = sign(ev_gap)
else:
    direction = 0  # WAIT
```

---

## 🔧 권장 설정값

### Score Threshold
```bash
# Option A/B용
export SCORE_ENTRY_THRESHOLD=0.5  # 양수이면서 의미있는 값

# Option C용 (현재)
export POLICY_MIN_SCORE_GAP=0.0001  # 0.01% gap
```

### Neighbor 가중치 조정
```bash
# Bonus: 같은 방향 지지 시 보너스
export POLICY_NEIGHBOR_BONUS_W=0.25
export POLICY_NEIGHBOR_BONUS_CAP=0.0015

# Penalty: 반대 방향 지지 시 패널티
export POLICY_NEIGHBOR_PENALTY_W=0.25
export POLICY_NEIGHBOR_PENALTY_CAP=0.0015

# Veto: 인접이 강하게 반대하면 거부
export POLICY_NEIGHBOR_OPPOSE_VETO_ABS=0.0007  # profit_target
```

---

## 📝 제거할 필터들

### 1. Entry Gate 필터 (entry_evaluation.py)
```python
# 라인 2631-2660: 비활성화
can_enter = True  # 항상 통과
blocked_by = []
```

### 2. Funnel Filter (decision.py)
```python
# 라인 272-382: 비활성화
# 모든 if 조건 제거, action은 direction_policy에서만 결정
```

### 3. Win/CVaR 게이트 (모든 곳)
- `min_win` 체크 제거
- `cvar_floor` 체크 제거
- `ev_floor` 체크 제거 (또는 0으로 설정)

---

## 🚀 구현 계획

### Step 1: 환경변수로 필터 비활성화
```bash
export DISABLE_ENTRY_GATE=1
export DISABLE_FUNNEL_FILTER=1
export SCORE_ONLY_MODE=1
```

### Step 2: params.py 수정
```python
# min_win = 0.0 (비활성화)
# profit_target = 0.0 (비활성화)
DEFAULT_PARAMS = {
    "bull": MCParams(min_win=0.0, profit_target=0.0, ...),
    "bear": MCParams(min_win=0.0, profit_target=0.0, ...),
    "chop": MCParams(min_win=0.0, profit_target=0.0, ...),
}
```

### Step 3: 코드 수정
1. `entry_evaluation.py` 라인 2631: `can_enter = True` 강제
2. `decision.py` 라인 272-382: Funnel Filter 제거
3. Score 기반 진입만 사용

---

## 📊 예상 Score 값 범위

현재 `new_objective` 모드:
```
J = EV / (|CVaR| + 2*StdDev) * (1/sqrt(T))

예시 (60s horizon):
- EV = 0.001 (0.1%)
- CVaR = -0.005 (0.5%)
- StdDev = 0.01 (1%)
- T = 60s

J = 0.001 / (0.005 + 0.02) * (1/sqrt(60))
  = 0.001 / 0.025 * 0.129
  ≈ 0.005

→ Score ≈ 0.005 + neighbor_bonus - neighbor_penalty
→ 합리적 임계값: 0.001~0.01
```

# 🔍 Score 이외의 진입/청산 필터 전체 목록

## 📊 현재 활성화된 필터

### ✅ 진입 필터 (Entry Filters)

#### 1. **Entry Gate** (entry_evaluation.py 라인 2631-2685)
**상태**: `SCORE_ONLY_MODE=1` 시 **비활성화됨** ✅
```python
if use_score_only:
    can_enter = True  # 우회
else:
    # 레거시 게이트:
    if ev <= ev_floor: blocked
    if win < win_floor: blocked
    if cvar <= -cvar_floor_abs: blocked
```

#### 2. **Funnel Filter** (decision.py 라인 272-384)
**상태**: `SCORE_ONLY_MODE=1` 시 **비활성화됨** ✅
```python
if use_score_only:
    # direction은 metrics의 policy_direction만 사용
    action = "LONG" if direction == 1 else "SHORT"
else:
    # 레거시 필터:
    if ev_for_filter <= 0.0: WAIT
    if win_rate < win_floor: WAIT (기본 OFF)
    if cvar1 < cvar_floor: WAIT
```

#### 3. **역선택 방지 필터** (entry_evaluation.py 라인 2830-2835)
**상태**: ⚠️ **항상 활성화됨** (SCORE_ONLY_MODE와 무관)
```python
# Adverse Selection Protection
pmaker_entry = ctx.get("pmaker_entry", 0.0)
pmaker_threshold = 0.3

if pmaker_entry > 0 and pmaker_entry < 0.3:
    logger.info(f"[ADVERSE_SELECTION] Entry blocked")
    can_enter = False  # 🚨 진입 차단!
```

**문제**: 
- PMaker fill rate가 낮으면 진입 차단
- Score와 무관하게 작동
- **현재도 활성화되어 있음!**

#### 4. **BTC 상관관계 필터** (entry_evaluation.py 라인 2823-2828)
**상태**: ✅ Kelly 비중만 축소 (진입 차단 아님)
```python
btc_corr = ctx.get("btc_corr", 0.0)
if btc_corr > 0.7:
    kelly *= 0.8  # Kelly 20% 축소
```

**영향**: 진입은 허용하되, 포지션 크기만 줄임

---

### 🚫 청산 필터 (Exit Filters)

#### 1. **Exit Policy** (별도 파일)
**위치**: `engines/mc/exit_policy.py` (추정)
```python
# TP/SL 도달
# 시간 만료
# Score 반전
# Unrealized DD
```

#### 2. **Min Hold Time**
```python
MIN_HOLD_SEC_DIRECTIONAL = 10초
```
- 진입 후 최소 10초는 유지
- 너무 빠른 청산 방지

---

## 🚨 현재 문제: 역선택 필터

### 문제점
```python
# entry_evaluation.py 라인 2830-2835
if pmaker_entry > 0 and pmaker_entry < 0.3:
    can_enter = False  # 진입 차단!
```

**시나리오**:
1. Score가 양수 → 진입 신호 ✅
2. 하지만 pmaker_entry = 0.2 (30% 미만)
3. → **역선택 필터가 진입 차단** ❌
4. → Score가 무용지물!

### 확인 필요
```bash
# 로그에서 ADVERSE_SELECTION 확인
tail -f engine_early_return.log | grep ADVERSE_SELECTION
```

---

## ✅ 권장 조치

### Option A: 역선택 필터 비활성화
```python
# entry_evaluation.py 라인 2830-2835 주석 처리
# if pmaker_entry > 0 and pmaker_entry < pmaker_threshold:
#     can_enter = False
```

### Option B: SCORE_ONLY_MODE에서 우회
```python
use_score_only = _env_bool("SCORE_ONLY_MODE", True)

if not use_score_only:  # Score 모드에서는 우회
    if pmaker_entry > 0 and pmaker_entry < pmaker_threshold:
        can_enter = False
```

### Option C: 임계값 완화
```python
pmaker_threshold = 0.1  # 0.3 → 0.1로 낮춤
```

---

## 📋 전체 필터 요약

| 필터 | 위치 | 상태 | 영향 |
|------|------|------|------|
| Entry Gate | entry_evaluation.py:2631 | ✅ 비활성화 | 없음 |
| Funnel Filter | decision.py:272 | ✅ 비활성화 | 없음 |
| **역선택 필터** | entry_evaluation.py:2830 | 🚨 **활성화** | **진입 차단!** |
| BTC 상관관계 | entry_evaluation.py:2823 | ✅ 활성화 | Kelly만 축소 |
| Exit Policy | exit_policy.py | ✅ 활성화 | 청산 |
| Min Hold | constants.py | ✅ 활성화 | 최소 10초 |

---

## 🔧 즉시 확인

### 1. 역선택 필터 작동 여부
```bash
tail -200 engine_early_return.log | grep -E "(ADVERSE_SELECTION|pmaker_entry)"
```

### 2. can_enter 값 확인
```bash
curl -s "http://localhost:9999/debug/payload" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for r in d.get('market', [])[:3]:
    print(f\"{r['symbol']}: Status={r['status']}\")
"
```

### 3. PMaker 상태 확인
```bash
curl -s "http://localhost:9999/debug/payload" | python3 -c "
import sys, json
d = json.load(sys.stdin)
r = d.get('market', [{}])[0]
print(f\"PMaker Entry: {r.get('pmaker_entry')}\")
print(f\"PMaker Fill Rate: {r.get('mu_alpha_pmaker_fill_rate')}\")
"
```

---

## ✅ 결론

**Score 이외의 필터**:
1. ✅ Entry Gate: 비활성화됨
2. ✅ Funnel Filter: 비활성화됨
3. 🚨 **역선택 필터**: **아직 활성화!** → 즉시 확인 필요
4. ✅ BTC 상관관계: Kelly만 조정 (문제 없음)
5. ✅ Exit Policy: 청산용 (진입과 무관)

**다음 단계**:
1. 역선택 필터 작동 확인
2. 필요시 SCORE_ONLY_MODE에서 우회

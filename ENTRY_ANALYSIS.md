# 진입 로직 분석 보고서

## 현재 진입 조건 (2단계 필터)

### 1단계: Entry Gate (entry_evaluation.py)
**목적**: 기본적인 통계적 유의성 검증

```python
# 라인 2615-2663
ev_floor = params.profit_target  # 기본값 확인 필요
win_floor = params.min_win       # 기본값 확인 필요  
cvar_floor_abs = cost_floor * 3.0
```

**진입 조건 (모두 만족해야 함)**:
1. ✅ **EV > ev_floor** (기대값이 profit_target보다 커야 함)
   - `params.profit_target` 값 확인 필요
   
2. ✅ **Win Rate >= win_floor** (승률이 min_win보다 높아야 함)
   - `params.min_win` 값 확인 필요
   
3. ✅ **CVaR > -cvar_floor_abs** (손실 한계가 cost_floor * 3.0보다 작아야 함)
   - `cvar_floor_abs = cost_floor * 3.0`
   - `cost_floor = fee_rt * leverage`

4. 🔹 **[선택] EV Cost Gate** (환경변수 `EV_COST_MULT_GATE > 0`인 경우)
   - `EV > k * fee_roundtrip_total * leverage`

5. 🔹 **[선택] Mid-term Gate** (ev_mid, win_mid 존재하는 경우)
   - `ev_mid >= 0.0`
   - `win_mid >= 0.50`

---

### 2단계: Funnel Filter (decision.py 라인 272-382)
**목적**: 리스크 관리 및 시장 상황 적합성 검증

#### 현재 임계값 (Regime별):

```python
# WIN RATE (기본: OFF)
use_winrate_filter = False  # 환경변수로 활성화 가능
win_floors = {
    "BULL": 0.48,    # 48%
    "BEAR": 0.48,    # 48%
    "CHOP": 0.50,    # 50%
    "VOLATILE": 0.50 # 50%
}

# CVaR (Leverage=1 기준)
cvar_floors = {
    "BULL": -0.12,     # -12%
    "BEAR": -0.12,     # -12%
    "CHOP": -0.10,     # -10%
    "VOLATILE": -0.09  # -9%
}

# Event CVaR (VOLATILE 장세만 적용)
event_cvar_floors = {
    "VOLATILE": -1.20  # -120% (R 단위)
}
```

**필터 순서**:
1. ✅ **EV 필터** (최우선): `ev_for_filter > 0.0`
   - ❌ **현재 문제**: 대부분 EV가 음수(-0.77%)여서 여기서 탈락
   
2. 🔹 **Win Rate 필터** (기본 OFF): `win_rate >= win_floor[regime]`
   
3. ✅ **CVaR 필터**: `cvar1 >= cvar_floor[regime]`
   
4. 🔹 **Event CVaR 필터** (VOLATILE만): `event_cvar_r >= -1.20`

5. ✅ **Direction 필터**: `direction != 0`

---

## 🔍 현재 문제점 분석

### 로그에서 확인된 값:
```
EV: -0.007290841321806548 (-0.73%)
Win Rate: 0.0 (0%)
CVaR: -0.009198621006638303 (-0.92%)
```

### 왜 진입이 안 되는가?

#### **1단계 실패**: 
- ❌ `EV = -0.0073 < ev_floor (profit_target)`
- ❌ `Win Rate = 0.0 < win_floor (min_win)`
- ✅ `CVaR = -0.0092` (아마 통과할 것으로 예상)

#### **2단계**: 
- ❌ `ev_for_filter = -0.0073 <= 0.0` → **즉시 WAIT**

---

## ⚠️ 문제의 근본 원인

### 1. **EV가 음수인 이유**
로그를 보면 대부분 코인에서:
```
[EV_VALIDATION_3] costs are 7738~16011% of gross_ev_approx_5min
cost_entry(0.000762~0.000820)가 expected_ev의 50% 이상
```

**→ 실행 비용이 예상 수익보다 훨씬 큼!**

#### 세부 분석:
- DOT: cost 7738% of gross EV
- ADA: cost 9709% of gross EV  
- DOGE: cost 16011% of gross EV

**문제**: 
- `mu_base` 또는 `sigma`가 `None`
- TP 기여도가 너무 작음 (p_tp = 0.0039~0.0078)
- SL 확률이 너무 높음 (p_sl = 0.9922~1.0000)

### 2. **Win Rate가 0.0인 이유**
- `mu_base=None` → 가격 드리프트를 계산할 수 없음
- 결과적으로 모든 path가 SL에 도달

---

## ✅ 권장 조치사항

### A. **즉시 조치 (긴급)**

1. **`mu_base` 및 `sigma` 계산 수정**
   ```python
   # 현재: mu_base=None, sigma=None
   # → orchestrator의 데이터 파이프라인 확인 필요
   ```

2. **임시 완화: ev_floor 낮추기** (테스트용)
   ```bash
   export FUNNEL_EV_FLOOR=-0.005  # -0.5%까지 허용 (임시)
   ```

### B. **중기 조치**

1. **실행 비용 재검토**
   - 현재 비용이 예상 수익의 77배~160배
   - `fee_roundtrip`, `slippage_dyn`, `expected_spread_cost` 점검

2. **TP/SL 비율 조정**
   ```python
   # 현재 (entry_evaluation.py 라인 2701-2702):
   tp_pct = max(params.profit_target, 0.0005)  # 0.05%
   sl_pct = max(tp_pct * 0.8, 0.0008)         # 0.08%
   ```
   → **너무 타이트함!** 단기(0~5분)에는 비현실적

3. **Exit Policy 시간 연장**
   - 현재 `exit_t_mean=10.0s` (min_hold와 동일)
   - → 너무 빨리 청산되어 수익 실현 기회 없음

### C. **장기 조치**

**진입 조건 완전 재설계**:

```python
# 제안: 적응형 임계값
ev_floor_adaptive = {
    "bull": max(cost * 1.5, 0.0008),  # 비용의 1.5배 또는 0.08%
    "chop": max(cost * 2.0, 0.0012),  # 비용의 2배 또는 0.12%
}

win_floor_adaptive = {
    "bull": 0.45,  # 상승장: 승률 낮아도 OK
    "chop": 0.52,  # 횡보장: 승률 중요
}
```

---

## 📈 Score 시스템 제안

현재는 **Binary Gate** (통과/차단)이지만, **Score 기반 우선순위**를 추가하면 더 나음:

```python
# 예시: 진입 점수 계산
entry_score = (
    ev / max(ev_floor, 0.001) * 0.4 +        # EV 기여도 40%
    (win - 0.5) * 2.0 * 0.3 +                # Win 기여도 30%
    -cvar / max(cvar_floor_abs, 0.001) * 0.3 # Risk 기여도 30%
)

# 진입 조건: score > 1.0
can_enter = (entry_score > 1.0) and (ev > 0)
```

---

## 🎯 결론

**현재 상태**: 
- ✅ 로직 자체는 건전함 (보수적임)
- ❌ `mu_base=None`으로 모든 계산이 망가짐
- ❌ 실행 비용이 예상 수익의 77~160배

**우선순위**:
1. 🔴 **HIGH**: `mu_base`, `sigma` 계산 버그 수정
2. 🟠 **MEDIUM**: TP/SL 비율 완화 (0.05%/0.08% → 0.2%/0.3%)  
3. 🟡 **LOW**: 적응형 임계값 도입

# MC Engine Objective Function Update

## 개요

MC 엔진의 목적함수를 개선하여 **생존 필터**, **신호 신뢰도 벌점**, **시간 효율성 가중치**를 적용했습니다.

## 변경 사항

### 1. 표준편차(Std_Dev) 계산 추가

- **위치**: `engines/mc/entry_evaluation.py` (라인 1650-1655)
- **내용**: 분산(variance)에서 표준편차(std_dev)를 계산하여 신호 신뢰도 벌점에 사용

```python
# Calculate standard deviation from variance
std_long_h = np.sqrt(np.maximum(var_long_h, 0.0))
std_short_h = np.sqrt(np.maximum(var_short_h, 0.0))
```

### 2. 새로운 목적함수 구현

- **위치**: `engines/mc/entry_evaluation.py` (라인 1660-1675)
- **내용**: 사용자 요청 수식에 맞는 새로운 목적함수 구현

```python
# New objective function: J = (EV_net) / (CVaR + (2.0 * Std_Dev)) * (1 / sqrt(T))
# This replaces the previous ratio-based approach
denominator_long = abs_cvar_long + (2.0 * std_long_h)
denominator_short = abs_cvar_short + (2.0 * std_short_h)

# Avoid division by zero
denominator_long = np.maximum(denominator_long, 1e-12)
denominator_short = np.maximum(denominator_short, 1e-12)

# Time efficiency weight: 1 / sqrt(T)
time_w = 1.0 / np.sqrt(np.maximum(h_arr_pol.astype(np.float64), 1.0))

# New objective function with signal reliability penalty (Std_Dev)
j_new_long = (ev_long_h / denominator_long) * time_w
j_new_short = (ev_short_h / denominator_short) * time_w
```

### 3. 환경 변수 모드 추가

- **위치**: `engines/mc/entry_evaluation.py` (라인 1720-1725)
- **내용**: 새로운 목적함수를 활성화하는 환경 변수 모드 추가

```python
# Available modes:
# - "ratio_time_var": default mode (EV/CVaR * time_weight - lambda*var)
# - "new_objective": new objective with signal reliability penalty (EV / (CVaR + 2*Std_Dev) * time_weight)
# - "signal_reliability": alias for "new_objective"
```

### 4. 모드 선택 로직

- **위치**: `engines/mc/entry_evaluation.py` (라인 1760-1770)
- **내용**: 환경 변수에 따라 적절한 목적함수 선택

```python
elif obj_mode in ("new_objective", "signal_reliability"):
    # Use the new objective function with signal reliability penalty
    obj_long_raw = j_new_long
    obj_short_raw = j_new_short
```

## 사용 방법

### 기본 모드 (기존 동작)
```bash
# 아무 설정도 하지 않으면 기본 모드 사용
# 또는 명시적으로 설정
export POLICY_OBJECTIVE_MODE=ratio_time_var
```

### 새로운 목적함수 활성화
```bash
# 새로운 목적함수 사용 (신호 신뢰도 벌점 포함)
export POLICY_OBJECTIVE_MODE=new_objective

# 또는 alias 사용
export POLICY_OBJECTIVE_MODE=signal_reliability
```

## 목적함수 수식

### 새로운 목적함수
```
J = (EV_net) / (CVaR + (2.0 * Std_Dev)) * (1 / sqrt(T))
```

### 구성 요소 설명
- **EV_net**: 순기대수익 (비용 차감 후)
- **CVaR**: Conditional Value at Risk (하위 5% 평균 손실)
- **Std_Dev**: 표준편차 (신호 변동성)
- **T**: 시간 (초 단위)
- **2.0**: 신호 신뢰도 벌점 가중치

### 적용되는 필터

#### 1. 생존 필터 (Constraints & Risk Penalty)
- **비용 장벽**: **Expected Profit / Total Cost > 1.2**
- **청산 방지**: **p_liq (청산 확률) < 0.01%**
- **리스크 반영**: 목적함수 분모에 **CVaR(하위 5% 평균 손실)**를 넣어 최악의 상황을 방어

#### 2. 신호 신뢰도 벌점 (Estimation Error Penalty)
- **목적**: **"결과가 들쭉날쭉한(운에 기댄) 수익은 점수를 깎는다."**
- **적용 로직**: **J = EV_net / (CVaR + 2.0 * Std_Dev)**
- **설명**: 표준편차가 클수록 분모가 커져 점수가 감소

#### 3. 시간 효율성 가중치 (Time-Horizon Efficiency)
- **목적**: **"빨리 먹고 나오는 것이 장기 보유보다 우월하다."**
- **적용 로직**: **J_final = J * (1 / sqrt(T))**
- **설명**: 노출 시간이 짧은 선택지를 우선

## 기대 효과

1. **비용 대비 수익이 충분하지 않은 거래 차단**: 비용 장벽 필터
2. **변동성이 큰 신호에 보수적 접근**: 신호 신뢰도 벌점
3. **자본 회전율 증가**: 시간 효율성 가중치
4. **청산 위험 감소**: 청산 방지 필터

## 테스트 방법

1. 기본 모드에서 실행 (기존 동작 확인)
2. 새로운 목적함수 모드로 전환
3. 결과 비교 분석
4. 필요에 따라 파라미터 조정

## 주의사항

- 새로운 목적함수는 보다 보수적인 접근을 합니다
- 신호 변동성이 큰 경우 거래 빈도가 감소할 수 있습니다
- 시간 효율성 가중치로 인해 단기 호라이즌이 선호됩니다
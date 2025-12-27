---
name: EV 계산 경로에서 diff 누락 지점 분석 및 수정
overview: EV_VALIDATION 1~4를 기준으로 diff 1~7의 의도가 실제 EV 계산 경로에 완전히 연결되지 않은 지점을 정확히 짚고, 각 문제점에 대한 수정 방안을 제시합니다.
todos: []
---

# EV 계

산 경로에서 diff 누락 지점 분석 및 수정

## 문제 요약

EV가 음수인 원인은 diff 1~7에서 의도한 변경사항들이 실제 EV 계산 경로에 완전히 연결되지 않았기 때문입니다. 각 검증 지점별로 누락된 부분을 정확히 짚었습니다.

## [EV_VALIDATION_1] mu_alpha가 여전히 주된 드리프트 항으로 사용됨 (우선순위 2)

### 문제점

1. **TP/SL hit 확률 기반 EV가 조건부로만 사용됨**

- 위치: `engines/mc_engine.py` line 2803-2821
- `alpha_hit`가 `None`이면 MC 시뮬레이션 결과(`ev_exit_long`, `ev_exit_short`)를 사용
- `alpha_hit`가 있으면 TP/SL 확률 기반 EV (`tp_pL * tp_r - sl_pL * sl_r - eff_cost`) 사용
- **문제**: `alpha_hit`가 없을 때 MC 시뮬레이션이 `mu_adj`(즉, `mu_alpha`)를 드리프트 항으로 사용

2. **MC 시뮬레이션이 mu_alpha를 직접 사용**

- 위치: `engines/mc_engine.py` line 2684, 2755, 2773, 1664
- `mu_exit = float(mu_adj) / float(SECONDS_PER_YEAR)` (line 2684)
- `mu_adj`는 `mu_alpha`에서 파생됨 (line 2251)
- `compute_exit_policy_metrics`에서 `mu=mu_exit`를 사용하여 MC 시뮬레이션 수행
- `simulate_exit_policy_rollforward`에서 `mu=float(mu_adj)`를 사용하여 price path 생성

### 수정 방안

- **절대 금지**: `alpha_hit`가 없을 때 `mu_alpha` 기반 EV를 사용하지 않음
- **Fallback 옵션 (환경변수로 선택)**:
- (A) `ALPHA_HIT_FALLBACK=deny`: TP/SL 확률이 없으면 거래 거부 (ev=0, direction=0)
- (B) `ALPHA_HIT_FALLBACK=mc_to_hitprob`: MC 시뮬레이션 결과(`compute_exit_policy_metrics`)에서 exit 분포를 분석하여 `p_tp`, `p_sl` 추정
- MC 결과에서 `exit_reason_counts`의 "tp", "sl" 비율 사용
- 또는 `exit_t` 분포와 TP/SL barrier를 비교하여 hit 확률 추정
- 추정된 `p_tp`, `p_sl`로 `ev = p_tp * tp_r - p_sl * sl_r - cost` 계산
- **원칙**: 모든 EV 계산은 반드시 `(p_tp, p_sl)` 기반 TP/SL EV 공식으로만 수행

## [EV_VALIDATION_2] exit 분포가 horizon mix에 반영되지 않음 (우선순위 4)

### 문제점

1. **horizon 가중치 계산에 exit 분포가 포함되지 않음**

- 위치: `engines/mc_engine.py` line 2851-2861
- `w_long = w_prior_arr * contrib_long` (line 2858)
- `contrib_long = np.log1p(np.exp(ev_long_h * beta))` (line 2855)
- **문제**: `exit_t_mean_sec`나 exit 분포가 `w_long` 또는 `ev_long_h` 계산에 직접 사용되지 않음

2. **exit 분포는 검증용으로만 사용됨**

- 위치: `engines/mc_engine.py` line 3116-3125
- `exit_t_mean_avg_best`는 검증 경고만 발생시키고, 실제 EV나 가중치에 반영되지 않음
- min_hold 근처에서 exit이 반복될 때 EV가 자동으로 깎이는 구조가 없음

### 수정 방안

- **exit 분포 기반 페널티 계산**:
- `p_early_exit = p(exit <= min_hold + Δ)` 계산 (예: Δ = 60초)
- `compute_exit_policy_metrics` 결과에서 `exit_t` 배열 사용
- `p_early_exit = (exit_t <= min_hold + Δ).mean()`
- 또는 `exit_reason_counts`에서 min_hold 직후 조기 exit 비율 계산
- `exit_reason_counts`에서 "hold_bad", "score_flip" 비율 추출 (min_hold 직후 발생하는 exit)
- `p_early_bad = (counts.get("hold_bad", 0) + counts.get("score_flip", 0)) / total`
- **페널티 적용**:
- `penalty = 1.0 - k * p_early_exit` (또는 `p_early_bad`)
- `w(h) = w_prior(h) * contrib(EV_h) * penalty` 또는 `ev_long_h *= penalty`
- k는 환경변수로 조절 가능 (예: `EXIT_EARLY_PENALTY_K=0.5`)

## [EV_VALIDATION_3] 비용이 중복 차감되고 maker delay가 제대로 반영되지 않음 (우선순위 3)

### 문제점

1. **eff_cost가 모든 horizon에 동일하게 적용됨**

- 위치: `engines/mc_engine.py` line 2653-2654, 2815-2816
- `eff_cost = fee_roundtrip_total + delay_penalty` (line 2654)
- 각 horizon별로 `evL = tp_pL * tp_r - sl_pL * sl_r - eff_cost` 계산 (line 2815)
- **문제**: `fee_roundtrip_total`은 roundtrip 비용인데, 각 horizon별로 중복 차감될 수 있음

2. **maker delay가 horizon shift로만 반영되고 EV 페널티로는 부분적으로만 반영됨**

- 위치: `engines/mc_engine.py` line 1586-1597, 2640
- `start_shift_steps`는 `compute_exit_policy_metrics` 내부에서만 적용됨
- `delay_penalty`는 `evaluate_entry_metrics`에서 한 번만 계산되고 모든 horizon에 동일하게 적용됨
- **문제**: maker delay가 horizon별로 다를 수 있는데, 하나의 `delay_penalty` 값을 모든 horizon에 사용

3. **entry fee가 모든 horizon에 중복 차감될 가능성**

- 위치: `engines/mc_engine.py` line 2759, 2777
- `compute_exit_policy_metrics`에 `fee_roundtrip=fee_rt_total_f`를 전달
- `fee_rt_total_f`는 이미 roundtrip 비용을 포함하는데, `compute_exit_policy_metrics` 내부에서도 비용을 차감함

### 수정 방안

- **(a) 비용 분리 및 단일 차감**:
- Entry 비용: `cost_entry = fee_entry + spread_entry + slippage_entry` (1회만 차감)
- Exit 비용: `cost_exit = fee_exit + spread_exit + slippage_exit` (1회만 차감)
- `ev = p_tp * tp_r - p_sl * sl_r - cost_entry - cost_exit`
- **주의**: `compute_exit_policy_metrics` 내부에서 이미 exit 비용이 차감되고 있으므로, 중복 차감 방지 필요
- 해결: `compute_exit_policy_metrics`에서는 exit 비용만 차감하고, entry 비용은 `evaluate_entry_metrics`에서만 차감
- **(b) delay를 horizon별로 스케일링**:
- Entry delay: `delay_penalty_entry_h = delay_entry * (delay_entry / h)` 또는 hit-prob 감소로 반영
- 예: `p_tp_adj = p_tp * exp(-k * delay_entry / h)`, `p_sl_adj = p_sl * exp(k * delay_entry / h)`
- Exit delay: `delay_penalty_exit_h = delay_exit * (delay_exit / h)` 또는 hit-prob 감소로 반영
- 또는 delay에 따른 시간 손실을 hit-prob 감소로 변환: `p_tp_adj = p_tp * (1 - delay/h)`

## [EV_VALIDATION_4] 방향 선택 로직이 완전히 교체되지 않음 (우선순위 1)

### 문제점

1. **use_hold_ev=True일 때 hold_ev_mix를 사용**

- 위치: `engines/mc_engine.py` line 3137-3166
- `direction = 1 if hold_ev_mix_long >= hold_ev_mix_short else -1` (line 3138)
- **문제**: `policy_ev_mix` 대신 `hold_ev_mix`를 사용하여 방향 결정
- `hold_ev_mix`는 legacy MC simulation 기반이고, `policy_ev_mix`는 TP/SL 확률 기반

2. **hold_best_ev가 메타에만 저장되고 실제 방향 결정에는 사용되지 않음**

- 위치: `engines/mc_engine.py` line 3653-3654
- `hold_best_ev_long`/`hold_best_ev_short`는 메타에 저장되지만, 실제 방향 결정에는 사용되지 않음

3. **방향 선택 및 게이트가 policy 기반으로 완전 고정되지 않음**

- `use_hold_ev` 플래그에 따라 다른 로직 사용
- 게이트(CVaR, EV threshold 등)도 `use_hold_ev`에 따라 다른 값 사용 가능

### 수정 방안

- **방향 선택은 항상 `policy_ev_mix` 기반으로 고정**:
- `direction = direction_policy` (line 2904에서 계산된 값 사용)
- `direction_policy = 1 if ev_gap > 0 else -1` (line 2904)
- `ev_gap = policy_ev_mix_long - policy_ev_mix_short` (line 2883)
- `use_hold_ev`와 관계없이 `policy_ev_mix`만 사용
- **게이트도 policy 기반으로 고정**:
- CVaR 게이트: `policy_cvar_mix` 사용 (line 2915)
- EV threshold: `policy_ev_mix` 사용
- Win rate: `policy_p_pos_mix` 사용
- **`hold_ev_mix`는 메타/진단용으로만 사용**:
- 방향 결정이나 게이트에 사용하지 않음
- 대시보드/로그에서만 참고용으로 표시

## 수정 우선순위

1. **EV_VALIDATION_4** (우선순위 1): 방향 선택/게이트가 policy 기반으로 완전 고정

- 방향 선택: 항상 `policy_ev_mix` 사용 (`use_hold_ev` 무관)
- 게이트: `policy_cvar_mix`, `policy_ev_mix`, `policy_p_pos_mix` 사용
- `hold_ev_mix`는 메타/진단용으로만 사용

2. **EV_VALIDATION_1** (우선순위 2): alpha_hit 없을 때 mu_alpha로 회귀 금지

- Fallback 옵션: (A) trade deny 또는 (B) MC→hit-prob 변환
- 모든 EV 계산은 반드시 `(p_tp, p_sl)` 기반 TP/SL EV 공식으로만 수행

3. **EV_VALIDATION_3** (우선순위 3): 비용/지연 정합

- 비용: entry 1회 + exit 1회로 분리하여 한 번만 차감
- delay: horizon별로 스케일링 (delay/h 또는 hit-prob 감소로 반영)

4. **EV_VALIDATION_2** (우선순위 4): exit 분포-가중치 정합

- 페널티: `p(exit <= min_hold+Δ)` 또는 `exit_reason_counts`(hold_bad/flip 비율) 기반
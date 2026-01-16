# 매매 로직의 메인 실행 파일

## main.py

`main.py` 파일은 매매 로직의 메인 실행 파일입니다. 이 파일은 `asyncio.run(main())`을 호출하여 비동기적으로 실행되는 메인 함수를 포함하고 있습니다.

## 구조도

전체 로직 구조도는 `ARCHITECTURE.md`를 참고하세요.

## 임포트된 모듈

`main.py` 파일은 다음과 같은 모듈을 임포트하여 모든 기능을 사용할 수 있습니다:

- `engines.evaluation_methods`: `evaluate_entry_metrics`, `decide`, `compute_exit_policy_metrics`
- `engines.running_stats_methods`: `_compress_reason_counts`
- `engines.cvar_methods`: `_cvar_empirical`, `_cvar_bootstrap`, `_cvar_tail_inflate`, `cvar_ensemble`, `_cvar_jnp`
- `engines.simulation_methods`: `_sample_noise`, `_mc_first_passage_tp_sl_jax_core`, `_generate_and_check_paths_jax_core`, `mc_first_passage_tp_sl_jax`, `simulate_paths_price`, `simulate_paths_netpnl`
- `engines.probability_methods`: `_norm_cdf`, `_approx_p_pos_and_ev_hold`, `_prob_max_geq`, `_prob_min_leq`
- `engines.exit_policy_methods`: `simulate_exit_policy_rollforward`, `simulate_exit_policy_rollforward_analytic`, `_weights_for_horizons`, `_execution_mix_from_survival`, `_sigma_per_sec`
- `engines.alpha_features_methods`: `_extract_alpha_hit_features`, `collect_alpha_hit_sample`, `_predict_horizon_hit_probs`
- `engines.mc.signal_features`: `ema`, `_annualize`, `_trend_direction`, `_signal_alpha_mu_annual_parts`, `_cluster_regime`
- `engines.mc.runtime_params`: `_get_params` (regime/ctx → `MCParams`)
- `engines.mc.execution_costs`: `_estimate_slippage`, `_estimate_p_maker`
- `engines.mc.tail_sampling`: `_sample_increments_np`, `_sample_increments_jax`
- `engines.mc.path_simulation`: `simulate_paths_price`, `simulate_paths_netpnl`
- `engines.mc.first_passage`: `mc_first_passage_tp_sl`
- `engines.mc.entry_evaluation`: `evaluate_entry_metrics`
- `engines.mc.policy_weights`: `_weights_for_horizons`, `_compute_ev_based_weights`
- `engines.mc.execution_mix`: `_execution_mix_from_survival`, `_sigma_per_sec`
- `engines.mc.exit_policy`: `compute_exit_policy_metrics`
- `engines.mc.decision`: decide() 정책 로직 (MonteCarloEngine 내부 구현)
- `engines.mc.alpha_hit`: AlphaHitMLP 관련 로직 (옵션)
- `engines.mc.monte_carlo_engine`: `MonteCarloEngine` (구현 본체)
- (호환) `engines.mc.paths`, `engines.mc.evaluation`: 기존 import 유지용 얇은 wrapper
- `dashboard`: `show_dashboard`

## 실행 방법

`main.py` 파일을 실행하려면 다음 명령어를 사용합니다:

### 1) 의존성 설치(권장: venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) 임포트 스모크 테스트(네트워크 없음)

```bash
python -u main.py --imports-only
```

### 3) 대시보드 실행(빠른 확인용)

```bash
SYMBOLS_CSV="BTC/USDT:USDT" \
LOG_STDOUT=1 \
MC_N_PATHS_LIVE=16384 \
MC_N_PATHS_EXIT=128 \
python -u main.py
```

- 기본 동작(안전 모드):
  - mainnet 데이터(`BYBIT_TESTNET=0`)
  - 주문 비활성(`ENABLE_LIVE_ORDERS=0`)
  - paper trading 활성(`PAPER_TRADING=1`)
  - paper에서 엔진 sizing/레버리지 사용(`PAPER_USE_ENGINE_SIZING=1`)

- 대시보드: `http://localhost:9999`
- 디버그 Payload: `http://localhost:9999/debug/payload`

참고:
- `--no-preload`를 주면 시작이 더 빠르지만, 초반에는 `candles=0`일 수 있습니다.

### 4) Paper trading(모의 포지션/PNL)

- 기본값: `ENABLE_LIVE_ORDERS=0`일 때 `PAPER_TRADING=1`로 동작합니다.
- 비활성화: `PAPER_TRADING=0` 또는 `--no-paper`
- 엔진 sizing이 너무 작게 나오면 아래 파라미터로 floor/scale 조절:
  - `PAPER_ENGINE_SIZE_MIN_FRAC` (기본 0.005)
  - `PAPER_ENGINE_SIZE_MULT` (기본 1.0)
  - `PAPER_ENGINE_SIZE_MAX_FRAC` (기본 0.20)

### 5) 런타임 튜닝(UI/API/CLI)

- 대시보드 상단의 `런타임 튜닝` 버튼에서 즉시 조정 가능
- API:
  - `GET /api/runtime`
  - `POST /api/runtime`
- CLI 예시:

```bash
python -u main.py \
  --symbols "BTC/USDT:USDT" \
  --port 9999 \
  --decision-refresh-sec 2 \
  --mc-n-paths-live 16384 \
  --mc-tail-mode student_t \
  --mc-student-t-df 6.0 \
  --paper
```

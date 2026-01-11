## 전체 구조도 (High-level)

```mermaid
flowchart TD
  main[main.py] --> orch[core/orchestrator.LiveOrchestrator]
  main --> dash[core/dashboard_server.DashboardServer]

  orch --> data[core/data_manager.DataManager]
  data -->|fetch_tickers / fetch_ohlcv / fetch_orderbook| orch

  orch --> hub[engines/engine_hub.EngineHub]
  hub --> mc[engines/mc/monte_carlo_engine.MonteCarloEngine]

  mc --> decide[engines/mc/decision.py: decide()]
  decide --> entry_eval[engines/mc/entry_evaluation.py: evaluate_entry_metrics()]
  entry_evaㅌ l --> paths[engines/mc/path_simulation.py: simulate_paths_*]
  entry_eval --> first_passage[engines/mc/first_passage.py: mc_first_passage_tp_sl()]
  entry_eval --> exit_policy[engines/mc/exit_policy.py: compute_exit_policy_metrics()]

  orch -->|broadcast(rows)| dash
```

## 실행 흐름 (main.py 기준)

1) `main.py`가 `LiveOrchestrator`를 만들고(필요 시) `DashboardServer`를 붙입니다.  
2) `DataManager`가 가격/캔들/호가 데이터를 비동기로 갱신합니다.  
3) `LiveOrchestrator._rows_snapshot()`에서 심볼별 `ctx`를 구성하고 `EngineHub.decide(ctx)`를 호출합니다.  
4) `EngineHub`가 등록된 엔진들을 순회하며 `engine.decide(ctx)`를 호출하고, 결과를 sanitize 후 최종 의사결정을 만듭니다.  
5) `DashboardServer.broadcast()`로 UI에 rows/payload를 전송합니다.

## MonteCarloEngine 내부 (engines/mc/)

- `engines/mc/monte_carlo_engine.py`: `MonteCarloEngine` 본체(파라미터/상수/초기화) + mixin 조합
- `engines/mc/decision.py`: `decide(ctx)` (엔트리 평가 호출, 최종 action/ev/meta 구성)
- `engines/mc/entry_evaluation.py`: `evaluate_entry_metrics()` (비용/슬리피지/경로샘플링/정책평가 포함)
- `engines/mc/exit_policy.py`: `compute_exit_policy_metrics()` (롤포워드 기반 정책 EV/p_pos 계산)
- `engines/mc/policy_weights.py`: `_weights_for_horizons()`, `_compute_ev_based_weights()` (지평 가중치)
- `engines/mc/execution_mix.py`: `_execution_mix_from_survival()`, `_sigma_per_sec()` (maker/taker 혼합 + 지연 패널티)
- `engines/mc/signal_features.py`: `ema()`, `_annualize()`, `_trend_direction()`, `_signal_alpha_mu_annual_parts()` (신호/레짐/알파μ)
- `engines/mc/runtime_params.py`: `_get_params()` (regime/ctx → `MCParams`)
- `engines/mc/execution_costs.py`: `_estimate_slippage()`, `_estimate_p_maker()` (체결/비용 근사)
- `engines/mc/tail_sampling.py`: `_sample_increments_{np,jax}()` (분포/부트스트랩 샘플링)
- `engines/mc/path_simulation.py`: `simulate_paths_{price,netpnl}()` (가격/PNL 경로 생성)
- `engines/mc/first_passage.py`: `mc_first_passage_tp_sl()` (TP/SL first-passage event)

## 호환(레거시) 모듈

- `engines/mc/paths.py`, `engines/mc/evaluation.py`는 기존 import를 깨지지 않게 하기 위한 얇은 wrapper(재-export)입니다.

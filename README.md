# 매매 로직의 메인 실행 파일

## main.py

`main.py` 파일은 매매 로직의 메인 실행 파일입니다. 이 파일은 `asyncio.run(main())`을 호출하여 비동기적으로 실행되는 메인 함수를 포함하고 있습니다.

## 임포트된 모듈

`main.py` 파일은 다음과 같은 모듈을 임포트하여 모든 기능을 사용할 수 있습니다:

- `engines.evaluation_methods`: `evaluate_entry_metrics`, `decide`, `compute_exit_policy_metrics`
- `engines.running_stats_methods`: `_compress_reason_counts`
- `engines.cvar_methods`: `_cvar_empirical`, `_cvar_bootstrap`, `_cvar_tail_inflate`, `cvar_ensemble`, `_cvar_jnp`
- `engines.simulation_methods`: `_sample_noise`, `_mc_first_passage_tp_sl_jax_core`, `_generate_and_check_paths_jax_core`, `mc_first_passage_tp_sl_jax`, `simulate_paths_price`, `simulate_paths_netpnl`
- `engines.probability_methods`: `_norm_cdf`, `_approx_p_pos_and_ev_hold`, `_prob_max_geq`, `_prob_min_leq`
- `engines.exit_policy_methods`: `simulate_exit_policy_rollforward`, `simulate_exit_policy_rollforward_analytic`, `_weights_for_horizons`, `_execution_mix_from_survival`, `_sigma_per_sec`
- `engines.alpha_features_methods`: `_extract_alpha_hit_features`, `collect_alpha_hit_sample`, `_predict_horizon_hit_probs`

## 실행 방법

`main.py` 파일을 실행하려면 다음 명령어를 사용합니다:


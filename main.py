from __future__ import annotations
import asyncio
from engines.evaluation_methods import evaluate_entry_metrics, decide, compute_exit_policy_metrics
from engines.running_stats_methods import _compress_reason_counts
from engines.cvar_methods import _cvar_empirical, _cvar_bootstrap, _cvar_tail_inflate, cvar_ensemble, _cvar_jnp
from engines.simulation_methods import _sample_noise, _mc_first_passage_tp_sl_jax_core, _generate_and_check_paths_jax_core, mc_first_passage_tp_sl_jax, simulate_paths_price, simulate_paths_netpnl
from engines.probability_methods import _norm_cdf, _approx_p_pos_and_ev_hold, _prob_max_geq, _prob_min_leq
from engines.exit_policy_methods import simulate_exit_policy_rollforward, simulate_exit_policy_rollforward_analytic, _weights_for_horizons, _execution_mix_from_survival, _sigma_per_sec
from engines.alpha_features_methods import _extract_alpha_hit_features, collect_alpha_hit_sample, _predict_horizon_hit_probs
from dashboard import show_dashboard  # 대시보드 관련 기능 임포트

async def main():
    # 메인 함수 정의
    pass

if __name__ == "__main__":
    asyncio.run(main())

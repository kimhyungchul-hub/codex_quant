import jax
import jax.numpy as jnp
import time
from engines.mc.exit_policy_jax import simulate_exit_policy_multi_symbol_jax

def test_multi_symbol_exit_policy_metal():
    n_symbols = 18
    n_paths = 1024
    h_pts = 3601
    batch_size = 2 # LONG/SHORT
    
    price_paths = jax.random.normal(jax.random.PRNGKey(0), shape=(n_symbols, n_paths, h_pts), dtype=jnp.float32) + 100.0
    s0 = jnp.ones(n_symbols) * 100.0
    mu_ps = jnp.ones(n_symbols) * 0.00001
    sigma_ps = jnp.ones(n_symbols) * 0.0001
    leverage = jnp.ones(n_symbols) * 10.0
    fee_roundtrip = jnp.ones(n_symbols) * 0.0006
    exec_oneway = jnp.ones(n_symbols) * 0.0003
    impact_cost = jnp.ones(n_symbols) * 0.0
    decision_dt_sec = 60
    max_horizon_sec = 3600
    
    side_now_batch = jnp.ones((n_symbols, batch_size), dtype=jnp.int32)
    horizon_sec_batch = jnp.ones((n_symbols, batch_size), dtype=jnp.int32) * 3600
    min_hold_sec_batch = jnp.ones((n_symbols, batch_size), dtype=jnp.int32) * 30
    tp_target_roe_batch = jnp.ones((n_symbols, batch_size), dtype=jnp.float32) * 0.006
    sl_target_roe_batch = jnp.ones((n_symbols, batch_size), dtype=jnp.float32) * 0.005
    dd_stop_roe_batch = jnp.ones((n_symbols, batch_size), dtype=jnp.float32) * -0.01
    
    # Thresholds
    p_pos_floor_enter = jnp.ones(n_symbols) * 0.5
    p_pos_floor_hold = jnp.ones(n_symbols) * 0.5
    p_sl_enter_ceiling = jnp.ones(n_symbols) * 0.05
    p_sl_hold_ceiling = jnp.ones(n_symbols) * 0.05
    p_sl_emergency = jnp.ones(n_symbols) * 0.1
    p_tp_floor_enter = jnp.ones(n_symbols) * 0.5
    p_tp_floor_hold = jnp.ones(n_symbols) * 0.5
    score_margin = jnp.ones(n_symbols) * 0.0
    soft_floor = jnp.ones(n_symbols) * -0.01
    
    print("Starting Multi-Symbol Exit Policy Metal test...")
    try:
        t0 = time.perf_counter()
        res = simulate_exit_policy_multi_symbol_jax(
            price_paths, s0, mu_ps, sigma_ps, leverage, fee_roundtrip, exec_oneway, impact_cost,
            decision_dt_sec, 1, max_horizon_sec,
            side_now_batch, horizon_sec_batch, min_hold_sec_batch, tp_target_roe_batch, sl_target_roe_batch,
            p_pos_floor_enter, p_pos_floor_hold, p_sl_enter_ceiling, p_sl_hold_ceiling, p_sl_emergency,
            p_tp_floor_enter, p_tp_floor_hold, score_margin, soft_floor,
            True, dd_stop_roe_batch, 3, 5, -1.0, 0.05
        )
        jax.block_until_ready(res)
        t1 = time.perf_counter()
        print(f"Multi-Symbol Exit Policy test successful in {t1-t0:.4f}s.")
    except Exception as e:
        print(f"Multi-Symbol Exit Policy test FAILED: {e}")

if __name__ == "__main__":
    test_multi_symbol_exit_policy_metal()

import jax
import jax.numpy as jnp
import time
from engines.mc.exit_policy_jax import simulate_exit_policy_rollforward_batched_vmap_jax

def test_exit_policy_metal():
    n_paths = 1024
    h_pts = 3601
    batch_size = 2 # LONG and SHORT
    
    price_paths = jax.random.normal(jax.random.PRNGKey(0), shape=(n_paths, h_pts), dtype=jnp.float32) + 100.0
    s0 = 100.0
    mu_ps = 0.00001
    sigma_ps = 0.0001
    leverage = 10.0
    fee_roundtrip = 0.0006
    exec_oneway = 0.0003
    impact_cost = 0.0
    decision_dt_sec = 60
    max_horizon_sec = 3600
    
    side_now_batch = jnp.array([1, -1], dtype=jnp.int32)
    horizon_sec_batch = jnp.array([3600, 3600], dtype=jnp.int32)
    min_hold_sec_batch = jnp.array([30, 30], dtype=jnp.int32)
    tp_target_roe_batch = jnp.array([0.006, 0.006], dtype=jnp.float32)
    sl_target_roe_batch = jnp.array([0.005, 0.005], dtype=jnp.float32)
    dd_stop_roe_batch = jnp.array([-0.01, -0.01], dtype=jnp.float32)
    
    print("Starting Exit Policy Metal test...")
    try:
        t0 = time.perf_counter()
        res = simulate_exit_policy_rollforward_batched_vmap_jax(
            price_paths, s0, mu_ps, sigma_ps, leverage, fee_roundtrip, exec_oneway, impact_cost,
            decision_dt_sec, max_horizon_sec,
            side_now_batch, horizon_sec_batch, min_hold_sec_batch, tp_target_roe_batch, sl_target_roe_batch,
            0.5, 0.5, 0.05, 0.05, 0.1, 0.5, 0.5, 0.0, -0.01,
            True, dd_stop_roe_batch, 3, 5
        )
        jax.block_until_ready(res)
        t1 = time.perf_counter()
        print(f"Exit Policy test successful in {t1-t0:.4f}s.")
    except Exception as e:
        print(f"Exit Policy test FAILED: {e}")

if __name__ == "__main__":
    test_exit_policy_metal()

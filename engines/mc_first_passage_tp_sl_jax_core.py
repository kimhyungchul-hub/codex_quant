import jax
import jax.numpy as jnp
from jax import lax

# _sample_noise 함수 정의
def _sample_noise(key, shape, dist="gaussian", df=6.0, boot=None):
    if dist == "gaussian":
        return jax.random.normal(key, shape)
    elif dist == "student_t":
        return jax.random.t(key, df, shape)
    elif dist == "bootstrap":
        return boot
    else:
        raise ValueError(f"Unknown distribution: {dist}")

# _cvar_jnp 함수 정의
def _cvar_jnp(x, alpha):
    sorted_x = jnp.sort(x)
    index = int(jnp.ceil(alpha * len(x)))
    return jnp.mean(sorted_x[index:])

# _JAX_OK 상수 정의
_JAX_OK = True

def _mc_first_passage_tp_sl_jax_core(
    key,
    s0: float,
    tp_pct: float,
    sl_pct: float,
    drift: float,
    vol: float,
    max_steps: int,
    n_paths: int,
    dist: str,
    df: float,
    boot_jnp,
    cvar_alpha: float,
):
    """
    JAX JIT 컴파일된 핵심 연산 (경로 생성 + TP/SL 체크)
    """
    eps = _sample_noise(
        key,
        (n_paths, max_steps),
        dist=dist,
        df=df,
        boot=boot_jnp,
    )

    log_inc = drift + vol * eps
    tp_price = s0 * (1.0 + tp_pct)
    sl_price = s0 * (1.0 - sl_pct)

    alive = jnp.ones(n_paths, dtype=bool)  # type: ignore[attr-defined]
    hit_tp = jnp.zeros(n_paths, dtype=bool)  # type: ignore[attr-defined]
    hit_sl = jnp.zeros(n_paths, dtype=bool)  # type: ignore[attr-defined]
    t_hit = -jnp.ones(n_paths, dtype=jnp.int32)  # type: ignore[attr-defined]
    logp = jnp.zeros(n_paths)  # type: ignore[attr-defined]

    def step(carry, t):
        logp, alive, hit_tp, hit_sl, t_hit = carry
        logp2 = logp + log_inc[:, t]
        price = s0 * jnp.exp(logp2)  # type: ignore[attr-defined]
        tp_now = alive & (price >= tp_price)
        sl_now = alive & (price <= sl_price)
        hit = tp_now | sl_now
        t_hit = jnp.where(hit & (t_hit < 0), t, t_hit)  # type: ignore[attr-defined]
        hit_tp = hit_tp | tp_now
        hit_sl = hit_sl | sl_now
        alive = alive & (~hit)
        return (logp2, alive, hit_tp, hit_sl, t_hit), None

    (logp, alive, hit_tp, hit_sl, t_hit), _ = lax.scan(  # type: ignore[attr-defined]
        step,
        (logp, alive, hit_tp, hit_sl, t_hit),
        jnp.arange(max_steps),  # type: ignore[attr-defined]
    )

    p_tp = jnp.mean(hit_tp)  # type: ignore[attr-defined]
    p_sl = jnp.mean(hit_sl)  # type: ignore[attr-defined]
    p_to = jnp.mean(alive)  # type: ignore[attr-defined]

    r_tp = tp_pct / sl_pct
    r = jnp.where(hit_tp, r_tp, jnp.where(hit_sl, -1.0, 0.0))  # type: ignore[attr-defined]

    ev_r = jnp.mean(r)  # type: ignore[attr-defined]
    cvar_r = _cvar_jnp(r, cvar_alpha)

    t_vals = jnp.where(t_hit >= 0, t_hit.astype(jnp.float32), jnp.nan)  # type: ignore[attr-defined]

    return p_tp, p_sl, p_to, ev_r, cvar_r, t_vals


# JIT 컴파일된 핵심 함수
if _JAX_OK:
    _mc_first_passage_tp_sl_jax_core_jit = jax.jit(_mc_first_passage_tp_sl_jax_core, static_argnames=("dist", "max_steps", "n_paths"))  # type: ignore[attr-defined]
else:
    _mc_first_passage_tp_sl_jax_core_jit = None

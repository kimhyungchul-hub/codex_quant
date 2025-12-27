# engines/mc_engine.py
import math
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from engines.base import BaseEngine
from regime import adjust_mu_sigma
from engines.mc_risk import kelly_with_cvar

# AlphaHitMLP for horizon-specific TP/SL prediction
try:
    from models.alpha_hit_mlp import AlphaHitMLP, AlphaHitConfig
    from trainers.online_alpha_trainer import OnlineAlphaTrainer, AlphaTrainerConfig
    import torch
    _ALPHA_HIT_MLP_OK = True
except Exception:
    _ALPHA_HIT_MLP_OK = False
    AlphaHitMLP = None
    AlphaHitConfig = None
    OnlineAlphaTrainer = None
    AlphaTrainerConfig = None
    torch = None

logger = logging.getLogger(__name__)

from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
MC_VERBOSE_PRINT = str(os.environ.get("MC_VERBOSE_PRINT", "0")).strip().lower() in ("1", "true", "yes")
MC_N_PATHS_LIVE = int(os.environ.get("MC_N_PATHS_LIVE", "10000"))
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0

# NOTE:
# - meta에 멀티호라이즌(예: 60s/180s) 보조 필드를 넣어
#   상위 오케스트레이터에서 mid_boost/필터를 더 세밀하게 적용할 수 있게 한다.

# optional JAX acceleration
try:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
    from jax import random, lax  # type: ignore
    from jax import random as jrand  # type: ignore
except Exception:  # pragma: no cover
    jax = None
    jnp = None
    random = None
    lax = None
    jrand = None

_JAX_OK = jax is not None
def _jax_mc_device():
    """Return a device to run MC kernels on.
    ✅ GPU 우선: 항상 default backend (GPU/Metal) 사용
    CPU로 강제하려면 env JAX_MC_DEVICE=cpu 설정
    """
    if jax is None:
        return None
    try:
        pref = os.environ.get("JAX_MC_DEVICE", "").strip().lower()
        # CPU로 강제하는 경우만 CPU 디바이스 반환
        if pref == "cpu":
            devs = jax.devices("cpu")
            if devs:
                return devs[0]
        # 그 외에는 None 반환하여 default backend (GPU/Metal) 사용
        return None
    except Exception:
        return None


# -----------------------------
# CVaR Estimation (Ensemble)
# -----------------------------
def _cvar_empirical(pnl: np.ndarray, alpha: float = 0.05) -> float:
    x = np.sort(np.asarray(pnl, dtype=np.float64))
    k = max(1, int(alpha * len(x)))
    return float(x[:k].mean())


def _cvar_bootstrap(pnl: np.ndarray, alpha: float = 0.05, n_boot: Optional[int] = None, sample_frac: float = 0.7, seed: int = 42) -> float:
    if n_boot is None:
        n_boot = int(os.environ.get("MC_N_BOOT", "40"))
    
    # Live mode optimization: if n_boot is very low, just use empirical
    if n_boot <= 1:
        return _cvar_empirical(pnl, alpha)

    rng = np.random.default_rng(seed)
    x = np.asarray(pnl, dtype=np.float64)
    n = len(x)
    m = max(30, int(n * sample_frac))
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(x, size=m, replace=True)
        vals.append(_cvar_empirical(samp, alpha))
    return float(np.median(vals))


def _cvar_tail_inflate(pnl: np.ndarray, alpha: float = 0.05, inflate: float = 1.15) -> float:
    x = np.asarray(pnl, dtype=np.float64)
    var = float(np.quantile(x, alpha))
    tail = x[x <= var]
    cvar = float(tail.mean()) if tail.size > 0 else var
    if tail.size < 100:
        cvar *= float(inflate)
    return float(cvar)


def cvar_ensemble(pnl: Sequence[float], alpha: float = 0.05) -> float:
    x = np.asarray(pnl, dtype=np.float64)
    if x.size < 50:
        return float(_cvar_empirical(x, alpha))
    a = _cvar_empirical(x, alpha)
    b = _cvar_bootstrap(x, alpha)
    c = _cvar_tail_inflate(x, alpha)
    return float(0.60 * b + 0.25 * a + 0.15 * c)


# -----------------------------
# Normal-approx helpers (policy roll-forward)
# -----------------------------
def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _approx_p_pos_and_ev_hold(
    mu: float,
    sigma: float,
    tau_sec: float,
    direction: int,
    leverage: float,
    fee_roundtrip: float,
) -> tuple[float, float]:
    """
    GBM 근사에서 tau 동안 hold 했을 때의
      net = direction * return * leverage - fee_roundtrip
    를 산술수익률 r ~ Normal(m, v) 근사로 계산:
      p_pos = P(net > 0)
      ev    = E(net)

    Notes:
    - mu/sigma는 per-second 스케일을 가정 (tau_sec도 seconds).
    - fee_roundtrip은 (enter+exit) 왕복 비용(비율)로 취급.
    """
    tau = max(0.0, float(tau_sec))
    sig = float(sigma)
    lev = float(leverage)
    fee = float(fee_roundtrip)

    if tau <= 0.0 or sig <= 0.0 or lev <= 0.0:
        ev = -fee
        p_pos = 0.0 if ev <= 0.0 else 1.0
        return float(p_pos), float(ev)

    m = float(mu) * tau
    v = (sig * sig) * tau
    s = math.sqrt(max(1e-12, v))

    thr = fee / max(1e-12, lev)
    if int(direction) == 1:
        z = (m - thr) / s
        p_pos = _norm_cdf(z)
    else:
        z = (-thr - m) / s
        p_pos = _norm_cdf(z)

    ev = float(direction) * m * lev - fee
    return float(p_pos), float(ev)


def simulate_exit_policy_rollforward(
    price_paths: np.ndarray,  # (n_paths, n_steps) 1초 단위 가격 경로
    s0: float,
    mu: float,
    sigma: float,
    leverage: float,
    fee_roundtrip: float,
    exec_oneway: float,
    impact_cost: float,
    regime: str,
    decision_dt_sec: int = 5,
    horizon_sec: int = 1800,
    min_hold_sec: int = 180,
    flip_confirm_ticks: int = 3,
    hold_bad_ticks: int = 3,
    # 아래는 엔진의 정책 테이블/상수 값을 호출부에서 주입
    p_pos_floor_enter: float = 0.52,
    p_pos_floor_hold: float = 0.50,
    p_sl_enter_ceiling: float = 0.20,
    p_sl_hold_ceiling: float = 0.25,
    p_sl_emergency: float = 0.38,
    p_tp_floor_enter: float = 0.15,
    p_tp_floor_hold: float = 0.12,
    score_margin: float = 0.0001,
    soft_floor: float = -0.001,
    *,
    side_now: int = 1,  # +1 long, -1 short
    enable_dd_stop: bool = False,
    dd_stop_roe: float = -0.02,
    tp_dyn: float = 0.0,
    sl_dyn: float = 0.0,
    meta_provider=None,
) -> Dict[str, Any]:
    """
    main_engine_mc_v2_final.py::_maybe_exit_position()의 PURE_MC_MODE 청산 규칙을
    price path 위에서 그대로 "롤포워드"로 모사하는 구현.

    구현 포인트:
    - 매 decision_dt_sec마다 남은 시간(tau)을 기준으로 "현재/대안"의 점수(score)와 확률(p_pos/p_sl/p_tp)을 가져와
      flip_ok/hold_ok/psl_emergency/누적 규칙을 적용하여 exit 시점을 결정한다.
    - 기본 구현은 score를 EV(정규근사)로 두고, p_sl/p_tp는 0으로 둬서(=없음) 엔진과 같은 gate-skip 동작을 만든다.
    - 더 정확히 맞추려면 meta_provider를 연결해서 score/p_pos/p_sl/p_tp를 네 실제 엔진 계산으로 넣으면 된다.

    meta_provider (선택):
      meta_provider(t_idx: int, tau_sec: float, s_t: float, side_now: int) -> dict
        - 기대 키(없으면 기본값 사용):
          score_long, score_short,
          p_pos_long, p_pos_short,
          p_sl_long, p_sl_short,
          p_tp_long, p_tp_short
    """
    # --- Vectorized Implementation ---
    price_paths = np.asarray(price_paths, dtype=np.float64)
    if price_paths.ndim != 2:
        raise ValueError("price_paths must be 2D (n_paths, n_steps)")
    n_paths, n_steps = price_paths.shape
    if n_paths <= 0 or n_steps <= 1:
        return {
            "p_pos_exit": 0.0,
            "ev_exit": float(-fee_roundtrip),
            "exit_t_mean_sec": 0.0,
            "exit_t_p50_sec": 0.0,
            "exit_reason_counts": {"no_paths": int(n_paths)},
            "exit_t": np.zeros((max(0, n_paths),), dtype=np.int64),
            "net_out": np.zeros((max(0, n_paths),), dtype=np.float64),
        }

    H = min(int(max(2, horizon_sec)), int(n_steps))
    dt_dec = int(max(1, decision_dt_sec))
    fee_exit_only = 0.5 * float(fee_roundtrip)
    s0_f = float(s0)
    side_now = 1 if int(side_now) >= 0 else -1
    alt_side = -side_now
    switch_cost = float(max(0.0, 2.0 * float(exec_oneway) + float(impact_cost)))
    # local copies for helper closure (mu/sigma are already per-second in caller)
    mu_ps = float(mu)
    sigma_ps = float(max(float(sigma), 1e-12))
    p_tp_valid_long = True
    p_tp_valid_short = True

    def _meta_default(tau: float) -> dict:
        ppos_cur, ev_cur = _approx_p_pos_and_ev_hold(mu_ps, sigma_ps, tau, side_now, leverage, fee_exit_only)
        ppos_alt, ev_alt = _approx_p_pos_and_ev_hold(mu_ps, sigma_ps, tau, alt_side, leverage, fee_exit_only)
        lev = float(max(1e-12, float(leverage)))
        tp = float(tp_dyn)
        sl = float(sl_dyn)

        p_tp_long = float("nan")
        p_tp_short = float("nan")
        if tp > 0.0:
            tp_under = tp / lev
            if bool(p_tp_valid_long):
                p_tp_long = _prob_max_geq(mu_ps, sigma_ps, tau, tp_under)
            if bool(p_tp_valid_short):
                p_tp_short = _prob_min_leq(mu_ps, sigma_ps, tau, -tp_under)

        if sl > 0.0:
            sl_under = sl / lev
            p_sl_long = _prob_min_leq(mu_ps, sigma_ps, tau, -sl_under)
            p_sl_short = _prob_max_geq(mu_ps, sigma_ps, tau, sl_under)
        else:
            p_sl_long = 0.0
            p_sl_short = 0.0

        if side_now == 1:
            return {
                "score_long": float(ev_cur),
                "score_short": float(ev_alt),
                "p_pos_long": float(ppos_cur),
                "p_pos_short": float(ppos_alt),
                "p_sl_long": float(p_sl_long),
                "p_sl_short": float(p_sl_short),
                "p_tp_long": float(p_tp_long) if math.isfinite(float(p_tp_long)) else float("nan"),
                "p_tp_short": float(p_tp_short) if math.isfinite(float(p_tp_short)) else float("nan"),
            }
        return {
            "score_long": float(ev_alt),
            "score_short": float(ev_cur),
            "p_pos_long": float(ppos_alt),
            "p_pos_short": float(ppos_cur),
            "p_sl_long": float(p_sl_long),
            "p_sl_short": float(p_sl_short),
            "p_tp_long": float(p_tp_long) if math.isfinite(float(p_tp_long)) else float("nan"),
            "p_tp_short": float(p_tp_short) if math.isfinite(float(p_tp_short)) else float("nan"),
        }

    # 1. Compute path-independent policy (flip/hold_bad)
    # Since meta_provider is None in the main flow, we can pre-calculate this once.
    policy_exit_t_global = H - 1
    policy_reason_global = "horizon_end"
    
    if meta_provider is None:
        flip_streak = 0
        hold_bad = 0
        grace_sec = min(60.0, 0.25 * max(0.0, float(min_hold_sec)))
        
        for t in range(0, H, dt_dec):
            tau = max(0.0, (H - 1) - t)
            m = _meta_default(tau)
            
            score_long = float(m.get("score_long", 0.0))
            score_short = float(m.get("score_short", 0.0))
            p_pos_L = float(m.get("p_pos_long", 0.0))
            p_pos_S = float(m.get("p_pos_short", 0.0))
            p_sl_L = float(m.get("p_sl_long", 0.0))
            p_sl_S = float(m.get("p_sl_short", 0.0))
            p_tp_L = float(m.get("p_tp_long", 0.0))
            p_tp_S = float(m.get("p_tp_short", 0.0))

            if side_now == 1:
                score_cur, score_alt_raw = score_long, score_short
                p_pos_cur, p_pos_alt = p_pos_L, p_pos_S
                p_sl_cur, p_sl_alt = p_sl_L, p_sl_S
                p_tp_cur, p_tp_alt = p_tp_L, p_tp_S
            else:
                score_cur, score_alt_raw = score_short, score_long
                p_pos_cur, p_pos_alt = p_pos_S, p_pos_L
                p_sl_cur, p_sl_alt = p_sl_S, p_sl_L
                p_tp_cur, p_tp_alt = p_tp_S, p_tp_L

            score_alt = score_alt_raw - switch_cost
            gap_eff = score_cur - score_alt
            alt_value_after_cost = score_alt_raw - exec_oneway
            
            flip_ok = (p_pos_alt >= p_pos_floor_enter) and (alt_value_after_cost > soft_floor)
            if (p_sl_L > 0.0) or (p_sl_S > 0.0):
                flip_ok = flip_ok and (p_sl_alt <= p_sl_enter_ceiling)
            if p_tp_alt > 0.0:
                flip_ok = flip_ok and (p_tp_alt >= p_tp_floor_enter)

            mgn = max(score_margin, 0.0)
            if (gap_eff < -mgn) and flip_ok:
                flip_streak += 1
            else:
                flip_streak = 0

            hold_ok = (gap_eff >= -mgn) and (p_pos_cur >= p_pos_floor_hold)
            if (p_sl_L > 0.0) or (p_sl_S > 0.0):
                hold_ok = hold_ok and (p_sl_cur <= p_sl_hold_ceiling)
            if p_tp_cur > 0.0:
                hold_ok = hold_ok and (p_tp_cur >= p_tp_floor_hold)

            if (p_sl_L > 0.0) or (p_sl_S > 0.0):
                if (p_sl_cur >= p_sl_emergency) and (t >= grace_sec):
                    policy_exit_t_global = t
                    policy_reason_global = "psl_emergency"
                    break

            if (score_cur < -mgn) or (not hold_ok):
                hold_bad += 1
            else:
                hold_bad = 0

            if t >= min_hold_sec:
                if flip_streak >= flip_confirm_ticks:
                    policy_exit_t_global = t
                    policy_reason_global = "score_flip"
                    break
                if hold_bad >= hold_bad_ticks:
                    policy_exit_t_global = t
                    policy_reason_global = "hold_bad"
                    break

    # 2. Path-dependent exits (unrealized_dd) - Vectorized
    exit_t = np.full((n_paths,), policy_exit_t_global, dtype=np.int64)
    exit_reason = np.full((n_paths,), policy_reason_global, dtype=object)
    
    if enable_dd_stop:
        # Compute ROE for all paths and steps
        # We only need to check up to policy_exit_t_global
        check_steps = np.arange(0, policy_exit_t_global + 1, dt_dec)
        prices_subset = price_paths[:, check_steps]
        roe_unreal = side_now * (prices_subset / s0_f - 1.0) * leverage
        
        dd_mask = roe_unreal <= dd_stop_roe
        # Find first occurrence of True along axis 1
        has_dd = np.any(dd_mask, axis=1)
        first_dd_idx = np.argmax(dd_mask, axis=1)
        
        # Update exit_t and reason for paths that hit DD
        exit_t[has_dd] = check_steps[first_dd_idx[has_dd]]
        exit_reason[has_dd] = "unrealized_dd"

    # 3. Final calculations
    px_exit = price_paths[np.arange(n_paths), exit_t]
    gross = (side_now * (px_exit - s0_f) / s0_f) * leverage
    net_out = gross - fee_roundtrip

    # Reason counts
    counts = {}
    for r in exit_reason:
        counts[r] = counts.get(r, 0) + 1

    net_out_arr = net_out
    is_tp = net_out_arr > 0.0
    is_sl = net_out_arr < 0.0
    is_other = net_out_arr == 0.0
    
    n_tp, n_sl, n_other = int(np.sum(is_tp)), int(np.sum(is_sl)), int(np.sum(is_other))
    n_total = len(net_out_arr)
    
    return {
        "p_pos_exit": float(np.mean(is_tp)),
        "ev_exit": float(np.mean(net_out_arr)),
        "exit_t_mean_sec": float(np.mean(exit_t)),
        "exit_t_p50_sec": float(np.median(exit_t)),
        "exit_reason_counts": counts,
        "exit_t": exit_t,
        "net_out": net_out_arr,
        "exit_reason": exit_reason,
        "p_tp": float(n_tp / n_total) if n_total > 0 else 0.0,
        "p_sl": float(n_sl / n_total) if n_total > 0 else 0.0,
        "p_other": float(n_other / n_total) if n_total > 0 else 0.0,
        "tp_r_actual": float(np.mean(net_out_arr[is_tp])) if n_tp > 0 else 0.0,
        "sl_r_actual": float(np.mean(net_out_arr[is_sl])) if n_sl > 0 else 0.0,
        "other_r_actual": float(np.mean(net_out_arr[is_other])) if n_other > 0 else 0.0,
        "n_tp": n_tp, "n_sl": n_sl, "n_other": n_other,
        "prob_sum_check": True,
    }


def simulate_exit_policy_rollforward_analytic(
    *,
    mu_ps: float,
    sigma_ps: float,
    leverage: float,
    fee_roundtrip: float,
    exec_oneway: float,
    impact_cost: float,
    horizon_sec: int,
    decision_dt_sec: int,
    min_hold_sec: int,
    flip_confirm_ticks: int,
    hold_bad_ticks: int,
    p_pos_floor_enter: float,
    p_pos_floor_hold: float,
    p_sl_enter_ceiling: float,
    p_sl_hold_ceiling: float,
    p_sl_emergency: float,
    p_tp_floor_enter: float,
    p_tp_floor_hold: float,
    score_margin: float,
    soft_floor: float,
    side_now: int,
    tp_dyn: float,
    sl_dyn: float,
    p_tp_valid_long: bool = True,
    p_tp_valid_short: bool = True,
) -> Dict[str, Any]:
    """
    simulate_exit_policy_rollforward의 "경량" 버전:
    - path를 만들지 않고, τ(남은 시간)에 대한 정규/반사원리 근사만으로
      동일한 flip/hold_bad/psl_emergency 규칙을 time-loop으로 적용해 exit 시점을 산출.
    - exit 시점의 realized net은 정상근사(hold)로 p_pos/EV를 계산해 반환한다.
    """
    H = int(max(1, int(horizon_sec)))
    dt_dec = int(max(1, int(decision_dt_sec)))
    side_now = 1 if int(side_now) >= 0 else -1
    alt_side = -side_now

    switch_cost = float(max(0.0, 2.0 * float(exec_oneway) + float(impact_cost)))
    fee_exit_only = 0.5 * float(fee_roundtrip)
    mgn = float(max(0.0, score_margin))
    soft_floor_f = float(soft_floor)

    def _meta_default(tau: float) -> dict:
        ppos_cur, ev_cur = _approx_p_pos_and_ev_hold(mu_ps, sigma_ps, tau, side_now, leverage, fee_exit_only)
        ppos_alt, ev_alt = _approx_p_pos_and_ev_hold(mu_ps, sigma_ps, tau, alt_side, leverage, fee_exit_only)
        lev = float(max(1e-12, float(leverage)))
        tp = float(tp_dyn)
        sl = float(sl_dyn)

        p_tp_long = float("nan")
        p_tp_short = float("nan")
        if tp > 0.0:
            tp_under = tp / lev
            if bool(p_tp_valid_long):
                p_tp_long = _prob_max_geq(mu_ps, sigma_ps, tau, tp_under)
            if bool(p_tp_valid_short):
                p_tp_short = _prob_min_leq(mu_ps, sigma_ps, tau, -tp_under)

        if sl > 0.0:
            sl_under = sl / lev
            p_sl_long = _prob_min_leq(mu_ps, sigma_ps, tau, -sl_under)
            p_sl_short = _prob_max_geq(mu_ps, sigma_ps, tau, sl_under)
        else:
            p_sl_long = 0.0
            p_sl_short = 0.0

        if side_now == 1:
            return {
                "score_long": float(ev_cur),
                "score_short": float(ev_alt),
                "p_pos_long": float(ppos_cur),
                "p_pos_short": float(ppos_alt),
                "p_sl_long": float(p_sl_long),
                "p_sl_short": float(p_sl_short),
                "p_tp_long": float(p_tp_long) if math.isfinite(float(p_tp_long)) else float("nan"),
                "p_tp_short": float(p_tp_short) if math.isfinite(float(p_tp_short)) else float("nan"),
            }
        return {
            "score_long": float(ev_alt),
            "score_short": float(ev_cur),
            "p_pos_long": float(ppos_alt),
            "p_pos_short": float(ppos_cur),
            "p_sl_long": float(p_sl_long),
            "p_sl_short": float(p_sl_short),
            "p_tp_long": float(p_tp_long) if math.isfinite(float(p_tp_long)) else float("nan"),
            "p_tp_short": float(p_tp_short) if math.isfinite(float(p_tp_short)) else float("nan"),
        }

    def _prob_max_geq(mu0: float, sig0: float, T: float, a: float) -> float:
        T = float(max(0.0, T))
        sig0 = float(sig0)
        a = float(a)
        if T <= 0.0:
            return 1.0 if a <= 0.0 else 0.0
        if sig0 <= 0.0:
            xT = float(mu0) * T
            return 1.0 if xT >= a else 0.0
        s = sig0 * math.sqrt(T)
        z1 = (a - float(mu0) * T) / max(1e-12, s)
        expo = (2.0 * float(mu0) * a) / max(1e-12, sig0 * sig0)
        expo = float(max(-80.0, min(80.0, expo)))
        term = math.exp(expo) * _norm_cdf((-a - float(mu0) * T) / max(1e-12, s))
        p = (1.0 - _norm_cdf(z1)) + term
        return float(max(0.0, min(1.0, p)))

    def _prob_min_leq(mu0: float, sig0: float, T: float, neg_a: float) -> float:
        a = float(-neg_a)
        if a < 0.0:
            a = -a
        return _prob_max_geq(-float(mu0), float(sig0), float(T), float(a))

    flip_streak = 0
    hold_bad = 0
    exit_t = H
    reason = "time_stop"

    for t in range(0, H + 1, dt_dec):
        age = int(t)
        tau = max(0.0, float(H - t))
        m = _meta_default(float(tau))

        score_long = float(m.get("score_long", 0.0))
        score_short = float(m.get("score_short", 0.0))
        p_pos_L = float(m.get("p_pos_long", 0.0))
        p_pos_S = float(m.get("p_pos_short", 0.0))
        p_sl_L = float(m.get("p_sl_long", 0.0))
        p_sl_S = float(m.get("p_sl_short", 0.0))
        p_tp_L = float(m.get("p_tp_long", float("nan")))
        p_tp_S = float(m.get("p_tp_short", float("nan")))

        if side_now == 1:
            score_cur = float(score_long)
            score_alt_raw = float(score_short)
            p_pos_cur, p_pos_alt = float(p_pos_L), float(p_pos_S)
            p_sl_cur, p_sl_alt = float(p_sl_L), float(p_sl_S)
            p_tp_cur, p_tp_alt = float(p_tp_L), float(p_tp_S)
        else:
            score_cur = float(score_short)
            score_alt_raw = float(score_long)
            p_pos_cur, p_pos_alt = float(p_pos_S), float(p_pos_L)
            p_sl_cur, p_sl_alt = float(p_sl_S), float(p_sl_L)
            p_tp_cur, p_tp_alt = float(p_tp_S), float(p_tp_L)

        score_alt = float(score_alt_raw) - float(switch_cost)
        gap_eff = float(score_cur - score_alt)

        alt_value_after_cost = float(score_alt_raw) - float(exec_oneway)
        flip_ok = bool(
            (p_pos_alt >= float(p_pos_floor_enter))
            and ((alt_value_after_cost > 0.0) or (alt_value_after_cost > float(soft_floor_f)))
        )
        if (float(p_sl_L) > 0.0) or (float(p_sl_S) > 0.0):
            flip_ok = bool(flip_ok and (float(p_sl_alt) <= float(p_sl_enter_ceiling)))
        if math.isfinite(float(p_tp_alt)) and float(p_tp_alt) > 0.0:
            flip_ok = bool(flip_ok and (float(p_tp_alt) >= float(p_tp_floor_enter)))

        pref_side = None
        if (gap_eff < -mgn) and flip_ok:
            pref_side = alt_side

        if pref_side is not None and pref_side != side_now:
            flip_streak += 1
        else:
            flip_streak = 0

        hold_value_ok = bool(gap_eff >= -mgn)
        hold_ok = bool(hold_value_ok and (p_pos_cur >= float(p_pos_floor_hold)))
        if (float(p_sl_L) > 0.0) or (float(p_sl_S) > 0.0):
            hold_ok = bool(hold_ok and (float(p_sl_cur) <= float(p_sl_hold_ceiling)))
        if math.isfinite(float(p_tp_cur)) and float(p_tp_cur) > 0.0:
            hold_ok = bool(hold_ok and (float(p_tp_cur) >= float(p_tp_floor_hold)))

        grace_sec = min(60.0, 0.25 * max(0.0, float(min_hold_sec)))
        if (float(p_sl_L) > 0.0) or (float(p_sl_S) > 0.0):
            if (float(p_sl_cur) >= float(p_sl_emergency)) and (float(age) >= float(grace_sec)) and age >= int(min_hold_sec):
                exit_t = age
                reason = "psl_emergency"
                break

        if (score_cur < -mgn) or (not hold_ok):
            hold_bad += 1
        else:
            hold_bad = 0

        if age >= int(min_hold_sec) and flip_streak >= int(flip_confirm_ticks):
            exit_t = age
            reason = "score_flip"
            break
        if age >= int(min_hold_sec) and hold_bad >= int(hold_bad_ticks):
            exit_t = age
            reason = "hold_bad"
            break

    p_pos_rt, ev_rt = _approx_p_pos_and_ev_hold(mu_ps, sigma_ps, float(exit_t), side_now, leverage, float(fee_roundtrip))
    p_pos_ow, ev_ow = _approx_p_pos_and_ev_hold(mu_ps, sigma_ps, float(exit_t), side_now, leverage, float(exec_oneway))
    return {
        "exit_t_sec": float(exit_t),
        "exit_reason": str(reason),
        "p_pos_exit_roundtrip": float(p_pos_rt),
        "ev_exit_roundtrip": float(ev_rt),
        "p_pos_exit_oneway": float(p_pos_ow),
        "ev_exit_oneway": float(ev_ow),
    }


# -----------------------------
# JAX helpers (optional)
# -----------------------------


def _cvar_jnp(x: "jnp.ndarray", alpha: float) -> "jnp.ndarray":  # type: ignore[name-defined]
    xs = jnp.sort(x)  # type: ignore[attr-defined]
    k = jnp.asarray(xs.shape[0] * alpha, dtype=jnp.int32)  # type: ignore[attr-defined]
    k = jnp.maximum(1, k)  # type: ignore[attr-defined]
    return jnp.mean(xs[:k])  # type: ignore[attr-defined]


def _sample_noise(key, shape, dist="gaussian", df=6.0, boot=None):
    # ✅ 해결: boot를 무조건 jnp.asarray로 정규화하여 device/host 섞임 방지
    if dist == "bootstrap" and boot is not None:
        boot_jnp = jnp.asarray(boot, dtype=jnp.float32)  # type: ignore[attr-defined]
        boot_size = int(boot_jnp.shape[0])  # Python int로 변환
        if boot_size > 16:
            idx = random.randint(key, shape, 0, boot_size)  # type: ignore[attr-defined]
            return boot_jnp[idx]
    if dist == "student_t":
        k1, k2 = random.split(key)  # type: ignore[attr-defined]
        z = random.normal(k1, shape)  # type: ignore[attr-defined]
        u = 2.0 * random.gamma(k2, df / 2.0, shape)  # type: ignore[attr-defined]
        return z / jnp.sqrt(u / df)  # type: ignore[attr-defined]
    return random.normal(key, shape)  # type: ignore[attr-defined]


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


# 경로 생성과 TP/SL 체크를 통합한 JIT 함수
def _generate_and_check_paths_jax_core(
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
    경로 생성과 TP/SL 체크를 하나의 JIT 함수로 통합하여 데이터 이동 오버헤드 최소화
    """
    # 노이즈 샘플링
    eps = _sample_noise(
        key,
        (n_paths, max_steps),
        dist=dist,
        df=df,
        boot=boot_jnp,
    )

    # GBM 경로 생성과 동시에 TP/SL 체크
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


# JIT 컴파일된 통합 함수
if _JAX_OK:
    _generate_and_check_paths_jax_core_jit = jax.jit(_generate_and_check_paths_jax_core, static_argnames=("dist", "max_steps", "n_paths"))  # type: ignore[attr-defined]
else:
    _generate_and_check_paths_jax_core_jit = None


def mc_first_passage_tp_sl_jax(
    s0: float,
    tp_pct: float,
    sl_pct: float,
    mu: float,
    sigma: float,
    dt: float,
    max_steps: int,
    n_paths: int,
    seed: int,
    dist: str = "gaussian",
    df: float = 6.0,
    boot_rets: np.ndarray | None = None,
    cvar_alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    JAX 가속 first-passage MC (TP/SL/timeout) - JIT 최적화 버전
    """
    if jax is None or tp_pct <= 0 or sl_pct <= 0 or sigma <= 0:
        return {}

    drift = (mu - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)

    key = random.PRNGKey(seed & 0xFFFFFFFF)  # type: ignore[attr-defined]
    # ✅ 해결: boot_rets를 미리 jnp.asarray로 정규화하여 _sample_noise에 전달
    boot_jnp = None if boot_rets is None else jnp.asarray(boot_rets, dtype=jnp.float32)  # type: ignore[attr-defined]
    
    # JIT 컴파일된 핵심 함수 호출
    if _mc_first_passage_tp_sl_jax_core_jit is not None:
        p_tp, p_sl, p_to, ev_r, cvar_r, t_vals = _mc_first_passage_tp_sl_jax_core_jit(
            key, s0, tp_pct, sl_pct, drift, vol, max_steps, n_paths, dist, df, boot_jnp, cvar_alpha
        )
    else:
        p_tp, p_sl, p_to, ev_r, cvar_r, t_vals = _mc_first_passage_tp_sl_jax_core(
            key, s0, tp_pct, sl_pct, drift, vol, max_steps, n_paths, dist, df, boot_jnp, cvar_alpha
        )

    # 결과를 Python 값으로 변환
    p_tp_f = float(p_tp)
    p_sl_f = float(p_sl)
    p_to_f = float(p_to)
    prob_sum = p_tp_f + p_sl_f + p_to_f
    if abs(prob_sum - 1.0) > 1e-3 and prob_sum > 0:
        p_tp_f /= prob_sum
        p_sl_f /= prob_sum
        p_to_f = max(0.0, 1.0 - p_tp_f - p_sl_f)

    # ✅ 해결: JAX 결과를 host 값으로 변환 (tracer -> concrete)
    # ✅ 해결: JAX scalar/tracer를 host 값으로 변환 (ConcretizationTypeError 방지)
    t_median_raw = jnp.nanmedian(t_vals)  # type: ignore[attr-defined]
    t_mean_raw = jnp.nanmean(t_vals)  # type: ignore[attr-defined]
    t_median = float(np.asarray(t_median_raw).item())
    t_mean = float(np.asarray(t_mean_raw).item())
    if not math.isfinite(t_median):
        t_median = None
    if not math.isfinite(t_mean):
        t_mean = None

    # ✅ 해결: JAX scalar/tracer를 host 값으로 변환
    ev_r_host = float(np.asarray(ev_r).item())
    cvar_r_host = float(np.asarray(cvar_r).item())

    return {
        "event_p_tp": p_tp_f,
        "event_p_sl": p_sl_f,
        "event_p_timeout": p_to_f,
        "event_ev_r": ev_r_host,
        "event_cvar_r": cvar_r_host,
        "event_t_median": t_median,
        "event_t_mean": t_mean,
    }


# -----------------------------
# Helpers
# -----------------------------
def ema(values: Sequence[float], period: int) -> Optional[float]:
    if values is None or len(values) < 2:
        return None
    v = np.asarray(values, dtype=np.float64)
    period = max(2, int(period))
    alpha = 2.0 / (period + 1.0)
    e = float(v[0])
    for x in v[1:]:
        e = alpha * float(x) + (1.0 - alpha) * e
    return float(e)


@dataclass
class MCParams:
    min_win: float
    profit_target: float
    ofi_weight: float
    max_kelly: float
    cvar_alpha: float
    cvar_scale: float
    n_paths: int


DEFAULT_PARAMS = {
    "bull": MCParams(min_win=0.55, profit_target=0.0012, ofi_weight=0.0015, max_kelly=0.25, cvar_alpha=0.05, cvar_scale=6.0, n_paths=16000),
    "bear": MCParams(min_win=0.57, profit_target=0.0012, ofi_weight=0.0018, max_kelly=0.20, cvar_alpha=0.05, cvar_scale=7.0, n_paths=16000),
    "chop": MCParams(min_win=0.60, profit_target=0.0010, ofi_weight=0.0022, max_kelly=0.10, cvar_alpha=0.05, cvar_scale=8.0, n_paths=16000),
}


class MonteCarloEngine(BaseEngine):
    """
    - ctx에서 regime_params를 받으면 그것을 우선 사용
    - 아니면 DEFAULT_PARAMS[regime]
    - 시뮬레이션은 numpy로 결정론 seed 사용(튜닝 안정)
    """
    name = "mc_barrier"
    weight = 1.0

    POLICY_P_POS_ENTER_BY_REGIME = {"bull": 0.52, "bear": 0.52, "chop": 0.52, "volatile": 0.52}
    POLICY_P_POS_HOLD_BY_REGIME = {"bull": 0.50, "bear": 0.50, "chop": 0.50, "volatile": 0.50}
    POLICY_P_SL_ENTER_MAX_BY_REGIME = {"bull": 0.20, "bear": 0.20, "chop": 0.20, "volatile": 0.20}
    POLICY_P_SL_HOLD_MAX_BY_REGIME = {"bull": 0.25, "bear": 0.25, "chop": 0.25, "volatile": 0.25}
    POLICY_P_TP_ENTER_MIN_BY_REGIME = {"bull": 0.15, "bear": 0.15, "chop": 0.15, "volatile": 0.15}
    POLICY_P_TP_HOLD_MIN_BY_REGIME = {"bull": 0.12, "bear": 0.12, "chop": 0.12, "volatile": 0.12}
    POLICY_P_SL_EMERGENCY = 0.38
    POLICY_HOLD_BAD_TICKS = 3
    FLIP_CONFIRM_TICKS = 3
    MIN_HOLD_SEC_DIRECTIONAL = 180
    POLICY_DECISION_DT_SEC = 5
    POLICY_HORIZON_SEC = 1800
    POLICY_MULTI_HORIZONS_SEC = (60, 180, 300, 600, 900, 1800)
    SCORE_MARGIN_DEFAULT = 0.0001
    POLICY_VALUE_SOFT_FLOOR_AFTER_COST = -0.001
    N_PATHS_EXIT_POLICY = int(os.environ.get("MC_N_PATHS_EXIT", "2048"))
    # [D] TP/SL EV configuration
    TP_R_BY_H = {60: 0.0012, 180: 0.0018, 300: 0.0020, 600: 0.0025, 900: 0.0030, 1800: 0.0035}  # TP return fraction per horizon
    SL_R_FIXED = 0.0020  # Fixed SL return fraction (fallback)
    TRAIL_ATR_MULT = 2.0  # ATR multiplier for trailing SL
    PMAKER_DELAY_PENALTY_K = 1.0  # Delay penalty scaling factor
    POLICY_W_EV_BETA = 200.0  # EV shaping factor for weight computation
    POLICY_MIN_EV_GAP = 0.0002  # Minimum EV gap for direction selection

    def __init__(self):
        # horizons(초) - GPU 실행 가정, 더 촘촘하게 확장
        self.horizons = (15, 30, 60, 120, 180, 300, 600, 900, 1200, 1800)
        self.dt = 1.0 / 31536000.0  # seconds/year
        # Bybit taker round-trip(0.06% * 2) 기준, 레버리지와 무관한 고정 비용
        self.fee_roundtrip_base = 0.0012
        # Bybit maker round-trip(0.01% * 2) 기준
        self.fee_roundtrip_maker_base = 0.0002
        self.slippage_perc = 0.0003
        # tail mode defaults
        self.default_tail_mode = "student_t"  # "gaussian" | "student_t" | "bootstrap"
        self.default_student_t_df = 6.0
        self._use_jax = True
        self._tail_mode = self.default_tail_mode
        self._student_t_df = self.default_student_t_df
        self._bootstrap_returns = None
        self._ofi_hist: Dict[Tuple[str, str], List[float]] = {}
        self._gate_log_count = 0  # 진입게이트 로그 카운터 (최대 3개)
        
        # [B] AlphaHitMLP for horizon-specific TP/SL prediction (TP/SL hit 확률 기반 EV)
        self.alpha_hit_enabled = _ALPHA_HIT_MLP_OK and str(os.environ.get("ALPHA_HIT_ENABLE", "1")).strip().lower() in ("1", "true", "yes")
        self.alpha_hit_beta = float(os.environ.get("ALPHA_HIT_BETA", "1.0"))  # EV scaling factor
        self.alpha_hit_model_path = str(os.environ.get("ALPHA_HIT_MODEL_PATH", "state/alpha_hit_mlp.pt"))
        self.alpha_hit_trainer = None
        
        if self.alpha_hit_enabled and _ALPHA_HIT_MLP_OK and OnlineAlphaTrainer is not None:
            try:
                policy_horizons = list(getattr(self, "POLICY_MULTI_HORIZONS_SEC", (60, 180, 300, 600, 900, 1800)))
                n_features = 20  # Match _extract_alpha_hit_features dimension
                
                trainer_cfg = AlphaTrainerConfig(
                    horizons_sec=policy_horizons,
                    n_features=n_features,
                    device=str(os.environ.get("ALPHA_HIT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")),
                    lr=float(os.environ.get("ALPHA_HIT_LR", "2e-4")),
                    batch_size=int(os.environ.get("ALPHA_HIT_BATCH_SIZE", "256")),
                    steps_per_tick=int(os.environ.get("ALPHA_HIT_STEPS_PER_TICK", "2")),
                    max_buffer=int(os.environ.get("ALPHA_HIT_MAX_BUFFER", "200000")),
                    data_half_life_sec=float(os.environ.get("ALPHA_HIT_DATA_HALF_LIFE_SEC", "3600.0")),
                    ckpt_path=self.alpha_hit_model_path,
                    enable=True,
                )
                self.alpha_hit_trainer = OnlineAlphaTrainer(trainer_cfg)
                self.alpha_hit_mlp = self.alpha_hit_trainer.model  # Access model through trainer
                logger.info(f"[ALPHA_HIT] Initialized OnlineAlphaTrainer with {len(policy_horizons)} horizons")
            except Exception as e:
                logger.warning(f"[ALPHA_HIT] Failed to initialize trainer: {e}")
                self.alpha_hit_trainer = None
                self.alpha_hit_mlp = None
                self.alpha_hit_enabled = False
        # ✅ 격리 테스트용 플래그
        self._skip_goal_jax = False  # goal 확률 계산(JAX)만 끄기
        self._skip_first_passage_jax = False  # first-passage JAX만 끄기
        self._force_gaussian_dist = False  # dist를 gaussian으로 고정
        # ✅ Step C: 원인 분리 A/B 테스트 플래그
        self._force_zero_cost = False  # 비용 0으로 강제 (fee_roundtrip=0, expected_spread_cost=0, slippage_dyn=0)
        self._force_horizon_600 = False  # horizon 600초 고정 (best_h 선택 무시하고 600만 평가)
        
        # PMAKER execution mixing parameters
        self.PMAKER_DELAY_PENALTY_MULT = float(os.getenv("PMAKER_DELAY_PENALTY_MULT", "1.0"))
        self.PMAKER_EXIT_DELAY_PENALTY_MULT = float(os.getenv("PMAKER_EXIT_DELAY_PENALTY_MULT", "1.0"))
        self.ALPHA_DELAY_DECAY_TAU_SEC = float(os.getenv("ALPHA_DELAY_DECAY_TAU_SEC", "30.0"))
        self.PMAKER_ENTRY_DELAY_SHIFT = os.getenv("PMAKER_ENTRY_DELAY_SHIFT", "1") == "1"
        self.PMAKER_STRICT = bool(os.getenv("PMAKER_STRICT", "0") == "1")

    # -----------------------------
    # First-passage TP/SL Monte Carlo (event-based)
    # -----------------------------
    def mc_first_passage_tp_sl(
        self,
        s0: float,
        tp_pct: float,
        sl_pct: float,
        mu: float,
        sigma: float,
        dt: float,
        max_steps: int,
        n_paths: int,
        cvar_alpha: float = 0.05,
        timeout_mode: str = "flat",
        seed: Optional[int] = None,
        side: str = "LONG",
    ) -> Dict[str, Any]:
        tp_pct = float(tp_pct)
        sl_pct = float(sl_pct)
        if tp_pct <= 0 or sl_pct <= 0 or sigma <= 0 or s0 <= 0:
            return {
                "event_p_tp": None,
                "event_p_sl": None,
                "event_p_timeout": None,
                "event_ev_r": None,
                "event_cvar_r": None,
                "event_t_median": None,
                "event_t_mean": None,
            }

        rng = np.random.default_rng(seed)
        max_steps = int(max(1, max_steps))
        # -----------------------------
        # Direction handling
        # LONG: 그대로, SHORT: log-return 반전
        # -----------------------------
        direction = 1.0
        if str(side).upper() == "SHORT":
            direction = -1.0

        drift = direction * (mu - 0.5 * sigma * sigma) * dt
        diffusion = sigma * math.sqrt(dt)

        mode = str(getattr(self, "_tail_mode", self.default_tail_mode))
        df = float(getattr(self, "_student_t_df", self.default_student_t_df))
        br = getattr(self, "_bootstrap_returns", None)
        use_jax = bool(getattr(self, "_use_jax", True)) and _JAX_OK

        prices_np: np.ndarray
        if use_jax:
            # ✅ GPU 우선: default backend (GPU/Metal) 사용
            force_cpu_dev = _jax_mc_device()
            try:
                if force_cpu_dev is None:
                    # GPU/Metal default backend 사용
                    key = jrand.PRNGKey(int(seed or 0) & 0xFFFFFFFF)  # type: ignore[attr-defined]
                    key, z_j = self._sample_increments_jax(
                        key,
                        (int(n_paths), int(max_steps)),
                        mode=mode,
                        df=df,
                        bootstrap_returns=br,
                    )
                    if z_j is None:
                        use_jax = False
                    else:
                        z_j = z_j.astype(jnp.float32)  # type: ignore[attr-defined]
                        logret_j = jnp.cumsum((drift + diffusion * z_j), axis=1)  # type: ignore[attr-defined]
                        prices_j = float(s0) * jnp.exp(direction * logret_j)  # type: ignore[attr-defined]
                        prices_np = np.asarray(jax.device_get(prices_j), dtype=np.float64)  # type: ignore[attr-defined]
                else:
                    # CPU로 강제된 경우만 CPU 사용 (env JAX_MC_DEVICE=cpu)
                    with jax.default_device(force_cpu_dev):  # type: ignore[attr-defined]
                        key = jrand.PRNGKey(int(seed or 0) & 0xFFFFFFFF)  # type: ignore[attr-defined]
                        key, z_j = self._sample_increments_jax(
                            key,
                            (int(n_paths), int(max_steps)),
                            mode=mode,
                            df=df,
                            bootstrap_returns=br,
                        )
                        if z_j is None:
                            use_jax = False
                        else:
                            z_j = z_j.astype(jnp.float32)  # type: ignore[attr-defined]
                            logret_j = jnp.cumsum((drift + diffusion * z_j), axis=1)  # type: ignore[attr-defined]
                            prices_j = float(s0) * jnp.exp(direction * logret_j)  # type: ignore[attr-defined]
                            prices_np = np.asarray(jax.device_get(prices_j), dtype=np.float64)  # type: ignore[attr-defined]
            except Exception:
                # Any JAX/XLA backend failure -> fall back to NumPy path simulation
                use_jax = False


        if not use_jax:
            z = self._sample_increments_np(rng, (n_paths, max_steps), mode=mode, df=df, bootstrap_returns=br)
            steps = drift + diffusion * z
            log_prices = np.cumsum(steps, axis=1) + math.log(s0)
            prices_np = np.exp(direction * log_prices)

        if str(side).upper() == "SHORT":
            tp_level = s0 * (1.0 - tp_pct)
            sl_level = s0 * (1.0 + sl_pct)
        else:
            tp_level = s0 * (1.0 + tp_pct)
            sl_level = s0 * (1.0 - sl_pct)

        hit_tp = prices_np >= tp_level
        hit_sl = prices_np <= sl_level

        tp_hit_idx = np.where(hit_tp.any(axis=1), hit_tp.argmax(axis=1) + 1, max_steps + 1)
        sl_hit_idx = np.where(hit_sl.any(axis=1), hit_sl.argmax(axis=1) + 1, max_steps + 1)

        first_hit_idx = np.minimum(tp_hit_idx, sl_hit_idx)
        hit_type = np.full(n_paths, "timeout", dtype=object)
        hit_type[(tp_hit_idx < sl_hit_idx) & (tp_hit_idx <= max_steps)] = "tp"
        hit_type[(sl_hit_idx < tp_hit_idx) & (sl_hit_idx <= max_steps)] = "sl"

        tp_R = float(tp_pct / sl_pct)
        returns_r = np.zeros(n_paths, dtype=np.float64)
        returns_r[hit_type == "tp"] = tp_R
        returns_r[hit_type == "sl"] = -1.0
        if timeout_mode in ("mark_to_market", "mtm"):
            # mark-to-market: use end-of-horizon return
            end_prices = prices_np[:, -1]
            returns_r[hit_type == "timeout"] = (end_prices[hit_type == "timeout"] - s0) / (s0 * sl_pct)

        event_p_tp = float(np.mean(hit_type == "tp"))
        event_p_sl = float(np.mean(hit_type == "sl"))
        event_p_timeout = float(np.mean(hit_type == "timeout"))
        event_ev_r = float(np.mean(returns_r))
        event_cvar_r = float(cvar_ensemble(returns_r, alpha=cvar_alpha))

        hit_mask = (hit_type == "tp") | (hit_type == "sl")
        hit_times = first_hit_idx[hit_mask]
        event_t_median = float(np.median(hit_times)) if hit_times.size > 0 else None
        event_t_mean = float(np.mean(hit_times)) if hit_times.size > 0 else None

        # sanity check
        prob_sum = event_p_tp + event_p_sl + event_p_timeout
        if abs(prob_sum - 1.0) > 1e-3:
            # normalize softly
            event_p_tp /= prob_sum
            event_p_sl /= prob_sum
            event_p_timeout = max(0.0, 1.0 - event_p_tp - event_p_sl)

        return {
            "event_p_tp": event_p_tp,
            "event_p_sl": event_p_sl,
            "event_p_timeout": event_p_timeout,
            "event_ev_r": event_ev_r,
            "event_cvar_r": event_cvar_r,
            "event_t_median": event_t_median,
            "event_t_mean": event_t_mean,
        }

    @staticmethod
    def _annualize(mu_bar: float, sigma_bar: float, bar_seconds: float) -> Tuple[float, float]:
        bars_per_year = (365.0 * 24.0 * 3600.0) / float(bar_seconds)
        mu_base = float(mu_bar) * bars_per_year
        sigma_ann = float(sigma_bar) * math.sqrt(bars_per_year)
        return float(mu_base), float(max(sigma_ann, 1e-6))

    @staticmethod
    def _trend_direction(price: float, closes: Sequence[float]) -> int:
        # EMA200 없으면 EMA50/20로 대체
        if closes is None or len(closes) < 30:
            return 1
        p = float(price)
        e_slow = ema(closes, 200) if len(closes) >= 200 else ema(closes, min(50, len(closes)))
        if e_slow is None:
            return 1
        return 1 if p >= float(e_slow) else -1

    @staticmethod
    def _signal_alpha_mu_annual(closes: Sequence[float], bar_seconds: float, ofi_score: float, regime: str) -> float:
        """
        최근 평균수익(mu_bar)을 쓰지 않고, "신호(모멘텀/OFI)"로부터 조건부 기대수익(알파) μ(연율)를 만든다.
        - 출력 단위: 연율(log-return drift, per-year)
        - 방향: 양수=가격상승 기대, 음수=가격하락 기대
        """
        parts = MonteCarloEngine._signal_alpha_mu_annual_parts(closes, bar_seconds, ofi_score, regime)
        try:
            return float(parts.get("mu_alpha") or 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _signal_alpha_mu_annual_parts(closes: Sequence[float], bar_seconds: float, ofi_score: float, regime: str) -> Dict[str, Any]:
        """
        _signal_alpha_mu_annual()의 분해/디버그 버전.
        - mu_mom_{15,30,60,120}: 각 모멘텀 창의 연율(로그수익 기울기) 추정치
        - mu_mom: 가중 평균
        - mu_ofi: OFI 기반 단기 알파 항
        - mu_alpha_raw: regime_scale 및 결합 후, cap 적용 전
        - mu_alpha: cap 적용 후 (실제 사용)
        """
        if not closes or len(closes) < 8:
            return {
                "bar_seconds": float(bar_seconds) if float(bar_seconds) > 0 else 60.0,
                "n_closes": int(len(closes) if closes else 0),
                "regime": str(regime or ""),
                "regime_scale": 1.0,
                "ofi_score_clipped": 0.0,
                "mu_ofi": 0.0,
                "mu_mom": 0.0,
                "mu_alpha_raw": 0.0,
                "mu_alpha_cap": 40.0,
                "mu_alpha": 0.0,
                "reason": "insufficient_closes",
            }

        bs = float(bar_seconds) if float(bar_seconds) > 0 else 60.0
        n = len(closes)

        def _mom_mu_ann(window_bars: int) -> tuple[float, float, float]:
            w = int(window_bars)
            if w <= 1 or n <= w:
                return 0.0, 0.0, 0.0
            p0 = float(closes[-w - 1])
            p1 = float(closes[-1])
            if p0 <= 0.0 or p1 <= 0.0:
                return 0.0, 0.0, 0.0
            try:
                lr = math.log(p1 / p0)
            except Exception:
                return 0.0, 0.0, 0.0
            # Stabilize annualization: short-window momentum can explode when scaled to per-year.
            # Use a tau floor (in seconds) and optional log-return clipping.
            try:
                lr_cap = float(os.environ.get("MU_MOM_LR_CAP", "0.10") or 0.10)  # default 10% cap per window
            except Exception:
                lr_cap = 0.10
            if lr_cap > 0:
                lr = float(max(-lr_cap, min(lr_cap, float(lr))))

            try:
                tau_floor = float(os.environ.get("MU_MOM_TAU_FLOOR_SEC", "21600") or 21600.0)  # default 6h floor
            except Exception:
                tau_floor = 21600.0
            tau = max(1e-9, float(w) * bs, float(max(0.0, tau_floor)))
            mu_ann = float((lr / tau) * SECONDS_PER_YEAR)
            return float(mu_ann), float(lr), float(tau)

        mom_cfg = ((15, 0.35), (30, 0.30), (60, 0.20), (120, 0.15))
        mom_terms_w = []
        mom_w = []
        mom_each: Dict[str, Any] = {}
        for w, wt in mom_cfg:
            if n > int(w) + 1:
                mu_w, lr_w, tau_w = _mom_mu_ann(int(w))
                mom_each[f"mu_mom_{int(w)}"] = float(mu_w)
                mom_each[f"lr_mom_{int(w)}"] = float(lr_w)
                mom_each[f"tau_mom_{int(w)}_sec"] = float(tau_w)
                mom_terms_w.append(float(mu_w) * float(wt))
                mom_w.append(float(wt))
        mu_mom = float(sum(mom_terms_w) / max(1e-12, float(sum(mom_w)))) if mom_terms_w else 0.0

        # OFI는 매우 단기 알파로 취급 (연율로 스케일)
        try:
            ofi = float(ofi_score)
        except Exception:
            ofi = 0.0
        ofi = float(max(-1.0, min(1.0, ofi)))
        ofi_scale = 8.0
        mu_ofi = float(ofi * ofi_scale)

        r = str(regime or "").lower()
        regime_scale = 1.0
        if r == "chop":
            regime_scale = 0.35
        elif r == "volatile":
            regime_scale = 0.55

        mu_alpha_raw = float(regime_scale * (0.70 * mu_mom + 0.30 * mu_ofi))

        try:
            mu_cap = float(os.environ.get("MU_ALPHA_CAP", "40.0") or 40.0)
        except Exception:
            mu_cap = 40.0
        mu_alpha = float(mu_alpha_raw)
        if mu_alpha > mu_cap:
            mu_alpha = mu_cap
        elif mu_alpha < -mu_cap:
            mu_alpha = -mu_cap

        out: Dict[str, Any] = {
            "bar_seconds": float(bs),
            "n_closes": int(n),
            "regime": str(regime or ""),
            "regime_scale": float(regime_scale),
            "mu_mom_tau_floor_sec": float(os.environ.get("MU_MOM_TAU_FLOOR_SEC", "21600") or 21600.0),
            "mu_mom_lr_cap": float(os.environ.get("MU_MOM_LR_CAP", "0.10") or 0.10),
            "ofi_score_clipped": float(ofi),
            "mu_ofi": float(mu_ofi),
            "mu_mom": float(mu_mom),
            "mu_alpha_raw": float(mu_alpha_raw),
            "mu_alpha_cap": float(mu_cap),
            "mu_alpha": float(mu_alpha),
        }
        out.update(mom_each)
        
        # Debug: Print mu_alpha calculation
        print(f"[ALPHA_DEBUG] mu_mom={mu_mom:.6f} mu_ofi={mu_ofi:.6f} regime_scale={regime_scale:.2f} mu_alpha_raw={mu_alpha_raw:.6f} mu_alpha={mu_alpha:.6f}")
        
        return out

    def _get_params(self, regime: str, ctx: Dict[str, Any]) -> MCParams:
        rp = ctx.get("regime_params")
        # lightweight runtime override for live mode responsiveness
        n_paths_override = ctx.get("n_paths")
        if n_paths_override is None:
            n_paths_override = MC_N_PATHS_LIVE
        try:
            n_paths_override = int(n_paths_override)
        except Exception:
            n_paths_override = MC_N_PATHS_LIVE
        n_paths_override = int(max(200, min(200000, n_paths_override)))
        if isinstance(rp, dict):
            # dict → MCParams로 안전 변환
            base = DEFAULT_PARAMS.get(regime, DEFAULT_PARAMS["chop"])
            return MCParams(
                min_win=float(rp.get("min_win", base.min_win)),
                profit_target=float(rp.get("profit_target", base.profit_target)),
                ofi_weight=float(rp.get("ofi_weight", base.ofi_weight)),
                max_kelly=float(rp.get("max_kelly", base.max_kelly)),
                cvar_alpha=float(rp.get("cvar_alpha", base.cvar_alpha)),
                cvar_scale=float(rp.get("cvar_scale", base.cvar_scale)),
                # ctx override가 항상 우선 (live에서 프리즈 방지)
                n_paths=int(n_paths_override),
            )
        base = DEFAULT_PARAMS.get(regime, DEFAULT_PARAMS["chop"])
        if n_paths_override != int(base.n_paths):
            return MCParams(
                min_win=float(base.min_win),
                profit_target=float(base.profit_target),
                ofi_weight=float(base.ofi_weight),
                max_kelly=float(base.max_kelly),
                cvar_alpha=float(base.cvar_alpha),
                cvar_scale=float(base.cvar_scale),
                n_paths=int(n_paths_override),
            )
        return base

    # -----------------------------
    # Regime clustering (lightweight K-Means)
    # -----------------------------
    @staticmethod
    def _cluster_regime(closes: Sequence[float]) -> str:
        if closes is None or len(closes) < 40:
            return "chop"
        x = np.asarray(closes, dtype=np.float64)
        rets = np.diff(np.log(x))
        if rets.size < 30:
            return "chop"
        # 특징: 단기 추세, 변동성
        slope = float(x[-1] - x[-10]) / max(1e-6, float(x[-10]))
        vol = float(rets[-30:].std())
        feats = np.array([[slope, vol]], dtype=np.float64)
        # 초기 중심 (bear/chop/bull 가정)
        centers = np.array([
            [-0.002, vol * 1.2],
            [0.0, vol],
            [0.002, max(vol * 0.8, 1e-6)]
        ], dtype=np.float64)
        # 미니 k-means 3회
        for _ in range(3):
            d = ((feats[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = d.argmin(axis=1)
            for k in range(3):
                mask = labels == k
                if mask.any():
                    centers[k] = feats[mask].mean(axis=0)
        label = int(labels[0])
        if label == 0:
            return "bear"
        if label == 2:
            return "bull"
        return "volatile" if vol > 0.01 else "chop"

    # -----------------------------
    # Slippage model
    # -----------------------------
    def _estimate_slippage(self, leverage: float, sigma: float, liq_score: float, ofi_z_abs: float = 0.0) -> float:
        base = self.slippage_perc
        vol_term = 1.0 + float(sigma) * 0.5
        liq_term = 1.0 if liq_score <= 0 else min(2.0, 1.0 + 1.0 / max(liq_score, 1.0))
        # ✅ 로그 스케일 적용: 레버리지 영향력을 대폭 줄임
        # 기존: lev_term = max(1.0, abs(leverage) / 5.0) (20배 레버리지 시 4배 증가)
        # 수정: 로그 스케일로 변경 (10배 레버리지 시 약 1.2배 정도만 증가)
        lev_term = 1.0 + 0.1 * math.log(1.0 + abs(leverage))
        adv_k = 1.0 + 0.6 * min(2.0, max(0.0, ofi_z_abs))
        slip = base * vol_term * liq_term * lev_term * adv_k
        slip_mult = float(os.environ.get("SLIPPAGE_MULT", "0.3"))
        slip_cap = float(os.environ.get("SLIPPAGE_CAP", "0.0003"))
        slip = max(0.0, float(slip) * slip_mult)
        if slip_cap > 0:
            slip = min(slip, slip_cap)
        return slip

    def _estimate_p_maker(self, *, spread_pct: float, liq_score: float, ofi_z_abs: float) -> float:
        """
        post-only maker 시도(짧은 timeout)에서 maker fill 성공 확률(0~1) 근사.
        - 기본: 유동성↑, 스프레드↓, OFI extreme↓일수록 maker 성공 확률↑
        - 너무 과도한 낙관을 막기 위해 [0.05, 0.95]로 클립
        """
        fixed = os.environ.get("P_MAKER_FIXED")
        if fixed is not None and str(fixed).strip() != "":
            try:
                return float(np.clip(float(fixed), 0.0, 1.0))
            except Exception:
                pass

        sp = float(max(0.0, spread_pct))
        liq = float(max(1.0, liq_score))
        ofi = float(max(0.0, ofi_z_abs))

        # liq_score는 scale이 크므로 log로 완만하게
        liq_term = math.log(liq)  # 0~(대략)
        # simple logistic
        x = 0.35 + 0.12 * liq_term - 900.0 * sp - 0.25 * ofi
        # numerical-stable sigmoid
        if x >= 0:
            p = 1.0 / (1.0 + math.exp(-x))
        else:
            ex = math.exp(x)
            p = ex / (1.0 + ex)
        return float(np.clip(p, 0.05, 0.95))

    # -----------------------------
    # Tail samplers
    # -----------------------------
    def _sample_increments_np(self, rng: np.random.Generator, shape, *, mode: str, df: float, bootstrap_returns: Optional[np.ndarray]):
        if mode == "bootstrap" and bootstrap_returns is not None and bootstrap_returns.size >= 16:
            idx = rng.integers(0, bootstrap_returns.size, size=shape)
            return bootstrap_returns[idx].astype(np.float64)
        if mode == "student_t":
            z = rng.standard_t(df=df, size=shape).astype(np.float64)
            if df > 2:
                z = z / np.sqrt(df / (df - 2.0))
            return z
        return rng.standard_normal(size=shape).astype(np.float64)

    def _sample_increments_jax(self, key, shape, *, mode: str, df: float, bootstrap_returns: Optional[np.ndarray]):
        if jrand is None:
            return key, None
        if mode == "bootstrap" and bootstrap_returns is not None:
            # ✅ 해결: boot를 무조건 jnp.asarray로 정규화하고 shape[0]을 Python int로 변환
            br = jnp.asarray(bootstrap_returns, dtype=jnp.float32)  # type: ignore[attr-defined]
            br_size = int(br.shape[0])  # Python int로 변환 (tracer 방지)
            if br_size >= 16:
                key, k1 = jrand.split(key)  # type: ignore[attr-defined]
                idx = jrand.randint(k1, shape=shape, minval=0, maxval=br_size)  # type: ignore[attr-defined]
                return key, br[idx]
        if mode == "student_t":
            key, k1 = jrand.split(key)  # type: ignore[attr-defined]
            z = jrand.t(k1, df=df, shape=shape)  # type: ignore[attr-defined]
            if df > 2:
                z = z / jnp.sqrt(df / (df - 2.0))  # type: ignore[attr-defined]
            return key, z
        key, k1 = jrand.split(key)  # type: ignore[attr-defined]
        return key, jrand.normal(k1, shape=shape)  # type: ignore[attr-defined]

    def simulate_paths_price(
        self,
        *,
        seed: int,
        s0: float,
        mu: float,
        sigma: float,
        n_paths: int,
        n_steps: int,
        dt: float,
    ) -> np.ndarray:
        """
        1초 단위 가격 경로를 생성한다 (JAX JIT 최적화 버전).
        - 반환 shape: (n_paths, n_steps+1)
          paths[:,0] = s0 (t=0), paths[:,t] = price at t seconds
        """
        n_paths_i = int(max(1, int(n_paths)))
        n_steps_i = int(max(1, int(n_steps)))

        drift = (float(mu) - 0.5 * float(sigma) * float(sigma)) * float(dt)
        diffusion = float(sigma) * math.sqrt(float(dt))

        mode = str(getattr(self, "_tail_mode", self.default_tail_mode))
        df = float(getattr(self, "_student_t_df", self.default_student_t_df))
        br = getattr(self, "_bootstrap_returns", None)
        use_jax = bool(getattr(self, "_use_jax", True)) and _JAX_OK

        if use_jax and _simulate_paths_price_jax_core_jit is not None:
            try:
                key = jrand.PRNGKey(int(seed) & 0xFFFFFFFF)  # type: ignore[attr-defined]
                boot_jnp = None if br is None else jnp.asarray(br, dtype=jnp.float32)  # type: ignore[attr-defined]
                
                # JIT 컴파일된 핵심 함수 호출
                paths_jnp = _simulate_paths_price_jax_core_jit(
                    key, s0, drift, diffusion, n_paths_i, n_steps_i, mode, df, boot_jnp
                )
                
                # JAX 배열을 NumPy로 변환
                paths = np.asarray(jax.device_get(paths_jnp), dtype=np.float64)  # type: ignore[attr-defined]
                return paths
            except Exception:
                use_jax = False

        # Fallback to NumPy implementation
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        z = self._sample_increments_np(rng, (n_paths_i, n_steps_i), mode=mode, df=df, bootstrap_returns=br)
        logret = np.cumsum(drift + diffusion * z, axis=1)
        prices_1 = float(s0) * np.exp(logret)

        paths = np.empty((n_paths_i, n_steps_i + 1), dtype=np.float64)
        paths[:, 0] = float(s0)
        paths[:, 1:] = prices_1
        return paths
    def evaluate_entry_metrics(self, ctx: Dict[str, Any], params: MCParams, seed: int) -> Dict[str, Any]:
        """
        튜너가 과거 ctx로 candidate 파라미터 평가할 때도 사용.
        """
        symbol = str(ctx.get("symbol", ""))
        # ✅ 함수 호출 확인 로그
        logger.info(f"[COST_DEBUG] {symbol} | evaluate_entry_metrics called")
        # ✅ [EV_DEBUG] evaluate_entry_metrics 시작 로그
        logger.info(f"[EV_DEBUG] {symbol} | evaluate_entry_metrics START: price={ctx.get('price')} mu_sim={ctx.get('mu_sim')} sigma_sim={ctx.get('sigma_sim')}")
        print(f"[EV_DEBUG] {symbol} | evaluate_entry_metrics START: price={ctx.get('price')} mu_sim={ctx.get('mu_sim')} sigma_sim={ctx.get('sigma_sim')}")
        def _s(val, default=0.0) -> float:
            try:
                if val is None:
                    return float(default)
                return float(val)
            except Exception:
                return float(default)

        # symbol은 위에서 이미 가져옴
        price = _s(ctx.get("price"), 0.0)
        
        # mu/sigma 입력은 유지하되, "μ는 최근 평균수익"이 아니라 아래에서 계산하는 알파 μ로 완전 대체한다.
        # (mu_base는 디버깅/추적용으로만 남긴다)
        mu_base = ctx.get("mu_sim")
        if mu_base is None:
            mu_base = ctx.get("mu_base")
        sigma = ctx.get("sigma_sim")  # sigma_sim 우선
        if sigma is None:
            sigma = ctx.get("sigma")  # fallback
        
        # ✅ [EV_DEBUG] mu_base, sigma 값 확인
        print(f"[EV_DEBUG] {symbol} | evaluate_entry_metrics: mu_base={mu_base} sigma={sigma} price={price}")
        if mu_base is None or sigma is None or sigma <= 0:
            print(f"[EV_DEBUG] {symbol} | ⚠️  WARNING: mu_base or sigma is invalid! mu_base={mu_base} sigma={sigma}")
            logger.warning(f"[EV_DEBUG] {symbol} | ⚠️  WARNING: mu_base or sigma is invalid! mu_base={mu_base} sigma={sigma}")
        
        closes = ctx.get("closes")
        liq_score = _s(ctx.get("liquidity_score"), 1.0)
        bar_seconds = _s(ctx.get("bar_seconds", 60.0), 60.0)
        # tail mode plumbing
        self._use_jax = bool(ctx.get("use_jax", True))
        self._tail_mode = str(ctx.get("tail_mode", self.default_tail_mode))
        self._student_t_df = _s(ctx.get("student_t_df", self.default_student_t_df), self.default_student_t_df)
        br = ctx.get("bootstrap_returns")
        if br is not None:
            try:
                self._bootstrap_returns = np.asarray(br, dtype=np.float64)
            except Exception:
                self._bootstrap_returns = None
        else:
            if self._tail_mode == "bootstrap" and closes is not None and len(closes) >= 64:
                x = np.asarray(closes, dtype=np.float64)
                rets = np.diff(np.log(np.maximum(x, 1e-12)))
                self._bootstrap_returns = rets[-512:].astype(np.float64) if rets.size >= 32 else None
            else:
                self._bootstrap_returns = None
        regime_ctx = str(ctx.get("regime", "chop"))

        # ✅ Step 1: mu_base, sigma 초기 계산 전 상태 기록
        mu_base_input = mu_base
        sigma_input = sigma
        closes_len = len(closes) if closes is not None else 0
        returns_window_len = None
        vol_src = "ctx_input"
        
        # sigma가 없는 경우만 closes에서 보충 (μ는 사용하지 않음)
        if (sigma is None) and closes is not None and len(closes) >= 10:
            rets = np.diff(np.log(np.asarray(closes, dtype=np.float64)))
            if rets.size >= 5:
                returns_window_len = rets.size
                vol_src = f"closes_diff_n={returns_window_len}"
                mu_bar = float(rets.mean())
                sigma_bar = float(rets.std())
                _, sigma = self._annualize(mu_bar, sigma_bar, bar_seconds=bar_seconds)

        # 다중 지평 블렌딩으로 추정치 안정화
        if closes is not None and len(closes) >= 30:
            rets = np.diff(np.log(np.asarray(closes, dtype=np.float64)))
            windows = [30, 90, 180]
            mu_blend = []
            sigma_blend = []
            for w in windows:
                if rets.size >= w:
                    rw = rets[-w:]
                    mu_blend.append(float(rw.mean()))
                    sigma_blend.append(float(rw.std()))
            if mu_blend and sigma_blend:
                returns_window_len = rets.size  # 업데이트
                vol_src = f"blend_windows={windows}_n={returns_window_len}"
                mu_bar_mix = float(np.mean(mu_blend))
                sigma_bar_mix = float(np.mean(sigma_blend))
                mu_mix, sigma_mix = self._annualize(mu_bar_mix, sigma_bar_mix, bar_seconds=bar_seconds)
                if sigma is None:
                    sigma = sigma_mix
                else:
                    sigma = 0.5 * float(sigma) + 0.5 * float(sigma_mix)

        if sigma is None or price <= 0:
            logger.warning(
                f"[MU_SIGMA_DEBUG] {symbol} | early return: mu_base_input={mu_base_input}, sigma_input={sigma_input}, "
                f"mu_base_after={mu_base}, sigma_after={sigma}, price={price}, closes_len={closes_len}, "
                f"returns_window_len={returns_window_len}, vol_src={vol_src}, regime={regime_ctx}"
            )
            print(f"[EARLY_RETURN_1] {symbol} | sigma={sigma} price={price} (returning ev=0)")
            return {"can_enter": False, "ev": 0.0, "ev_raw": 0.0, "win": 0.0, "cvar": 0.0, "best_h": 0, "direction": 1, "kelly": 0.0, "size_frac": 0.0}

        sigma = float(sigma) if sigma is not None else 0.0
        
        # ✅ Step 1: sigma가 0이면 경고 및 키 확인
        if sigma <= 0:
            ctx_keys = list(ctx.keys())
            mu_sigma_keys = [k for k in ctx_keys if 'mu' in k.lower() or 'sigma' in k.lower()]
            logger.error(
                f"[MU_SIGMA_ERROR] {symbol} | sigma <= 0 before adjust! "
                f"sigma={sigma}, mu_base={mu_base}, "
                f"ctx_keys_with_mu_sigma={mu_sigma_keys}, "
                f"mu_sim={ctx.get('mu_sim')}, sigma_sim={ctx.get('sigma_sim')}, "
                f"mu_base_ctx={ctx.get('mu_base')}, sigma_ctx={ctx.get('sigma')}"
            )
            # 절대 0 fallback하지 않음 - 에러를 명확히 보고
            return {"can_enter": False, "ev": 0.0, "ev_raw": 0.0, "win": 0.0, "cvar": 0.0, "best_h": 0, "direction": 1, "kelly": 0.0, "size_frac": 0.0}
        
        sigma = max(sigma, 1e-6)  # 최소값 보정 (0이 아닌 경우에만)
        
        # 레짐 기반 기대수익/변동성 조정
        mu_base_before_adjust = float(mu_base) if mu_base is not None else 0.0
        sigma_before_adjust = sigma
        # ✅ μ는 신호 조건부 기대수익(알파)로 완전 대체
        ofi_score = _s(ctx.get("ofi_score"), 0.0)
        mu_alpha_parts = self._signal_alpha_mu_annual_parts(closes or [], bar_seconds, ofi_score, regime_ctx)
        mu_alpha_raw = float(mu_alpha_parts.get("mu_alpha") or 0.0)
        
        # ✅ [FIX 2] PMaker fill 결과를 mu_alpha에 반영하여 개선
        # fill rate가 높으면 alpha 신뢰도 증가 → mu_alpha 증가
        # fill rate가 낮으면 alpha 신뢰도 감소 → mu_alpha 감소
        mu_alpha_pmaker_adjusted = mu_alpha_raw
        pmaker_surv = ctx.get("pmaker_surv")
        pmaker_mu_alpha_boost_enabled = str(os.environ.get("PMAKER_MU_ALPHA_BOOST_ENABLED", "1")).strip().lower() in ("1", "true", "yes")
        pmaker_mu_alpha_boost_k = float(os.environ.get("PMAKER_MU_ALPHA_BOOST_K", "0.15"))  # fill rate 변화 1당 mu_alpha 변화량 (연율)
        
        
        if pmaker_mu_alpha_boost_enabled and pmaker_surv is not None:
            try:
                # 심볼별 평균 fill rate 가져오기
                fill_rate_mean = pmaker_surv.sym_fill_mean(symbol)
                # 중립 기준을 0.5로 잡되, fill_rate가 낮다고 mu_alpha를 추가로 깎지는 않는다(부정적 신호와 이중 반영 방지).
                fill_rate_bias = max(0.0, (fill_rate_mean - 0.5) * 2.0)  # [0, 1] 범위로 변환
                # fill_rate가 0.5보다 높을 때만 alpha를 보정(추가 가산), 낮을 때는 0으로 두어 악화 방지
                mu_alpha_boost = fill_rate_bias * pmaker_mu_alpha_boost_k * abs(mu_alpha_raw)
                mu_alpha_pmaker_adjusted = mu_alpha_raw + mu_alpha_boost
                
                # mu_alpha_parts에 조정값 저장
                mu_alpha_parts["mu_alpha_pmaker_fill_rate"] = float(fill_rate_mean)
                mu_alpha_parts["mu_alpha_pmaker_boost"] = float(mu_alpha_boost)
                mu_alpha_parts["mu_alpha_before_pmaker"] = float(mu_alpha_raw)
                
                logger.info(
                    f"[MU_ALPHA_PMAKER_BOOST] {symbol} | "
                    f"fill_rate={fill_rate_mean:.4f} fill_rate_bias={fill_rate_bias:.4f} | "
                    f"mu_alpha_raw={mu_alpha_raw:.6f} boost={mu_alpha_boost:.6f} mu_alpha_adjusted={mu_alpha_pmaker_adjusted:.6f}"
                )
            except Exception as e:
                logger.warning(f"[MU_ALPHA_PMAKER_BOOST] {symbol} | Failed to apply PMaker boost: {e}")
                mu_alpha_parts["mu_alpha_pmaker_fill_rate"] = None
                mu_alpha_parts["mu_alpha_pmaker_boost"] = 0.0
        
        # mu_alpha cap 적용 (pmaker 조정 후)
        try:
            mu_cap = float(os.environ.get("MU_ALPHA_CAP", "40.0") or 40.0)
        except Exception:
            mu_cap = 40.0
        mu_alpha_final = float(mu_alpha_pmaker_adjusted)
        if mu_alpha_final > mu_cap:
            mu_alpha_final = mu_cap
        elif mu_alpha_final < -mu_cap:
            mu_alpha_final = -mu_cap
        
        mu_alpha_parts["mu_alpha"] = float(mu_alpha_final)  # 최종 mu_alpha 업데이트
        try:
            ctx["mu_alpha"] = float(mu_alpha_final)
        except Exception:
            ctx["mu_alpha"] = 0.0
        
        mu_base, sigma = adjust_mu_sigma(float(mu_alpha_final), sigma, regime_ctx)
        
        # ✅ Step 1: 조정 후에도 sigma가 0이면 경고
        if sigma <= 0:
            logger.error(
                f"[MU_SIGMA_ERROR] {symbol} | sigma <= 0 after adjust_mu_sigma! "
                f"sigma_before={sigma_before_adjust:.8f}, sigma_after={sigma:.8f}, "
                f"mu_before={mu_base_before_adjust:.8f}, mu_after={mu_base:.8f}, regime={regime_ctx}"
            )
            print(f"[EARLY_RETURN_2] {symbol} | sigma={sigma} after adjust_mu_sigma (returning ev=0)")
            return {"can_enter": False, "ev": 0.0, "ev_raw": 0.0, "win": 0.0, "cvar": 0.0, "best_h": 0, "direction": 1, "kelly": 0.0, "size_frac": 0.0}
        
        sigma = max(sigma, 1e-6)  # 조정 후 최소값 보정
        
        # ✅ [FIX] meta 변수를 초기화 (line 2413에서 사용되기 전에)
        meta = {}
        
        # ✅ Step 1: mu_sim, sigma_sim 추적 로그 (조정 후)
        logger.info(
            f"[MU_SIGMA_DEBUG] {symbol} | "
            f"mu_base_input={mu_base_input}, sigma_input={sigma_input} | "
            f"mu_base_before_adjust={mu_base_before_adjust:.8f}, sigma_before_adjust={sigma_before_adjust:.8f} | "
            f"mu_base_after_adjust={mu_base:.8f}, sigma_after_adjust={sigma:.8f} | "
            f"closes_len={closes_len}, returns_window_len={returns_window_len}, vol_src={vol_src} | "
            f"regime={regime_ctx}"
        )

        direction = int(ctx.get("direction") or self._trend_direction(price, closes or []))
        regime_ctx_for_cluster = ctx.get("regime")
        if regime_ctx_for_cluster == "chop" and closes:
            regime_ctx_for_cluster = self._cluster_regime(closes)

        mu_adj = float(mu_base)
        signal_mu = float(mu_alpha_parts.get("mu_alpha", mu_alpha_raw))  # ✅ FIX: pmaker 조정된 mu_alpha 사용
        
        # ✅ [EV_VALIDATION 1] mu_alpha 자체가 음수/미약 검증 (pmaker 조정 후)
        mu_alpha_value = float(mu_alpha_parts.get("mu_alpha", 0.0))
        mu_alpha_mom = float(mu_alpha_parts.get("mu_mom", 0.0))
        mu_alpha_ofi = float(mu_alpha_parts.get("mu_ofi", 0.0))
        mu_alpha_warning = []
        if abs(mu_alpha_value) < 0.01:  # 1% annualized threshold
            mu_alpha_warning.append(f"mu_alpha={mu_alpha_value:.6f} is too weak (|mu_alpha|<0.01)")
        if mu_alpha_value < -5.0:
            mu_alpha_warning.append(f"mu_alpha={mu_alpha_value:.6f} is strongly negative (<-5.0)")
        if abs(mu_alpha_mom) < 0.005 and abs(mu_alpha_ofi) < 0.5:
            mu_alpha_warning.append(f"Both mu_mom={mu_alpha_mom:.6f} and mu_ofi={mu_alpha_ofi:.6f} are weak")
        if mu_alpha_warning:
            logger.warning(f"[EV_VALIDATION_1] {symbol} | mu_alpha issues: {'; '.join(mu_alpha_warning)}")
            print(f"[EV_VALIDATION_1] {symbol} | mu_alpha issues: {'; '.join(mu_alpha_warning)}")
        
        leverage = _s(ctx.get("leverage", 5.0), 5.0)

        # OFI z-score by regime/session (abs) for slippage adverse factor
        key = (str(ctx.get("regime", "chop")), str(ctx.get("session", "OFF")))
        hist = self._ofi_hist.setdefault(key, [])
        hist.append(ofi_score)
        if len(hist) > 500:
            hist.pop(0)
        mean = 0.0
        std = 0.05
        ofi_z_abs = 0.0
        if len(hist) >= 5:
            arr = np.asarray(hist, dtype=np.float64)
            mean = float(arr.mean())
            std = float(arr.std())
            std = std if std > 1e-6 else 0.05
            ofi_z_abs = abs(ofi_score - mean) / std

        # 레버리지/변동성/유동성 기반 슬리피지 모델
        slippage_dyn_raw = self._estimate_slippage(leverage, sigma, liq_score, ofi_z_abs=ofi_z_abs)
        slippage_base_raw = self._estimate_slippage(1.0, sigma, liq_score, ofi_z_abs=ofi_z_abs)
        fee_fee_taker = float(self.fee_roundtrip_base)
        fee_fee_maker = float(getattr(self, "fee_roundtrip_maker_base", 0.0002))
        spread_pct = ctx.get("spread_pct")
        if spread_pct is None:
            spread_pct = 0.0002  # 2bp fallback
        try:
            spread_pct = float(spread_pct)
        except Exception:
            spread_pct = 0.0002
        spread_cap = float(os.environ.get("SPREAD_PCT_MAX", "0.0005"))
        if spread_cap > 0.0 and spread_pct > spread_cap:
            spread_pct = spread_cap
        expected_spread_cost_raw = 0.5 * float(spread_pct) * 1.0  # adverse selection factor=1.0

        # maker_then_market 모드: 기대비용을 (maker/taker 혼합)으로 근사
        exec_mode = str(os.environ.get("EXEC_MODE", "market")).strip().lower()
        p_maker = 0.0
        fee_fee_mix = fee_fee_taker
        slippage_dyn = float(slippage_dyn_raw)
        slippage_base = float(slippage_base_raw)
        expected_spread_cost = float(expected_spread_cost_raw)
        if exec_mode == "maker_then_market":
            p_maker = float(self._estimate_p_maker(spread_pct=float(spread_pct), liq_score=float(liq_score), ofi_z_abs=float(ofi_z_abs)))
            fee_fee_mix = float(p_maker * fee_fee_maker + (1.0 - p_maker) * fee_fee_taker)
            # maker fill 시 spread/slippage를 대부분 피한다고 가정(보수적으로 taker 비중만 반영)
            slippage_dyn = float((1.0 - p_maker) * float(slippage_dyn_raw))
            slippage_base = float((1.0 - p_maker) * float(slippage_base_raw))
            expected_spread_cost = float((1.0 - p_maker) * float(expected_spread_cost_raw))

        # --- PMAKER survival override (from orchestrator decision.meta) ---
        pmaker_entry = 0.0
        pmaker_entry_delay_sec = 0.0
        pmaker_exit = 0.0
        pmaker_exit_delay_sec = 0.0
        pmaker_override_used = False
        try:
            meta_in = ctx.get("meta") or {}
            pmaker_entry = float(ctx.get("pmaker_entry") or meta_in.get("pmaker_entry") or 0.0)
            print(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_entry={pmaker_entry:.4f} from ctx.get={ctx.get('pmaker_entry')} meta_in.get={meta_in.get('pmaker_entry')}")
            logger.info(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_entry={pmaker_entry:.4f} from ctx.get={ctx.get('pmaker_entry')} meta_in.get={meta_in.get('pmaker_entry')}")
        except Exception as e:
            print(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_entry read failed: {e}")
            logger.warning(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_entry read failed: {e}")
            pmaker_entry = 0.0
        try:
            meta_in = ctx.get("meta") or {}
            pmaker_entry_delay_sec = float(ctx.get("pmaker_entry_delay_sec") or meta_in.get("pmaker_entry_delay_sec") or 0.0)
            logger.info(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_entry_delay_sec={pmaker_entry_delay_sec:.4f}")
        except Exception as e:
            logger.warning(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_entry_delay_sec read failed: {e}")
            pmaker_entry_delay_sec = 0.0
        try:
            meta_in = ctx.get("meta") or {}
            pmaker_exit = float(ctx.get("pmaker_exit") or meta_in.get("pmaker_exit") or pmaker_entry)
            print(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_exit={pmaker_exit:.4f} from ctx.get={ctx.get('pmaker_exit')} meta_in.get={meta_in.get('pmaker_exit')}")
            logger.info(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_exit={pmaker_exit:.4f} from ctx.get={ctx.get('pmaker_exit')} meta_in.get={meta_in.get('pmaker_exit')}")
        except Exception as e:
            print(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_exit read failed: {e}")
            logger.warning(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_exit read failed: {e}")
            pmaker_exit = pmaker_entry
        try:
            meta_in = ctx.get("meta") or {}
            pmaker_exit_delay_sec = float(ctx.get("pmaker_exit_delay_sec") or meta_in.get("pmaker_exit_delay_sec") or pmaker_entry_delay_sec)
            logger.info(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_exit_delay_sec={pmaker_exit_delay_sec:.4f}")
        except Exception as e:
            logger.warning(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_exit_delay_sec read failed: {e}")
            pmaker_exit_delay_sec = pmaker_entry_delay_sec

        # Guard against NaN/Inf leaking from ctx/meta (would silently become null in JSON).
        try:
            if not math.isfinite(float(pmaker_entry)):
                pmaker_entry = 0.0
        except Exception:
            pmaker_entry = 0.0
        try:
            if not math.isfinite(float(pmaker_entry_delay_sec)):
                pmaker_entry_delay_sec = 0.0
        except Exception:
            pmaker_entry_delay_sec = 0.0
        try:
            if not math.isfinite(float(pmaker_exit)):
                pmaker_exit = float(pmaker_entry)
        except Exception:
            pmaker_exit = float(pmaker_entry)
        try:
            if not math.isfinite(float(pmaker_exit_delay_sec)):
                pmaker_exit_delay_sec = float(pmaker_entry_delay_sec)
        except Exception:
            pmaker_exit_delay_sec = float(pmaker_entry_delay_sec)

        if exec_mode == "maker_then_market" and pmaker_entry > 0.0:
            # Use PMaker's calibrated P(fill<=timeout) instead of heuristic p_maker.
            pmaker_entry = float(min(1.0, max(0.0, pmaker_entry)))
            p_maker = pmaker_entry
            fee_fee_mix = float(p_maker * fee_fee_maker + (1.0 - p_maker) * fee_fee_taker)
            slippage_dyn = float((1.0 - p_maker) * float(slippage_dyn_raw))
            slippage_base = float((1.0 - p_maker) * float(slippage_base_raw))
            expected_spread_cost = float((1.0 - p_maker) * float(expected_spread_cost_raw))
            pmaker_override_used = True

        # [DIFF 5 VALIDATION] Validate Maker → Market 혼합 실행 + 기대 비용 모델
        diff5_validation_warnings = []
        
        # Validation point 1: exec_mode should be "maker_then_market" when using maker mode
        if exec_mode == "maker_then_market":
            # Validation point 2: pmaker_entry should exist and be > 0 when using maker_then_market
            # ✅ [FIX] pmaker_entry_local이 아직 정의되지 않았으므로 pmaker_entry를 직접 사용
            if pmaker_entry <= 0.0:
                diff5_validation_warnings.append(
                    f"pmaker_entry={pmaker_entry:.6f} <= 0.0 in maker_then_market mode"
                )
            
            # Validation point 3: fee_roundtrip_fee_mix should be < fee_taker (maker mix should be cheaper)
            if fee_fee_mix >= fee_fee_taker:
                diff5_validation_warnings.append(
                    f"fee_roundtrip_fee_mix={fee_fee_mix:.8f} >= fee_taker={fee_fee_taker:.8f} "
                    f"(p_maker={p_maker:.4f} fee_maker={fee_fee_maker:.8f})"
                )
            elif p_maker > 0.0 and fee_fee_maker >= fee_fee_taker:
                # If p_maker > 0 but fee_maker >= fee_taker, then fee_mix cannot be < fee_taker
                diff5_validation_warnings.append(
                    f"fee_maker={fee_fee_maker:.8f} >= fee_taker={fee_fee_taker:.8f} "
                    f"but p_maker={p_maker:.4f} > 0 (fee_mix cannot be cheaper)"
                )
        
        # Additional validation: fee_roundtrip_fee_mix calculation
        if exec_mode == "maker_then_market" and p_maker > 0.0:
            fee_mix_expected = float(p_maker * fee_fee_maker + (1.0 - p_maker) * fee_fee_taker)
            fee_mix_diff = abs(fee_fee_mix - fee_mix_expected)
            if fee_mix_diff > 1e-6:
                diff5_validation_warnings.append(
                    f"fee_roundtrip_fee_mix calculation mismatch: computed={fee_fee_mix:.8f} "
                    f"expected={fee_mix_expected:.8f} diff={fee_mix_diff:.8f}"
                )
        
        # Log validation warnings
        if diff5_validation_warnings:
            logger.warning(
                f"[DIFF5_VALIDATION] {symbol} | exec_mode={exec_mode} "
                f"pmaker_entry={pmaker_entry:.6f} p_maker={p_maker:.4f} "
                f"fee_roundtrip_fee_mix={fee_fee_mix:.8f} fee_taker={fee_fee_taker:.8f} "
                f"fee_maker={fee_fee_maker:.8f} | Warnings: {'; '.join(diff5_validation_warnings)}"
            )
        
        # Store validation metrics in meta
        if diff5_validation_warnings:
            meta["diff5_validation_warnings"] = diff5_validation_warnings

        # 수수료는 레버리지와 무관하게 고정, 슬리피지는 노출(lev) 가중
        fee_rt = float(fee_fee_mix + slippage_dyn)
        # gate용 baseline(lev=1)
        fee_rt_base = float(fee_fee_mix + slippage_base)
        
        # ✅ Step C: 원인 분리 A/B 테스트 플래그
        _force_zero_cost = getattr(self, "_force_zero_cost", False)  # 비용 0으로 강제
        _force_horizon_600 = getattr(self, "_force_horizon_600", False)  # horizon 600초 고정
        
        if _force_zero_cost:
            fee_rt = 0.0
            fee_rt_base = 0.0
            expected_spread_cost = 0.0
            expected_spread_cost_raw = 0.0
            slippage_dyn = 0.0
            slippage_base = 0.0
            slippage_dyn_raw = 0.0
            slippage_base_raw = 0.0
            p_maker = 0.0
            fee_fee_mix = 0.0

        # ✅ total execution cost (roundtrip) used in net simulation
        # NOTE: simulate_paths_netpnl already subtracts fee_roundtrip, so we MUST NOT subtract it again later.
        # ✅ slippage_dyn도 비용에 포함 (entry + exit 각각 발생하므로 roundtrip 기준으로 포함)
        # slippage는 entry와 exit에서 각각 발생하므로, roundtrip에서는 2배가 아니라 1회만 반영 (이미 각각 계산됨)
        fee_roundtrip_total = float(fee_rt) + float(expected_spread_cost) + float(slippage_dyn)
        fee_roundtrip_total_base = float(fee_rt_base) + float(expected_spread_cost) + float(slippage_base)
        
        # ✅ 하드 캡 적용: 비정상적으로 높은 수수료 방지
        MAX_FEE_ROUNDTRIP = 0.003  # 0.3% 상한선
        fee_roundtrip_total = min(fee_roundtrip_total, MAX_FEE_ROUNDTRIP)
        fee_roundtrip_total_base = min(fee_roundtrip_total_base, MAX_FEE_ROUNDTRIP)
        
        execution_cost = float(fee_roundtrip_total)
        
        # ✅ 모든 심볼의 비용 검증 (LINK 제외, 비용이 너무 큰 경우)
        if not symbol.startswith("LINK"):
            # cost_entry는 fee_roundtrip_total의 절반이므로, fee_roundtrip_total이 크면 cost_entry도 큼
            # 일반적으로 fee_roundtrip_total은 0.001~0.002 정도여야 함 (0.1%~0.2%)
            if fee_roundtrip_total > 0.002:  # 0.2% 이상이면 비정상적으로 큼
                logger.warning(f"[EV_VALIDATION_NEG] {symbol} | ⚠️  비정상적으로 큰 fee_roundtrip_total: {fee_roundtrip_total:.6f} (일반적으로 0.001~0.002)")
                logger.warning(f"[EV_VALIDATION_NEG] {symbol} | fee_rt={fee_rt:.6f} expected_spread_cost={expected_spread_cost:.6f} fee_fee_mix={fee_fee_mix:.6f} slippage_dyn={slippage_dyn:.6f}")
                print(f"[EV_VALIDATION_NEG] {symbol} | ⚠️  비정상적으로 큰 fee_roundtrip_total: {fee_roundtrip_total:.6f}")
                print(f"[EV_VALIDATION_NEG] {symbol} | fee_rt={fee_rt:.6f} expected_spread_cost={expected_spread_cost:.6f} fee_fee_mix={fee_fee_mix:.6f} slippage_dyn={slippage_dyn:.6f}")
        
        # ✅ Step C: horizon 선택
        horizons_for_sim = (600,) if _force_horizon_600 else self.horizons
        
        # ✅ 즉시 확인: fee_roundtrip_total (execution_cost) 값
        logger.info(
            f"[COST_QUICK] {symbol} | exec_mode={exec_mode} p_maker={p_maker:.3f} | "
            f"execution_cost={execution_cost:.6f} | fee_taker={fee_fee_taker:.6f} fee_maker={fee_fee_maker:.6f} fee_mix={fee_fee_mix:.6f} | "
            f"slippage_dyn_eff={slippage_dyn:.6f} spread_eff={expected_spread_cost:.6f} | "
            f"slippage_dyn_raw={float(slippage_dyn_raw):.6f} spread_raw={float(expected_spread_cost_raw):.6f} | "
            f"_force_zero_cost={_force_zero_cost}"
        )
        # ✅ 즉시 확인: self.horizons 값
        logger.info(f"[HORIZON_DEBUG] {symbol} | self.horizons={self.horizons} | horizons_for_sim={horizons_for_sim} | _force_horizon_600={_force_horizon_600}")
        
        # ✅ Step 1: direction이 왜 SHORT로 잡히는지 확인
        logger.info(f"[DIR_DEBUG] {symbol} direction={direction} mu_adj={mu_adj:.10f} sigma={sigma:.10f} lev={leverage} dt={self.dt} fee_rt={fee_rt:.6f}")
        if MC_VERBOSE_PRINT:
            print(f"[DIR_DEBUG] {symbol} direction={direction} mu_adj={mu_adj:.10f} sigma={sigma:.10f} lev={leverage} dt={self.dt} fee_rt={fee_rt:.6f}")

        # ✅ Step 2: LONG/SHORT 둘 다 평가
        max_steps = int(max(horizons_for_sim)) if horizons_for_sim else 0
        price_paths = self.simulate_paths_price(
            seed=seed,
            s0=float(price),
            mu=float(mu_adj),
            sigma=float(sigma),
            n_paths=int(params.n_paths),
            n_steps=max_steps,
            dt=float(self.dt),
        )
        # net = direction * ((S_t - S0)/S0) * leverage - fee_roundtrip_total
        s0_f = float(price)
        lev_f = float(leverage)
        fee_rt_total_f = float(fee_roundtrip_total)
        fee_rt_total_base_f = float(fee_roundtrip_total_base)

        net_by_h_long: Dict[int, np.ndarray] = {}
        net_by_h_short: Dict[int, np.ndarray] = {}
        net_by_h_long_base: Dict[int, np.ndarray] = {}
        net_by_h_short_base: Dict[int, np.ndarray] = {}
        for h in horizons_for_sim:
            hh = int(h)
            # price_paths includes s0 at index 0, so horizon h seconds => index h
            tp = np.asarray(price_paths[:, hh], dtype=np.float64)
            gross_ret = (tp - s0_f) / max(1e-12, s0_f)
            gross_long = gross_ret * lev_f
            gross_short = -gross_ret * lev_f
            net_by_h_long[hh] = (gross_long - fee_rt_total_f).astype(np.float64)
            net_by_h_short[hh] = (gross_short - fee_rt_total_f).astype(np.float64)
            net_by_h_long_base[hh] = (gross_ret - fee_rt_total_base_f).astype(np.float64)
            net_by_h_short_base[hh] = (-gross_ret - fee_rt_total_base_f).astype(np.float64)
        
        # ✅ Step 2: per-horizon stats를 계산하는 함수
        def summarize(net_by_h):
            ev_list, win_list, cvar_list, h_list = [], [], [], []
            best_h, best_ev = None, -1e18
            for h, net in net_by_h.items():
                ev_h = float(net.mean())
                win_h = float((net > 0).mean())
                cvar_h = float(cvar_ensemble(net, alpha=params.cvar_alpha))
                ev_list.append(ev_h)
                win_list.append(win_h)
                cvar_list.append(cvar_h)
                h_list.append(int(h))
                if ev_h > best_ev:
                    best_ev = ev_h
                    best_h = int(h)
            ev_mean = float(np.mean(ev_list)) if ev_list else 0.0
            win_mean = float(np.mean(win_list)) if win_list else 0.0
            cvar_mean = float(np.mean(cvar_list)) if cvar_list else 0.0
            return best_h, best_ev, ev_mean, win_mean, cvar_mean, (ev_list, win_list, cvar_list, h_list)
        
        best_h_L, best_ev_L, ev_L, win_L, cvar_L, dbg_L = summarize(net_by_h_long)
        best_h_S, best_ev_S, ev_S, win_S, cvar_S, dbg_S = summarize(net_by_h_short)

        policy_horizons = list(getattr(self, "POLICY_MULTI_HORIZONS_SEC", (60, 180, 300, 600, 900, 1800)))
        if not policy_horizons:
            policy_horizons = [int(getattr(self, "POLICY_HORIZON_SEC", 1800))]
        max_policy_h = int(max(policy_horizons)) if policy_horizons else int(getattr(self, "POLICY_HORIZON_SEC", 1800))
        
        # ✅ [EV_DEBUG] policy_horizons 확인 (성능 개선: MC_VERBOSE_PRINT로 조건부)
        if MC_VERBOSE_PRINT:
            logger.info(f"[EV_DEBUG] {symbol} | policy_horizons={policy_horizons} max_policy_h={max_policy_h}")
            print(f"[EV_DEBUG] {symbol} | policy_horizons={policy_horizons} max_policy_h={max_policy_h}")

        # ✅ Check 1) horizon별로 path를 새로 생성하지 않도록, 최대 horizon 길이 경로를 1번만 만들고 재사용
        # - 기본: 이미 생성한 price_paths(hold/horizon용)를 재사용 (동일하게 1초 스텝 경로임)
        # - 단, 디버그로 horizons_for_sim이 짧아졌을 때(_force_horizon_600), policy horizon이 더 길면 별도 생성
        price_paths_policy = price_paths
        paths_reused = True
        paths_seed_base = int(seed)
        if int(price_paths_policy.shape[1]) < int(max_policy_h + 1):
            price_paths_policy = self.simulate_paths_price(
                seed=paths_seed_base,
                s0=float(price),
                mu=float(mu_adj),
                sigma=float(sigma),
                n_paths=int(params.n_paths),
                n_steps=int(max_policy_h),
                dt=float(self.dt),
            )
            paths_reused = False

        momentum_z = 0.0
        if closes is not None and len(closes) >= 6:
            rets = np.diff(np.log(np.asarray(closes, dtype=np.float64)))
            if rets.size >= 5:
                window = int(min(20, rets.size))
                subset = rets[-window:]
                mean_r = float(subset.mean())
                std_r = float(subset.std())
                if std_r > 1e-9:
                    momentum_z = float((subset[-1] - mean_r) / std_r)

        ofi_z = 0.0
        if len(hist) >= 5:
            std_local = float(max(std, 1e-9))
            ofi_z = float((ofi_score - mean) / std_local) if std_local > 1e-9 else 0.0

        signal_strength = float(abs(momentum_z) + 0.7 * abs(ofi_z))
        s_clip = float(np.clip(signal_strength, 0.0, 4.0))
        policy_half_life_sec = float(1800.0 / (1.0 + s_clip))
        
        # Compute prior weights (rule-based) - used as base for EV-based weights
        w_prior = self._weights_for_horizons(policy_horizons, signal_strength)
        
        # [D] AlphaHitMLP: Predict horizon-specific TP/SL probabilities using OnlineAlphaTrainer
        alpha_hit = None
        features_np = None
        if self.alpha_hit_enabled and self.alpha_hit_trainer is not None:
            try:
                features = self._extract_alpha_hit_features(
                    symbol=symbol,
                    price=price,
                    mu=mu_adj,
                    sigma=sigma,
                    momentum_z=momentum_z,
                    ofi_z=ofi_z,
                    regime=regime_ctx,
                    leverage=leverage,
                    ctx=ctx,
                )
                if features is not None:
                    features_np = features[0].detach().cpu().numpy()  # [F]
                    # Use trainer's predict method
                    pred = self.alpha_hit_trainer.predict(features_np)
                    # Convert to expected format (arrays aligned with horizons)
                    alpha_hit = {
                        "p_tp_long": pred["p_tp_long"],
                        "p_sl_long": pred["p_sl_long"],
                        "p_tp_short": pred["p_tp_short"],
                        "p_sl_short": pred["p_sl_short"],
                    }
                    # Train on previous samples (async, non-blocking)
                    try:
                        train_stats = self.alpha_hit_trainer.train_tick()
                        if train_stats.get("loss") is not None:
                            logger.debug(f"[ALPHA_HIT] Training loss: {train_stats['loss']:.6f}, buffer: {train_stats['buffer_n']}")
                    except Exception as e:
                        logger.warning(f"[ALPHA_HIT] Training failed: {e}")
            except Exception as e:
                logger.warning(f"[ALPHA_HIT] Failed to predict: {e}")
                alpha_hit = None
        # [D] TP/SL 확률 기반 EV 계산 (완전 전환)
        # --- NEW: alpha hit model predictions (per horizon, long/short) ---
        # Costs: maker survival + delay are supplied per symbol (from LiveOrchestrator)
        # Note: pmaker_entry, pmaker_entry_delay_sec, pmaker_exit, pmaker_exit_delay_sec are already read above (line 2270-2294)
        # We use them here for delay penalty calculation
        # ✅ Use already-read values from line 2301-2314 instead of re-reading from ctx
        pmaker_entry_local = float(pmaker_entry)  # Use value read at line 2301
        pmaker_delay_sec_local = float(pmaker_entry_delay_sec)  # Use value read at line 2310
        
        # [C] PMaker delay penalty: k * sigma_per_sqrt_sec * sqrt(delay)
        # (approximates adverse selection / drift uncertainty during waiting)
        delay_penalty_k = float(os.environ.get("PMAKER_DELAY_PENALTY_K", getattr(self, "PMAKER_DELAY_PENALTY_K", 1.0)))
        # sigma_bar is per-sec vol proxy (from meta or computed)
        mu_sigma_meta = ctx.get("meta", {})
        sigma_bar = float(mu_sigma_meta.get("sigma_bar", 0.0) or 0.0)
        if sigma_bar <= 0:
            # Fallback: use annualized sigma converted to per-second
            sigma_bar = float(sigma) / math.sqrt(float(SECONDS_PER_YEAR))
        delay_penalty = delay_penalty_k * sigma_bar * math.sqrt(max(0.0, pmaker_delay_sec_local))
        
        # [D] TP/SL EV per horizon
        # TP table per horizon (horizon-specific TP returns)
        tp_r_by_h = getattr(self, "TP_R_BY_H", {})
        # Trailing SL is ATR based; in EV space we approximate SL_r as risk-bound
        sl_r_fixed = float(os.environ.get("SL_R_FIXED", getattr(self, "SL_R_FIXED", 0.0020)))
        atr_frac = float(ctx.get("atr_frac", 0.0) or 0.0)  # ATR/price from orchestrator
        sl_atr_mult = float(os.environ.get("TRAIL_ATR_MULT", getattr(self, "TRAIL_ATR_MULT", 2.0)))
        sl_r = (sl_atr_mult * atr_frac) if atr_frac > 0 else sl_r_fixed
        
        # ✅ [EV_VALIDATION_3] 비용 분리: entry 1회 + exit 1회로 분리하여 한 번만 차감
        # fee_rt_total_f는 roundtrip 비용 (entry + exit 합계)
        # Entry 비용: fee_entry + spread_entry + slippage_entry (1회만 차감)
        # Exit 비용: fee_exit + spread_exit + slippage_exit (1회만 차감)
        # 계획: entry 비용은 여기서 차감, exit 비용은 compute_exit_policy_metrics에서 exec_oneway로 차감
        # 따라서 compute_exit_policy_metrics에는 exit 비용만 전달 (fee_roundtrip에 exit 비용만)
        fee_roundtrip_total = float(fee_rt_total_f)
        # Entry 비용: roundtrip의 절반 (entry fee + spread_entry + slippage_entry)
        # 실제로는 fee_rt_total_f가 이미 entry+exit 합계이므로, 절반을 entry로 사용
        cost_entry = fee_roundtrip_total / 2.0  # Entry 비용 (roundtrip의 절반)
        cost_exit = fee_roundtrip_total / 2.0  # Exit 비용 (roundtrip의 절반, compute_exit_policy_metrics에서 사용)
        # delay_penalty는 horizon별로 스케일링되어 적용됨 (아래에서 처리)
        
        # ✅ [EV_VALIDATION_3] maker delay + spread + slippage가 gross EV를 잠식 검증
        # Gross EV approximation: mu_adj * leverage * time_horizon (rough estimate for 10min)
        gross_ev_approx_10min = float(mu_adj) / float(SECONDS_PER_YEAR) * float(leverage) * 600.0  # 10min
        cost_fraction = cost_entry / max(1e-6, abs(gross_ev_approx_10min)) if gross_ev_approx_10min != 0.0 else float('inf')
        cost_warning_val = None
        if cost_fraction > 0.5:  # 비용이 gross EV의 50% 이상
            cost_warning_val = f"costs are {cost_fraction*100:.1f}% of gross_ev_approx_10min (cost_entry={cost_entry:.6f}, gross={gross_ev_approx_10min:.6f})"
            logger.warning(f"[EV_VALIDATION_3] {symbol} | {cost_warning_val}")
            print(f"[EV_VALIDATION_3] {symbol} | {cost_warning_val}")
            # Store in meta (meta will be initialized/accessed later)
            meta["ev_validation_3_warning"] = cost_warning_val
        
        # [D] Compute EV_h for long/short based on predicted hit probs
        ev_long_h = []
        ev_short_h = []
        ppos_long_h = []
        ppos_short_h = []
        
        # Keep MC simulation results for meta/diagnostics (exit-policy rollforward)
        evs_long = []
        evs_short = []
        pps_long = []
        pps_short = []
        cvars_long = []
        cvars_short = []
        per_h_long = []
        per_h_short = []
        mc_p_tp_long = []
        mc_p_sl_long = []
        mc_p_tp_short = []
        mc_p_sl_short = []

        mu_exit = float(mu_adj) / float(SECONDS_PER_YEAR)
        sigma_exit = float(sigma) / math.sqrt(float(SECONDS_PER_YEAR))
        impact_cost = 0.0
        exec_oneway = float(execution_cost) / 2.0
        
        # [C] PMaker survival function and delay penalty per horizon
        pmaker_surv = ctx.get("pmaker_surv")  # PMakerSurvivalMLP instance from orchestrator
        pmaker_timeout_ms = int(ctx.get("pmaker_timeout_ms", 1500))  # 1.5s default
        pmaker_features = ctx.get("pmaker_features")  # Features for PMaker prediction
        
        # [DIFF 6] Prepare decision_meta for compute_exit_policy_metrics (pmaker_entry/exit from ctx)
        # Use already-read values from evaluate_entry_metrics (line 2270-2294)
        # These are already extracted from ctx, so we can use them directly
        decision_meta_for_exit = {}
        # Always include pmaker_entry/exit even if 0 (compute_exit_policy_metrics needs to know they exist)
        if pmaker_entry is not None:
            decision_meta_for_exit["pmaker_entry"] = float(pmaker_entry)
        if pmaker_entry_delay_sec is not None:
            decision_meta_for_exit["pmaker_entry_delay_sec"] = float(pmaker_entry_delay_sec)
        if pmaker_exit is not None:
            decision_meta_for_exit["pmaker_exit"] = float(pmaker_exit)
        if pmaker_exit_delay_sec is not None:
            decision_meta_for_exit["pmaker_exit_delay_sec"] = float(pmaker_exit_delay_sec)
        print(f"[PMAKER_DEBUG] {symbol} | decision_meta_for_exit keys={list(decision_meta_for_exit.keys())} values={decision_meta_for_exit}")
        logger.info(f"[PMAKER_DEBUG] {symbol} | decision_meta_for_exit keys={list(decision_meta_for_exit.keys())} values={decision_meta_for_exit}")
        
        # ✅ [FIX] meta는 이미 초기화됨 (line 2229에서)
        # meta = {}  # 이미 초기화되어 있으므로 주석 처리
        
        # [D] Compute TP/SL 확률 기반 EV_h for each horizon
        # ✅ [EV_DEBUG] alpha_hit 상태 확인 (성능 개선: MC_VERBOSE_PRINT로 조건부)
        if MC_VERBOSE_PRINT:
            logger.info(f"[EV_DEBUG] {symbol} | alpha_hit_enabled={self.alpha_hit_enabled} alpha_hit_trainer={self.alpha_hit_trainer is not None} alpha_hit={alpha_hit is not None}")
            print(f"[EV_DEBUG] {symbol} | alpha_hit_enabled={self.alpha_hit_enabled} alpha_hit_trainer={self.alpha_hit_trainer is not None} alpha_hit={alpha_hit is not None}")
            if alpha_hit is None:
                logger.warning(f"[EV_DEBUG] {symbol} | alpha_hit is None - will use compute_exit_policy_metrics results only")
                print(f"[EV_DEBUG] {symbol} | alpha_hit is None - will use compute_exit_policy_metrics results only")
        
        # ✅ [EV_DEBUG] horizon 루프 시작 확인 (성능 개선: MC_VERBOSE_PRINT로 조건부)
        if MC_VERBOSE_PRINT:
            logger.info(f"[EV_DEBUG] {symbol} | Starting horizon loop: policy_horizons={policy_horizons} len={len(policy_horizons)}")
            print(f"[EV_DEBUG] {symbol} | Starting horizon loop: policy_horizons={policy_horizons} len={len(policy_horizons)}")
        
        # Debug: Confirm horizon loop starts
        print(f"[HORIZON_LOOP_DEBUG] {symbol} | Starting horizon loop with {len(policy_horizons)} horizons: {policy_horizons}")
        
        for idx, h in enumerate(policy_horizons):
            # Get TP return for this horizon
            tp_r = float(tp_r_by_h.get(int(h), tp_r_by_h.get(str(int(h)), 0.0)))
            
            # [D] Get predicted TP/SL hit probabilities from AlphaHitMLP
            if alpha_hit is None:
                # Fallback: keep old policy rollforward EV if alpha not available
                tp_pL = 0.0
                sl_pL = 0.0
                tp_pS = 0.0
                sl_pS = 0.0
            else:
                # alpha_hit has per-horizon arrays aligned with horizons list
                h_idx = list(map(int, policy_horizons)).index(int(h))
                if h_idx < len(alpha_hit["p_tp_long"]):
                    tp_pL = float(alpha_hit["p_tp_long"][h_idx])
                    sl_pL = float(alpha_hit["p_sl_long"][h_idx])
                    tp_pS = float(alpha_hit["p_tp_short"][h_idx])
                    sl_pS = float(alpha_hit["p_sl_short"][h_idx])
                else:
                    tp_pL = 0.0
                    sl_pL = 0.0
                    tp_pS = 0.0
                    sl_pS = 0.0
            
            # [D] Keep MC simulation results for meta/diagnostics (exit-policy rollforward)
            seed_h = int(seed ^ (idx * 0x9E3779B1))
            m_long = self.compute_exit_policy_metrics(
                symbol=symbol,
                price=price,
                mu=mu_exit,
                sigma=sigma_exit,
                leverage=leverage,
                direction=1,
                fee_roundtrip=cost_exit,  # Exit 비용만 전달 (entry는 별도 차감)
                exec_oneway=exec_oneway,
                impact_cost=impact_cost,
                regime=regime_ctx,
                horizon_sec=int(h),
                decision_dt_sec=int(getattr(self, "POLICY_DECISION_DT_SEC", self.POLICY_DECISION_DT_SEC)),
                seed=seed_h,
                cvar_alpha=params.cvar_alpha,
                price_paths=price_paths_policy,
                decision_meta=decision_meta_for_exit,
            )
            m_short = self.compute_exit_policy_metrics(
                symbol=symbol,
                price=price,
                mu=mu_exit,
                sigma=sigma_exit,
                leverage=leverage,
                direction=-1,
                fee_roundtrip=cost_exit,  # Exit 비용만 전달 (entry는 별도 차감)
                exec_oneway=exec_oneway,
                impact_cost=impact_cost,
                regime=regime_ctx,
                horizon_sec=int(h),
                decision_dt_sec=int(getattr(self, "POLICY_DECISION_DT_SEC", self.POLICY_DECISION_DT_SEC)),
                seed=seed_h ^ 0x55555555,
                cvar_alpha=params.cvar_alpha,
                price_paths=price_paths_policy,
                decision_meta=decision_meta_for_exit,
            )
            per_h_long.append(m_long)
            per_h_short.append(m_short)
            mc_p_tp_long.append(float(m_long.get("p_tp", 0.0)))
            mc_p_sl_long.append(float(m_long.get("p_sl", 0.0)))
            mc_p_tp_short.append(float(m_short.get("p_tp", 0.0)))
            mc_p_sl_short.append(float(m_short.get("p_sl", 0.0)))
            ev_exit_long = float(m_long["ev_exit_policy"])
            ev_exit_short = float(m_short["ev_exit_policy"])
            evs_long.append(ev_exit_long)
            evs_short.append(ev_exit_short)
            pps_long.append(float(m_long["p_pos_exit_policy"]))
            pps_short.append(float(m_short["p_pos_exit_policy"]))
            cvars_long.append(float(m_long["cvar_exit_policy"]))
            cvars_short.append(float(m_short["cvar_exit_policy"]))
            
            # ✅ [EV_DEBUG] compute_exit_policy_metrics 결과 로그 (성능 개선: MC_VERBOSE_PRINT로 조건부)
            if MC_VERBOSE_PRINT:
                logger.info(f"[EV_DEBUG] {symbol} | h={h}s compute_exit_policy_metrics: ev_exit_long={ev_exit_long:.6f} ev_exit_short={ev_exit_short:.6f}")
                print(f"[EV_DEBUG] {symbol} | h={h}s compute_exit_policy_metrics: ev_exit_long={ev_exit_long:.6f} ev_exit_short={ev_exit_short:.6f}")
            
            # ✅ [EV_VALIDATION_1] alpha_hit가 None일 때 mu_alpha 기반 EV 사용 금지
            # Fallback 옵션: (A) trade deny 또는 (B) MC→hit-prob 변환
            if alpha_hit is None:
                alpha_hit_fallback = str(os.environ.get("ALPHA_HIT_FALLBACK", "mc_to_hitprob")).strip().lower()
                
                if alpha_hit_fallback == "deny":
                    # (A) Trade deny: ev=0, direction=0으로 처리 (horizon별로 ev=0 설정)
                    evL = 0.0
                    evS = 0.0
                    pposL = 0.0
                    pposS = 0.0
                    if MC_VERBOSE_PRINT:
                        logger.warning(f"[EV_VALIDATION_1] {symbol} | h={h}s: alpha_hit=None, fallback=deny: evL=0.0 evS=0.0 (trade denied)")
                        print(f"[EV_VALIDATION_1] {symbol} | h={h}s: alpha_hit=None, fallback=deny: evL=0.0 evS=0.0 (trade denied)")
                else:
                    # (B) MC→hit-prob 변환: simulate_exit_policy_rollforward에서 직접 집계한 p_tp/p_sl/p_other 사용
                    # delay는 entry shift로 시뮬에 이미 반영됨 (start_shift_steps), 사후 delay_scale 보정 없음
                    p_tp_L = float(m_long.get("p_tp", 0.0))
                    p_sl_L = float(m_long.get("p_sl", 0.0))
                    p_other_L = float(m_long.get("p_other", 0.0))
                    tp_r_actual_L = float(m_long.get("tp_r_actual", 0.0))  # 실제 TP 평균 수익률 (net, exit cost 포함)
                    sl_r_actual_L = float(m_long.get("sl_r_actual", 0.0))  # 실제 SL 평균 수익률 (net, exit cost 포함)
                    other_r_actual_L = float(m_long.get("other_r_actual", 0.0))  # 실제 other 평균 수익률 (net, exit cost 포함)
                    prob_sum_check_L = bool(m_long.get("prob_sum_check", False))
                    
                    p_tp_S = float(m_short.get("p_tp", 0.0))
                    p_sl_S = float(m_short.get("p_sl", 0.0))
                    p_other_S = float(m_short.get("p_other", 0.0))
                    tp_r_actual_S = float(m_short.get("tp_r_actual", 0.0))
                    sl_r_actual_S = float(m_short.get("sl_r_actual", 0.0))
                    other_r_actual_S = float(m_short.get("other_r_actual", 0.0))
                    prob_sum_check_S = bool(m_short.get("prob_sum_check", False))
                    
                    # ✅ EV 계산: 3항 (tp/sl/other)
                    # ev = p_tp * tp_r_actual + p_sl * sl_r_actual + p_other * other_r_actual - cost_entry
                    # (tp_r_actual, sl_r_actual, other_r_actual은 이미 exit cost 포함된 net 수익률)
                    evL = p_tp_L * tp_r_actual_L + p_sl_L * sl_r_actual_L + p_other_L * other_r_actual_L - cost_entry
                    evS = p_tp_S * tp_r_actual_S + p_sl_S * sl_r_actual_S + p_other_S * other_r_actual_S - cost_entry
                    pposL = p_tp_L  # p_pos는 TP hit 확률로 사용
                    pposS = p_tp_S
                    
                    logger.info(f"[EV_VALIDATION_1] {symbol} | h={h}s: alpha_hit=None, fallback=mc_to_hitprob: p_tp_L={p_tp_L:.4f} p_sl_L={p_sl_L:.4f} p_other_L={p_other_L:.4f} tp_r_L={tp_r_actual_L:.6f} sl_r_L={sl_r_actual_L:.6f} other_r_L={other_r_actual_L:.6f} prob_sum_check_L={prob_sum_check_L} cost_entry={cost_entry:.6f} evL={evL:.6f}")
                    logger.info(f"[EV_VALIDATION_1] {symbol} | h={h}s: alpha_hit=None, fallback=mc_to_hitprob: p_tp_S={p_tp_S:.4f} p_sl_S={p_sl_S:.4f} p_other_S={p_other_S:.4f} tp_r_S={tp_r_actual_S:.6f} sl_r_S={sl_r_actual_S:.6f} other_r_S={other_r_actual_S:.6f} prob_sum_check_S={prob_sum_check_S} cost_entry={cost_entry:.6f} evS={evS:.6f}")
                    print(f"[EV_VALIDATION_1] {symbol} | h={h}s: alpha_hit=None, fallback=mc_to_hitprob: p_tp_L={p_tp_L:.4f} p_sl_L={p_sl_L:.4f} p_other_L={p_other_L:.4f} tp_r_L={tp_r_actual_L:.6f} sl_r_L={sl_r_actual_L:.6f} other_r_L={other_r_actual_L:.6f} prob_sum_check_L={prob_sum_check_L} cost_entry={cost_entry:.6f} evL={evL:.6f}")
                    print(f"[EV_VALIDATION_1] {symbol} | h={h}s: alpha_hit=None, fallback=mc_to_hitprob: p_tp_S={p_tp_S:.4f} p_sl_S={p_sl_S:.4f} p_other_S={p_other_S:.4f} tp_r_S={tp_r_actual_S:.6f} sl_r_S={sl_r_actual_S:.6f} other_r_S={other_r_actual_S:.6f} prob_sum_check_S={prob_sum_check_S} cost_entry={cost_entry:.6f} evS={evS:.6f}")
                    
                    # ✅ alpha_hit=None일 때도 음수 EV 상세 진단 로그 추가
                    if not symbol.startswith("LINK") and (evL < 0 or evS < 0) and h == 60:
                        # EV 계산 상세 분석
                        expected_evL = p_tp_L * tp_r_actual_L + p_sl_L * sl_r_actual_L + p_other_L * other_r_actual_L
                        expected_evS = p_tp_S * tp_r_actual_S + p_sl_S * sl_r_actual_S + p_other_S * other_r_actual_S
                        
                        # ✅ 각 항목별 기여도 분석
                        tp_contrib_L = p_tp_L * tp_r_actual_L
                        sl_contrib_L = p_sl_L * sl_r_actual_L
                        other_contrib_L = p_other_L * other_r_actual_L
                        tp_contrib_S = p_tp_S * tp_r_actual_S
                        sl_contrib_S = p_sl_S * sl_r_actual_S
                        other_contrib_S = p_other_S * other_r_actual_S
                        
                        # ✅ exit_reason_counts 가져오기
                        exit_reason_counts_L = m_long.get("exit_reason_counts", {}) or {}
                        exit_reason_counts_S = m_short.get("exit_reason_counts", {}) or {}
                        
                        # ✅ 원인 분석
                        causes = []
                        if cost_entry > abs(expected_evL) * 0.5:
                            causes.append(f"cost_entry({cost_entry:.6f})가 expected_evL({expected_evL:.6f})의 50% 이상")
                        if sl_contrib_L < -0.001:
                            causes.append(f"SL 기여도({sl_contrib_L:.6f})가 너무 음수 (p_sl={p_sl_L:.4f}, sl_r={sl_r_actual_L:.6f})")
                        if tp_contrib_L < 0.001:
                            causes.append(f"TP 기여도({tp_contrib_L:.6f})가 너무 작음 (p_tp={p_tp_L:.4f}, tp_r={tp_r_actual_L:.6f})")
                        if p_sl_L > 0.5:
                            causes.append(f"SL 확률({p_sl_L:.4f})이 50% 이상으로 너무 높음")
                        if p_tp_L < 0.15:
                            causes.append(f"TP 확률({p_tp_L:.4f})이 15% 미만으로 너무 낮음")
                        
                        logger.warning(f"[EV_VALIDATION_NEG] {symbol} | ⚠️  음수 EV 발견: evL={evL:.6f} evS={evS:.6f}")
                        logger.warning(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: cost_entry={cost_entry:.6f} (비용이 너무 큰가?)")
                        logger.warning(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: tp_pL={p_tp_L:.4f} sl_pL={p_sl_L:.4f} p_other_L={p_other_L:.4f}")
                        logger.warning(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: tp_r_actual_L={tp_r_actual_L:.6f} sl_r_actual_L={sl_r_actual_L:.6f} other_r_actual_L={other_r_actual_L:.6f}")
                        logger.warning(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: expected_evL(비용 제외)={expected_evL:.6f} expected_evS(비용 제외)={expected_evS:.6f}")
                        logger.warning(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: cost_entry 비율: evL에서 {abs(cost_entry/expected_evL*100) if expected_evL != 0 else 0:.1f}%, evS에서 {abs(cost_entry/expected_evS*100) if expected_evS != 0 else 0:.1f}%")
                        logger.warning(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: 기여도 분석 - TP={tp_contrib_L:.6f} SL={sl_contrib_L:.6f} Other={other_contrib_L:.6f} (Long)")
                        logger.warning(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: 기여도 분석 - TP={tp_contrib_S:.6f} SL={sl_contrib_S:.6f} Other={other_contrib_S:.6f} (Short)")
                        if exit_reason_counts_L:
                            logger.warning(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: exit_reason_counts (Long): {exit_reason_counts_L}")
                        if exit_reason_counts_S:
                            logger.warning(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: exit_reason_counts (Short): {exit_reason_counts_S}")
                        if causes:
                            logger.warning(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: 🔍 음수 EV 원인: {'; '.join(causes)}")
                        
                        print(f"[EV_VALIDATION_NEG] {symbol} | ⚠️  음수 EV 발견: evL={evL:.6f} evS={evS:.6f}")
                        print(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: cost_entry={cost_entry:.6f} (비용이 너무 큰가?)")
                        print(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: tp_pL={p_tp_L:.4f} sl_pL={p_sl_L:.4f} p_other_L={p_other_L:.4f}")
                        print(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: tp_r_actual_L={tp_r_actual_L:.6f} sl_r_actual_L={sl_r_actual_L:.6f} other_r_actual_L={other_r_actual_L:.6f}")
                        print(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: expected_evL(비용 제외)={expected_evL:.6f} expected_evS(비용 제외)={expected_evS:.6f}")
                        print(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: cost_entry 비율: evL에서 {abs(cost_entry/expected_evL*100) if expected_evL != 0 else 0:.1f}%, evS에서 {abs(cost_entry/expected_evS*100) if expected_evS != 0 else 0:.1f}%")
                        print(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: 기여도 분석 - TP={tp_contrib_L:.6f} SL={sl_contrib_L:.6f} Other={other_contrib_L:.6f} (Long)")
                        print(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: 기여도 분석 - TP={tp_contrib_S:.6f} SL={sl_contrib_S:.6f} Other={other_contrib_S:.6f} (Short)")
                        if exit_reason_counts_L:
                            print(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: exit_reason_counts (Long): {exit_reason_counts_L}")
                        if exit_reason_counts_S:
                            print(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: exit_reason_counts (Short): {exit_reason_counts_S}")
                        if causes:
                            print(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: 🔍 음수 EV 원인: {'; '.join(causes)}")
            else:
                # ✅ alpha_hit가 있는 경우: AlphaHitMLP에서 예측한 TP/SL 확률 사용
                # delay는 entry shift로 시뮬에 이미 반영됨 (compute_exit_policy_metrics의 start_shift_steps)
                # 사후 delay_scale 보정 없음
                
                # AlphaHitMLP 예측값 사용 (이미 시뮬에서 delay 반영됨)
                tp_pL_val = float(tp_pL)
                sl_pL_val = float(sl_pL)
                tp_pS_val = float(tp_pS)
                sl_pS_val = float(sl_pS)
                
                # other 확률 계산 (p_tp + p_sl + p_other = 1)
                p_other_L = max(0.0, 1.0 - tp_pL_val - sl_pL_val)
                p_other_S = max(0.0, 1.0 - tp_pS_val - sl_pS_val)
                
                # ✅ 실제 발생한 수익률 사용 (MC 시뮬 결과에서)
                # compute_exit_policy_metrics에서 실제 tp_r_actual, sl_r_actual을 가져옴
                tp_r_actual_L = float(m_long.get("tp_r_actual", tp_r))  # 실제 TP 평균 수익률, 없으면 고정값 사용
                sl_r_actual_L = float(m_long.get("sl_r_actual", sl_r))  # 실제 SL 평균 수익률, 없으면 고정값 사용
                other_r_actual_L = float(m_long.get("other_r_actual", 0.0))  # 실제 other 평균 수익률
                
                tp_r_actual_S = float(m_short.get("tp_r_actual", tp_r))
                sl_r_actual_S = float(m_short.get("sl_r_actual", sl_r))
                other_r_actual_S = float(m_short.get("other_r_actual", 0.0))
                
                # ✅ LINK 심볼의 비정상적으로 큰 tp_r_actual, sl_r_actual 값 검증
                if symbol.startswith("LINK"):
                    # tp_r_actual, sl_r_actual이 비정상적으로 큰 경우 (절댓값 > 0.1)
                    if abs(tp_r_actual_L) > 0.1 or abs(sl_r_actual_L) > 0.1 or abs(tp_r_actual_S) > 0.1 or abs(sl_r_actual_S) > 0.1:
                        logger.warning(f"[EV_VALIDATION_LINK] {symbol} | ⚠️  비정상적으로 큰 tp_r/sl_r 값 발견:")
                        logger.warning(f"[EV_VALIDATION_LINK] {symbol} | h={h}s: tp_r_actual_L={tp_r_actual_L:.6f} sl_r_actual_L={sl_r_actual_L:.6f} tp_r_actual_S={tp_r_actual_S:.6f} sl_r_actual_S={sl_r_actual_S:.6f}")
                        logger.warning(f"[EV_VALIDATION_LINK] {symbol} | m_long keys={list(m_long.keys())[:20]} m_short keys={list(m_short.keys())[:20]}")
                        print(f"[EV_VALIDATION_LINK] {symbol} | ⚠️  비정상적으로 큰 tp_r/sl_r 값 발견:")
                        print(f"[EV_VALIDATION_LINK] {symbol} | h={h}s: tp_r_actual_L={tp_r_actual_L:.6f} sl_r_actual_L={sl_r_actual_L:.6f} tp_r_actual_S={tp_r_actual_S:.6f} sl_r_actual_S={sl_r_actual_S:.6f}")
                        print(f"[EV_VALIDATION_LINK] {symbol} | m_long keys={list(m_long.keys())[:20]} m_short keys={list(m_short.keys())[:20]}")
                
                # ✅ EV 계산: 3항 (tp/sl/other)
                # ev = p_tp * tp_r_actual + p_sl * sl_r_actual + p_other * other_r_actual - cost_entry
                # (tp_r_actual, sl_r_actual, other_r_actual은 이미 exit cost 포함된 net 수익률)
                evL = tp_pL_val * tp_r_actual_L + sl_pL_val * sl_r_actual_L + p_other_L * other_r_actual_L - cost_entry
                evS = tp_pS_val * tp_r_actual_S + sl_pS_val * sl_r_actual_S + p_other_S * other_r_actual_S - cost_entry
                pposL = tp_pL_val  # p_pos는 TP hit 확률로 사용
                pposS = tp_pS_val
                
                # ✅ [EV_DEBUG] horizon별 EV 계산 로그 (성능 개선: MC_VERBOSE_PRINT로 조건부)
                if MC_VERBOSE_PRINT:
                    logger.info(f"[EV_DEBUG] {symbol} | h={h}s: tp_pL={tp_pL_val:.4f} sl_pL={sl_pL_val:.4f} p_other_L={p_other_L:.4f} tp_r_actual_L={tp_r_actual_L:.6f} sl_r_actual_L={sl_r_actual_L:.6f} other_r_actual_L={other_r_actual_L:.6f} cost_entry={cost_entry:.6f} evL={evL:.6f}")
                    logger.info(f"[EV_DEBUG] {symbol} | h={h}s: tp_pS={tp_pS_val:.4f} sl_pS={sl_pS_val:.4f} p_other_S={p_other_S:.4f} tp_r_actual_S={tp_r_actual_S:.6f} sl_r_actual_S={sl_r_actual_S:.6f} other_r_actual_S={other_r_actual_S:.6f} cost_entry={cost_entry:.6f} evS={evS:.6f}")
                    print(f"[EV_DEBUG] {symbol} | h={h}s: tp_pL={tp_pL_val:.4f} sl_pL={sl_pL_val:.4f} p_other_L={p_other_L:.4f} tp_r_actual_L={tp_r_actual_L:.6f} sl_r_actual_L={sl_r_actual_L:.6f} other_r_actual_L={other_r_actual_L:.6f} cost_entry={cost_entry:.6f} evL={evL:.6f}")
                    print(f"[EV_DEBUG] {symbol} | h={h}s: tp_pS={tp_pS_val:.4f} sl_pS={sl_pS_val:.4f} p_other_S={p_other_S:.4f} tp_r_actual_S={tp_r_actual_S:.6f} sl_r_actual_S={sl_r_actual_S:.6f} other_r_actual_S={other_r_actual_S:.6f} cost_entry={cost_entry:.6f} evS={evS:.6f}")
                
                # ✅ LINK 심볼의 비정상적으로 큰 EV 값 검증
                if symbol.startswith("LINK"):
                    if abs(evL) > 0.1 or abs(evS) > 0.1:
                        logger.warning(f"[EV_VALIDATION_LINK] {symbol} | ⚠️  큰 EV 값 발견: evL={evL:.6f} evS={evS:.6f}")
                        logger.warning(f"[EV_VALIDATION_LINK] {symbol} | h={h}s: tp_r_actual_L={tp_r_actual_L:.6f} sl_r_actual_L={sl_r_actual_L:.6f} tp_r_actual_S={tp_r_actual_S:.6f} sl_r_actual_S={sl_r_actual_S:.6f}")
                        logger.warning(f"[EV_VALIDATION_LINK] {symbol} | h={h}s: cost_entry={cost_entry:.6f} p_tp_L={tp_pL_val:.4f} p_sl_L={sl_pL_val:.4f}")
                        print(f"[EV_VALIDATION_LINK] {symbol} | ⚠️  큰 EV 값 발견: evL={evL:.6f} evS={evS:.6f}")
                        print(f"[EV_VALIDATION_LINK] {symbol} | h={h}s: tp_r_actual_L={tp_r_actual_L:.6f} sl_r_actual_L={sl_r_actual_L:.6f} tp_r_actual_S={tp_r_actual_S:.6f} sl_r_actual_S={sl_r_actual_S:.6f}")
                        print(f"[EV_VALIDATION_LINK] {symbol} | h={h}s: cost_entry={cost_entry:.6f} p_tp_L={tp_pL_val:.4f} p_sl_L={sl_pL_val:.4f}")
                
                # ✅ 모든 심볼의 음수 EV 값 검증 (LINK 제외)
                if not symbol.startswith("LINK") and (evL < 0 or evS < 0):
                    # 모든 horizon에서 음수인지 확인하기 위해 로그만 출력 (너무 많으면 성능 문제)
                    if h == 60:  # 첫 번째 horizon만 상세 로그
                        # EV 계산 상세 분석
                        expected_evL = tp_pL_val * tp_r_actual_L + sl_pL_val * sl_r_actual_L + p_other_L * other_r_actual_L
                        expected_evS = tp_pS_val * tp_r_actual_S + sl_pS_val * sl_r_actual_S + p_other_S * other_r_actual_S
                        
                        # ✅ 각 항목별 기여도 분석
                        tp_contrib_L = tp_pL_val * tp_r_actual_L
                        sl_contrib_L = sl_pL_val * sl_r_actual_L
                        other_contrib_L = p_other_L * other_r_actual_L
                        tp_contrib_S = tp_pS_val * tp_r_actual_S
                        sl_contrib_S = sl_pS_val * sl_r_actual_S
                        other_contrib_S = p_other_S * other_r_actual_S
                        
                        # ✅ 원인 분석
                        causes = []
                        if cost_entry > abs(expected_evL) * 0.5:
                            causes.append(f"cost_entry({cost_entry:.6f})가 expected_evL({expected_evL:.6f})의 50% 이상")
                        if sl_contrib_L < -0.001:
                            causes.append(f"SL 기여도({sl_contrib_L:.6f})가 너무 음수 (p_sl={sl_pL_val:.4f}, sl_r={sl_r_actual_L:.6f})")
                        if tp_contrib_L < 0.001:
                            causes.append(f"TP 기여도({tp_contrib_L:.6f})가 너무 작음 (p_tp={tp_pL_val:.4f}, tp_r={tp_r_actual_L:.6f})")
                        if sl_pL_val > 0.5:
                            causes.append(f"SL 확률({sl_pL_val:.4f})이 50% 이상으로 너무 높음")
                        if tp_pL_val < 0.15:
                            causes.append(f"TP 확률({tp_pL_val:.4f})이 15% 미만으로 너무 낮음")
                        
                        logger.warning(f"[EV_VALIDATION_NEG] {symbol} | ⚠️  음수 EV 발견: evL={evL:.6f} evS={evS:.6f}")
                        logger.warning(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: cost_entry={cost_entry:.6f} (비용이 너무 큰가?)")
                        logger.warning(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: tp_pL={tp_pL_val:.4f} sl_pL={sl_pL_val:.4f} p_other_L={p_other_L:.4f}")
                        logger.warning(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: tp_r_actual_L={tp_r_actual_L:.6f} sl_r_actual_L={sl_r_actual_L:.6f} other_r_actual_L={other_r_actual_L:.6f}")
                        logger.warning(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: expected_evL(비용 제외)={expected_evL:.6f} expected_evS(비용 제외)={expected_evS:.6f}")
                        logger.warning(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: cost_entry 비율: evL에서 {abs(cost_entry/expected_evL*100) if expected_evL != 0 else 0:.1f}%, evS에서 {abs(cost_entry/expected_evS*100) if expected_evS != 0 else 0:.1f}%")
                        logger.warning(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: 기여도 분석 - TP={tp_contrib_L:.6f} SL={sl_contrib_L:.6f} Other={other_contrib_L:.6f} (Long)")
                        logger.warning(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: 기여도 분석 - TP={tp_contrib_S:.6f} SL={sl_contrib_S:.6f} Other={other_contrib_S:.6f} (Short)")
                        if causes:
                            logger.warning(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: 🔍 음수 EV 원인: {'; '.join(causes)}")
                        
                        print(f"[EV_VALIDATION_NEG] {symbol} | ⚠️  음수 EV 발견: evL={evL:.6f} evS={evS:.6f}")
                        print(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: cost_entry={cost_entry:.6f} (비용이 너무 큰가?)")
                        print(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: tp_pL={tp_pL_val:.4f} sl_pL={sl_pL_val:.4f} p_other_L={p_other_L:.4f}")
                        print(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: tp_r_actual_L={tp_r_actual_L:.6f} sl_r_actual_L={sl_r_actual_L:.6f} other_r_actual_L={other_r_actual_L:.6f}")
                        print(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: expected_evL(비용 제외)={expected_evL:.6f} expected_evS(비용 제외)={expected_evS:.6f}")
                        print(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: cost_entry 비율: evL에서 {abs(cost_entry/expected_evL*100) if expected_evL != 0 else 0:.1f}%, evS에서 {abs(cost_entry/expected_evS*100) if expected_evS != 0 else 0:.1f}%")
                        print(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: 기여도 분석 - TP={tp_contrib_L:.6f} SL={sl_contrib_L:.6f} Other={other_contrib_L:.6f} (Long)")
                        print(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: 기여도 분석 - TP={tp_contrib_S:.6f} SL={sl_contrib_S:.6f} Other={other_contrib_S:.6f} (Short)")
                        if causes:
                            print(f"[EV_VALIDATION_NEG] {symbol} | h={h}s: 🔍 음수 EV 원인: {'; '.join(causes)}")
            
            ev_long_h.append(evL)
            ev_short_h.append(evS)
            ppos_long_h.append(pposL)
            ppos_short_h.append(pposS)
            
            # Debug: Print per-horizon EV values
            print(f"[EV_PER_H_DEBUG] {symbol} | h={h}s: evL={evL:.6f} evS={evS:.6f} pposL={pposL:.4f} pposS={pposS:.4f}")
            
            # ✅ TP/SL/Other 확률 및 실제 수익률을 메타에 저장 (horizon별)
            if alpha_hit is None:
                # MC fallback: 직접 집계한 값 사용
                meta[f"policy_p_tp_per_h_{h}_long"] = p_tp_L
                meta[f"policy_p_sl_per_h_{h}_long"] = p_sl_L
                meta[f"policy_p_other_per_h_{h}_long"] = p_other_L
                meta[f"policy_tp_r_actual_per_h_{h}_long"] = tp_r_actual_L
                meta[f"policy_sl_r_actual_per_h_{h}_long"] = sl_r_actual_L
                meta[f"policy_other_r_actual_per_h_{h}_long"] = other_r_actual_L
                meta[f"policy_prob_sum_check_per_h_{h}_long"] = prob_sum_check_L
                
                meta[f"policy_p_tp_per_h_{h}_short"] = p_tp_S
                meta[f"policy_p_sl_per_h_{h}_short"] = p_sl_S
                meta[f"policy_p_other_per_h_{h}_short"] = p_other_S
                meta[f"policy_tp_r_actual_per_h_{h}_short"] = tp_r_actual_S
                meta[f"policy_sl_r_actual_per_h_{h}_short"] = sl_r_actual_S
                meta[f"policy_other_r_actual_per_h_{h}_short"] = other_r_actual_S
                meta[f"policy_prob_sum_check_per_h_{h}_short"] = prob_sum_check_S
            else:
                # AlphaHitMLP: 예측 확률과 실제 수익률 조합
                meta[f"policy_p_tp_per_h_{h}_long"] = tp_pL_val
                meta[f"policy_p_sl_per_h_{h}_long"] = sl_pL_val
                meta[f"policy_p_other_per_h_{h}_long"] = p_other_L
                meta[f"policy_tp_r_actual_per_h_{h}_long"] = tp_r_actual_L
                meta[f"policy_sl_r_actual_per_h_{h}_long"] = sl_r_actual_L
                meta[f"policy_other_r_actual_per_h_{h}_long"] = other_r_actual_L
                
                meta[f"policy_p_tp_per_h_{h}_short"] = tp_pS_val
                meta[f"policy_p_sl_per_h_{h}_short"] = sl_pS_val
                meta[f"policy_p_other_per_h_{h}_short"] = p_other_S
                meta[f"policy_tp_r_actual_per_h_{h}_short"] = tp_r_actual_S
                meta[f"policy_sl_r_actual_per_h_{h}_short"] = sl_r_actual_S
                meta[f"policy_other_r_actual_per_h_{h}_short"] = other_r_actual_S

            # [DIFF 6] Extract delay penalty and horizon effective from first horizon (they should be similar across horizons)
            if idx == 0:  # First horizon
                m_long_meta = m_long.get("meta", {}) or {}
                m_short_meta = m_short.get("meta", {}) or {}
                # Use long direction meta (or short if direction is short, but we'll use long for consistency)
                # Always extract even if 0 (for visibility in payload)
                meta["pmaker_entry_delay_penalty_r"] = float(m_long_meta.get("pmaker_entry_delay_penalty_r", 0.0))
                meta["pmaker_exit_delay_penalty_r"] = float(m_long_meta.get("pmaker_exit_delay_penalty_r", 0.0))
                shift_steps_raw = m_long_meta.get("policy_entry_shift_steps", 0)
                # Cap shift_steps to reasonable value (max 1000 steps = 5000 seconds with dt=5)
                meta["policy_entry_shift_steps"] = int(min(int(shift_steps_raw) if shift_steps_raw is not None else 0, 1000))
                meta["policy_horizon_eff_sec"] = int(m_long_meta.get("policy_horizon_eff_sec", int(h)))
                print(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: extracted from first horizon - entry_penalty={meta['pmaker_entry_delay_penalty_r']:.6f} exit_penalty={meta['pmaker_exit_delay_penalty_r']:.6f} shift_steps={meta['policy_entry_shift_steps']} horizon_eff={meta['policy_horizon_eff_sec']}")
                print(f"[PMAKER_DEBUG] {symbol} | m_long_meta keys={list(m_long_meta.keys())}")
                logger.info(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: extracted from first horizon - entry_penalty={meta['pmaker_entry_delay_penalty_r']:.6f} exit_penalty={meta['pmaker_exit_delay_penalty_r']:.6f} shift_steps={meta['policy_entry_shift_steps']} horizon_eff={meta['policy_horizon_eff_sec']}")
                logger.info(f"[PMAKER_DEBUG] {symbol} | m_long_meta keys={list(m_long_meta.keys())}")

        if self.alpha_hit_enabled and self.alpha_hit_trainer is not None and features_np is not None:
            try:
                y_tp_long = np.clip(np.asarray(mc_p_tp_long, dtype=np.float32), 0.0, 1.0)
                y_sl_long = np.clip(np.asarray(mc_p_sl_long, dtype=np.float32), 0.0, 1.0)
                y_tp_short = np.clip(np.asarray(mc_p_tp_short, dtype=np.float32), 0.0, 1.0)
                y_sl_short = np.clip(np.asarray(mc_p_sl_short, dtype=np.float32), 0.0, 1.0)
                if len(y_tp_long) == len(policy_horizons):
                    self.alpha_hit_trainer.add_sample(
                        x=features_np,
                        y={
                            "tp_long": y_tp_long,
                            "sl_long": y_sl_long,
                            "tp_short": y_tp_short,
                            "sl_short": y_sl_short,
                        },
                        ts_ms=int(time.time() * 1000),
                        symbol=symbol,
                    )
            except Exception as e:
                logger.warning(f"[ALPHA_HIT] Failed to add MC soft-label sample: {e}")
        
        # [D] Convert to numpy arrays
        ev_long_h = np.asarray(ev_long_h, dtype=np.float64)
        ev_short_h = np.asarray(ev_short_h, dtype=np.float64)
        ppos_long_h = np.asarray(ppos_long_h, dtype=np.float64)
        ppos_short_h = np.asarray(ppos_short_h, dtype=np.float64)
        
        # [D] Learned contribution -> w(h) uses EV_h (but keep rule-based half-life prior)
        w_prior_arr = np.asarray(w_prior, dtype=np.float64)
        # EV shaping: positive-only contribution (avoid negative EV horizon hijacking weight)
        beta = float(os.environ.get("POLICY_W_EV_BETA", getattr(self, "POLICY_W_EV_BETA", 200.0)))
        contrib_long = np.log1p(np.exp(ev_long_h * beta))
        contrib_short = np.log1p(np.exp(ev_short_h * beta))
        
        # ✅ [EV_VALIDATION_2] exit 분포 기반 페널티 계산 및 적용
        # 페널티: p(exit <= min_hold+Δ) 또는 exit_reason_counts(hold_bad/flip 비율) 기반
        min_hold_sec_val = float(self.MIN_HOLD_SEC_DIRECTIONAL)
        exit_early_delta_sec = float(os.environ.get("EXIT_EARLY_DELTA_SEC", "60.0"))  # 기본값 60초
        exit_early_penalty_k = float(os.environ.get("EXIT_EARLY_PENALTY_K", "0.5"))  # 기본값 0.5
        
        penalty_long = np.ones(len(per_h_long), dtype=np.float64)
        penalty_short = np.ones(len(per_h_short), dtype=np.float64)
        
        for i, (m_long_h, m_short_h) in enumerate(zip(per_h_long, per_h_short)):
            # Option 1: p(exit <= min_hold + Δ) 계산
            exit_t_long = m_long_h.get("exit_t")
            exit_t_short = m_short_h.get("exit_t")
            
            if exit_t_long is not None:
                exit_t_arr_long = np.asarray(exit_t_long, dtype=np.float64)
                if exit_t_arr_long.size > 0:
                    # p_early_exit = p(exit <= min_hold + Δ)
                    p_early_exit_long = float(np.mean(exit_t_arr_long <= (min_hold_sec_val + exit_early_delta_sec)))
                    penalty_long[i] = max(0.0, 1.0 - exit_early_penalty_k * p_early_exit_long)
            
            if exit_t_short is not None:
                exit_t_arr_short = np.asarray(exit_t_short, dtype=np.float64)
                if exit_t_arr_short.size > 0:
                    p_early_exit_short = float(np.mean(exit_t_arr_short <= (min_hold_sec_val + exit_early_delta_sec)))
                    penalty_short[i] = max(0.0, 1.0 - exit_early_penalty_k * p_early_exit_short)
            
            # Option 2: exit_reason_counts에서 hold_bad/flip 비율 계산 (보조 지표)
            exit_reason_counts_long = m_long_h.get("exit_reason_counts", {}) or {}
            exit_reason_counts_short = m_short_h.get("exit_reason_counts", {}) or {}
            
            total_exits_long = sum(int(v) for v in exit_reason_counts_long.values()) if exit_reason_counts_long else 1
            total_exits_short = sum(int(v) for v in exit_reason_counts_short.values()) if exit_reason_counts_short else 1
            
            if total_exits_long > 0:
                p_early_bad_long = float((exit_reason_counts_long.get("hold_bad", 0) + exit_reason_counts_long.get("score_flip", 0))) / float(total_exits_long)
                # Option 1과 Option 2 중 더 보수적인 값 사용 (더 낮은 penalty = 더 큰 페널티)
                penalty_long[i] = min(penalty_long[i], max(0.0, 1.0 - exit_early_penalty_k * p_early_bad_long))
            
            if total_exits_short > 0:
                p_early_bad_short = float((exit_reason_counts_short.get("hold_bad", 0) + exit_reason_counts_short.get("score_flip", 0))) / float(total_exits_short)
                penalty_short[i] = min(penalty_short[i], max(0.0, 1.0 - exit_early_penalty_k * p_early_bad_short))
        
        penalty_long = np.asarray(penalty_long, dtype=np.float64)
        penalty_short = np.asarray(penalty_short, dtype=np.float64)
        
        # ✅ [EV_VALIDATION_2] 페널티를 w(h)에 적용: w(h) = w_prior(h) * contrib(EV_h) * penalty
        w_long = w_prior_arr * contrib_long * penalty_long
        w_short = w_prior_arr * contrib_short * penalty_short
        w_long = w_long / (w_long.sum() + 1e-12)
        w_short = w_short / (w_short.sum() + 1e-12)
        
        # ✅ [EV_VALIDATION_2] 페널티 로그
        logger.info(f"[EV_VALIDATION_2] {symbol} | exit_early_penalty: penalty_long={penalty_long.tolist()} penalty_short={penalty_short.tolist()} (k={exit_early_penalty_k}, delta={exit_early_delta_sec}s)")
        print(f"[EV_VALIDATION_2] {symbol} | exit_early_penalty: penalty_long={penalty_long.tolist()} penalty_short={penalty_short.tolist()} (k={exit_early_penalty_k}, delta={exit_early_delta_sec}s)")
        
        # [D] Mix EV using learned weights
        # ✅ [EV_DEBUG] 가중치 및 EV 배열 확인
        logger.info(f"[EV_DEBUG] {symbol} | Before mix: ev_long_h={ev_long_h.tolist()} ev_short_h={ev_short_h.tolist()}")
        logger.info(f"[EV_DEBUG] {symbol} | Before mix: w_long={w_long.tolist()} w_short={w_short.tolist()}")
        logger.info(f"[EV_DEBUG] {symbol} | Before mix: evs_long={evs_long} evs_short={evs_short}")
        print(f"[EV_DEBUG] {symbol} | Before mix: ev_long_h={ev_long_h.tolist()} ev_short_h={ev_short_h.tolist()}")
        print(f"[EV_DEBUG] {symbol} | Before mix: w_long={w_long.tolist()} w_short={w_short.tolist()}")
        print(f"[EV_DEBUG] {symbol} | Before mix: evs_long={evs_long} evs_short={evs_short}")
        
        policy_ev_mix_long = float((w_long * ev_long_h).sum())
        policy_ev_mix_short = float((w_short * ev_short_h).sum())
        policy_p_pos_mix_long = float((w_long * ppos_long_h).sum())
        policy_p_pos_mix_short = float((w_short * ppos_short_h).sum())
        
        # ✅ [EV_DEBUG] policy_ev_mix 계산 결과 로그
        print(f"[EV_DEBUG] {symbol} | policy_ev_mix calculation: ev_long_h={ev_long_h.tolist()} ev_short_h={ev_short_h.tolist()}")
        print(f"[EV_DEBUG] {symbol} | policy_ev_mix calculation: w_long={w_long.tolist()} w_short={w_short.tolist()}")
        print(f"[EV_DEBUG] {symbol} | policy_ev_mix calculation: policy_ev_mix_long={policy_ev_mix_long:.8f} policy_ev_mix_short={policy_ev_mix_short:.8f}")
        logger.info(f"[EV_DEBUG] {symbol} | policy_ev_mix calculation: policy_ev_mix_long={policy_ev_mix_long:.8f} policy_ev_mix_short={policy_ev_mix_short:.8f}")
        
        # ✅ 모든 심볼의 policy_ev_mix가 음수인 경우 검증 (LINK 제외)
        if not symbol.startswith("LINK") and policy_ev_mix_long < 0 and policy_ev_mix_short < 0:
            logger.warning(f"[EV_VALIDATION_NEG] {symbol} | ⚠️  policy_ev_mix가 모두 음수: long={policy_ev_mix_long:.6f} short={policy_ev_mix_short:.6f}")
            logger.warning(f"[EV_VALIDATION_NEG] {symbol} | ev_long_h={ev_long_h.tolist()} ev_short_h={ev_short_h.tolist()}")
            logger.warning(f"[EV_VALIDATION_NEG] {symbol} | w_long={w_long.tolist()} w_short={w_short.tolist()}")
            print(f"[EV_VALIDATION_NEG] {symbol} | ⚠️  policy_ev_mix가 모두 음수: long={policy_ev_mix_long:.6f} short={policy_ev_mix_short:.6f}")
            print(f"[EV_VALIDATION_NEG] {symbol} | ev_long_h={ev_long_h.tolist()} ev_short_h={ev_short_h.tolist()}")
            print(f"[EV_VALIDATION_NEG] {symbol} | w_long={w_long.tolist()} w_short={w_short.tolist()}")
        
        # ✅ 모든 심볼의 policy_ev_mix 통계 (LINK 제외)
        if not symbol.startswith("LINK"):
            # horizon별 EV 값이 모두 음수인지 확인
            all_negative_long = all(ev < 0 for ev in ev_long_h) if len(ev_long_h) > 0 else False
            all_negative_short = all(ev < 0 for ev in ev_short_h) if len(ev_short_h) > 0 else False
            
            if all_negative_long and all_negative_short:
                logger.warning(f"[EV_VALIDATION_NEG] {symbol} | ⚠️  모든 horizon에서 EV가 음수: ev_long_h={ev_long_h.tolist()} ev_short_h={ev_short_h.tolist()}")
                print(f"[EV_VALIDATION_NEG] {symbol} | ⚠️  모든 horizon에서 EV가 음수: ev_long_h={ev_long_h.tolist()} ev_short_h={ev_short_h.tolist()}")
        
        # ✅ [EV_DEBUG] policy_ev_mix 계산 결과 로그
        logger.info(f"[EV_DEBUG] {symbol} | policy_ev_mix_long={policy_ev_mix_long:.6f} policy_ev_mix_short={policy_ev_mix_short:.6f}")
        print(f"[EV_DEBUG] {symbol} | policy_ev_mix_long={policy_ev_mix_long:.6f} policy_ev_mix_short={policy_ev_mix_short:.6f}")
        
        # ✅ [EV_DEBUG] policy_ev_mix가 0인 경우 원인 파악
        if abs(policy_ev_mix_long) < 1e-6 and abs(policy_ev_mix_short) < 1e-6:
            logger.warning(f"[EV_DEBUG] {symbol} | ⚠️  policy_ev_mix_long and policy_ev_mix_short are both near 0!")
            logger.warning(f"[EV_DEBUG] {symbol} | ev_long_h={ev_long_h.tolist()} ev_short_h={ev_short_h.tolist()}")
            logger.warning(f"[EV_DEBUG] {symbol} | w_long={w_long.tolist()} w_short={w_short.tolist()}")
            print(f"[EV_DEBUG] {symbol} | ⚠️  policy_ev_mix_long and policy_ev_mix_short are both near 0!")
            print(f"[EV_DEBUG] {symbol} | ev_long_h={ev_long_h.tolist()} ev_short_h={ev_short_h.tolist()}")
            print(f"[EV_DEBUG] {symbol} | w_long={w_long.tolist()} w_short={w_short.tolist()}")
        
        # [D] Direction choice based on EV gap constraint (>=min_gap)
        min_gap = float(os.environ.get("POLICY_MIN_EV_GAP", getattr(self, "POLICY_MIN_EV_GAP", 0.005)))
        ev_gap = policy_ev_mix_long - policy_ev_mix_short
        p_pos_gap = policy_p_pos_mix_long - policy_p_pos_mix_short
        # ✅ [EV_DEBUG] direction_policy 결정 로그
        logger.info(f"[EV_DEBUG] {symbol} | ev_gap={ev_gap:.6f} min_gap={min_gap:.6f} abs(ev_gap)={abs(ev_gap):.6f}")
        print(f"[EV_DEBUG] {symbol} | ev_gap={ev_gap:.6f} min_gap={min_gap:.6f} abs(ev_gap)={abs(ev_gap):.6f}")
        
        # ✅ [EV_VALIDATION 4] SHORT 쪽 EV가 상대적으로 더 좋은데 long 편향 검증
        if ev_gap < 0 and abs(ev_gap) >= min_gap:
            # SHORT이 더 좋은데 LONG이 선택될 수 있는 경우
            logger.warning(f"[EV_VALIDATION_4] {symbol} | SHORT is better: ev_gap={ev_gap:.6f} (long={policy_ev_mix_long:.6f} short={policy_ev_mix_short:.6f}), but may choose LONG due to gap constraint")
            print(f"[EV_VALIDATION_4] {symbol} | SHORT is better: ev_gap={ev_gap:.6f} (long={policy_ev_mix_long:.6f} short={policy_ev_mix_short:.6f}), p_pos_gap={p_pos_gap:.6f}")
        elif ev_gap > 0 and abs(ev_gap) < min_gap and policy_ev_mix_short > policy_ev_mix_long:
            # gap이 작아서 no trade인데 실제로는 SHORT가 더 좋은 경우
            logger.warning(f"[EV_VALIDATION_4] {symbol} | No trade due to small gap={ev_gap:.6f}, but SHORT={policy_ev_mix_short:.6f} > LONG={policy_ev_mix_long:.6f}")
            print(f"[EV_VALIDATION_4] {symbol} | No trade due to small gap={ev_gap:.6f}, but SHORT={policy_ev_mix_short:.6f} > LONG={policy_ev_mix_long:.6f}, p_pos_gap={p_pos_gap:.6f}")
        
        if abs(ev_gap) < min_gap:
            direction_policy = 0  # no trade
            logger.warning(f"[EV_DEBUG] {symbol} | direction_policy=0 (no trade) because abs(ev_gap)={abs(ev_gap):.6f} < min_gap={min_gap:.6f}")
            print(f"[EV_DEBUG] {symbol} | direction_policy=0 (no trade) because abs(ev_gap)={abs(ev_gap):.6f} < min_gap={min_gap:.6f}")
        else:
            direction_policy = 1 if ev_gap > 0 else -1
            logger.info(f"[EV_DEBUG] {symbol} | direction_policy={direction_policy} (ev_gap={ev_gap:.6f})")
            print(f"[EV_DEBUG] {symbol} | direction_policy={direction_policy} (ev_gap={ev_gap:.6f})")
        
        # [D] Overwrite the decision-driving fields to use new mix
        policy_ev_mix = policy_ev_mix_long if direction_policy == 1 else policy_ev_mix_short
        policy_p_pos_mix = policy_p_pos_mix_long if direction_policy == 1 else policy_p_pos_mix_short
        
        # ✅ [EV_DEBUG] policy_ev_mix 설정 로그
        logger.info(f"[EV_DEBUG] {symbol} | policy_ev_mix set to {policy_ev_mix:.6f} (direction_policy={direction_policy}, long={policy_ev_mix_long:.6f}, short={policy_ev_mix_short:.6f})")
        print(f"[EV_DEBUG] {symbol} | policy_ev_mix set to {policy_ev_mix:.6f} (direction_policy={direction_policy}, long={policy_ev_mix_long:.6f}, short={policy_ev_mix_short:.6f})")
        
        # Use MC simulation CVaR for risk gate (can be refined later)
        w_arr_for_cvar = w_long if direction_policy == 1 else w_short
        cvars_arr = np.asarray(cvars_long if direction_policy == 1 else cvars_short, dtype=np.float64)
        policy_cvar_mix = float((w_arr_for_cvar * cvars_arr).sum()) if cvars_arr.size > 0 else 0.0
        
        # [D] Expose per-horizon and weights in meta (meta already initialized before horizon loop)
        meta["policy_ev_by_h_long"] = ev_long_h.tolist()
        meta["policy_ev_by_h_short"] = ev_short_h.tolist()
        meta["policy_w_h_long"] = w_long.tolist()
        meta["policy_w_h_short"] = w_short.tolist()
        meta["policy_ev_mix_long"] = policy_ev_mix_long
        meta["policy_ev_mix_short"] = policy_ev_mix_short
        meta["policy_p_pos_mix_long"] = policy_p_pos_mix_long
        meta["policy_p_pos_mix_short"] = policy_p_pos_mix_short
        meta["policy_direction"] = direction_policy
        meta["policy_min_ev_gap"] = min_gap
        meta["policy_ev_gap"] = float(ev_gap)
        meta["policy_p_pos_gap"] = float(p_pos_gap)
        
        # Store validation warnings in meta
        if mu_alpha_warning:
            meta["ev_validation_1_warnings"] = mu_alpha_warning
        # cost_warning will be added to meta later after it's computed (around line 2670)
        
        # [C] Cost diagnostics (maker delay integrated)
        meta["pmaker_entry"] = pmaker_entry_local
        meta["pmaker_entry_delay_sec"] = pmaker_delay_sec_local
        meta["pmaker_delay_penalty"] = delay_penalty
        meta["policy_cost_entry"] = cost_entry  # ✅ [EV_VALIDATION_3] entry 비용만 저장 (exit은 compute_exit_policy_metrics에서 처리)
        meta["sl_r_used"] = sl_r
        meta["fee_roundtrip_total"] = float(fee_roundtrip_total)
        if 'gross_ev_approx_10min' in locals():
            meta["gross_ev_approx_10min"] = float(gross_ev_approx_10min)
        
        # [D] Compute policy_h_eff_sec and policy_w_short_sum from learned weights
        policy_h_eff_sec = 0.0
        policy_w_short_sum = 0.0
        policy_h_eff_sec_prior = 0.0  # For validation: w_prior only (rule-based)
        try:
            h_arr_pol = np.asarray(policy_horizons, dtype=np.float64)
            w_arr_pol = w_long if direction_policy == 1 else w_short
            policy_h_eff_sec = float(np.sum(w_arr_pol * h_arr_pol)) if w_arr_pol.size else 0.0
            policy_w_short_sum = float(np.sum(w_arr_pol[h_arr_pol <= 300.0])) if w_arr_pol.size else 0.0
            # Compute policy_h_eff_sec from w_prior only (for validation)
            if w_prior_arr.size > 0 and h_arr_pol.size == w_prior_arr.size:
                policy_h_eff_sec_prior = float(np.sum(w_prior_arr * h_arr_pol))
        except Exception:
            pass  # Already initialized to 0.0
        
        # [DIFF 3 VALIDATION] Validate dynamic weight behavior
        # Validation point 1: signal_strength ↑ → policy_h_eff_sec ↓
        # Check: w_prior-based policy_h_eff_sec should decrease as signal_strength increases
        # (Note: Final weights include EV contribution, so we validate w_prior separately)
        validation_warning = []
        if policy_h_eff_sec_prior > 0.0:
            # Expected: higher signal_strength → lower half_life → lower policy_h_eff_sec_prior
            # half_life = 1800.0 / (1.0 + s_clip)
            # For s_clip=0: half_life=1800, for s_clip=4: half_life=360
            # Higher s_clip → lower half_life → more weight on short horizons → lower policy_h_eff_sec_prior
            expected_half_life = policy_half_life_sec
            if expected_half_life > 0.0:
                # Sanity check: policy_h_eff_sec_prior should be reasonable given half_life
                # For exponential decay, effective horizon ≈ half_life * ln(2) for uniform distribution
                # But with multiple horizons, it's more complex. Just log for monitoring.
                pass  # Will be logged in meta
        
        # Validation point 2: policy_w_short_sum should be in [0, 1]
        if policy_w_short_sum < 0.0 or policy_w_short_sum > 1.0:
            validation_warning.append(
                f"policy_w_short_sum={policy_w_short_sum:.6f} is out of range [0, 1]"
            )
        
        # Log validation warnings
        if validation_warning:
            logger.warning(
                f"[DIFF3_VALIDATION] {symbol} | signal_strength={signal_strength:.4f} "
                f"policy_h_eff_sec={policy_h_eff_sec:.2f} policy_h_eff_sec_prior={policy_h_eff_sec_prior:.2f} "
                f"policy_w_short_sum={policy_w_short_sum:.6f} | Warnings: {'; '.join(validation_warning)}"
            )
        
        # Store validation metrics in meta
        meta["policy_h_eff_sec_prior"] = float(policy_h_eff_sec_prior)
        if validation_warning:
            meta["diff3_validation_warnings"] = validation_warning

        # [DIFF 2 VALIDATION] Validate Multi-Horizon Policy Mix
        diff2_validation_warnings = []
        
        # Validation point 1: policy_w_h normalization (sum should be ~1.0)
        w_arr = w_long if direction_policy == 1 else w_short
        w_arr_sum = float(np.sum(w_arr)) if w_arr.size > 0 else 0.0
        if abs(w_arr_sum - 1.0) > 1e-5:
            diff2_validation_warnings.append(
                f"policy_w_h sum={w_arr_sum:.8f} is not ~1.0 (expected: 1.0)"
            )
        
        # Validation point 2: policy_ev_mix_long/short calculation
        # Verify: policy_ev_mix_long = sum(w_long[i] * ev_long_h[i])
        if ev_long_h.size > 0 and w_long.size == ev_long_h.size:
            ev_mix_long_manual = float((w_long * ev_long_h).sum())
            ev_mix_long_diff = abs(ev_mix_long_manual - policy_ev_mix_long)
            if ev_mix_long_diff > 1e-6:
                diff2_validation_warnings.append(
                    f"policy_ev_mix_long calculation mismatch: computed={policy_ev_mix_long:.8f} "
                    f"manual={ev_mix_long_manual:.8f} diff={ev_mix_long_diff:.8f}"
                )
        
        if ev_short_h.size > 0 and w_short.size == ev_short_h.size:
            ev_mix_short_manual = float((w_short * ev_short_h).sum())
            ev_mix_short_diff = abs(ev_mix_short_manual - policy_ev_mix_short)
            if ev_mix_short_diff > 1e-6:
                diff2_validation_warnings.append(
                    f"policy_ev_mix_short calculation mismatch: computed={policy_ev_mix_short:.8f} "
                    f"manual={ev_mix_short_manual:.8f} diff={ev_mix_short_diff:.8f}"
                )
        
        # Validation point 3: policy_horizons should be [60, 180, 300, 600, 900, 1800]
        expected_horizons = [60, 180, 300, 600, 900, 1800]
        if list(policy_horizons) != expected_horizons:
            diff2_validation_warnings.append(
                f"policy_horizons={list(policy_horizons)} != expected {expected_horizons}"
            )
        
        # Validation point 4: policy_w_h length should match policy_horizons length
        if len(w_arr) != len(policy_horizons):
            diff2_validation_warnings.append(
                f"policy_w_h length={len(w_arr)} != policy_horizons length={len(policy_horizons)}"
            )
        
        # Log validation warnings
        if diff2_validation_warnings:
            logger.warning(
                f"[DIFF2_VALIDATION] {symbol} | policy_horizons={list(policy_horizons)} "
                f"policy_w_h_sum={w_arr_sum:.8f} policy_ev_mix_long={policy_ev_mix_long:.8f} "
                f"policy_ev_mix_short={policy_ev_mix_short:.8f} paths_reused={paths_reused} | "
                f"Warnings: {'; '.join(diff2_validation_warnings)}"
            )
        
        # Store validation metrics in meta
        if diff2_validation_warnings:
            meta["diff2_validation_warnings"] = diff2_validation_warnings
        meta["policy_w_h_sum"] = float(w_arr_sum)  # For validation

        # [D] Use TP/SL 확률 기반 EV for decision (already computed above)
        # MC simulation results are kept for meta/diagnostics only
        evs_long_arr = np.asarray(evs_long, dtype=np.float64)  # MC simulation results (for meta)
        evs_short_arr = np.asarray(evs_short, dtype=np.float64)
        pps_long_arr = np.asarray(pps_long, dtype=np.float64)
        pps_short_arr = np.asarray(pps_short, dtype=np.float64)
        cvars_long_arr = np.asarray(cvars_long, dtype=np.float64)
        cvars_short_arr = np.asarray(cvars_short, dtype=np.float64)

        # [D] Legacy MC simulation mix (for meta/diagnostics only)
        ev_mix_long = float(np.sum(w_arr * evs_long_arr)) if direction_policy == 1 else float(np.sum(w_arr * evs_short_arr))
        ev_mix_short = float(np.sum(w_arr * evs_short_arr)) if direction_policy == -1 else float(np.sum(w_arr * evs_long_arr))
        ppos_mix_long = float(np.sum(w_arr * pps_long_arr)) if direction_policy == 1 else float(np.sum(w_arr * pps_short_arr))
        ppos_mix_short = float(np.sum(w_arr * pps_short_arr)) if direction_policy == -1 else float(np.sum(w_arr * pps_long_arr))
        cvar_mix_long = float(np.sum(w_arr * cvars_long_arr)) if direction_policy == 1 else float(np.sum(w_arr * cvars_short_arr))
        cvar_mix_short = float(np.sum(w_arr * cvars_short_arr)) if direction_policy == -1 else float(np.sum(w_arr * cvars_long_arr))

        # -----------------------------
        # EV model choice:
        # - legacy: "exit_policy" (compute_exit_policy_metrics rollforward)
        # - unified exit mode: approximate with "hold" EV so entry/exit stay consistent
        # -----------------------------
        exit_mode = str(os.environ.get("EXIT_MODE", "") or ctx.get("exit_mode", "")).strip().lower()
        use_hold_ev = exit_mode in ("unified", "entry", "entry_like", "symmetric")

        # Pre-compute hold aggregates for both directions (from net_by_h_* summaries).
        def _agg_from_dbg(dbg):
            ev_list0, win_list0, cvar_list0, h_list0 = dbg
            if not h_list0:
                return 0.0, 0.0, 0.0
            h_arr0 = np.asarray(h_list0, dtype=np.float64)
            w0 = np.exp(-h_arr0 * math.log(2) / 120.0)
            w0 = w0 / max(1e-12, float(np.sum(w0)))
            evs0 = np.asarray(ev_list0, dtype=np.float64)
            wins0 = np.asarray(win_list0, dtype=np.float64)
            cvars0 = np.asarray(cvar_list0, dtype=np.float64)
            ev_agg0 = float(np.sum(w0 * evs0))
            win_agg0 = float(np.sum(w0 * wins0))
            cvar_agg0 = float(np.quantile(cvars0, 0.25)) if cvars0.size else 0.0
            return ev_agg0, win_agg0, cvar_agg0

        hold_ev_mix_long, hold_win_mix_long, hold_cvar_mix_long = _agg_from_dbg(dbg_L)
        hold_ev_mix_short, hold_win_mix_short, hold_cvar_mix_short = _agg_from_dbg(dbg_S)
        # ✅ direction_policy=0일 때도 cvars_best를 정의 (더 나은 방향 사용)
        if direction_policy == 1:
            per_h_best = per_h_long
            evs_best = evs_long
            pps_best = pps_long
            cvars_best = cvars_long
        elif direction_policy == -1:
            per_h_best = per_h_short
            evs_best = evs_short
            pps_best = pps_short
            cvars_best = cvars_short
        else:  # direction_policy == 0
            # 더 나은 방향 선택 (더 큰 EV)
            if policy_ev_mix_long > policy_ev_mix_short:
                per_h_best = per_h_long
                evs_best = evs_long
                pps_best = pps_long
                cvars_best = cvars_long
            else:
                per_h_best = per_h_short
                evs_best = evs_short
                pps_best = pps_short
                cvars_best = cvars_short
        exit_reason_counts_per_h_long = [
            self._compress_reason_counts(m.get("exit_reason_counts"), top_k=3) for m in per_h_long
        ]
        exit_reason_counts_per_h_short = [
            self._compress_reason_counts(m.get("exit_reason_counts"), top_k=3) for m in per_h_short
        ]
        exit_reason_counts_per_h_best = exit_reason_counts_per_h_long if direction_policy == 1 else exit_reason_counts_per_h_short
        exit_t_mean_per_h_long = [float(m.get("exit_t_mean_sec", 0.0) or 0.0) for m in per_h_long]
        exit_t_mean_per_h_short = [float(m.get("exit_t_mean_sec", 0.0) or 0.0) for m in per_h_short]
        exit_t_p50_per_h_long = [float(m.get("exit_t_p50_sec", 0.0) or 0.0) for m in per_h_long]
        exit_t_p50_per_h_short = [float(m.get("exit_t_p50_sec", 0.0) or 0.0) for m in per_h_short]
        exit_t_mean_per_h_best = exit_t_mean_per_h_long if direction_policy == 1 else exit_t_mean_per_h_short
        exit_t_p50_per_h_best = exit_t_p50_per_h_long if direction_policy == 1 else exit_t_p50_per_h_short
        
        # ✅ [EV_VALIDATION 2] exit이 min_hold 근처에서 반복적으로 발생 검증
        min_hold_sec_val = float(self.MIN_HOLD_SEC_DIRECTIONAL)
        exit_t_mean_avg_long = float(np.mean(exit_t_mean_per_h_long)) if exit_t_mean_per_h_long else 0.0
        exit_t_mean_avg_short = float(np.mean(exit_t_mean_per_h_short)) if exit_t_mean_per_h_short else 0.0
        exit_t_mean_avg_best = exit_t_mean_avg_long if direction_policy == 1 else exit_t_mean_avg_short
        if exit_t_mean_avg_best > 0:
            hold_ratio = exit_t_mean_avg_best / max(1.0, min_hold_sec_val)
            if 0.9 <= hold_ratio <= 1.2:  # min_hold의 90%~120% 범위
                logger.warning(f"[EV_VALIDATION_2] {symbol} | exit_t_mean_avg={exit_t_mean_avg_best:.1f}s is near min_hold={min_hold_sec_val:.1f}s (ratio={hold_ratio:.2f}) - frequent early exits")
                print(f"[EV_VALIDATION_2] {symbol} | exit_t_mean_avg={exit_t_mean_avg_best:.1f}s is near min_hold={min_hold_sec_val:.1f}s (ratio={hold_ratio:.2f}) - frequent early exits")
        exit_reason_counts_policy = per_h_best[-1].get("exit_reason_counts") if per_h_best else {}
        weight_peak_h = int(policy_horizons[int(np.argmax(w_arr))]) if policy_horizons else 0
        best_h = int(max_policy_h)

        # [D] Drive return values from TP/SL 확률 기반 policy mix (not hold EV)
        # ✅ [EV_VALIDATION_4 FIX] 방향 선택 및 게이트는 항상 policy_ev_mix 기반으로 고정
        # use_hold_ev는 backward compatibility를 위해 유지되지만, 실제 방향 결정이나 게이트에는 사용되지 않음
        # hold_ev_mix는 메타/진단용으로만 사용됨
        logger.info(f"[EV_DEBUG] {symbol} | use_hold_ev={use_hold_ev} exit_mode={exit_mode} (policy_ev_mix 기반으로 고정)")
        print(f"[EV_DEBUG] {symbol} | use_hold_ev={use_hold_ev} exit_mode={exit_mode} (policy_ev_mix 기반으로 고정)")
        
        # ✅ [EV_VALIDATION_4] 방향 선택은 항상 policy_ev_mix 기반으로 고정 (use_hold_ev 무관)
        direction = direction_policy
        
        # ✅ [EV_VALIDATION_4] direction_policy=0일 때도 policy_ev_mix 값을 반환 (대시보드에서 표시하기 위해)
        # can_enter=False로 설정되어 거래는 하지 않지만, EV 값은 표시됨
        if direction_policy == 0:
            # direction_policy=0이면 더 나은 방향의 policy_ev_mix 사용 (더 큰 값, 즉 덜 나쁜 쪽)
            if policy_ev_mix_long > policy_ev_mix_short:
                ev = float(policy_ev_mix_long)
                win = float(policy_p_pos_mix_long)
            else:
                ev = float(policy_ev_mix_short)
                win = float(policy_p_pos_mix_short)
            # CVaR는 보수적으로 더 나쁜 쪽 사용
            cvar_gate = float(min(cvars_best)) if cvars_best else float(policy_cvar_mix)
            cvar = float(cvar_gate)
            logger.info(f"[EV_DEBUG] {symbol} | direction_policy=0: using better policy_ev_mix={ev:.6f} win={win:.4f} cvar={cvar:.6f} (no trade, but ev shown)")
            print(f"[EV_DEBUG] {symbol} | direction_policy=0: using better policy_ev_mix={ev:.6f} win={win:.4f} cvar={cvar:.6f} (no trade, but ev shown)")
        else:
            # ✅ [EV_VALIDATION_4] EV, win, CVaR는 항상 policy 기반 값 사용
            ev = float(policy_ev_mix)
            win = float(policy_p_pos_mix)
            # ✅ [EV_VALIDATION_4] CVaR 게이트는 policy_cvar_mix 사용 (보수적으로 최소값 사용)
            cvar_gate = float(min(cvars_best)) if cvars_best else float(policy_cvar_mix)
            cvar = float(cvar_gate)
            
            # ✅ [EV_DEBUG] 최종 ev 값 확인
            logger.info(f"[EV_DEBUG] {symbol} | Final ev={ev:.6f} (policy_ev_mix={policy_ev_mix:.6f}) win={win:.4f} cvar={cvar:.6f} direction={direction}")
            print(f"[EV_DEBUG] {symbol} | Final ev={ev:.6f} (policy_ev_mix={policy_ev_mix:.6f}) win={win:.4f} cvar={cvar:.6f} direction={direction}")
            
            # ✅ [EV_DEBUG] ev가 0인 경우 경고
            if abs(ev) < 1e-6:
                logger.warning(f"[EV_DEBUG] {symbol} | ⚠️  Final ev is near 0! policy_ev_mix={policy_ev_mix:.6f} direction_policy={direction_policy}")
                print(f"[EV_DEBUG] {symbol} | ⚠️  Final ev is near 0! policy_ev_mix={policy_ev_mix:.6f} direction_policy={direction_policy}")
        
        # exit-policy metrics는 항상 policy 기반 (direction_policy 기준)
        exit_t_mean_sec = float(per_h_best[-1].get("exit_t_mean_sec", 0.0)) if per_h_best else 0.0
        exit_reason_counts_policy = per_h_best[-1].get("exit_reason_counts") if per_h_best else {}
        
        # ✅ [EV_DEBUG] 최종 ev 결정 로그
        logger.info(f"[EV_DEBUG] {symbol} | Final ev decision: direction={direction} ev={ev:.6f} policy_ev_mix={policy_ev_mix:.6f} direction_policy={direction_policy} (policy 기반 고정)")
        print(f"[EV_DEBUG] {symbol} | Final ev decision: direction={direction} ev={ev:.6f} policy_ev_mix={policy_ev_mix:.6f} direction_policy={direction_policy} (policy 기반 고정)")
        
        # ✅ [EV_VALIDATION_4] hold_ev_mix는 메타/진단용으로만 저장 (방향 결정이나 게이트에 사용하지 않음)
        # hold_ev_mix 값은 이미 계산되어 메타에 저장됨 (아래에서 저장)

        # Exit-policy diagnostics: which rule dominates exits?
        policy_exit_unrealized_dd_frac = None
        policy_exit_hold_bad_frac = None
        policy_exit_score_flip_frac = None
        try:
            cnt = exit_reason_counts_policy or {}
            tot = float(sum(int(v) for v in cnt.values())) if isinstance(cnt, dict) else 0.0
            if tot > 0:
                policy_exit_unrealized_dd_frac = float(cnt.get("unrealized_dd", 0)) / tot
                policy_exit_hold_bad_frac = float(cnt.get("hold_bad", 0)) / tot
                policy_exit_score_flip_frac = float(cnt.get("score_flip", 0)) / tot
        except Exception:
            policy_exit_unrealized_dd_frac = None
            policy_exit_hold_bad_frac = None
            policy_exit_score_flip_frac = None
        
        # [DIFF 4 VALIDATION] Validate Exit Reason statistics
        diff4_validation_warnings = []
        
        # Validation point 1: exit reason counts structure
        # Check that exit_reason_counts_per_h_long/short are lists with correct length
        if exit_reason_counts_per_h_long is not None:
            if not isinstance(exit_reason_counts_per_h_long, list):
                diff4_validation_warnings.append(
                    f"exit_reason_counts_per_h_long is not a list: {type(exit_reason_counts_per_h_long)}"
                )
            elif len(exit_reason_counts_per_h_long) != len(policy_horizons):
                diff4_validation_warnings.append(
                    f"exit_reason_counts_per_h_long length={len(exit_reason_counts_per_h_long)} != "
                    f"policy_horizons length={len(policy_horizons)}"
                )
        
        if exit_reason_counts_per_h_short is not None:
            if not isinstance(exit_reason_counts_per_h_short, list):
                diff4_validation_warnings.append(
                    f"exit_reason_counts_per_h_short is not a list: {type(exit_reason_counts_per_h_short)}"
                )
            elif len(exit_reason_counts_per_h_short) != len(policy_horizons):
                diff4_validation_warnings.append(
                    f"exit_reason_counts_per_h_short length={len(exit_reason_counts_per_h_short)} != "
                    f"policy_horizons length={len(policy_horizons)}"
                )
        
        # Validation point 2: min_hold_sec should be ≈180s
        min_hold_sec_expected = 180
        min_hold_sec_actual = int(self.MIN_HOLD_SEC_DIRECTIONAL)
        if abs(min_hold_sec_actual - min_hold_sec_expected) > 10:  # Allow 10s tolerance
            diff4_validation_warnings.append(
                f"min_hold_sec={min_hold_sec_actual} is not ≈{min_hold_sec_expected}s"
            )
        
        # Validation point 3: Check exit reason fractions sum to reasonable value
        # (They should be fractions of total exits, so sum should be <= 1.0)
        if policy_exit_unrealized_dd_frac is not None and policy_exit_hold_bad_frac is not None and policy_exit_score_flip_frac is not None:
            frac_sum = policy_exit_unrealized_dd_frac + policy_exit_hold_bad_frac + policy_exit_score_flip_frac
            if frac_sum > 1.0 + 1e-5:  # Allow small floating point error
                diff4_validation_warnings.append(
                    f"Exit reason fractions sum={frac_sum:.6f} > 1.0 "
                    f"(unrealized_dd={policy_exit_unrealized_dd_frac:.6f} "
                    f"hold_bad={policy_exit_hold_bad_frac:.6f} "
                    f"score_flip={policy_exit_score_flip_frac:.6f})"
                )
        
        # Validation point 4: Check that exit times respect min_hold_sec
        # (This is a soft check - we can't enforce it strictly as some exits may be before min_hold)
        # But we can log if there are many early exits
        if exit_t_mean_per_h_long and exit_t_mean_per_h_short:
            min_hold_sec_float = float(min_hold_sec_actual)
            early_exits_long = sum(1 for t in exit_t_mean_per_h_long if t < min_hold_sec_float)
            early_exits_short = sum(1 for t in exit_t_mean_per_h_short if t < min_hold_sec_float)
            if early_exits_long > len(exit_t_mean_per_h_long) * 0.5:  # More than 50% early exits
                diff4_validation_warnings.append(
                    f"Many early exits in LONG: {early_exits_long}/{len(exit_t_mean_per_h_long)} "
                    f"exits before min_hold_sec={min_hold_sec_actual}s"
                )
            if early_exits_short > len(exit_t_mean_per_h_short) * 0.5:
                diff4_validation_warnings.append(
                    f"Many early exits in SHORT: {early_exits_short}/{len(exit_t_mean_per_h_short)} "
                    f"exits before min_hold_sec={min_hold_sec_actual}s"
                )
        
        # Log validation warnings
        if diff4_validation_warnings:
            logger.warning(
                f"[DIFF4_VALIDATION] {symbol} | min_hold_sec={min_hold_sec_actual} "
                f"policy_exit_unrealized_dd_frac={policy_exit_unrealized_dd_frac} "
                f"policy_exit_hold_bad_frac={policy_exit_hold_bad_frac} "
                f"policy_exit_score_flip_frac={policy_exit_score_flip_frac} | "
                f"Warnings: {'; '.join(diff4_validation_warnings)}"
            )
        
        # Store validation metrics in meta
        if diff4_validation_warnings:
            meta["diff4_validation_warnings"] = diff4_validation_warnings
        meta["min_hold_sec"] = int(min_hold_sec_actual)  # For validation
        
        picked_side = "LONG" if direction == 1 else "SHORT"
        ev_model = "hold" if use_hold_ev else "exit_policy"
        logger.info(
            f"[SIDE_CHOICE] {symbol} | model={ev_model} picked={picked_side} | "
            f"policy_ev_mix={policy_ev_mix:.6f} policy_p_pos_mix={policy_p_pos_mix:.4f} | "
            f"hold_best_ev_L={best_ev_L:.6f} hold_best_ev_S={best_ev_S:.6f} | "
            f"horizon_policy={best_h}"
        )
        if MC_VERBOSE_PRINT:
            print(
                f"[SIDE_CHOICE] {symbol} | model={ev_model} picked={picked_side} | "
                f"policy_ev_mix={policy_ev_mix:.6f} policy_p_pos_mix={policy_p_pos_mix:.4f} | "
                f"horizon_policy={best_h}"
            )
        
        # hold-to-horizon debug lists (match picked direction for consistency)
        ev_list, win_list, cvar_list, h_list = dbg_L if int(direction) == 1 else dbg_S

        # ✅ Step B: ev_raw가 어디서 만들어지는지 확인 (horizon별 ev_h, win_h 출력)
        if ev_list and win_list:
            log_msg = f"[NET_STATS] {symbol} | ev_h={ev_list} win_h={win_list} fee_rt={fee_rt:.6f} horizons={h_list}"
            logger.info(log_msg)
            if MC_VERBOSE_PRINT:
                print(log_msg)
        
        # gate용 baseline(lev=1): 동일 price path에서 direction만 반영
        net_by_h_base = net_by_h_long_base if int(direction) == 1 else net_by_h_short_base
        net_by_h = net_by_h_long if int(direction) == 1 else net_by_h_short

        if not h_list:
            logger.warning(f"[COST_DEBUG] {symbol} | h_list is empty, early return")
            print(f"[EARLY_RETURN_3] {symbol} | fee_roundtrip_total={fee_roundtrip_total} > 0.01 (returning ev=0)")
            return {"can_enter": False, "ev": 0.0, "ev_raw": 0.0, "win": 0.0, "cvar": 0.0, "best_h": 0, "direction": direction, "kelly": 0.0, "size_frac": 0.0}

        # exp-decay weights (half-life=120s)
        h_arr = np.asarray(h_list, dtype=np.float64)
        w = np.exp(-h_arr * math.log(2) / 120.0)
        w = w / np.sum(w)
        evs = np.asarray(ev_list, dtype=np.float64)
        wins = np.asarray(win_list, dtype=np.float64)
        cvars = np.asarray(cvar_list, dtype=np.float64)

        ev_agg = float(np.sum(w * evs))
        win_agg = float(np.sum(w * wins))
        cvar_agg = float(np.quantile(cvars, 0.25))
        horizon_weights = {int(h): float(w_i) for h, w_i in zip(h_list, w)}

        # ✅ Step C: horizon 600초 고정 테스트
        _force_horizon_600 = getattr(self, "_force_horizon_600", False)
        if _force_horizon_600:
            if 600 in h_list:
                best_h = 600
                best_ev = ev_list[h_list.index(600)]
            else:
                # 600초가 없으면 첫 번째 horizon 사용
                if h_list:
                    best_h = h_list[0]
                    best_ev = ev_list[0]
        
        if best_h is None:
            logger.warning(f"[COST_DEBUG] {symbol} | best_h is None, early return")
            print(f"[EARLY_RETURN_4] {symbol} | MC validation failed (returning ev=0)")
            return {"can_enter": False, "ev": 0.0, "ev_raw": 0.0, "win": 0.0, "cvar": 0.0, "best_h": 0, "direction": direction, "kelly": 0.0, "size_frac": 0.0}

        # ✅ Step 2: best_h에 해당하는 net_mat로 검증 로그 출력
        net_mat_for_verification = None
        if best_h is not None and best_h in net_by_h:
            net_mat_for_verification = net_by_h[best_h]
            net_arr = np.asarray(net_mat_for_verification, dtype=np.float64)
            net_mat_mean = float(net_arr.mean())
            net_mat_win = float((net_arr > 0).mean())
            net_mat_std = float(net_arr.std())
            net_mat_min = float(net_arr.min())
            net_mat_max = float(net_arr.max())
            logger.info(
                f"[DBG_NET] {symbol} | best_h={best_h} | "
                f"net_mat_mean={net_mat_mean:.8f} net_mat_win={net_mat_win:.4f} net_mat_std={net_mat_std:.8f} | "
                f"net_mat_min={net_mat_min:.8f} net_mat_max={net_mat_max:.8f} | "
                f"sigma_used={sigma:.8f} | "
                f"ev_agg={ev_agg:.8f} win_agg={win_agg:.4f} cvar_agg={cvar_agg:.8f}"
            )
        
        # hold-to-horizon metrics
        ev_hold = float(ev_agg)
        p_pos_hold = float(win_agg)
        cvar_hold = float(cvar_agg)

        # ✅ "정책-청산 반영 EV/승률"을 진입/홀드 판단에 사용 (완전 대체)

        exit_reason_counts = exit_reason_counts_policy

        # ev is net already (simulation subtracts fee_roundtrip_total inside the path net),
        # so reconstruct a gross proxy by adding execution_cost back.
        ev_gross = float(ev) + float(execution_cost)
        ev_raw = float(ev_gross)
        
        # ✅ [EV_DEBUG] ev_raw 계산 로그
        logger.info(f"[EV_DEBUG] {symbol} | ev_raw calculation: ev={ev:.6f} execution_cost={execution_cost:.6f} ev_gross={ev_gross:.6f} ev_raw={ev_raw:.6f}")
        print(f"[EV_DEBUG] {symbol} | ev_raw calculation: ev={ev:.6f} execution_cost={execution_cost:.6f} ev_gross={ev_gross:.6f} ev_raw={ev_raw:.6f}")

        # [C] PMaker delay penalty is now applied per-horizon in the loop above
        # Legacy discount logic removed - delay penalty is computed horizon-specifically
        # EV already includes horizon-specific delay penalty from MC simulation
        pmaker_discount = 1.0  # Kept for backward compatibility in meta

        # EV decompose (quick sanity): drift-only gross vs fee for 10m/30m
        try:
            mu_exit_per_sec = float(mu_adj) / float(SECONDS_PER_YEAR)
        except Exception:
            mu_exit_per_sec = 0.0
        gross_long_600 = float(mu_exit_per_sec * 600.0 * float(leverage))
        gross_long_1800 = float(mu_exit_per_sec * 1800.0 * float(leverage))
        gross_short_600 = float(-gross_long_600)
        gross_short_1800 = float(-gross_long_1800)
        fee_rt_total_for_decomp = float(fee_rt_total_f)
        ev_net_approx_long_600 = float(gross_long_600 - fee_rt_total_for_decomp)
        ev_net_approx_long_1800 = float(gross_long_1800 - fee_rt_total_for_decomp)
        ev_net_approx_short_600 = float((-gross_long_600) - fee_rt_total_for_decomp)
        ev_net_approx_short_1800 = float((-gross_long_1800) - fee_rt_total_for_decomp)
        # Breakeven drift (annual) required to offset execution costs (ignores sigma/tails; drift-only proxy).
        # mu_req = fee * SECONDS_PER_YEAR / (h * leverage)
        mu_req_600 = None
        mu_req_1800 = None
        mu_req_exit_mean = None
        try:
            lev_eff = max(1e-9, float(leverage))
            mu_req_600 = float(fee_rt_total_for_decomp) * float(SECONDS_PER_YEAR) / max(1e-9, 600.0 * lev_eff)
            mu_req_1800 = float(fee_rt_total_for_decomp) * float(SECONDS_PER_YEAR) / max(1e-9, 1800.0 * lev_eff)
            # Use the *policy*'s typical exit time (mean of chosen horizon rollforward) to contextualize EV negativity.
            # If exit_mean is short, required drift becomes huge and EV will often be negative even when |mu| looks large.
            exit_mean = float(exit_t_mean_sec) if (exit_t_mean_sec is not None) else 0.0
            if exit_mean > 1e-9:
                mu_req_exit_mean = float(fee_rt_total_for_decomp) * float(SECONDS_PER_YEAR) / max(1e-9, exit_mean * lev_eff)
        except Exception:
            mu_req_600 = None
            mu_req_1800 = None
            mu_req_exit_mean = None
        # gate용(base lev=1): use base simulation (already net after base costs)
        try:
            ev1_by_h = [float(net_by_h_base[int(h)].mean()) for h in h_list]
            win1_by_h = [float((net_by_h_base[int(h)] > 0).mean()) for h in h_list]
            cvar1_by_h = [float(cvar_ensemble(net_by_h_base[int(h)], alpha=params.cvar_alpha)) for h in h_list]
            ev1 = float(np.sum(w * np.asarray(ev1_by_h, dtype=np.float64)))
            win1 = float(np.sum(w * np.asarray(win1_by_h, dtype=np.float64)))
            cvar1 = float(np.quantile(np.asarray(cvar1_by_h, dtype=np.float64), 0.25))
        except Exception:
            ev1 = float(ev)  # fallback
            win1 = float(win)
            cvar1 = float(cvar)
        
        # ✅ A) 비용 디버그 로그 (각 심볼마다 출력)
        execution_cost_oneway = float(execution_cost) / 2.0  # oneway는 roundtrip의 절반
        logger.info(
            f"[COST_DEBUG] {symbol} | "
            f"fee_roundtrip={fee_rt:.6f} | "
            f"execution_cost_oneway={execution_cost_oneway:.6f} | "
            f"expected_spread_cost={expected_spread_cost:.6f} | "
            f"slippage_dyn={slippage_dyn:.6f} | "
            f"leverage={leverage:.2f} | "
            f"ev_hold={ev_hold:.6f} p_pos_hold={p_pos_hold:.4f} | "
            f"policy_ev_mix={policy_ev_mix:.6f} policy_p_pos_mix={policy_p_pos_mix:.4f} exit_t_mean={exit_t_mean_sec:.1f}s"
        )
        
        # ✅ B) win 계산 디버그 로그 (각 심볼마다 출력)
        w_sum = float(np.sum(w))
        logger.info(
            f"[WIN_DEBUG] {symbol} | "
            f"best_h={best_h} | "
            f"horizons={h_list} | "
            f"w_sum={w_sum:.6f} | "
            f"w={[f'{w_i:.4f}' for w_i in w]} | "
            f"win_h={[f'{win_h:.4f}' for win_h in win_list]} | "
            f"ev_h={[f'{ev_h:.6f}' for ev_h in ev_list]} | "
            f"win_agg={win_agg:.4f}"
        )
        mid_h = []
        ev_mid = None
        win_mid = None
        cvar_mid = None
        if mid_h:
            mid_concat = np.concatenate(mid_h, axis=0)
            ev_mid = float(mid_concat.mean())
            win_mid = float((mid_concat > 0).mean())
            cvar_mid = float(cvar_ensemble(mid_concat, alpha=params.cvar_alpha))

        # entry gating: 비용을 충분히 이길 때만, 승률/꼬리/중기 필터
        cost_floor = float(fee_rt)
        # 더 높은 EV 기준: 비용 + 추가 버퍼
        ev_floor = max(params.profit_target, cost_floor * 1.0 + 0.0008)
        win_floor = max(params.min_win - 0.02, 0.53)
        cvar_floor_abs = cost_floor * 3.0  # 요구: cvar_agg > -cvar_floor_abs
        # 비용 대비 마진 게이트(옵션): ev_for_gate >= k * fee_roundtrip_total
        ev_cost_mult_gate = 0.0
        ev_cost_floor = None
        try:
            ev_cost_mult_gate = float(os.environ.get("EV_COST_MULT_GATE", "0") or 0.0)
        except Exception:
            ev_cost_mult_gate = 0.0
        if ev_cost_mult_gate > 0:
            ev_cost_floor = float(ev_cost_mult_gate) * float(fee_roundtrip_total)

        can_enter = False
        blocked_by = []
        
        # 기본 게이트 체크
        if ev <= ev_floor:
            blocked_by.append(f"ev_gate(ev={ev:.6f} <= ev_floor={ev_floor:.6f})")
        if ev_cost_floor is not None and ev <= float(ev_cost_floor):
            blocked_by.append(f"ev_cost_gate(ev={ev:.6f} <= k*fee={float(ev_cost_floor):.6f}, k={ev_cost_mult_gate:.2f}, fee_rt_total={float(fee_roundtrip_total):.6f})")
        if win < win_floor:
            blocked_by.append(f"win_gate(win={win:.4f} < win_floor={win_floor:.4f})")
        if cvar <= -cvar_floor_abs:
            blocked_by.append(f"cvar_gate(cvar={cvar:.6f} <= -cvar_floor_abs={-cvar_floor_abs:.6f})")
        
        ev_ok = ev > ev_floor
        if ev_cost_floor is not None:
            ev_ok = ev_ok and (ev > float(ev_cost_floor))
        if ev_ok and win >= win_floor and cvar > -cvar_floor_abs:
            mid_ok = True
            mid_cut_reason = None
            if ev_mid is not None:
                if ev_mid < 0.0:
                    mid_ok = False
                    mid_cut_reason = f"ev_mid={ev_mid:.6f} < 0.0"
            if win_mid is not None and mid_ok:
                if win_mid < 0.50:
                    mid_ok = False
                    mid_cut_reason = f"win_mid={win_mid:.4f} < 0.50"
            if not mid_ok and mid_cut_reason is not None:
                blocked_by.append(f"mid_gate({mid_cut_reason})")
            can_enter = mid_ok
        else:
            # 기본 게이트에서 막힌 경우 mid 체크는 스킵
            pass
        
        # 진입게이트 상세 로그 (항상 출력)
        if True:  # ✅ 로그 제한 제거: 항상 출력
            direction_str = "LONG" if direction == 1 else "SHORT"
            aggressive = abs(leverage) > 5.0  # 레버리지 기반 aggressive 판단
            
            # ✅ f-string 조건부 포맷팅 수정: None 체크 후 포맷팅
            win_mid_str = f"{win_mid:.4f}" if win_mid is not None else "None"
            ev_mid_str = f"{ev_mid:.6f}" if ev_mid is not None else "None"
            log_msg = (
                f"[ENTRY_GATE] 선택방향={direction_str} | "
                f"p_pos_for_gate={win:.4f}, win_floor={win_floor:.4f} | "
                f"ev_for_gate={ev:.6f}, ev_floor={ev_floor:.6f}, cost_p90={cost_floor:.6f}, profit_target={params.profit_target:.6f} | "
                f"cvar={cvar:.6f}, cvar_floor_abs={cvar_floor_abs:.6f} | "
                f"aggressive={aggressive}, win_mid={win_mid_str}, ev_mid={ev_mid_str}, mid_cut={not can_enter and len(blocked_by) > 0 and any('mid_gate' in b for b in blocked_by)} | "
                f"최종 blocked_by={', '.join(blocked_by) if blocked_by else 'PASS'}"
            )
            logger.info(log_msg)
            self._gate_log_count += 1

        # Kelly raw (EV / variance proxy)
        variance_proxy = float(sigma * sigma)
        kelly_raw = max(0.0, ev / max(variance_proxy, 1e-6))

        # CVaR 기반 축소 (레버리지 고려)
        leverage_penalty = max(1.0, abs(leverage) / 5.0)
        cvar_penalty = max(0.05, 1.0 - params.cvar_scale * abs(cvar) * leverage_penalty)

        # 고레버리지일수록 Kelly 상한을 자동 축소
        kelly_cap = params.max_kelly / leverage_penalty

        kelly = min(kelly_raw * cvar_penalty, kelly_cap)

        confidence = float(win)  # hub confidence는 win 기반 유지
        size_frac = float(max(0.0, kelly * confidence))

        # Event-based MC (first passage TP/SL)
        tp_pct = float(max(params.profit_target, 0.0005))
        sl_pct = float(max(tp_pct * 0.8, 0.0008))
        # ✅ first-passage JAX 사용 여부:
        # - ctx/use_jax를 존중 (simulate_paths_price와 동일하게)
        # - JAX 오류 시 항상 numpy fallback (엔진 전체가 실패하면 meta가 비어서 디버깅이 불가능해짐)
        use_first_passage_jax = bool(getattr(self, "_use_jax", True)) and (jax is not None) and (not getattr(self, "_skip_first_passage_jax", False))
        # ✅ 격리 테스트: dist를 gaussian으로 고정
        dist_mode = "gaussian" if getattr(self, "_force_gaussian_dist", False) else str(getattr(self, "_tail_mode", self.default_tail_mode))
        if use_first_passage_jax:
            try:
                event_metrics = mc_first_passage_tp_sl_jax(
                    s0=price,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                    mu=mu_adj,
                    sigma=sigma,
                    dt=self.dt,
                    max_steps=int(max(self.horizons)),
                    n_paths=int(params.n_paths),
                    cvar_alpha=params.cvar_alpha,
                    seed=seed,
                    dist=dist_mode,
                    df=float(getattr(self, "_student_t_df", self.default_student_t_df)),
                    boot_rets=getattr(self, "_bootstrap_returns", None),
                )
            except Exception:
                event_metrics = None
            if not event_metrics:
                event_metrics = self.mc_first_passage_tp_sl(
                    s0=price,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                    mu=mu_adj,
                    sigma=sigma,
                    dt=self.dt,
                    max_steps=int(max(self.horizons)),
                    n_paths=int(params.n_paths),
                    cvar_alpha=params.cvar_alpha,
                    seed=seed,
                )
        else:
            event_metrics = self.mc_first_passage_tp_sl(
                s0=price,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                mu=mu_adj,
                sigma=sigma,
                dt=self.dt,
                max_steps=int(max(self.horizons)),
                n_paths=int(params.n_paths),
                cvar_alpha=params.cvar_alpha,
                seed=seed,
            )

        # event EV/CVaR를 % 수익률 단위로 환산 (R * SL%)
        event_ev_pct = None
        event_cvar_pct = None
        try:
            if event_metrics.get("event_ev_r") is not None:
                event_ev_pct = float(event_metrics["event_ev_r"]) * sl_pct
        except Exception:
            event_ev_pct = None
        try:
            if event_metrics.get("event_cvar_r") is not None:
                event_cvar_pct = float(event_metrics["event_cvar_r"]) * sl_pct
        except Exception:
            event_cvar_pct = None

        # ✅ BTC 상관관계 반영 (리스크 관리)
        btc_corr = _s(ctx.get("btc_corr"), 0.0)
        if btc_corr > 0.7:
            # 상관관계가 높으면 Kelly 비중 축소 (분산투자 효과 감소)
            kelly *= 0.8
            logger.info(f"[RISK] {symbol} High BTC corr={btc_corr:.2f} -> Kelly reduced to {kelly:.2f}")

        # ✅ 역선택 방지 (Adverse Selection Protection)
        pmaker_entry = _s(ctx.get("pmaker_entry"), 0.0)
        pmaker_threshold = 0.3  # 임계값 (필요시 환경변수화)
        if pmaker_entry > 0 and pmaker_entry < pmaker_threshold:
            logger.info(f"[ADVERSE_SELECTION] {symbol} pmaker_entry={pmaker_entry:.2f} < {pmaker_threshold} -> Entry blocked")
            can_enter = False

        # [EV_DEBUG] 최종 EV 값 확인
        print(f"[EV_DEBUG] evaluate_entry_metrics: symbol={symbol} ev={ev} win={win} cvar={cvar} can_enter={can_enter} direction={direction}")
        logger.info(f"[EV_DEBUG] evaluate_entry_metrics: symbol={symbol} ev={ev} win={win} cvar={cvar} can_enter={can_enter} direction={direction}")
        
        # [EV_FINAL_DEBUG] 최종 반환값 확인
        print(f"[EV_FINAL_DEBUG] {symbol} | Returning: ev={ev:.6f} policy_ev_mix={policy_ev_mix:.6f} ev_mix_long={ev_mix_long:.6f} ev_mix_short={ev_mix_short:.6f} direction={direction}")
        
        return {
            "can_enter": bool(can_enter),
            "ev": float(ev) if not math.isnan(float(ev)) else 0.0,
            "ev_raw": float(ev_raw) if not math.isnan(float(ev_raw)) else 0.0,
            "win": win,
            "cvar": cvar,
            "best_h": int(best_h),
            "direction": int(direction),
            "kelly": float(kelly),
            "size_frac": float(size_frac),
            # hold-to-horizon (기존 p_pos 계산 유지)
            "ev_hold": float(ev_hold),
            "p_pos_hold": float(p_pos_hold),
            "cvar_hold": float(cvar_hold),
            # 정책-청산 반영 지표 (mix)
            "policy_ev_mix": float(policy_ev_mix),
            "policy_p_pos_mix": float(policy_p_pos_mix),
            "policy_cvar_mix": float(policy_cvar_mix),
            "policy_cvar_gate": float(cvar_gate),
            "policy_cvar_gate_mode": "min",
            "ev_model": ev_model,
            "policy_ev_mix_long": float(ev_mix_long),
            "policy_ev_mix_short": float(ev_mix_short),
            "policy_p_pos_mix_long": float(ppos_mix_long),
            "policy_p_pos_mix_short": float(ppos_mix_short),
            "hold_best_h_long": int(best_h_L) if best_h_L is not None else None,
            "hold_best_h_short": int(best_h_S) if best_h_S is not None else None,
            "hold_best_ev_long": float(best_ev_L) if best_ev_L is not None else None,
            "hold_best_ev_short": float(best_ev_S) if best_ev_S is not None else None,
            "policy_exit_unrealized_dd_frac": float(policy_exit_unrealized_dd_frac) if policy_exit_unrealized_dd_frac is not None else None,
            "policy_exit_hold_bad_frac": float(policy_exit_hold_bad_frac) if policy_exit_hold_bad_frac is not None else None,
            "policy_exit_score_flip_frac": float(policy_exit_score_flip_frac) if policy_exit_score_flip_frac is not None else None,
            "policy_signal_strength": float(signal_strength),
            "policy_half_life_sec": float(policy_half_life_sec),
            "policy_h_eff_sec": float(policy_h_eff_sec),
            "policy_h_eff_sec_prior": float(policy_h_eff_sec_prior),  # [DIFF 3] Rule-based only (for validation)
            "policy_w_short_sum": float(policy_w_short_sum),
            "policy_momentum_z": float(momentum_z),
            "policy_ofi_z": float(ofi_z),
            "paths_seed_base": int(paths_seed_base),
            "paths_reused": bool(paths_reused),
            "policy_weight_peak_h": int(weight_peak_h),
            "policy_paths_mu_annual": float(mu_adj),
            "policy_paths_sigma_annual": float(sigma),
            "policy_paths_dt_years": float(self.dt),
            "policy_ev_per_h": [float(x) for x in evs_best],
            "policy_p_pos_per_h": [float(x) for x in pps_best],
            "policy_cvar_per_h": [float(x) for x in cvars_best],
            "policy_exit_t_mean_per_h": [float(x) for x in exit_t_mean_per_h_best],
            "policy_exit_t_p50_per_h": [float(x) for x in exit_t_p50_per_h_best],
            "policy_exit_t_mean_per_h_long": [float(x) for x in exit_t_mean_per_h_long],
            "policy_exit_t_mean_per_h_short": [float(x) for x in exit_t_mean_per_h_short],
            "policy_exit_reason_counts_per_h": exit_reason_counts_per_h_best,
            "policy_exit_reason_counts_per_h_long": exit_reason_counts_per_h_long,
            "policy_exit_reason_counts_per_h_short": exit_reason_counts_per_h_short,
            "policy_horizons": [int(h) for h in policy_horizons],
            "policy_w_h": [float(x) for x in w_arr],
            "policy_direction": int(direction_policy),
            "policy_exit_reason_counts": exit_reason_counts_policy,
            "exit_time_mean_sec": float(exit_t_mean_sec),
            "exit_reason_counts": exit_reason_counts,
            # backward-compat aliases (기존 필드명 유지)
            "ev_exit_policy_30m": float(policy_ev_mix),
            "p_pos_exit_policy_30m": float(policy_p_pos_mix),
            "cvar_exit_policy_30m": float(cvar_gate),
            "ev1": ev1,
            "win1": win1,
            "cvar1": cvar1,
            "mu_adj": float(mu_adj),
            "mu_sim": float(mu_adj),  # ✅ Step 1/2: 추가 (실제 시뮬레이션에 사용되는 mu)
            "signal_mu": float(signal_mu),
            # --- μ(alpha) decomposition (debug) ---
            "mu_alpha": float(mu_alpha_parts.get("mu_alpha") or 0.0) if isinstance(mu_alpha_parts, dict) else float(signal_mu),
            "mu_alpha_raw": float(mu_alpha_parts.get("mu_alpha_raw") or 0.0) if isinstance(mu_alpha_parts, dict) else None,
            "mu_alpha_cap": float(mu_alpha_parts.get("mu_alpha_cap") or 0.0) if isinstance(mu_alpha_parts, dict) else None,
            "mu_alpha_regime_scale": float(mu_alpha_parts.get("regime_scale") or 0.0) if isinstance(mu_alpha_parts, dict) else None,
            "mu_alpha_mom": float(mu_alpha_parts.get("mu_mom") or 0.0) if isinstance(mu_alpha_parts, dict) else None,
            "mu_alpha_ofi": float(mu_alpha_parts.get("mu_ofi") or 0.0) if isinstance(mu_alpha_parts, dict) else None,
            "mu_alpha_ofi_score_clipped": float(mu_alpha_parts.get("ofi_score_clipped") or 0.0) if isinstance(mu_alpha_parts, dict) else None,
            "mu_alpha_mom_15": float(mu_alpha_parts.get("mu_mom_15") or 0.0) if isinstance(mu_alpha_parts, dict) else None,
            "mu_alpha_mom_30": float(mu_alpha_parts.get("mu_mom_30") or 0.0) if isinstance(mu_alpha_parts, dict) else None,
            "mu_alpha_mom_60": float(mu_alpha_parts.get("mu_mom_60") or 0.0) if isinstance(mu_alpha_parts, dict) else None,
            "mu_alpha_mom_120": float(mu_alpha_parts.get("mu_mom_120") or 0.0) if isinstance(mu_alpha_parts, dict) else None,
            # ✅ [FIX 2] PMaker mu_alpha boost 정보 추가
            "mu_alpha_pmaker_fill_rate": float(mu_alpha_parts.get("mu_alpha_pmaker_fill_rate")) if isinstance(mu_alpha_parts, dict) and mu_alpha_parts.get("mu_alpha_pmaker_fill_rate") is not None else None,
            "mu_alpha_pmaker_boost": float(mu_alpha_parts.get("mu_alpha_pmaker_boost") or 0.0) if isinstance(mu_alpha_parts, dict) else None,
            "mu_alpha_before_pmaker": float(mu_alpha_parts.get("mu_alpha_before_pmaker") or 0.0) if isinstance(mu_alpha_parts, dict) else None,
            "sigma_sim": float(sigma),  # ✅ Step 1/2: 추가 (실제 시뮬레이션에 사용되는 sigma)
            # quick EV decomposition (drift-only approx; leverage-scaled)
            "ev_decomp_mu_annual": float(mu_adj),
            "ev_decomp_mu_per_sec": float(mu_exit_per_sec),
            "ev_decomp_fee_roundtrip_total": float(fee_rt_total_for_decomp),
            "ev_decomp_gross_long_600": float(gross_long_600),
            "ev_decomp_gross_long_1800": float(gross_long_1800),
            "ev_decomp_gross_short_600": float(gross_short_600),
            "ev_decomp_gross_short_1800": float(gross_short_1800),
            "ev_decomp_net_long_600": float(ev_net_approx_long_600),
            "ev_decomp_net_long_1800": float(ev_net_approx_long_1800),
            "ev_decomp_net_short_600": float(ev_net_approx_short_600),
            "ev_decomp_net_short_1800": float(ev_net_approx_short_1800),
            "ev_decomp_mu_req_annual_600": float(mu_req_600) if mu_req_600 is not None else None,
            "ev_decomp_mu_req_annual_1800": float(mu_req_1800) if mu_req_1800 is not None else None,
            "ev_decomp_mu_req_annual_exit_mean": float(mu_req_exit_mean) if mu_req_exit_mean is not None else None,
            # execution-mode aware cost meta (maker_then_market 혼합)
            "exec_mode": str(exec_mode),
            "p_maker": float(p_maker),
            "pmaker_override_used": bool(pmaker_override_used),
            "pmaker_entry": float(pmaker_entry_local),
            "pmaker_entry_delay_sec": float(pmaker_delay_sec_local),
            "pmaker_exit": float(pmaker_exit),
            "pmaker_exit_delay_sec": float(pmaker_exit_delay_sec),
            "pmaker_discount": float(pmaker_discount),
            "pmaker_ev_gross": float(ev_gross),
            "fee_roundtrip_fee_taker": float(fee_fee_taker),
            "fee_roundtrip_fee_maker": float(fee_fee_maker),
            "fee_roundtrip_fee_mix": float(fee_fee_mix),
            "slippage_dyn_raw": float(slippage_dyn_raw),
            "expected_spread_cost_raw": float(expected_spread_cost_raw),
            "fee_rt": float(fee_rt),
            "spread_pct": float(spread_pct),
            "expected_spread_cost": float(expected_spread_cost),
            "execution_cost": float(execution_cost),
            "fee_roundtrip_total": float(execution_cost),  # ✅ Step B: 추가
            "fee_roundtrip_base": float(self.fee_roundtrip_base),  # ✅ Step B: 추가
            "slippage_dyn": float(slippage_dyn),  # ✅ Step B: 추가 (기존 slippage_pct와 동일하지만 명확성을 위해)
            "liq_score": float(liq_score),  # ✅ Step B: 추가
            "ev_raw": float(ev_raw),  # ✅ Step 2: MC에서 나온 가중합 (비용 차감 전)
            "ev_for_gate": float(ev),  # ✅ Step 2: 게이트에 쓰는 최종 EV
            "ev_floor": float(ev_floor),
            "win_floor": float(win_floor),
            "cvar_floor": float(cvar_floor_abs),
            "ev_cost_mult_gate": float(ev_cost_mult_gate),
            "ev_cost_floor": float(ev_cost_floor) if ev_cost_floor is not None else None,
            "ev_mid": ev_mid,
            "win_mid": win_mid,
            "cvar_mid": cvar_mid,
            "slippage_pct": float(slippage_dyn),
            "ev_by_horizon": [float(x) for x in ev_list],
            "win_by_horizon": [float(x) for x in win_list],
            "cvar_by_horizon": [float(x) for x in cvar_list],
            "horizon_seq": [int(h) for h in h_list],
            "event_p_tp": event_metrics.get("event_p_tp"),
            "event_p_sl": event_metrics.get("event_p_sl"),
            "event_p_timeout": event_metrics.get("event_p_timeout"),
            "event_ev_r": event_metrics.get("event_ev_r"),
            "event_cvar_r": event_metrics.get("event_cvar_r"),
            "event_ev_pct": event_ev_pct,
            "event_cvar_pct": event_cvar_pct,
            "event_t_median": event_metrics.get("event_t_median"),
            "event_t_mean": event_metrics.get("event_t_mean"),
            "horizon_weights": horizon_weights,
            # [DIFF 6] PMaker delay penalty and horizon effective from compute_exit_policy_metrics
            "pmaker_entry_delay_penalty_r": meta.get("pmaker_entry_delay_penalty_r"),
            "pmaker_exit_delay_penalty_r": meta.get("pmaker_exit_delay_penalty_r"),
            "policy_entry_shift_steps": meta.get("policy_entry_shift_steps"),
            "policy_horizon_eff_sec": meta.get("policy_horizon_eff_sec"),
        }
    def decide(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(ctx.get("symbol", ""))
        price = float(ctx.get("price", 0.0))
        regime_ctx = str(ctx.get("regime", "chop"))
        params = self._get_params(regime_ctx, ctx)

        # deterministic seed for stability (per symbol, per minute-ish)
        seed = int((hash(symbol) ^ int(time.time() // 3)) & 0xFFFFFFFF)

        # ✅ [EV_DEBUG] mc_engine.decide 시작 로그
        print(f"[EV_DEBUG] mc_engine.decide: START symbol={symbol} price={price} regime={regime_ctx}")
        logger.info(f"[EV_DEBUG] mc_engine.decide: START symbol={symbol} price={price} regime={regime_ctx}")

        # Debug: log ctx keys and pmaker_entry value before calling evaluate_entry_metrics
        print(f"[PMAKER_DEBUG] {symbol} | mc_engine.decide: ctx keys={list(ctx.keys())} pmaker_entry={ctx.get('pmaker_entry')} pmaker_entry_delay_sec={ctx.get('pmaker_entry_delay_sec')}")
        logger.info(f"[PMAKER_DEBUG] {symbol} | mc_engine.decide: ctx keys={list(ctx.keys())} pmaker_entry={ctx.get('pmaker_entry')} pmaker_entry_delay_sec={ctx.get('pmaker_entry_delay_sec')}")

        # ✅ 최적 레버리지 자동 산출
        leverage_val = ctx.get("leverage")
        optimal_leverage = float(leverage_val if leverage_val is not None else 5.0)  # 기본값 (None 처리)
        optimal_size = 0.0
        best_net_ev = None  # ✅ 변수 스코프 확보: 레버리지 최적화에서 계산한 net EV
        
        try:
            # 1. 기초 자산(leverage=1.0) 기준으로 순수 EV와 리스크 계산
            # evaluate_entry_metrics를 leverage=1.0으로 호출하기 위해 ctx 복사
            ctx_base = ctx.copy()
            ctx_base["leverage"] = 1.0
            
            metrics_base = self.evaluate_entry_metrics(ctx_base, params, seed=seed)
            
            # ✅ can_enter 여부와 관계없이 EV 계산 (EV 값은 항상 표시되어야 함)
            # 기초 자산의 순수 EV와 리스크 추출
            ev_raw = float(metrics_base.get("ev_raw", metrics_base.get("ev", 0.0)) or 0.0)
            sigma_raw = float(ctx.get("sigma_sim") or ctx.get("sigma", 0.0) or 0.0)
            win_rate = float(metrics_base.get("win", 0.0) or 0.0)
            cvar_raw = float(metrics_base.get("cvar", 0.0) or 0.0)
            can_enter_base = metrics_base.get("can_enter", False)
            
            print(f"[EV_DEBUG] mc_engine.decide: metrics_base can_enter={can_enter_base} ev_raw={ev_raw} sigma_raw={sigma_raw} win_rate={win_rate}")
            
            # ✅ ev_raw가 유효한 값이 있으면 레버리지 최적화 수행 (can_enter와 무관)
            if ev_raw != 0.0 or sigma_raw > 0:
                
                # 수수료율과 슬리피지 추출 (레버리지와 무관한 기초 비용)
                # metrics_base에서 execution_cost를 가져오거나, meta에서 fee 정보 추출
                execution_cost_base = float(metrics_base.get("execution_cost", 0.0) or 0.0)
                fee_roundtrip_total_base = float(metrics_base.get("fee_roundtrip_total", execution_cost_base) or execution_cost_base)
                
                # 레버리지=1.0일 때의 비용이므로, 이를 기초 비용으로 사용
                fee_rate_total = fee_roundtrip_total_base
                
                # max_leverage 확인
                max_leverage = float(ctx.get("max_leverage", 20.0) or 20.0)
                
                # 2. 레버리지 후보 리스트 생성
                candidate_leverages = [1, 2, 3, 4, 5, 7, 10, 15, 20]
                candidate_leverages = [L for L in candidate_leverages if L <= max_leverage]
                
                best_leverage = 1.0
                best_kelly = 0.0
                best_net_ev_local = -1e18  # ✅ 로컬 변수명 변경 (외부 best_net_ev와 구분)
                best_size = 0.0
                
                # 3. 각 레버리지 후보에 대해 최적화
                for L in candidate_leverages:
                    # 레버리지 적용된 EV와 비용
                    adjusted_ev = ev_raw * L
                    adjusted_cost = fee_rate_total * L  # 수수료도 레버리지 배수만큼 커짐
                    net_ev = adjusted_ev - adjusted_cost
                    
                    # 레버리지 적용된 변동성
                    adjusted_sigma = sigma_raw * L
                    
                    # net_ev가 양수인 경우만 고려
                    if net_ev > 0 and adjusted_sigma > 1e-6:
                        # 켈리 공식 적용 (단순화된 근사식)
                        # kelly_f = net_ev / (adjusted_sigma^2)
                        kelly_f = net_ev / (adjusted_sigma ** 2)
                        
                        # CVaR 기반 켈리 보정 (기존 kelly_with_cvar 활용)
                        # TP/SL 비율 추정 (win_rate 기반)
                        tp_estimate = abs(ev_raw) * 2.0  # 대략적인 TP 추정
                        sl_estimate = abs(cvar_raw) if cvar_raw < 0 else abs(ev_raw) * 0.5  # 대략적인 SL 추정
                        if sl_estimate < 1e-6:
                            sl_estimate = abs(ev_raw) * 0.5
                        
                        kelly_cvar = kelly_with_cvar(win_rate, tp_estimate, sl_estimate, cvar_raw * L)
                        
                        # 성장률 = kelly_f * net_ev (또는 kelly_cvar 기반)
                        growth_rate = kelly_cvar * net_ev
                        
                        # 최적값 선택: net_ev가 양수이면서 growth_rate가 가장 높은 레버리지
                        if growth_rate > best_kelly:
                            best_leverage = float(L)
                            best_kelly = growth_rate
                            best_net_ev_local = net_ev
                            best_size = min(kelly_cvar, 1.0)  # 켈리 비율을 사이즈로 사용
                
                # 4. 안전 장치 적용
                MAX_POSITION_SIZE_CAP = 0.2  # 최대 포지션 사이즈 제한
                if best_size > MAX_POSITION_SIZE_CAP:
                    best_size = MAX_POSITION_SIZE_CAP
                    # 사이즈가 제한되면 레버리지도 조정
                    if best_leverage > 10.0:
                        best_leverage = min(best_leverage, 10.0)
                
                # 최종 결정
                if best_net_ev_local > 0:
                    optimal_leverage = best_leverage
                    optimal_size = best_size
                    # ✅ best_net_ev를 외부 스코프에 저장
                    best_net_ev = best_net_ev_local
                    logger.info(
                        f"[OPTIMAL_LEVERAGE] {symbol} | "
                        f"ev_raw={ev_raw:.6f} sigma_raw={sigma_raw:.6f} | "
                        f"optimal_leverage={optimal_leverage:.2f} optimal_size={optimal_size:.4f} | "
                        f"best_net_ev={best_net_ev:.6f} best_kelly={best_kelly:.6f}"
                    )
                else:
                    # ✅ best_net_ev_local이 양수가 아니면, ev_raw를 그대로 사용하지 않음 (음수일 수 있음)
                    # ✅ LINK 심볼의 경우 ev_raw가 음수인데 양수로 표시되는 문제 확인
                    if symbol.startswith("LINK"):
                        logger.warning(f"[EV_VALIDATION_LINK] {symbol} | ⚠️  레버리지 최적화 실패: ev_raw={ev_raw:.6f} best_net_ev_local={best_net_ev_local:.6f}")
                        print(f"[EV_VALIDATION_LINK] {symbol} | ⚠️  레버리지 최적화 실패: ev_raw={ev_raw:.6f} best_net_ev_local={best_net_ev_local:.6f}")
                    best_net_ev = None  # ✅ 음수 ev_raw를 그대로 사용하지 않음
                    logger.info(f"[OPTIMAL_LEVERAGE] {symbol} | No positive net_ev found, ev_raw={ev_raw:.6f}, best_net_ev=None, using default leverage={optimal_leverage:.2f}")
            else:
                # ✅ ev_raw가 0이거나 sigma가 0이면 레버리지 최적화 불가
                # ✅ ev_raw가 음수일 수 있으므로 그대로 사용하지 않음
                if symbol.startswith("LINK"):
                    logger.warning(f"[EV_VALIDATION_LINK] {symbol} | ⚠️  레버리지 최적화 불가: ev_raw={ev_raw:.6f} sigma_raw={sigma_raw:.6f}")
                    print(f"[EV_VALIDATION_LINK] {symbol} | ⚠️  레버리지 최적화 불가: ev_raw={ev_raw:.6f} sigma_raw={sigma_raw:.6f}")
                best_net_ev = None  # ✅ ev_raw가 음수일 수 있으므로 None으로 설정
                logger.info(f"[OPTIMAL_LEVERAGE] {symbol} | ev_raw={ev_raw} sigma_raw={sigma_raw} - cannot optimize leverage, best_net_ev=None, using default leverage={optimal_leverage:.2f}")
        except Exception as e:
            best_net_ev = None  # 예외 발생 시 None
            logger.warning(f"[OPTIMAL_LEVERAGE] {symbol} | Failed to compute optimal leverage: {e}, using default")
            import traceback
            traceback.print_exc()
        
        # 최적 레버리지로 ctx 업데이트
        ctx["leverage"] = optimal_leverage

        try:
            print(f"[EV_DEBUG] mc_engine.decide: About to call evaluate_entry_metrics for {symbol}")
            print(f"[EV_DEBUG] mc_engine.decide: ctx keys={list(ctx.keys())[:20]} mu_base={ctx.get('mu_base')} sigma={ctx.get('sigma')} price={ctx.get('price')}")
            metrics = self.evaluate_entry_metrics(ctx, params, seed=seed)
            print(f"[EV_DEBUG] mc_engine.decide: evaluate_entry_metrics returned: can_enter={metrics.get('can_enter')} ev={metrics.get('ev')} win={metrics.get('win')}")
        except Exception as e:
            print(f"[EV_DEBUG] mc_engine.decide: evaluate_entry_metrics failed: {e}")
            logger.error(f"[EV_DEBUG] mc_engine.decide: evaluate_entry_metrics failed: {e}")
            import traceback
            traceback.print_exc()
            # 예외 발생 시 기본값 반환
            metrics = {
                "can_enter": False,
                "ev": 0.0,
                "win": 0.0,
                "cvar": 0.0,
                "best_h": 0,
                "direction": 1,
                "kelly": 0.0,
                "size_frac": 0.0,
            }

        # Debug: log metrics keys and pmaker_entry value
        print(f"[PMAKER_DEBUG] {symbol} | mc_engine.decide: metrics keys (sample): {list(metrics.keys())[:20]}... pmaker_entry={metrics.get('pmaker_entry')} pmaker_entry_delay_sec={metrics.get('pmaker_entry_delay_sec')}")
        logger.info(f"[PMAKER_DEBUG] {symbol} | mc_engine.decide: metrics pmaker_entry={metrics.get('pmaker_entry')} pmaker_entry_delay_sec={metrics.get('pmaker_entry_delay_sec')} pmaker_exit={metrics.get('pmaker_exit')} pmaker_exit_delay_sec={metrics.get('pmaker_exit_delay_sec')}")

        # ============================================================
        # 깔때기(Funnel) 구조 진입 필터 로직
        # ============================================================
        
        # [Step 0] Regime 및 문턱값 정의
        regime = str(regime_ctx).upper() if regime_ctx else "CHOP"
        if regime not in ["BULL", "BEAR", "CHOP", "VOLATILE"]:
            regime = "CHOP"  # 기본값
        
        # Regime별 문턱값 딕셔너리
        win_floors = {
            "BULL": 0.50,
            "BEAR": 0.50,
            "CHOP": 0.52,
            "VOLATILE": 0.53
        }
        cvar_floors = {
            "BULL": -0.010,
            "BEAR": -0.011,
            "CHOP": -0.008,
            "VOLATILE": -0.007
        }
        event_cvar_floors = {
            "BULL": -1.20,
            "BEAR": -1.15,
            "CHOP": -1.05,
            "VOLATILE": -0.95
        }
        
        # 필터링에 사용할 메트릭 추출
        policy_ev_mix = float(metrics.get("policy_ev_mix", 0.0) or 0.0)
        ev_pct = float(metrics.get("ev", 0.0) or 0.0)
        # EV는 policy_ev_mix 우선, 없으면 ev_pct 사용
        ev_for_filter = policy_ev_mix if policy_ev_mix != 0.0 else ev_pct
        
        win_rate = float(metrics.get("policy_p_pos_mix", metrics.get("win", 0.0)) or 0.0)
        cvar1 = float(metrics.get("cvar1", metrics.get("cvar", 0.0)) or 0.0)
        event_cvar_r = metrics.get("event_cvar_r")
        direction = int(metrics.get("direction", 0))
        
        # 필터 단계별 체크
        action = "WAIT"
        filter_reason = None
        
        # [Step 1] EV(기대값) 필터 (최우선)
        if ev_for_filter <= 0.0:
            filter_reason = f"EV_FILTER: ev_for_filter={ev_for_filter:.6f} <= 0.0"
            logger.info(f"[FUNNEL_FILTER] {symbol} | {filter_reason}")
        else:
            # [Step 2] Win Rate 필터
            win_floor = win_floors.get(regime, 0.52)
            if win_rate < win_floor:
                filter_reason = f"WIN_RATE_FILTER: win_rate={win_rate:.4f} < win_floor[{regime}]={win_floor:.4f}"
                logger.info(f"[FUNNEL_FILTER] {symbol} | {filter_reason}")
            else:
                # [Step 3] 리스크 방어 (CVaR)
                cvar_floor = cvar_floors.get(regime, -0.008)
                if cvar1 < cvar_floor:
                    filter_reason = f"CVAR_FILTER: cvar1={cvar1:.6f} < cvar_floor[{regime}]={cvar_floor:.6f}"
                    logger.info(f"[FUNNEL_FILTER] {symbol} | {filter_reason}")
                else:
                    # VOLATILE 장세일 때만 event_cvar_r 추가 체크
                    if regime == "VOLATILE" and event_cvar_r is not None:
                        event_cvar_floor = event_cvar_floors.get(regime, -0.95)
                        event_cvar_r_val = float(event_cvar_r)
                        if event_cvar_r_val < event_cvar_floor:
                            filter_reason = f"EVENT_CVAR_FILTER: event_cvar_r={event_cvar_r_val:.6f} < event_cvar_floor[{regime}]={event_cvar_floor:.6f}"
                            logger.info(f"[FUNNEL_FILTER] {symbol} | {filter_reason}")
                        else:
                            # [Pass] 최종 승인: 모든 필터 통과
                            if direction != 0:
                                action = "LONG" if direction == 1 else "SHORT"
                                filter_reason = "PASS: All filters passed"
                                logger.info(f"[FUNNEL_FILTER] {symbol} | {filter_reason} | direction={direction} action={action}")
                            else:
                                filter_reason = "DIRECTION_FILTER: direction=0 (no trade)"
                                logger.info(f"[FUNNEL_FILTER] {symbol} | {filter_reason}")
                    else:
                        # [Pass] 최종 승인: 모든 필터 통과 (VOLATILE이 아니거나 event_cvar_r 없음)
                        if direction != 0:
                            action = "LONG" if direction == 1 else "SHORT"
                            filter_reason = "PASS: All filters passed"
                            logger.info(f"[FUNNEL_FILTER] {symbol} | {filter_reason} | direction={direction} action={action}")
                        else:
                            filter_reason = "DIRECTION_FILTER: direction=0 (no trade)"
                            logger.info(f"[FUNNEL_FILTER] {symbol} | {filter_reason}")
        
        # 필터링 결과 로그
        logger.info(
            f"[FUNNEL_FILTER] {symbol} | regime={regime} | "
            f"ev_for_filter={ev_for_filter:.6f} (policy_ev_mix={policy_ev_mix:.6f}, ev_pct={ev_pct:.6f}) | "
            f"win_rate={win_rate:.4f} (floor={win_floors.get(regime, 0.52):.4f}) | "
            f"cvar1={cvar1:.6f} (floor={cvar_floors.get(regime, -0.008):.6f}) | "
            f"event_cvar_r={event_cvar_r} | "
            f"action={action} | reason={filter_reason}"
        )

        # ✅ 검증 포인트: mc_h_desc="exit-policy" 표시
        # best_h가 None이거나 0이어도 "exit-policy"는 항상 포함
        best_h_val = metrics.get("best_h") or 0
        best_desc = f"{best_h_val}초(exit-policy)" if best_h_val > 0 else "exit-policy"
        
        # ✅ Step A: MC(15초) 원인 확인 로그
        picked_h = metrics.get("best_h")
        horizons_used = self.horizons
        dt_used = self.dt
        # horizon index 찾기
        idx = None
        if picked_h is not None:
            try:
                idx = list(horizons_used).index(picked_h) if picked_h in horizons_used else -1
            except Exception:
                idx = -1
        logger.info(
            f"[MC_SEC_DEBUG] {symbol} | "
            f"picked_h={picked_h} | "
            f"horizons={horizons_used} | "
            f"dt={dt_used:.10f} | "
            f"idx={idx} | "
            f"best_desc='{best_desc}'"
        )

        # 멀티호라이즌 보조 메타(예: 60s/180s) 추출/생성
        ev_60 = None
        win_60 = None
        cvar_60 = None
        ev_180 = metrics.get("ev_mid")
        win_180 = metrics.get("win_mid")
        cvar_180 = metrics.get("cvar_mid")
        ensemble_mid_boost = 1.0
        if win_180 is not None:
            try:
                ensemble_mid_boost = 0.85 + 0.6 * max(0.0, min(1.0, (win_180 - 0.52) / 0.18))
            except Exception:
                ensemble_mid_boost = 1.0

        # [EV_DEBUG] metrics에서 EV 값 확인
        ev_value = float(metrics.get("ev", 0.0) or 0.0)
        win_value = float(metrics.get("win", 0.0) or 0.0)
        cvar_value = float(metrics.get("cvar", 0.0) or 0.0)
        policy_ev_mix = float(metrics.get("policy_ev_mix", 0.0) or 0.0)
        direction_policy = int(metrics.get("direction", 0))
        
        # ✅ BTC 상관관계 반영 (리스크 관리)
        btc_corr = float(metrics.get("btc_corr", 0.0) or 0.0)
        if btc_corr > 0.7:
            # 상관관계가 높으면 Kelly 비중 축소 (분산투자 효과 감소)
            kelly_value *= 0.8
            logger.info(f"[RISK] {symbol} High BTC corr={btc_corr:.2f} -> Kelly reduced to {kelly_value:.2f}")

        
        # ✅ 레버리지 최적화에서 계산한 best_net_ev를 사용 (레버리지와 수수료를 고려한 net EV)
        # best_net_ev가 계산되었고 양수이면 이를 사용, 아니면 metrics의 ev 사용
        # ✅ metrics의 ev가 0이면 policy_ev_mix를 직접 사용
        metrics_ev = float(metrics.get("ev", 0.0) or 0.0)
        policy_ev_mix_from_meta = float(metrics.get("policy_ev_mix", 0.0) or 0.0)
        
        # ✅ LINK 심볼의 경우 EV 값 검증
        if symbol.startswith("LINK"):
            logger.warning(f"[EV_VALIDATION_LINK] {symbol} | EV 선택 전: best_net_ev={best_net_ev} metrics_ev={metrics_ev:.6f} policy_ev_mix={policy_ev_mix_from_meta:.6f}")
            print(f"[EV_VALIDATION_LINK] {symbol} | EV 선택 전: best_net_ev={best_net_ev} metrics_ev={metrics_ev:.6f} policy_ev_mix={policy_ev_mix_from_meta:.6f}")
        
        if best_net_ev is not None and best_net_ev > 0:
            ev_value = float(best_net_ev)
            if symbol.startswith("LINK"):
                logger.warning(f"[EV_VALIDATION_LINK] {symbol} | ⚠️  Using best_net_ev={best_net_ev:.6f} (leverage-optimized)")
                print(f"[EV_VALIDATION_LINK] {symbol} | ⚠️  Using best_net_ev={best_net_ev:.6f} (leverage-optimized)")
            print(f"[EV_DEBUG] mc_engine.decide: Using best_net_ev={best_net_ev:.6f} (leverage-optimized) instead of metrics.ev={metrics_ev:.6f}")
            logger.info(f"[EV_DEBUG] mc_engine.decide: Using best_net_ev={best_net_ev:.6f} (leverage-optimized)")
        elif metrics_ev == 0.0 and policy_ev_mix_from_meta != 0.0:
            # metrics.ev가 0이지만 policy_ev_mix가 있으면 사용
            ev_value = float(policy_ev_mix_from_meta)
            if symbol.startswith("LINK"):
                logger.warning(f"[EV_VALIDATION_LINK] {symbol} | ⚠️  Using policy_ev_mix={policy_ev_mix_from_meta:.6f} (metrics.ev=0.0)")
                print(f"[EV_VALIDATION_LINK] {symbol} | ⚠️  Using policy_ev_mix={policy_ev_mix_from_meta:.6f} (metrics.ev=0.0)")
            print(f"[EV_DEBUG] mc_engine.decide: metrics.ev=0.0, using policy_ev_mix={policy_ev_mix_from_meta:.6f} instead")
            logger.info(f"[EV_DEBUG] mc_engine.decide: metrics.ev=0.0, using policy_ev_mix={policy_ev_mix_from_meta:.6f}")
        elif metrics_ev == 0.0:
            # ✅ metrics.ev가 0이고 policy_ev_mix도 0이면, policy_ev_mix_long/short를 직접 확인
            policy_long = float(metrics.get("policy_ev_mix_long", 0.0) or 0.0)
            policy_short = float(metrics.get("policy_ev_mix_short", 0.0) or 0.0)
            # 절대값이 큰 쪽 선택 (음수여도 표시)
            if abs(policy_long) > abs(policy_short):
                ev_value = policy_long
            else:
                ev_value = policy_short
            print(f"[EV_DEBUG] mc_engine.decide: metrics.ev=0, policy_ev_mix=0, using policy_ev_mix_long={policy_long:.6f} or policy_ev_mix_short={policy_short:.6f} -> selected={ev_value:.6f}")
            logger.info(f"[EV_DEBUG] mc_engine.decide: using policy_ev from long/short: {ev_value:.6f}")
        else:
            ev_value = float(metrics_ev)
            if symbol.startswith("LINK") and abs(ev_value) > 0.1:
                logger.warning(f"[EV_VALIDATION_LINK] {symbol} | ⚠️  Using metrics.ev={ev_value:.6f} (큰 값!)")
                print(f"[EV_VALIDATION_LINK] {symbol} | ⚠️  Using metrics.ev={ev_value:.6f} (큰 값!)")
            print(f"[EV_DEBUG] mc_engine.decide: Using metrics.ev={ev_value:.6f} (best_net_ev={best_net_ev} policy_ev_mix={policy_ev_mix_from_meta:.6f})")
        
        print(f"[EV_DEBUG] mc_engine.decide: symbol={symbol} metrics type={type(metrics)} metrics keys={list(metrics.keys())[:50]}")
        print(f"[EV_DEBUG] mc_engine.decide: symbol={symbol} ev={ev_value} (final) win={win_value} cvar={cvar_value} action={action} policy_ev_mix={policy_ev_mix} direction_policy={direction_policy}")
        logger.info(f"[EV_DEBUG] mc_engine.decide: symbol={symbol} ev={ev_value} win={win_value} cvar={cvar_value} action={action} policy_ev_mix={policy_ev_mix} direction_policy={direction_policy}")
        
        # ✅ 모든 심볼의 최종 EV 값 검증 (LINK 제외)
        if not symbol.startswith("LINK") and ev_value < 0:
            logger.warning(f"[EV_VALIDATION_NEG] {symbol} | ⚠️  최종 EV가 음수: ev_value={ev_value:.6f} metrics_ev={metrics_ev:.6f} policy_ev_mix={policy_ev_mix_from_meta:.6f} best_net_ev={best_net_ev}")
            print(f"[EV_VALIDATION_NEG] {symbol} | ⚠️  최종 EV가 음수: ev_value={ev_value:.6f} metrics_ev={metrics_ev:.6f} policy_ev_mix={policy_ev_mix_from_meta:.6f} best_net_ev={best_net_ev}")
        
        return {
            "action": action,
            "ev": ev_value,
            "ev_raw": float(metrics.get("ev_raw", ev_value)),
            "confidence": win_value,
            "reason": f"MC({best_desc}) {regime_ctx} EV {ev_value*100:.2f}% Win {win_value*100:.1f}% CVaR {cvar_value*100:.2f}%",
            "meta": {
                "regime": regime_ctx,
                "best_horizon_desc": best_desc,
                "best_horizon_steps": int(metrics["best_h"]),
                "ev": float(metrics["ev"]),
                "ev_raw": float(metrics.get("ev_raw", metrics["ev"])),
                "ev_for_gate": float(metrics.get("ev_for_gate", metrics["ev"])),
                "ev1": float(metrics.get("ev1", metrics["ev"])),
                "win_rate": float(metrics["win"]),
                "win_rate1": float(metrics.get("win1", metrics["win"])),
                "cvar05": float(metrics["cvar"]),
                "cvar05_lev1": float(metrics.get("cvar1", metrics["cvar"])),
                # hold-to-horizon (기존)
                "ev_hold": float(metrics.get("ev_hold", 0.0) or 0.0),
                "p_pos_hold": float(metrics.get("p_pos_hold", 0.0) or 0.0),
                "cvar_hold": float(metrics.get("cvar_hold", 0.0) or 0.0),
                # 정책-청산 반영 (multi-horizon mix)
                "policy_ev_mix": float(metrics.get("policy_ev_mix", metrics["ev"]) or 0.0),
                "policy_p_pos_mix": float(metrics.get("policy_p_pos_mix", metrics["win"]) or 0.0),
                "policy_cvar_mix": float(metrics.get("policy_cvar_mix", 0.0) or 0.0),
                "policy_cvar_gate": float(metrics.get("policy_cvar_gate", metrics["cvar"]) or 0.0),
                "policy_cvar_gate_mode": metrics.get("policy_cvar_gate_mode"),
                "policy_ev_mix_long": float(metrics.get("policy_ev_mix_long", 0.0) or 0.0),
                "policy_ev_mix_short": float(metrics.get("policy_ev_mix_short", 0.0) or 0.0),
                "policy_p_pos_mix_long": float(metrics.get("policy_p_pos_mix_long", 0.0) or 0.0),
                "policy_p_pos_mix_short": float(metrics.get("policy_p_pos_mix_short", 0.0) or 0.0),
                "policy_signal_strength": float(metrics.get("policy_signal_strength", 0.0) or 0.0),
                "policy_half_life_sec": float(metrics.get("policy_half_life_sec", 0.0) or 0.0),
                "policy_h_eff_sec": float(metrics.get("policy_h_eff_sec", 0.0) or 0.0),
                "policy_w_short_sum": float(metrics.get("policy_w_short_sum", 0.0) or 0.0),
                "policy_momentum_z": float(metrics.get("policy_momentum_z", 0.0) or 0.0),
                "policy_ofi_z": float(metrics.get("policy_ofi_z", 0.0) or 0.0),
                "paths_seed_base": metrics.get("paths_seed_base"),
                "paths_reused": metrics.get("paths_reused"),
                "policy_weight_peak_h": metrics.get("policy_weight_peak_h"),
                "policy_horizons": metrics.get("policy_horizons"),
                "policy_w_h": metrics.get("policy_w_h"),
                # hold-to-horizon vs exit-policy diagnostics
                "hold_best_h_long": metrics.get("hold_best_h_long"),
                "hold_best_h_short": metrics.get("hold_best_h_short"),
                "hold_best_ev_long": metrics.get("hold_best_ev_long"),
                "hold_best_ev_short": metrics.get("hold_best_ev_short"),
                "policy_exit_unrealized_dd_frac": metrics.get("policy_exit_unrealized_dd_frac"),
                "policy_exit_hold_bad_frac": metrics.get("policy_exit_hold_bad_frac"),
                "policy_exit_score_flip_frac": metrics.get("policy_exit_score_flip_frac"),
                "ev_decomp_mu_annual": metrics.get("ev_decomp_mu_annual"),
                "ev_decomp_mu_per_sec": metrics.get("ev_decomp_mu_per_sec"),
                "ev_decomp_fee_roundtrip_total": metrics.get("ev_decomp_fee_roundtrip_total"),
                "ev_decomp_gross_long_600": metrics.get("ev_decomp_gross_long_600"),
                "ev_decomp_gross_long_1800": metrics.get("ev_decomp_gross_long_1800"),
                "ev_decomp_gross_short_600": metrics.get("ev_decomp_gross_short_600"),
                "ev_decomp_gross_short_1800": metrics.get("ev_decomp_gross_short_1800"),
                "ev_decomp_net_long_600": metrics.get("ev_decomp_net_long_600"),
                "ev_decomp_net_long_1800": metrics.get("ev_decomp_net_long_1800"),
                "ev_decomp_net_short_600": metrics.get("ev_decomp_net_short_600"),
                "ev_decomp_net_short_1800": metrics.get("ev_decomp_net_short_1800"),
                "ev_decomp_mu_req_annual_600": metrics.get("ev_decomp_mu_req_annual_600"),
                "ev_decomp_mu_req_annual_1800": metrics.get("ev_decomp_mu_req_annual_1800"),
                "ev_decomp_mu_req_annual_exit_mean": metrics.get("ev_decomp_mu_req_annual_exit_mean"),
                "policy_ev_per_h": metrics.get("policy_ev_per_h"),
                "policy_p_pos_per_h": metrics.get("policy_p_pos_per_h"),
                "policy_cvar_per_h": metrics.get("policy_cvar_per_h"),
                "policy_exit_reason_counts": metrics.get("policy_exit_reason_counts"),
                "exit_time_mean_sec": float(metrics.get("exit_time_mean_sec", 0.0) or 0.0),
                # backward-compat aliases
                "ev_exit_policy_30m": float(metrics.get("ev_exit_policy_30m", metrics.get("policy_ev_mix", metrics["ev"])) or 0.0),
                "p_pos_exit_policy_30m": float(metrics.get("p_pos_exit_policy_30m", metrics.get("policy_p_pos_mix", metrics["win"])) or 0.0),
                "cvar_exit_policy_30m": float(metrics.get("cvar_exit_policy_30m", metrics.get("policy_cvar_gate", metrics["cvar"])) or 0.0),
                "exit_reason_counts": metrics.get("exit_reason_counts"),
                "ev_mid": metrics.get("ev_mid"),
                "win_mid": metrics.get("win_mid"),
                "cvar_mid": metrics.get("cvar_mid"),
                "ev_by_horizon": metrics.get("ev_by_horizon"),
                "win_by_horizon": metrics.get("win_by_horizon"),
                "cvar_by_horizon": metrics.get("cvar_by_horizon"),
                "horizon_seq": metrics.get("horizon_seq"),
                "horizon_weights": metrics.get("horizon_weights"),
                # 비용/유동성 메타(대시보드/로그용)
                "execution_cost": float(metrics.get("execution_cost", 0.0) or 0.0),
                "fee_roundtrip_total": float(metrics.get("fee_roundtrip_total", metrics.get("execution_cost", 0.0)) or 0.0),
                "fee_roundtrip_base": float(metrics.get("fee_roundtrip_base", 0.0) or 0.0),
                "slippage_dyn": float(metrics.get("slippage_dyn", metrics.get("slippage_pct", 0.0)) or 0.0),
                "expected_spread_cost": float(metrics.get("expected_spread_cost", 0.0) or 0.0),
                "spread_pct": float(metrics.get("spread_pct", 0.0) or 0.0),
                "liq_score": float(metrics.get("liq_score", 0.0) or 0.0),
                # maker_then_market 비용 모델 디버그
                "exec_mode": metrics.get("exec_mode"),
                "p_maker": metrics.get("p_maker"),
                "pmaker_override_used": metrics.get("pmaker_override_used"),
                "pmaker_entry": metrics.get("pmaker_entry"),
                "pmaker_entry_delay_sec": metrics.get("pmaker_entry_delay_sec"),
                "pmaker_exit": metrics.get("pmaker_exit"),
                "pmaker_exit_delay_sec": metrics.get("pmaker_exit_delay_sec"),
                "pmaker_entry_delay_penalty_r": metrics.get("pmaker_entry_delay_penalty_r"),
                "pmaker_exit_delay_penalty_r": metrics.get("pmaker_exit_delay_penalty_r"),
                "policy_entry_shift_steps": metrics.get("policy_entry_shift_steps"),
                "policy_horizon_eff_sec": metrics.get("policy_horizon_eff_sec"),
                "pmaker_discount": metrics.get("pmaker_discount"),
                "fee_roundtrip_fee_mix": metrics.get("fee_roundtrip_fee_mix"),
                "fee_roundtrip_fee_taker": metrics.get("fee_roundtrip_fee_taker"),
                "fee_roundtrip_fee_maker": metrics.get("fee_roundtrip_fee_maker"),
                "slippage_dyn_raw": metrics.get("slippage_dyn_raw"),
                "expected_spread_cost_raw": metrics.get("expected_spread_cost_raw"),
                "ev_60": ev_60,
                "win_60": win_60,
                "cvar_60": cvar_60,
                "ev_180": ev_180,
                "win_180": win_180,
                "cvar_180": cvar_180,
                "ensemble_mid_boost": ensemble_mid_boost,
                "ev_entry_threshold": metrics.get("ev_floor"),
                "win_entry_threshold": metrics.get("win_floor"),
                "cvar_entry_threshold": metrics.get("cvar_floor"),
                "kelly": float(metrics["kelly"]),
                "size_fraction": float(metrics["size_frac"]),
                "direction": int(metrics["direction"]),
                "mu_adjusted": float(metrics.get("mu_adj", 0.0)),
                # μ(alpha) decomposition (debug / diagnosis)
                "mu_alpha": metrics.get("mu_alpha"),
                "mu_alpha_raw": metrics.get("mu_alpha_raw"),
                "mu_alpha_cap": metrics.get("mu_alpha_cap"),
                "mu_alpha_regime_scale": metrics.get("mu_alpha_regime_scale"),
                "mu_alpha_mom": metrics.get("mu_alpha_mom"),
                "mu_alpha_ofi": metrics.get("mu_alpha_ofi"),
                "mu_alpha_ofi_score_clipped": metrics.get("mu_alpha_ofi_score_clipped"),
                "mu_alpha_mom_15": metrics.get("mu_alpha_mom_15"),
                "mu_alpha_mom_30": metrics.get("mu_alpha_mom_30"),
                "mu_alpha_mom_60": metrics.get("mu_alpha_mom_60"),
                "mu_alpha_mom_120": metrics.get("mu_alpha_mom_120"),
                "mu_alpha_pmaker_fill_rate": metrics.get("mu_alpha_pmaker_fill_rate"),
                "mu_alpha_pmaker_boost": metrics.get("mu_alpha_pmaker_boost"),
                "mu_alpha_before_pmaker": metrics.get("mu_alpha_before_pmaker"),
                "event_p_tp": metrics.get("event_p_tp"),
                "event_p_sl": metrics.get("event_p_sl"),
                "event_p_timeout": metrics.get("event_p_timeout"),
                "event_ev_r": metrics.get("event_ev_r"),
                "event_cvar_r": metrics.get("event_cvar_r"),
                "event_ev_pct": metrics.get("event_ev_pct"),
                "event_cvar_pct": metrics.get("event_cvar_pct"),
                "event_t_median": metrics.get("event_t_median"),
                "event_t_mean": metrics.get("event_t_mean"),
                "params": {
                    "min_win": params.min_win,
                    "profit_target": params.profit_target,
                    "ofi_weight": params.ofi_weight,
                    "max_kelly": params.max_kelly,
                    "cvar_alpha": params.cvar_alpha,
                    "cvar_scale": params.cvar_scale,
                    "n_paths": params.n_paths,
                },
            },
            # EngineHub가 그대로 받게
            # 필터를 통과한 경우에만 size 계산, 통과하지 못한 경우 0
            "size_frac": float(optimal_size) if (action != "WAIT" and optimal_size > 0) else (float(metrics["size_frac"]) if action != "WAIT" else 0.0),
            # 최적 레버리지 정보 추가
            "optimal_leverage": float(optimal_leverage) if action != "WAIT" else 0.0,
            "optimal_size": float(optimal_size) if action != "WAIT" else 0.0,
        }
        
        # 최적 레버리지 정보를 meta에 추가
        ret["meta"]["optimal_leverage"] = float(optimal_leverage)
        ret["meta"]["optimal_size"] = float(optimal_size)
        ret["meta"]["leverage_used"] = float(optimal_leverage)
        
        # Debug: log final return meta keys and pmaker_entry value
        ret_meta = ret.get("meta", {})
        print(f"[PMAKER_DEBUG] {symbol} | mc_engine.decide: return meta keys (sample): {list(ret_meta.keys())[:20]}... pmaker_entry={ret_meta.get('pmaker_entry')} pmaker_entry_delay_sec={ret_meta.get('pmaker_entry_delay_sec')}")
        logger.info(f"[PMAKER_DEBUG] {symbol} | mc_engine.decide: return meta pmaker_entry={ret_meta.get('pmaker_entry')} pmaker_entry_delay_sec={ret_meta.get('pmaker_entry_delay_sec')} pmaker_exit={ret_meta.get('pmaker_exit')} pmaker_exit_delay_sec={ret_meta.get('pmaker_exit_delay_sec')}")
        
        return ret

    @staticmethod
    def _weights_for_horizons(hs, signal_strength: float):
        """
        Rule-based prior weights for horizons (used as base/prior when AlphaHitMLP is enabled).
        """
        h_arr = np.asarray(hs if hs else [], dtype=np.float64)
        if h_arr.size == 0:
            return np.asarray([], dtype=np.float64)
        s = float(np.clip(signal_strength, 0.0, 4.0))
        half_life = 1800.0 / (1.0 + s)
        decay = np.exp(-h_arr / max(1e-9, half_life))
        total = float(np.sum(decay))
        if total <= 0.0:
            return np.full(h_arr.shape, 1.0 / float(h_arr.size), dtype=np.float64)
        return decay / total
    def _execution_mix_from_survival(
        self,
        meta: Dict[str, Any],
        fee_maker: float,
        fee_taker: float,
        horizon_sec: int,
        sigma_per_sec: float,
        prefix: str,
        delay_penalty_mult: float,
    ) -> Dict[str, float]:
        """
        Uses survival-based pmaker + delay from decision.meta with a prefix:
          prefix="pmaker_entry" or "pmaker_exit"
        """
        p_fill = float(meta.get(prefix) or 0.0)
        delay_sec = float(meta.get(f"{prefix}_delay_sec") or 0.0)
        delay_cond_sec = float(meta.get(f"{prefix}_delay_cond_sec") or delay_sec)

        fee_mix = p_fill * fee_maker + (1.0 - p_fill) * fee_taker
        delay_penalty_r = float(delay_penalty_mult) * sigma_per_sec * math.sqrt(max(0.0, delay_sec))
        horizon_eff = max(1, int(round(horizon_sec - delay_sec)))

        return {
            "p_fill": p_fill,
            "delay_sec": delay_sec,
            "delay_cond_sec": delay_cond_sec,
            "fee_mix": fee_mix,
            "delay_penalty_r": delay_penalty_r,
            "horizon_eff_sec": float(horizon_eff),
        }

    def _sigma_per_sec(self, sigma: float, dt: float) -> float:
        """
        Convert annualized sigma to per-second sigma.
        """
        if dt <= 0:
            return 0.0
        # sigma is annualized, dt is in seconds/year
        # per-second sigma = sigma * sqrt(dt)
        return float(sigma) * math.sqrt(float(dt))
    def compute_exit_policy_metrics(
        self,
        *,
        symbol: str,
        price: float,
        mu: float,
        sigma: float,
        leverage: float,
        direction: int,
        fee_roundtrip: float,
        exec_oneway: float,
        impact_cost: float,
        regime: str,
        horizon_sec: int,
        decision_dt_sec: int = 5,
        seed: int = 0,
        cvar_alpha: float = 0.05,
        price_paths: Optional[np.ndarray] = None,
        decision_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        정책 롤포워드를 단일 horizon에서 돌려 p_pos/ev 기준을 반환한다.
        """
        print(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: decision_meta={decision_meta} keys={list(decision_meta.keys()) if decision_meta else []}")
        logger.info(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: decision_meta={decision_meta} keys={list(decision_meta.keys()) if decision_meta else []}")
        h = int(horizon_sec)
        dt = float(self.dt)
        # decision_dt_sec defaults to 5 seconds per step (used for shift calculation)
        decision_dt_sec = int(getattr(self, "POLICY_DECISION_DT_SEC", 5))
        h_pts = int(max(2, h + 1))  # t=0 포함 (0..h_sec)

        # ---- NEW: survival-based execution mixing (entry + exit) ----
        entry_mix = None
        exit_mix = None
        sigma_per_sec = float(self._sigma_per_sec(sigma=sigma, dt=dt))

        extra_entry_delay_penalty_r = 0.0
        extra_exit_delay_penalty_r = 0.0
        h_eff = h
        start_shift_steps = 0
        mu_adj = float(mu)

        if decision_meta is not None and ("pmaker_entry" in decision_meta):
            fee_maker = float(decision_meta.get("fee_roundtrip_maker", fee_roundtrip))
            fee_taker = float(decision_meta.get("fee_roundtrip_taker", fee_roundtrip))
            entry_mix = self._execution_mix_from_survival(
                meta=decision_meta,
                fee_maker=fee_maker,
                fee_taker=fee_taker,
                horizon_sec=h,
                sigma_per_sec=sigma_per_sec,
                prefix="pmaker_entry",
                delay_penalty_mult=self.PMAKER_DELAY_PENALTY_MULT,
            )
            # entry는 roundtrip fee의 "진입+청산"이 섞여 들어가면 과중첩될 수 있어.
            # 여기서는 fee_roundtrip를 entry_mix로 덮되, exit는 아래에서 exec_oneway/추가페널티로 반영.
            fee_roundtrip = float(entry_mix["fee_mix"])
            extra_entry_delay_penalty_r = float(entry_mix["delay_penalty_r"])

            # entry delay는 시뮬레이션 시작을 늦추는 방식(가장 현실적)
            # dt is in years (1.0 / SECONDS_PER_YEAR), but we need seconds per step
            # Use decision_dt_sec (default 5 seconds per step) for shift calculation
            if self.PMAKER_ENTRY_DELAY_SHIFT:
                delay_sec_val = float(entry_mix["delay_sec"])
                dt_step_sec = float(decision_dt_sec)  # decision_dt_sec is in seconds per step (default 5)
                if dt_step_sec > 0:
                    start_shift_steps = int(round(delay_sec_val / dt_step_sec))
                    start_shift_steps = max(0, min(start_shift_steps, h_pts - 2))  # Cap at horizon
                    print(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: delay_sec={delay_sec_val:.4f} dt_step_sec={dt_step_sec} start_shift_steps={start_shift_steps}")
                else:
                    start_shift_steps = 0
                if start_shift_steps > 0:
                    # horizon도 그만큼 줄어듦
                    h_eff = max(1, h - int(round(delay_sec_val)))

            # alpha decay: delay 동안 알파가 죽는다 -> mu를 약화
            tau = max(1e-6, float(self.ALPHA_DELAY_DECAY_TAU_SEC))
            decay = math.exp(-float(entry_mix["delay_sec"]) / tau)
            mu_adj = float(mu_adj) * decay

        if decision_meta is not None and ("pmaker_exit" in decision_meta):
            # exit는 one-way 성격이므로 fee/penalty를 별도로 더 보수적으로 반영
            fee_maker_exit = float(decision_meta.get("fee_oneway_maker", exec_oneway))
            fee_taker_exit = float(decision_meta.get("fee_oneway_taker", exec_oneway))
            exit_mix = self._execution_mix_from_survival(
                meta=decision_meta,
                fee_maker=fee_maker_exit,
                fee_taker=fee_taker_exit,
                horizon_sec=h_eff,
                sigma_per_sec=sigma_per_sec,
                prefix="pmaker_exit",
                delay_penalty_mult=self.PMAKER_EXIT_DELAY_PENALTY_MULT,
            )
            # exit one-way 예상비용을 덮어쓰기
            exec_oneway = float(exit_mix["fee_mix"])
            extra_exit_delay_penalty_r = float(exit_mix["delay_penalty_r"])
            # exit delay는 실질적으로 청산이 늦어져 변동성 노출이 늘어난다고 보고 horizon을 추가 감소
            h_eff = max(1, int(round(float(exit_mix["horizon_eff_sec"]))))

        # ✅ [HORIZON_SLICING] 전달된 price_paths를 슬라이싱만 사용 (시뮬레이션 재생성 방지)
        # evaluate_entry_metrics에서 이미 최대 horizon 길이로 생성하여 전달하므로,
        # 여기서는 슬라이싱만 수행하여 Horizon Slicing 기법 적용
        paths = None
        if price_paths is not None:
            try:
                arr = np.asarray(price_paths, dtype=np.float64)
                if arr.ndim == 2 and arr.shape[0] > 0:
                    # ✅ 슬라이싱: 전달된 full_paths에서 현재 horizon 길이만큼만 사용
                    if arr.shape[1] >= h_pts:
                        paths = arr[:, :h_pts]
                    else:
                        # 전달된 경로가 현재 horizon보다 짧으면 사용 가능한 만큼만 사용
                        paths = arr[:, :]
                        logger.warning(f"[HORIZON_SLICING] {symbol} | price_paths shape={arr.shape} < h_pts={h_pts}, using available length")
            except Exception as e:
                logger.warning(f"[HORIZON_SLICING] {symbol} | Failed to slice price_paths: {e}")
                paths = None

        # ✅ [HORIZON_SLICING] price_paths가 전달되지 않았거나 유효하지 않은 경우에만 시뮬레이션 생성
        # (일반적으로는 evaluate_entry_metrics에서 전달되므로 이 분기는 거의 실행되지 않음)
        if paths is None:
            n_paths = int(self.N_PATHS_EXIT_POLICY)
            # simulate_paths_price는 1초 단위 경로를 생성하므로 dt=1.0을 사용하고,
            # mu와 sigma는 초당 단위로 변환해서 전달해야 함
            dt_path = 1.0  # 1초 단위
            mu_ps_for_paths = float(mu_adj) * dt  # 연율 -> 초당 변환
            sigma_ps_for_paths = float(sigma) * math.sqrt(dt)  # 연율 -> 초당 변환
            # simulate_paths_price returns shape (n_paths, n_steps+1) with index == seconds
            n_steps = int(h)
            logger.info(f"[HORIZON_SLICING] {symbol} | price_paths not provided, generating new paths for h={h}s (n_steps={n_steps})")
            paths = self.simulate_paths_price(
                seed=seed,
                s0=price,
                mu=mu_ps_for_paths,  # 초당 단위로 변환된 mu
                sigma=sigma_ps_for_paths,  # 초당 단위로 변환된 sigma
                n_paths=n_paths,
                n_steps=n_steps,
                dt=dt_path,
            )[:, :h_pts]
        else:
            logger.debug(f"[HORIZON_SLICING] {symbol} | Using sliced price_paths: original_shape={price_paths.shape if price_paths is not None else None}, sliced_shape={paths.shape}, h_pts={h_pts}")

        # ---- apply entry delay as path shift (entry happens later) ----
        pp = paths
        if pp is not None and start_shift_steps > 0:
            # pp shape: (n_paths, n_steps)
            if pp.shape[1] > start_shift_steps + 2:
                pp = pp[:, start_shift_steps:]
            else:
                # 너무 짧으면 그대로 두되, horizon_eff는 최소 1로
                h_eff = max(1, min(h_eff, pp.shape[1] - 1))

        h_eff_pts = int(max(2, h_eff + 1))
        if pp is not None and pp.shape[1] > h_eff_pts:
            pp = pp[:, :h_eff_pts]

        # mu와 sigma를 연율(Annualized)에서 초당(Per-second) 단위로 변환
        mu_ps = float(mu_adj) * dt  # mu_adj * self.dt
        sigma_ps = float(sigma) * math.sqrt(dt)  # sigma * sqrt(self.dt)

        res = simulate_exit_policy_rollforward(
            price_paths=pp,
            s0=float(price),
            mu=mu_ps,  # 초당 단위로 변환된 mu
            sigma=sigma_ps,  # 초당 단위로 변환된 sigma
            leverage=float(leverage),
            fee_roundtrip=float(fee_roundtrip),
            exec_oneway=float(exec_oneway),
            impact_cost=float(impact_cost),
            regime=str(regime),
            decision_dt_sec=int(decision_dt_sec),
            horizon_sec=int(h_eff_pts),
            min_hold_sec=int(self.MIN_HOLD_SEC_DIRECTIONAL),
            flip_confirm_ticks=int(self.FLIP_CONFIRM_TICKS),
            hold_bad_ticks=int(self.POLICY_HOLD_BAD_TICKS),
            p_pos_floor_enter=float(self.POLICY_P_POS_ENTER_BY_REGIME.get(regime, 0.52)),
            p_pos_floor_hold=float(self.POLICY_P_POS_HOLD_BY_REGIME.get(regime, 0.50)),
            p_sl_enter_ceiling=float(self.POLICY_P_SL_ENTER_MAX_BY_REGIME.get(regime, 0.20)),
            p_sl_hold_ceiling=float(self.POLICY_P_SL_HOLD_MAX_BY_REGIME.get(regime, 0.25)),
            p_sl_emergency=float(self.POLICY_P_SL_EMERGENCY),
            p_tp_floor_enter=float(self.POLICY_P_TP_ENTER_MIN_BY_REGIME.get(regime, 0.15)),
            p_tp_floor_hold=float(self.POLICY_P_TP_HOLD_MIN_BY_REGIME.get(regime, 0.12)),
            score_margin=float(self.SCORE_MARGIN_DEFAULT),
            soft_floor=float(self.POLICY_VALUE_SOFT_FLOOR_AFTER_COST),
            side_now=int(direction),
            enable_dd_stop=True,
            dd_stop_roe=-0.02,
        )

        ev = float(res.get("ev_exit", 0.0))
        # delay penalties: entry + exit
        ev_adj = ev - float(extra_entry_delay_penalty_r) - float(extra_exit_delay_penalty_r)
        res["ev_exit"] = ev_adj

        # strict maker-only: entry fill이 안되면 거래가 성립 안 함.
        if self.PMAKER_STRICT and entry_mix is not None:
            p_fill = float(entry_mix["p_fill"])
            res["ev_exit"] = float(res["ev_exit"]) * p_fill
            res["p_pos_exit"] = float(res.get("p_pos_exit", 0.0)) * p_fill

        net_out_raw = res.get("net_out")
        if net_out_raw is None:
            net_out_raw = np.zeros((0,), dtype=np.float64)
        net_out = np.asarray(net_out_raw, dtype=np.float64)
        cvar_exit = (
            float(cvar_ensemble(net_out, alpha=float(cvar_alpha))) if net_out.size else 0.0
        )

        out_meta = res.get("meta", {}) or {}
        if entry_mix is not None:
            out_meta.update({
                "pmaker_entry_used": float(entry_mix["p_fill"]),
                "pmaker_entry_delay_used_sec": float(entry_mix["delay_sec"]),
                "pmaker_entry_delay_cond_used_sec": float(entry_mix["delay_cond_sec"]),
                "pmaker_entry_fee_mix_used": float(entry_mix["fee_mix"]),
                "pmaker_entry_delay_penalty_r": float(extra_entry_delay_penalty_r),
                "policy_entry_shift_steps": int(start_shift_steps),
            })
            print(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: entry_mix added, delay_penalty_r={extra_entry_delay_penalty_r:.6f} shift_steps={start_shift_steps}")
            logger.info(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: entry_mix added, delay_penalty_r={extra_entry_delay_penalty_r:.6f} shift_steps={start_shift_steps}")
        else:
            print(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: entry_mix is None")
            logger.info(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: entry_mix is None")
        if exit_mix is not None:
            out_meta.update({
                "pmaker_exit_used": float(exit_mix["p_fill"]),
                "pmaker_exit_delay_used_sec": float(exit_mix["delay_sec"]),
                "pmaker_exit_delay_cond_used_sec": float(exit_mix["delay_cond_sec"]),
                "pmaker_exit_fee_mix_used": float(exit_mix["fee_mix"]),
                "pmaker_exit_delay_penalty_r": float(extra_exit_delay_penalty_r),
            })
            print(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: exit_mix added, delay_penalty_r={extra_exit_delay_penalty_r:.6f}")
            logger.info(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: exit_mix added, delay_penalty_r={extra_exit_delay_penalty_r:.6f}")
        else:
            print(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: exit_mix is None")
            logger.info(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: exit_mix is None")
        out_meta["policy_horizon_eff_sec"] = int(h_eff)
        print(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: out_meta keys={list(out_meta.keys())} horizon_eff={h_eff}")
        logger.info(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: out_meta keys={list(out_meta.keys())} horizon_eff={h_eff}")
        res["meta"] = out_meta

        return {
            "ok": True,
            "symbol": symbol,
            "dir": int(direction),
            "horizon_sec": int(horizon_sec),
            "ev_exit_policy": float(res.get("ev_exit", 0.0)),
            "p_pos_exit_policy": float(res.get("p_pos_exit", 0.0)),
            "exit_t_mean_sec": float(res.get("exit_t_mean_sec", 0.0)),
            "exit_t_p50_sec": float(res.get("exit_t_p50_sec", 0.0)),
            "exit_reason_counts": res.get("exit_reason_counts"),
            "cvar_exit_policy": float(cvar_exit),
            "meta": out_meta,
            # ✅ TP/SL/Other 직접 집계 결과
            "p_tp": float(res.get("p_tp", 0.0)),
            "p_sl": float(res.get("p_sl", 0.0)),
            "p_other": float(res.get("p_other", 0.0)),
            "tp_r_actual": float(res.get("tp_r_actual", 0.0)),  # 실제 TP 평균 수익률 (net)
            "sl_r_actual": float(res.get("sl_r_actual", 0.0)),  # 실제 SL 평균 수익률 (net)
            "other_r_actual": float(res.get("other_r_actual", 0.0)),  # 실제 other 평균 수익률 (net)
            "prob_sum_check": bool(res.get("prob_sum_check", False)),  # p_tp + p_sl + p_other == 1 검증
        }

    @staticmethod
    def _compress_reason_counts(counts: Any, *, top_k: int = 3) -> Dict[str, int]:
        """
        exit_reason_counts는 사유 문자열이 길어질 수 있으니, 상위 K개만 남기고 나머지는 _other로 합친다.
        """
        if not isinstance(counts, dict) or not counts:
            return {}
        items = []
        other = 0
        for k, v in counts.items():
            try:
                key = str(k)
                val = int(v)
            except Exception:
                continue
            if val <= 0:
                continue
            items.append((key, val))
        if not items:
            return {}
        items.sort(key=lambda x: x[1], reverse=True)
        top = items[: max(0, int(top_k))]
        if len(items) > len(top):
            other = sum(v for _, v in items[len(top) :])
        out = {k: int(v) for k, v in top}
        if other > 0:
            out["_other"] = int(other)
        return out

    def _extract_alpha_hit_features(
        self,
        symbol: str,
        price: float,
        mu: float,
        sigma: float,
        momentum_z: float,
        ofi_z: float,
        regime: str,
        leverage: float,
        ctx: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        """
        Extract features for AlphaHitMLP prediction.
        Returns [1, n_features] tensor or None if model not available.
        """
        if not self.alpha_hit_enabled or self.alpha_hit_trainer is None:
            return None

        try:
            def _f(val, default=0.0) -> float:
                try:
                    if val is None:
                        return float(default)
                    return float(val)
                except Exception:
                    return float(default)

            mu_alpha = _f(ctx.get("mu_alpha"), 0.0)
            features = [
                float(mu) * SECONDS_PER_YEAR,
                float(sigma) * math.sqrt(SECONDS_PER_YEAR),
                float(momentum_z),
                float(ofi_z),
                float(leverage),
                float(price),
                1.0 if regime == "bull" else 0.0,
                1.0 if regime == "bear" else 0.0,
                1.0 if regime == "chop" else 0.0,
                1.0 if regime == "volatile" else 0.0,
                _f(ctx.get("spread_pct"), 0.0),
                _f(ctx.get("kelly"), 0.0),
                _f(ctx.get("confidence"), 0.0),
                _f(ctx.get("ev"), 0.0),
                float(mu_alpha),
                0.0, 0.0, 0.0, 0.0, 0.0,
            ]
            n_feat = self.alpha_hit_trainer.model.cfg.n_features
            features = features[:n_feat] + [0.0] * max(0, n_feat - len(features))
            return torch.tensor([features], dtype=torch.float32)
        except Exception as e:
            logger.warning(f"[ALPHA_HIT] Feature extraction failed: {e}")
            return None

    def collect_alpha_hit_sample(
        self,
        symbol: str,
        features_np: np.ndarray,
        entry_ts_ms: int,
        exit_ts_ms: int,
        direction: int,
        exit_reason: str,
        tp_level: float = 0.01,
        sl_level: float = 0.01,
    ):
        """
        Collect training sample for AlphaHitMLP from realized trade outcomes.
        """
        if not self.alpha_hit_enabled or self.alpha_hit_trainer is None or features_np is None:
            return

        try:
            duration_sec = (exit_ts_ms - entry_ts_ms) / 1000.0
            policy_horizons = list(getattr(self, "POLICY_MULTI_HORIZONS_SEC", (60, 180, 300, 600, 900, 1800)))
            H = len(policy_horizons)

            y_tp_long = np.zeros(H, dtype=np.float32)
            y_sl_long = np.zeros(H, dtype=np.float32)
            y_tp_short = np.zeros(H, dtype=np.float32)
            y_sl_short = np.zeros(H, dtype=np.float32)

            for i, h in enumerate(policy_horizons):
                if duration_sec <= h:
                    if exit_reason == "TP":
                        if direction == 1:
                            y_tp_long[i] = 1.0
                        else:
                            y_tp_short[i] = 1.0
                    elif exit_reason == "SL":
                        if direction == 1:
                            y_sl_long[i] = 1.0
                        else:
                            y_sl_short[i] = 1.0

            self.alpha_hit_trainer.add_sample(
                x=features_np,
                y={
                    "tp_long": y_tp_long,
                    "sl_long": y_sl_long,
                    "tp_short": y_tp_short,
                    "sl_short": y_sl_short,
                },
                ts_ms=entry_ts_ms,
                symbol=symbol,
            )
        except Exception as e:
            logger.warning(f"[ALPHA_HIT] Failed to collect sample: {e}")

    def _predict_horizon_hit_probs(
        self,
        features: torch.Tensor,
        horizons: List[int],
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Predict TP/SL hit probabilities per horizon using AlphaHitMLP.
        """
        if not self.alpha_hit_enabled or self.alpha_hit_mlp is None or features is None:
            return None

        try:
            with torch.no_grad():
                pred = self.alpha_hit_mlp.predict(features)

            model_horizons = self.alpha_hit_mlp.cfg.horizons_sec
            H = len(horizons)
            p_tp_long = np.zeros(H, dtype=np.float64)
            p_sl_long = np.zeros(H, dtype=np.float64)
            p_tp_short = np.zeros(H, dtype=np.float64)
            p_sl_short = np.zeros(H, dtype=np.float64)

            pred_tp_long = pred["p_tp_long"].cpu().numpy()[0]
            pred_sl_long = pred["p_sl_long"].cpu().numpy()[0]
            pred_tp_short = pred["p_tp_short"].cpu().numpy()[0]
            pred_sl_short = pred["p_sl_short"].cpu().numpy()[0]

            for i, h_req in enumerate(horizons):
                closest_idx = min(range(len(model_horizons)), key=lambda j: abs(model_horizons[j] - h_req))
                p_tp_long[i] = float(pred_tp_long[closest_idx])
                p_sl_long[i] = float(pred_sl_long[closest_idx])
                p_tp_short[i] = float(pred_tp_short[closest_idx])
                p_sl_short[i] = float(pred_sl_short[closest_idx])

            return {
                "p_tp_long": p_tp_long,
                "p_sl_long": p_sl_long,
                "p_tp_short": p_tp_short,
                "p_sl_short": p_sl_short,
            }
        except Exception as e:
            logger.warning(f"[ALPHA_HIT] Prediction failed: {e}")
            return None

# JIT 가능한 경로 생성 핵심 함수
def _simulate_paths_price_jax_core(
    key,
    s0: float,
    drift: float,
    diffusion: float,
    n_paths: int,
    n_steps: int,
    mode: str,
    df: float,
    boot_jnp,
) -> "jnp.ndarray":  # type: ignore[name-defined]
    """
    JAX JIT 컴파일된 GBM 경로 생성 핵심 연산
    """
    # 노이즈 샘플링
    if mode == "bootstrap" and boot_jnp is not None:
        br_size = int(boot_jnp.shape[0])
        if br_size >= 16:
            key, k1 = jrand.split(key)  # type: ignore[attr-defined]
            idx = jrand.randint(k1, shape=(n_paths, n_steps), minval=0, maxval=br_size)  # type: ignore[attr-defined]
            z = boot_jnp[idx]
        else:
            key, k1 = jrand.split(key)  # type: ignore[attr-defined]
            z = jrand.normal(k1, shape=(n_paths, n_steps))  # type: ignore[attr-defined]
    elif mode == "student_t":
        key, k1 = jrand.split(key)  # type: ignore[attr-defined]
        z = jrand.t(k1, df=df, shape=(n_paths, n_steps))  # type: ignore[attr-defined]
        if df > 2:
            z = z / jnp.sqrt(df / (df - 2.0))  # type: ignore[attr-defined]
    else:
        key, k1 = jrand.split(key)  # type: ignore[attr-defined]
        z = jrand.normal(k1, shape=(n_paths, n_steps))  # type: ignore[attr-defined]
    
    z = z.astype(jnp.float32)  # type: ignore[attr-defined]
    
    # GBM 경로 생성: logret = cumsum(drift + diffusion * z)
    logret = jnp.cumsum(drift + diffusion * z, axis=1)  # type: ignore[attr-defined]
    prices_1 = s0 * jnp.exp(logret)  # type: ignore[attr-defined]
    
    # 초기 가격 추가: shape (n_paths, n_steps+1)
    paths = jnp.concatenate(  # type: ignore[attr-defined]
        [jnp.full((n_paths, 1), s0, dtype=jnp.float32), prices_1],  # type: ignore[attr-defined]
        axis=1
    )
    return paths


# JIT 컴파일된 경로 생성 함수
if _JAX_OK:
    _simulate_paths_price_jax_core_jit = jax.jit(_simulate_paths_price_jax_core, static_argnames=("mode", "n_paths", "n_steps"))  # type: ignore[attr-defined]
else:
    _simulate_paths_price_jax_core_jit = None

    

    
    def _extract_alpha_hit_features(
        self,
        symbol: str,
        price: float,
        mu: float,
        sigma: float,
        momentum_z: float,
        ofi_z: float,
        regime: str,
        leverage: float,
        ctx: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        """
        Extract features for AlphaHitMLP prediction.
        Returns [1, n_features] tensor or None if model not available.
        """
        if not self.alpha_hit_enabled or self.alpha_hit_trainer is None:
            return None
        
        try:
            # Feature engineering: combine market state, regime, momentum, etc.
            # [B] mu_alpha는 보조 피처로만 사용 (방향 결정의 중심이 아님)
            mu_alpha = float(ctx.get("mu_alpha", 0.0))
            features = [
                float(mu) * SECONDS_PER_YEAR,  # annualized drift
                float(sigma) * math.sqrt(SECONDS_PER_YEAR),  # annualized vol
                float(momentum_z),
                float(ofi_z),
                float(leverage),
                float(price),  # normalized later if needed
                1.0 if regime == "bull" else 0.0,
                1.0 if regime == "bear" else 0.0,
                1.0 if regime == "chop" else 0.0,
                1.0 if regime == "volatile" else 0.0,
                # Add more features from ctx if available
                float(ctx.get("spread_pct", 0.0)),
                float(ctx.get("kelly", 0.0)),
                float(ctx.get("confidence", 0.0)),
                float(ctx.get("ev", 0.0)),
                float(mu_alpha),  # [B] mu_alpha as auxiliary feature
                # Placeholder for additional features (expand as needed)
                0.0, 0.0, 0.0, 0.0, 0.0,
            ]
            # Pad/truncate to match model's n_features
            n_feat = self.alpha_hit_trainer.model.cfg.n_features
            features = features[:n_feat] + [0.0] * max(0, n_feat - len(features))
            return torch.tensor([features], dtype=torch.float32)
        except Exception as e:
            logger.warning(f"[ALPHA_HIT] Feature extraction failed: {e}")
            return None
    
    def collect_alpha_hit_sample(
        self,
        symbol: str,
        features_np: np.ndarray,
        entry_ts_ms: int,
        exit_ts_ms: int,
        direction: int,  # 1 for long, -1 for short
        exit_reason: str,  # "TP", "SL", "timeout", etc.
        tp_level: float = 0.01,  # TP level (e.g., 0.01 = 1%)
        sl_level: float = 0.01,  # SL level (e.g., 0.01 = 1%)
    ):
        """
        [B] Collect training sample for AlphaHitMLP.
        Called when position is closed to record actual TP/SL hit outcomes.
        """
        if not self.alpha_hit_enabled or self.alpha_hit_trainer is None or features_np is None:
            return
        
        try:
            duration_sec = (exit_ts_ms - entry_ts_ms) / 1000.0
            policy_horizons = list(getattr(self, "POLICY_MULTI_HORIZONS_SEC", (60, 180, 300, 600, 900, 1800)))
            H = len(policy_horizons)
            
            # Determine if TP/SL was hit for each horizon
            y_tp_long = np.zeros(H, dtype=np.float32)
            y_sl_long = np.zeros(H, dtype=np.float32)
            y_tp_short = np.zeros(H, dtype=np.float32)
            y_sl_short = np.zeros(H, dtype=np.float32)
            
            # Simple heuristic: if exit happened before horizon and reason matches, mark as hit
            for i, h in enumerate(policy_horizons):
                if duration_sec <= h:
                    if exit_reason == "TP":
                        if direction == 1:
                            y_tp_long[i] = 1.0
                        else:
                            y_tp_short[i] = 1.0
                    elif exit_reason == "SL":
                        if direction == 1:
                            y_sl_long[i] = 1.0
                        else:
                            y_sl_short[i] = 1.0
            
            self.alpha_hit_trainer.add_sample(
                x=features_np,
                y={
                    "tp_long": y_tp_long,
                    "sl_long": y_sl_long,
                    "tp_short": y_tp_short,
                    "sl_short": y_sl_short,
                },
                ts_ms=entry_ts_ms,
                symbol=symbol,
            )
        except Exception as e:
            logger.warning(f"[ALPHA_HIT] Failed to collect sample: {e}")

    def _predict_horizon_hit_probs(
        self,
        features: torch.Tensor,
        horizons: List[int],
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Predict TP/SL hit probabilities per horizon using AlphaHitMLP.
        Returns dict with keys: p_tp_long, p_sl_long, p_tp_short, p_sl_short
        Each value is [H] numpy array.
        """
        if not self.alpha_hit_enabled or self.alpha_hit_mlp is None or features is None:
            return None
        
        try:
            with torch.no_grad():
                pred = self.alpha_hit_mlp.predict(features)
            
            # Extract probabilities for the requested horizons
            model_horizons = self.alpha_hit_mlp.cfg.horizons_sec
            H = len(horizons)
            
            # Map model horizons to requested horizons (simple nearest match)
            p_tp_long = np.zeros(H, dtype=np.float64)
            p_sl_long = np.zeros(H, dtype=np.float64)
            p_tp_short = np.zeros(H, dtype=np.float64)
            p_sl_short = np.zeros(H, dtype=np.float64)
            
            pred_tp_long = pred["p_tp_long"].cpu().numpy()[0]  # [H_model]
            pred_sl_long = pred["p_sl_long"].cpu().numpy()[0]
            pred_tp_short = pred["p_tp_short"].cpu().numpy()[0]
            pred_sl_short = pred["p_sl_short"].cpu().numpy()[0]
            
            for i, h_req in enumerate(horizons):
                # Find closest model horizon
                closest_idx = min(range(len(model_horizons)), key=lambda j: abs(model_horizons[j] - h_req))
                p_tp_long[i] = float(pred_tp_long[closest_idx])
                p_sl_long[i] = float(pred_sl_long[closest_idx])
                p_tp_short[i] = float(pred_tp_short[closest_idx])
                p_sl_short[i] = float(pred_sl_short[closest_idx])
            
            return {
                "p_tp_long": p_tp_long,
                "p_sl_long": p_sl_long,
                "p_tp_short": p_tp_short,
                "p_sl_short": p_sl_short,
            }
        except Exception as e:
            logger.warning(f"[ALPHA_HIT] Prediction failed: {e}")
            return None

    def _compute_ev_based_weights(
        self,
        horizons: List[int],
        w_prior: np.ndarray,
        evs_long: np.ndarray,
        evs_short: np.ndarray,
        beta: float = 1.0,
    ) -> np.ndarray:
        """
        Compute EV-based horizon weights: w(h) = normalize( w_prior(h) * softplus(EV_h * beta) )
        
        Args:
            horizons: List of horizon seconds
            w_prior: Prior weights from rule-based method [H]
            evs_long: EV per horizon for long [H]
            evs_short: EV per horizon for short [H]
            beta: Scaling factor for EV
        
        Returns:
            Normalized weights [H]
        """
        h_arr = np.asarray(horizons, dtype=np.float64)
        w_prior_arr = np.asarray(w_prior, dtype=np.float64)
        evs_long_arr = np.asarray(evs_long, dtype=np.float64)
        evs_short_arr = np.asarray(evs_short, dtype=np.float64)
        
        # Use max EV (long vs short) per horizon
        evs_max = np.maximum(evs_long_arr, evs_short_arr)
        
        # softplus(x) = log(1 + exp(x))
        # For numerical stability, use: softplus(x) ≈ max(0, x) + log(1 + exp(-|x|))
        ev_scaled = evs_max * float(beta)
        softplus_ev = np.maximum(0.0, ev_scaled) + np.log1p(np.exp(-np.abs(ev_scaled)))
        
        # Combine prior with EV-based scaling
        w_combined = w_prior_arr * softplus_ev
        
        # Normalize
        total = float(np.sum(w_combined))
        if total <= 0.0:
            # Fallback to uniform if all weights are zero/negative
            return np.full(h_arr.shape, 1.0 / float(h_arr.size), dtype=np.float64)
        
        return w_combined / total

    def simulate_paths_netpnl(
        self,
        seed: int,
        s0: float,
        mu: float,
        sigma: float,
        direction: int,
        leverage: float,
        n_paths: int,
        horizons: Sequence[int],
        dt: float,
        fee_roundtrip: float,
    ) -> Dict[int, np.ndarray]:
        """
        horizon별 net_pnl paths 반환
        """
        # ✅ Step A: MC 입력이 진짜 0인지 확인
        if not hasattr(self, "_sim_input_logged"):
            self._sim_input_logged = True
            logger.info(
                f"[SIM_INPUT] mu={mu:.10f} sigma={sigma:.10f} fee_rt={fee_roundtrip:.6f} "
                f"dir={direction} dt={dt} horizons={list(horizons)} s0={s0:.2f} n_paths={n_paths}"
            )
        
        rng = np.random.default_rng(seed)
        max_steps = int(max(horizons))
        drift = (mu - 0.5 * sigma * sigma) * dt
        diffusion = sigma * math.sqrt(dt)

        mode = str(getattr(self, "_tail_mode", self.default_tail_mode))
        df = float(getattr(self, "_student_t_df", self.default_student_t_df))
        br = getattr(self, "_bootstrap_returns", None)
        use_jax = bool(getattr(self, "_use_jax", True)) and _JAX_OK

        prices: np.ndarray
        if use_jax:
            # ✅ GPU 우선: default backend (GPU/Metal) 사용
            force_cpu_dev = _jax_mc_device()
            try:
                if force_cpu_dev is None:
                    # GPU/Metal default backend 사용
                    key = jrand.PRNGKey(int(seed) & 0xFFFFFFFF)  # type: ignore[attr-defined]
                    key, z_j = self._sample_increments_jax(
                        key,
                        (int(n_paths), int(max_steps)),
                        mode=mode,
                        df=df,
                        bootstrap_returns=br,
                    )
                    if z_j is None:
                        use_jax = False
                    else:
                        z_j = z_j.astype(jnp.float32)  # type: ignore[attr-defined]
                        logret = jnp.cumsum((drift + diffusion * z_j), axis=1)  # type: ignore[attr-defined]
                        prices = float(s0) * jnp.exp(logret)  # type: ignore[attr-defined]
                        prices = np.asarray(jax.device_get(prices), dtype=np.float64)  # type: ignore[attr-defined]
                else:
                    # CPU로 강제된 경우만 CPU 사용 (env JAX_MC_DEVICE=cpu)
                    with jax.default_device(force_cpu_dev):  # type: ignore[attr-defined]
                        key = jrand.PRNGKey(int(seed) & 0xFFFFFFFF)  # type: ignore[attr-defined]
                        key, z_j = self._sample_increments_jax(
                            key,
                            (int(n_paths), int(max_steps)),
                            mode=mode,
                            df=df,
                            bootstrap_returns=br,
                        )
                        if z_j is None:
                            use_jax = False
                        else:
                            z_j = z_j.astype(jnp.float32)  # type: ignore[attr-defined]
                            logret = jnp.cumsum((drift + diffusion * z_j), axis=1)  # type: ignore[attr-defined]
                            prices = float(s0) * jnp.exp(logret)  # type: ignore[attr-defined]
                            prices = np.asarray(jax.device_get(prices), dtype=np.float64)  # type: ignore[attr-defined]
            except Exception:
                use_jax = False


        if not use_jax:
            z = self._sample_increments_np(rng, (n_paths, max_steps), mode=mode, df=df, bootstrap_returns=br)
            logret = np.cumsum(drift + diffusion * z, axis=1)
            prices = s0 * np.exp(logret)

        out = {}
        for h in horizons:
            idx = int(h) - 1
            tp = prices[:, idx]
            gross = direction * (tp - s0) / s0 * float(leverage)
            net = gross - fee_roundtrip
            out[int(h)] = net.astype(np.float64)
        return out

    

    

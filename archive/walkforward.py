from __future__ import annotations

from typing import Callable, Dict, List, Tuple
import math
import numpy as np


def walkforward_backtest(
    closes: List[float],
    strategy_fn: Callable[[List[float]], int],
    *,
    train_window: int = 500,
    test_window: int = 100,
) -> Dict[str, float]:
    """
    Simple walk-forward backtest.

    - closes: list of prices
    - strategy_fn: takes train_window closes, returns signal {-1, 0, 1} for next period
    - returns aggregate metrics on test periods.
    """
    if closes is None or len(closes) < train_window + test_window + 1:
        return {}

    rets = []
    hits = []
    idx = train_window
    while idx + test_window < len(closes):
        train = closes[idx - train_window : idx]
        signal = strategy_fn(train)
        for j in range(idx, idx + test_window):
            r = (closes[j + 1] / closes[j]) - 1.0
            pnl = signal * r
            rets.append(pnl)
            hits.append(1 if pnl > 0 else 0)
        idx += test_window

    if not rets:
        return {}
    arr = np.asarray(rets, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std()) if arr.size else None
    sharpe = (mean / std) if std and std > 1e-9 else None
    downside = arr[arr < 0]
    sortino = (mean / (float(np.std(downside)) + 1e-9)) if downside.size else None
    # CVaR 5%
    sorted_rets = np.sort(arr)
    cvar = float(np.mean(sorted_rets[: max(1, int(0.05 * len(sorted_rets)))]))
    # Max drawdown from equity curve
    eq = [1.0]
    for r in arr:
        eq.append(eq[-1] * (1 + r))
    peak = eq[0]
    max_dd = 0.0
    for v in eq:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak if peak else 0.0)

    return {
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "sortino": sortino,
        "cvar": cvar,
        "hit_rate": float(np.mean(hits)) if hits else None,
        "max_dd": max_dd,
    }


def run_walkforward_suite(
    closes: List[float],
    signal_fns: Dict[str, Callable[[List[float]], int]],
    *,
    train_window: int = 500,
    test_window: int = 100,
) -> Dict[str, Dict[str, float]]:
    """
    Run multiple signals through walk-forward and return per-signal metrics.
    """
    results: Dict[str, Dict[str, float]] = {}
    for name, fn in signal_fns.items():
        try:
            res = walkforward_backtest(
                closes,
                fn,
                train_window=train_window,
                test_window=test_window,
            )
            results[name] = res
        except Exception as e:
            results[name] = {"error": str(e)}
    return results

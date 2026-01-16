from __future__ import annotations

"""
Simple 1m OHLCV backtester with transaction cost/slippage/spread modelling.
Input: CSV with columns at least ["timestamp","open","high","low","close","volume"].
Strategy: plug in any signal function that returns {-1,0,1} given a rolling window of closes.
Outputs: summary metrics (CAGR, Sharpe, Sortino, MDD/Calmar, CVaR, hit rate) and trade log.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Dict, Any

import numpy as np
import pandas as pd


# -----------------------------
# Strategy interface
# -----------------------------
def momentum_breakout_signal(closes: List[float], lookback: int = 50, band: float = 0.003) -> int:
    if len(closes) < lookback + 1:
        return 0
    window = np.asarray(closes[-lookback:], dtype=float)
    last = float(window[-1])
    hi = float(window.max())
    lo = float(window.min())
    if last >= hi * (1 + band):
        return 1
    if last <= lo * (1 - band):
        return -1
    return 0


def mean_reversion_zscore_signal(closes: List[float], lookback: int = 60, z_entry: float = 1.5) -> int:
    if len(closes) < lookback:
        return 0
    arr = np.asarray(closes[-lookback:], dtype=float)
    mean = float(arr.mean())
    std = float(arr.std()) or 1e-9
    z = (float(arr[-1]) - mean) / std
    if z >= z_entry:
        return -1
    if z <= -z_entry:
        return 1
    return 0


SIGNALS: Dict[str, Callable[[List[float]], int]] = {
    "momentum_breakout": lambda xs: momentum_breakout_signal(xs, lookback=50, band=0.003),
    "mean_reversion_z": lambda xs: mean_reversion_zscore_signal(xs, lookback=60, z_entry=1.5),
}


# -----------------------------
# Backtester
# -----------------------------
@dataclass
class CostModel:
    fee_bps: float = 6.0   # taker 0.06% per side
    slippage_bps: float = 5.0
    spread_bps: float = 2.0
    latency_ticks: int = 0  # simulate delayed execution (ticks)

    def price_with_cost(self, price: float, side: int) -> float:
        # side: 1 buy, -1 sell
        slip = self.slippage_bps / 10_000.0
        spread = self.spread_bps / 10_000.0
        adj = 1.0 + side * (slip + spread)
        return float(price * adj)

    def fee(self, notional: float) -> float:
        return float(notional * self.fee_bps / 10_000.0)


def run_backtest(df: pd.DataFrame, signal_fn: Callable[[List[float]], int], cost: CostModel, notional: float = 1.0) -> Dict[str, Any]:
    closes = df["close"].astype(float).tolist()
    positions = []
    equity_curve = [1.0]
    trades = []
    pos = 0  # -1 short, 0 flat, 1 long
    entry_px = 0.0

    for i in range(len(closes) - 1):
        px = closes[i]
        sig = signal_fn(closes[: i + 1])

        # optional latency
        exec_idx = min(len(closes) - 1, i + cost.latency_ticks)
        exec_px = cost.price_with_cost(closes[exec_idx], 1 if sig == 1 else -1)

        # exit if signal flips
        if pos != 0 and sig != pos:
            exit_px = cost.price_with_cost(exec_px, -pos)
            ret = (exit_px / entry_px - 1.0) * pos
            fee = cost.fee(notional) * 2
            pnl = ret - fee
            equity_curve.append(equity_curve[-1] * (1 + pnl))
            trades.append({"entry_px": entry_px, "exit_px": exit_px, "side": pos, "ret": ret, "pnl": pnl})
            pos = 0

        # enter if flat and signal != 0
        if pos == 0 and sig != 0:
            pos = sig
            entry_px = exec_px

    equity = np.asarray(equity_curve, dtype=float)
    rets = np.diff(equity) / equity[:-1]
    metrics = compute_metrics(rets, equity, trades)
    return {"metrics": metrics, "trades": trades, "equity": equity.tolist()}


def compute_metrics(rets: np.ndarray, equity: np.ndarray, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if rets.size == 0:
        return out
    mean = float(np.mean(rets))
    std = float(np.std(rets)) or None
    downside = rets[rets < 0]
    out["cagr"] = float(equity[-1] ** (1 / max(len(equity) / 1440, 1)) - 1) if equity.size else None  # rough daily->annualized
    out["sharpe"] = (mean / std) if std and std > 1e-12 else None
    out["sortino"] = (mean / (float(np.std(downside)) + 1e-12)) if downside.size else None
    # drawdown
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak if peak else 0.0)
    out["max_dd"] = max_dd
    out["calmar"] = (out["cagr"] / max_dd) if (max_dd and out.get("cagr") is not None) else None
    # CVaR 5%
    sorted_rets = np.sort(rets)
    out["cvar_5"] = float(np.mean(sorted_rets[: max(1, int(0.05 * len(sorted_rets)))]))
    # hit rate
    hits = [1 if t.get("pnl", 0) > 0 else 0 for t in trades]
    out["hit_rate"] = float(np.mean(hits)) if hits else None
    return out


def main():
    parser = argparse.ArgumentParser(description="1m OHLCV backtester with cost model.")
    parser.add_argument("--csv", required=True, help="CSV with 1m OHLCV (columns: timestamp,open,high,low,close,volume)")
    parser.add_argument("--signal", choices=list(SIGNALS.keys()), default="momentum_breakout")
    parser.add_argument("--fee_bps", type=float, default=6.0, help="Per-side fee in bps")
    parser.add_argument("--slip_bps", type=float, default=5.0, help="Slippage bps")
    parser.add_argument("--spread_bps", type=float, default=2.0, help="Half-spread bps")
    parser.add_argument("--latency", type=int, default=0, help="Execution latency in ticks")
    args = parser.parse_args()

    df = pd.read_csv(Path(args.csv))
    cost = CostModel(fee_bps=args.fee_bps, slippage_bps=args.slip_bps, spread_bps=args.spread_bps, latency_ticks=args.latency)
    res = run_backtest(df, SIGNALS[args.signal], cost=cost)
    print("Metrics:", res["metrics"])
    print("Trades:", len(res["trades"]))


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from walkforward import run_walkforward_suite
from strategies.wf_signals import (
    momentum_breakout_signal,
    mean_reversion_zscore_signal,
    ma_cross_signal,
)


def load_closes(path: Path, col: str) -> List[float]:
    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"column '{col}' not found in {path}")
    closes = df[col].astype(float).tolist()
    if len(closes) < 600:
        raise ValueError("need at least 600 rows for a meaningful walk-forward")
    return closes


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward backtests on simple rule-based signals.")
    parser.add_argument("--csv", type=str, required=True, help="CSV file with price data")
    parser.add_argument("--col", type=str, default="close", help="Column name for close price")
    parser.add_argument("--train", type=int, default=500, help="Train window size")
    parser.add_argument("--test", type=int, default=100, help="Test window size")
    args = parser.parse_args()

    closes = load_closes(Path(args.csv), args.col)
    signals = {
        "momentum_breakout": lambda xs: momentum_breakout_signal(xs, lookback=50, band=0.003),
        "mean_reversion_z": lambda xs: mean_reversion_zscore_signal(xs, lookback=60, z_entry=1.5),
        "ma_cross": lambda xs: ma_cross_signal(xs, short=20, long=60),
    }
    res = run_walkforward_suite(closes, signals, train_window=args.train, test_window=args.test)
    for name, metrics in res.items():
        print(f"\n[{name}]")
        for k, v in metrics.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()


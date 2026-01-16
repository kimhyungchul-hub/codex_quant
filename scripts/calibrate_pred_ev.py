#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    try:
        x = json.load(path.open("r", encoding="utf-8"))
    except FileNotFoundError:
        return []
    except Exception:
        return []
    if not isinstance(x, list):
        return []
    rows = [t for t in x if isinstance(t, dict) and t.get("type") == "EXIT"]
    return rows


def _print_bin(i: int, n: int, lo: float, hi: float, ev: np.ndarray, pnl: np.ndarray, hit: np.ndarray, m: np.ndarray):
    if n <= 0:
        print(i, "n", 0, "ev_range", float(lo), float(hi), "(empty)")
        return
    print(
        i,
        "n",
        int(n),
        "ev_range",
        float(lo),
        float(hi),
        "pred_ev_mean",
        float(ev[m].mean()),
        "pnl_mean",
        float(pnl[m].mean()),
        "win",
        float(hit[m].mean()),
        "pnl_p50",
        float(np.quantile(pnl[m], 0.50)),
    )


def main():
    ap = argparse.ArgumentParser(description="Calibrate pred_ev against realized pnl using EXIT trades.")
    ap.add_argument("--path", default="state/trade_tape.json", help="trade tape path (default: state/trade_tape.json)")
    ap.add_argument(
        "--x",
        default="pred_ev",
        choices=["pred_ev", "pred_win"],
        help="predictor field to bin by (default: pred_ev)",
    )
    ap.add_argument(
        "--y",
        default="pnl",
        choices=["pnl", "roe", "realized_r"],
        help="target field to evaluate (default: pnl; use roe/realized_r to normalize across notionals)",
    )
    ap.add_argument("--bins", type=int, default=2, choices=[2, 3, 5, 10], help="number of quantile bins")
    ap.add_argument("--min_n", type=int, default=1, help="min rows per bin to print")
    args = ap.parse_args()

    path = Path(args.path)
    rows_all = _load_rows(path)
    print("EXIT_total", len(rows_all), "path", str(path))
    if not rows_all:
        return

    x_key = str(args.x)
    y_key = str(args.y)
    rows = [t for t in rows_all if t.get(x_key) is not None and t.get(y_key) is not None]
    print("EXIT_usable", len(rows))
    if not rows:
        return

    x = np.array([t.get(x_key) for t in rows], dtype=np.float64)
    y = np.array([t.get(y_key) for t in rows], dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    n2 = int(x.size)
    if n2 != len(rows):
        print("filtered_nonfinite", len(rows) - n2)
    if n2 == 0:
        return
    hit = (y > 0).astype(np.float64)

    print(f"{x_key}_min/mean/max", float(x.min()), float(x.mean()), float(x.max()))
    print(f"{y_key}_min/mean/max", float(y.min()), float(y.mean()), float(y.max()))

    if args.bins == 2 and x.size >= 4:
        thr = float(np.median(x))
        print("median_thr", thr)
        lo = x < thr
        hi = x >= thr
        for name, mask in (("LOW", lo), ("HIGH", hi)):
            k = int(mask.sum())
            if k < int(args.min_n):
                print(name, "n", k, "(skip)")
                continue
            print(
                name,
                "n",
                k,
                f"{x_key}_mean",
                float(x[mask].mean()),
                f"{y_key}_mean",
                float(y[mask].mean()),
                "win",
                float(hit[mask].mean()),
                f"{y_key}_p50",
                float(np.quantile(y[mask], 0.50)),
            )
    else:
        qs = np.quantile(x, np.linspace(0.0, 1.0, int(args.bins) + 1))
        for i in range(int(args.bins)):
            lo, hi = float(qs[i]), float(qs[i + 1])
            if i < int(args.bins) - 1:
                m = (x >= lo) & (x < hi)
            else:
                m = (x >= lo) & (x <= hi)
            k = int(m.sum())
            if k < int(args.min_n):
                print(i, "n", k, "ev_range", float(lo), float(hi), "(skip)")
                continue
            if x_key == "pred_ev":
                _print_bin(i, k, lo, hi, x, y, hit, m)
            else:
                print(
                    i,
                    "n",
                    int(k),
                    "x_range",
                    float(lo),
                    float(hi),
                    f"{x_key}_mean",
                    float(x[m].mean()),
                    "win",
                    float(hit[m].mean()),
                    f"{y_key}_mean",
                    float(y[m].mean()),
                    f"{y_key}_p50",
                    float(np.quantile(y[m], 0.50)),
                )

    if x.size >= 3:
        try:
            corr = float(np.corrcoef(x, y)[0, 1])
        except Exception:
            corr = float("nan")
        print(f"corr({x_key}, {y_key})", corr)


if __name__ == "__main__":
    main()

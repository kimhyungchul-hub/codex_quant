from __future__ import annotations

import numpy as np


def momentum_breakout_signal(closes: list[float], lookback: int = 50, band: float = 0.003) -> int:
    """
    돌파 기반 모멘텀: 최근 종가가 lookback 최고 대비 band 이상이면 롱, 최저 대비 band 이상 하회면 숏.
    """
    if closes is None or len(closes) < lookback + 1:
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


def mean_reversion_zscore_signal(closes: list[float], lookback: int = 60, z_entry: float = 1.5) -> int:
    """
    평균회귀: z-score가 +z_entry 이상이면 숏, -z_entry 이하이면 롱.
    """
    if closes is None or len(closes) < lookback:
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


def ma_cross_signal(closes: list[float], short: int = 20, long: int = 60) -> int:
    """
    이동평균 크로스: 단기>장기 롱, 단기<장기 숏.
    """
    if closes is None or len(closes) < long + 1:
        return 0
    arr = np.asarray(closes, dtype=float)
    ma_s = float(arr[-short:].mean())
    ma_l = float(arr[-long:].mean())
    if ma_s > ma_l:
        return 1
    if ma_s < ma_l:
        return -1
    return 0

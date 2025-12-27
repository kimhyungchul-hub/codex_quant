import datetime
from typing import Optional, Tuple


def time_regime() -> str:
    """
    Rough session classifier based on UTC hour.
    """
    h = datetime.datetime.utcnow().hour
    if 0 <= h < 6:
        return "ASIA"
    if 6 <= h < 13:
        return "EU"
    if 13 <= h < 21:
        return "US"
    return "OFF"


def adjust_mu_sigma(mu: float, sigma: float, regime: Optional[str] = None) -> Tuple[float, float]:
    """
    Lightweight regime-based tweak for expected return/vol.
    Falls back to identity if regime is unknown.
    """
    r = (regime or "").lower()
    mu_adj = float(mu)
    sigma_adj = float(max(sigma, 1e-9))

    if r == "bull":
        mu_adj *= 1.05
        sigma_adj *= 0.95
    elif r == "bear":
        mu_adj *= 0.95
        sigma_adj *= 1.05
    elif r == "volatile":
        sigma_adj *= 1.15
    elif r == "chop":
        mu_adj *= 0.9
        sigma_adj *= 1.1

    return mu_adj, sigma_adj


def get_regime_mu_sigma(regime: str, session: str, *, symbol: str | None = None) -> Tuple[Optional[float], Optional[float]]:
    """
    Placeholder lookup for per-regime/session drift/vol; returns None to skip override.
    Extend with calibrated tables as they become available.
    """
    return None, None


__all__ = ["time_regime", "adjust_mu_sigma", "get_regime_mu_sigma"]

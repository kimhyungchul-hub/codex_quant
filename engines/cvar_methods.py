def _cvar_empirical(pnl: np.ndarray, alpha: float = 0.05) -> float:
    # Implementation of _cvar_empirical
    pass

def _cvar_bootstrap(pnl: np.ndarray, alpha: float = 0.05, n_boot: Optional[int] = None, sample_frac: float = 0.7, seed: int = 42) -> float:
    # Implementation of _cvar_bootstrap
    pass

def _cvar_tail_inflate(pnl: np.ndarray, alpha: float = 0.05, inflate: float = 1.15) -> float:
    # Implementation of _cvar_tail_inflate
    pass

def cvar_ensemble(pnl: Sequence[float], alpha: float = 0.05) -> float:
    # Implementation of cvar_ensemble
    pass

def _cvar_jnp(x: "jnp.ndarray", alpha: float) -> "jnp.ndarray":  # type: ignore[name-defined]
    # Implementation of _cvar_jnp
    pass

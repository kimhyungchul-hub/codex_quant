"""JAX backend initialization with Metal GPU support (Python 3.11 + JAX 0.4.35)."""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# JAX 0.4.35 with Metal GPU support
_JAX_OK = False
jax: Any = None
jnp: Any = None
jrand: Any = None
lax: Any = None

try:
    import jax as _jax_module
    import jax.numpy as _jnp_module
    from jax import random as _jrand_module
    from jax import lax as _lax_module

    jax = _jax_module
    jnp = _jnp_module
    jrand = _jrand_module
    lax = _lax_module
    _JAX_OK = True
    
    # Log configuration
    backend = jax.default_backend()
    devices = jax.devices()
    
    logger.info(f"✅ [JAX] Version {jax.__version__} loaded")
    logger.info(f"✅ [JAX] Backend: {backend}")
    logger.info(f"✅ [JAX] Devices: {devices}")
    
    # Check if Metal GPU is available
    metal_devices = [d for d in devices if str(d).upper().startswith('METAL')]
    if metal_devices:
        logger.info(f"🚀 [JAX] Metal GPU acceleration enabled: {metal_devices}")
    else:
        logger.info(f"💻 [JAX] Running on CPU")
        
except ImportError as e:
    logger.warning(f"⚠️  [JAX] Not available: {e}")
    _JAX_OK = False
except Exception as e:
    logger.warning(f"⚠️  [JAX] Initialization error: {e}")
    _JAX_OK = False


from functools import partial

@partial(jax.jit, static_argnames=("alpha",))
def _cvar_jax(returns: jnp.ndarray, alpha: float = 0.05) -> jnp.ndarray:
    """
    JAX-native CVaR (Conditional Value at Risk) 
    - returns: (n_paths,)
    - alpha: confidence level (default 0.05)
    """
    n = returns.shape[0]
    sorted_returns = jnp.sort(returns)
    
    # Robust CVaR calculation using masking to handle Tracers
    # This avoids 'int()' conversion errors during JIT compilation
    k_float = n * alpha
    indices = jnp.arange(n)
    mask = indices < k_float # Select bottom alpha%
    
    count = jnp.sum(mask)
    cvar = jnp.where(
        count > 0,
        jnp.sum(sorted_returns * mask) / count,
        sorted_returns[0]  # Fallback to minimum if mask is empty
    )
    return cvar


@partial(jax.jit, static_argnames=("alpha",))
def summarize_gbm_horizons_jax(
    price_paths: jnp.ndarray, 
    s0: float, 
    leverage: float, 
    fee_rt_total_roe: float, 
    horizons_indices: jnp.ndarray, 
    alpha: float = 0.05
):
    """
    GPU-side summary of GBM price paths for multiple horizons.
    - price_paths: (n_paths, n_steps)
    - horizons_indices: (n_horizons,)
    """
    # Select rows for each horizon
    # Shape: (n_paths, n_horizons)
    tp = price_paths[:, horizons_indices]
    
    gross_ret = (tp - s0) / jnp.maximum(1e-12, s0)
    
    # Long stats
    net_long = gross_ret * leverage - fee_rt_total_roe
    ev_long = jnp.mean(net_long, axis=0) # (n_horizons,)
    win_long = jnp.mean(net_long > 0, axis=0)
    # vmap CVaR over horizons (column-wise)
    cvar_long = jax.vmap(_cvar_jax, in_axes=(1, None))(net_long, alpha)
    
    # Short stats
    net_short = -gross_ret * leverage - fee_rt_total_roe
    ev_short = jnp.mean(net_short, axis=0)
    win_short = jnp.mean(net_short > 0, axis=0)
    cvar_short = jax.vmap(_cvar_jax, in_axes=(1, None))(net_short, alpha)
    
    return {
        "ev_long": ev_long,
        "win_long": win_long,
        "cvar_long": cvar_long,
        "ev_short": ev_short,
        "win_short": win_short,
        "cvar_short": cvar_short
    }


# ✅ GLOBAL BATCHING: Multi-symbol version of GBM summary
# in_axes: (price_paths, s0, leverage, fee, horizons, alpha)
# We vmap over first 4 arguments (symbol-specific), horizons and alpha are shared.
summarize_gbm_horizons_multi_symbol_jax = jax.vmap(
    summarize_gbm_horizons_jax,
    in_axes=(0, 0, 0, 0, None, None)
)


@jax.jit
def jax_covariance(returns: jnp.ndarray) -> jnp.ndarray:
    """
    JAX-accelerated covariance matrix calculation.
    - returns: (n_observations, n_assets)
    Returns: (n_assets, n_assets)
    """
    # jnp.cov expects (n_features, n_observations)
    return jnp.cov(returns.T)


def _jax_mc_device() -> Optional[Any]:


    """
    Returns the device for MC simulations.
    
    - If JAX_MC_DEVICE=cpu is set, forces CPU
    - Otherwise uses default backend (Metal GPU if available)
    """
    if not _JAX_OK or jax is None:
        return None
    
    try:
        forced_cpu = str(os.environ.get("JAX_MC_DEVICE", "")).strip().lower() == "cpu"
        if forced_cpu:
            cpu_devices = jax.devices("cpu")
            if cpu_devices:
                logger.info(f"🔧 [JAX] Forced CPU mode for MC simulations")
                return cpu_devices[0]
        
        # Use default device (Metal GPU if available)
        return None
    except Exception as e:
        logger.warning(f"⚠️  [JAX] Device selection error: {e}")
        return None

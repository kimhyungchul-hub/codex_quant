"""
Example usage of Portfolio Joint Simulation Engine

This demonstrates how to use the new portfolio joint simulation feature
to evaluate correlated risk across multiple symbols.
"""

from core.orchestrator import LiveOrchestrator

# Assuming you have an orchestrator instance running
# orch = LiveOrchestrator(exchange, symbols)

# Option 1: Evaluate all symbols in the portfolio
def example_full_portfolio_evaluation(orch):
    """Evaluate entire portfolio with joint simulation"""
    
    # This will automatically use all symbols in orch.symbols
    report = orch.evaluate_portfolio_joint()
    
    print("=" * 60)
    print("PORTFOLIO JOINT SIMULATION REPORT")
    print("=" * 60)
    print(f"Weights allocated: {report['weights']}")
    print(f"Total leverage: {report['total_leverage_allocated']:.2f}x")
    print(f"\nExpected portfolio PnL: {report['expected_portfolio_pnl']:.4f}")
    print(f"VaR (5%): {report['var']:.4f}")
    print(f"CVaR (5%): {report['cvar']:.4f}")
    print(f"\nP(any position liquidated): {report['prob_any_position_liquidated']:.2%}")
    print(f"P(account liquidation proxy): {report['prob_account_liquidation_proxy']:.2%}")
    print(f"\nSample PnL outcomes: {report['portfolio_pnl_samples_head'][:5]}")
    
    return report


# Option 2: Evaluate a subset of symbols
def example_subset_evaluation(orch):
    """Evaluate specific symbols with custom TP/SL"""
    
    # Focus on high-cap symbols
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    
    # Custom take-profit and stop-loss multipliers
    report = orch.evaluate_portfolio_joint(
        symbols=symbols,
        tp_mult=5.0,  # TP at 5x volatility
        sl_mult=2.5,  # SL at 2.5x volatility
    )
    
    print(f"Subset evaluation complete: {len(report['weights'])} positions")
    return report


# Option 3: Integration into decision loop
def example_periodic_portfolio_check(orch):
    """
    Periodically run portfolio joint simulation to check overall risk.
    
    This could be called every hour or when positions change significantly.
    """
    
    # Only run if portfolio mode is enabled
    from engines.mc.config import config as mc_config
    
    if not mc_config.portfolio_enabled:
        print("Portfolio joint simulation is disabled in config")
        return None
    
    # Check if we have open positions
    if len(orch.positions) < 3:
        print("Not enough positions for meaningful portfolio simulation")
        return None
    
    # Run joint simulation
    report = orch.evaluate_portfolio_joint()
    
    # Risk check: if CVaR is too bad, raise alert
    if report['cvar'] < -0.15:  # -15% CVaR threshold
        orch._log(f"⚠️ [PORTFOLIO_RISK] High tail risk detected! CVaR={report['cvar']:.2%}")
    
    # Liquidation risk check
    if report['prob_account_liquidation_proxy'] > 0.05:  # >5% chance
        orch._log(
            f"🚨 [PORTFOLIO_RISK] Account liquidation risk elevated! "
            f"P(liq)={report['prob_account_liquidation_proxy']:.2%}"
        )
    
    return report


# Configuration example
def example_config_setup():
    """
    Example of setting up portfolio simulation via environment variables.
    
    Add these to your .env or state/bybit.env file:
    """
    config_example = """
    # Enable portfolio joint simulation
    PORTFOLIO_JOINT_SIM_ENABLED=1
    
    # Simulation parameters
    PORTFOLIO_DAYS=3                    # 3-day horizon
    PORTFOLIO_SIMULATIONS=30000         # 30k simulations
    PORTFOLIO_BATCH_SIZE=6000           # Memory control
    
    # Jump modeling
    PORTFOLIO_USE_JUMPS=1
    PORTFOLIO_P_JUMP_MARKET=0.005       # 0.5% daily market crash prob
    PORTFOLIO_P_JUMP_IDIO=0.007         # 0.7% daily idio jump prob
    
    # Portfolio construction
    PORTFOLIO_TARGET_LEVERAGE=10.0      # Target 10x total leverage
    PORTFOLIO_INDIVIDUAL_CAP=3.0        # Max 3x per symbol
    PORTFOLIO_RISK_AVERSION=0.5         # Penalize liquidation risk
    
    # Risk metrics
    PORTFOLIO_VAR_ALPHA=0.05            # 5% VaR/CVaR
    """
    print(config_example)


if __name__ == "__main__":
    print(__doc__)
    print("\nSee example functions above for usage patterns.")
    example_config_setup()

from utils.helpers import greeting
from archive.legacy_architecture.data.feeds.bybit_api import BybitRESTClient
from archive.legacy_architecture.portfolio.positions import Portfolio
from archive.legacy_architecture.risk.risk_manager import _abs
from archive.legacy_architecture.strategies.base import Strategy
from engines.mc_risk import kelly_fraction
from engines.running_stats import RunningStats
from refactor_data import refactor_data_manager
from tools.analyze_pmaker_impact import analyze_pmaker_impact
from tools.verify_payload_keys import verify_payload_keys
from utils.alpha_features import build_alpha_features

def main():
    name = "User"
    greeting(name)

    # Add your quant trading system functions here
    bybit_client = BybitRESTClient()
    portfolio = Portfolio()
    strategy = Strategy()
    running_stats = RunningStats()
    refactor_data_manager("path/to/file")
    analyze_pmaker_impact("path/to/log_file", "symbol")
    verify_payload_keys()

    # Define the variables needed for build_alpha_features
    closes = [1, 2, 3, 4, 5]
    vols = [0.1, 0.2, 0.3, 0.4, 0.5]
    returns = [0.01, 0.02, 0.03, 0.04, 0.05]
    ofi_z = 1.0
    spread_pct = 0.01
    pmaker_entry = 0.001
    pmaker_delay_sec = 0.1
    regime_id = 1

    build_alpha_features(closes, vols, returns, ofi_z, spread_pct, pmaker_entry, pmaker_delay_sec, regime_id)

if __name__ == '__main__':
    main()

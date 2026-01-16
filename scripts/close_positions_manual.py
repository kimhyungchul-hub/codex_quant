#!/usr/bin/env python3
"""
Manual position closer with extreme limit orders (maker conditions).
Closes SUI and TRX positions using extreme prices to ensure fills.
"""
import os
import ccxt
import time
import config
from utils.helpers import _load_env_file

# Load local .env (developer convenience)
_load_env_file(str(config.BASE_DIR / ".env"))

# Initialize Bybit exchange
BYBIT_TESTNET = bool(getattr(config, "BYBIT_TESTNET", True))
exchange = ccxt.bybit({
    "apiKey": getattr(config, "BYBIT_API_KEY", ""),
    "secret": getattr(config, "BYBIT_API_SECRET", ""),
    "enableRateLimit": True,
})

if BYBIT_TESTNET:
    exchange.set_sandbox_mode(True)
    print("🟡 Using BYBIT TESTNET")
else:
    print("🔴 Using BYBIT MAINNET")

exchange.load_markets()

# Symbols to close
SYMBOLS_TO_CLOSE = ["SUI/USDT:USDT", "TRX/USDT:USDT"]

def get_extreme_price(symbol: str, side: str, current_price: float) -> float:
    """
    Get extreme price for maker order that will fill immediately.
    
    Args:
        symbol: Trading symbol
        side: 'buy' or 'sell'
        current_price: Current market price
    
    Returns:
        Extreme price for limit order
    """
    if side.lower() == "buy":
        # For closing SHORT: buy at much higher price (guaranteed fill)
        return round(current_price * 1.10, 6)  # 10% higher
    else:
        # For closing LONG: sell at much lower price (guaranteed fill)
        return round(current_price * 0.90, 6)  # 10% lower


def close_position(symbol: str):
    """Close position for given symbol with extreme limit order."""
    try:
        # Get current position
        positions = exchange.fetch_positions([symbol])
        position = None
        for pos in positions:
            if pos['symbol'] == symbol and float(pos.get('contracts', 0)) != 0:
                position = pos
                break
        
        if not position:
            print(f"❌ {symbol}: No open position found")
            return
        
        contracts = float(position['contracts'])
        side = position['side']  # 'long' or 'short'
        entry_price = float(position.get('entryPrice', 0))
        unrealized_pnl = float(position.get('unrealizedPnl', 0))
        
        print(f"\n📊 {symbol} Position:")
        print(f"   Side: {side.upper()}")
        print(f"   Size: {contracts}")
        print(f"   Entry Price: {entry_price}")
        print(f"   Unrealized PnL: {unrealized_pnl:.2f} USDT")
        
        # Get current market price
        ticker = exchange.fetch_ticker(symbol)
        current_price = float(ticker['last'])
        
        # Determine order side (opposite of position)
        order_side = "buy" if side == "short" else "sell"
        
        # Get extreme price for quick fill
        limit_price = get_extreme_price(symbol, order_side, current_price)
        
        print(f"\n🎯 Closing Order:")
        print(f"   Side: {order_side.upper()}")
        print(f"   Size: {abs(contracts)}")
        print(f"   Current Price: {current_price}")
        print(f"   Limit Price: {limit_price} (extreme)")
        print(f"   Price Diff: {((limit_price/current_price - 1) * 100):.2f}%")
        
        # Place limit order with reduceOnly
        order = exchange.create_order(
            symbol=symbol,
            type='limit',
            side=order_side,
            amount=abs(contracts),
            price=limit_price,
            params={
                'reduceOnly': True,
                'timeInForce': 'PostOnly',  # Maker only
                'positionIdx': 0  # One-way mode
            }
        )
        
        print(f"\n✅ Order placed successfully!")
        print(f"   Order ID: {order.get('id', 'N/A')}")
        print(f"   Status: {order.get('status', 'N/A')}")
        
        # Wait a bit and check if filled
        time.sleep(2)
        
        if order.get('id'):
            order_status = exchange.fetch_order(order['id'], symbol)
            print(f"\n📈 Order Status: {order_status.get('status', 'N/A')}")
            if order_status.get('status') == 'closed':
                print(f"   ✅ Position closed successfully!")
            else:
                print(f"   ⏳ Order pending... (may fill as maker)")
        
    except Exception as e:
        print(f"\n❌ Error closing {symbol}: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 60)
    print("🔧 Manual Position Closer - Extreme Limit Orders")
    print("=" * 60)
    
    for symbol in SYMBOLS_TO_CLOSE:
        close_position(symbol)
        print("\n" + "-" * 60)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

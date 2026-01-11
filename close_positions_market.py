#!/usr/bin/env python3
"""
Market order closer - instantly close positions at market price.
"""
import os
import ccxt
import time

# Load .env file manually
env_file = ".env"
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

# Initialize Bybit exchange
BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
exchange = ccxt.bybit({
    "apiKey": os.environ.get("BYBIT_API_KEY", ""),
    "secret": os.environ.get("BYBIT_API_SECRET", ""),
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


def cancel_all_orders(symbol: str):
    """Cancel all open orders for symbol."""
    try:
        open_orders = exchange.fetch_open_orders(symbol)
        print(f"\n🧹 Cancelling {len(open_orders)} open orders...")
        for order in open_orders:
            try:
                exchange.cancel_order(order['id'], symbol)
                print(f"   ❌ Cancelled: {order['id']}")
            except Exception as e:
                print(f"   ⚠️  Failed to cancel {order['id']}: {e}")
    except Exception as e:
        print(f"   ⚠️  Error fetching orders: {e}")


def close_position_market(symbol: str):
    """Close position immediately with market order."""
    print(f"\n{'='*60}")
    print(f"💥 MARKET CLOSE: {symbol}")
    print(f"{'='*60}")
    
    try:
        # Get current position
        positions = exchange.fetch_positions([symbol])
        position = None
        for pos in positions:
            if pos['symbol'] == symbol and float(pos.get('contracts', 0)) != 0:
                position = pos
                break
        
        if not position:
            print(f"✅ {symbol}: No open position")
            return True
        
        contracts = float(position.get('contracts', 0) or 0)
        side = position.get('side', 'unknown')
        entry_price = float(position.get('entryPrice', 0) or 0)
        unrealized_pnl = float(position.get('unrealizedPnl', 0) or 0)
        
        print(f"\n📊 Position to close:")
        print(f"   Side: {side.upper()}")
        print(f"   Size: {contracts}")
        print(f"   Entry Price: {entry_price}")
        print(f"   Unrealized PnL: {unrealized_pnl:.2f} USDT")
        
        # Cancel existing orders first
        cancel_all_orders(symbol)
        time.sleep(1)
        
        # Determine order side
        order_side = "buy" if side == "short" else "sell"
        
        print(f"\n💥 Placing MARKET order:")
        print(f"   Side: {order_side.upper()}")
        print(f"   Size: {abs(contracts)}")
        print(f"   Type: MARKET (immediate execution)")
        
        # Place market order
        order = exchange.create_order(
            symbol=symbol,
            type='market',
            side=order_side,
            amount=abs(contracts),
            params={
                'reduceOnly': True,
                'positionIdx': 0
            }
        )
        
        print(f"\n✅ Market order executed!")
        print(f"   Order ID: {order.get('id', 'N/A')}")
        print(f"   Status: {order.get('status', 'N/A')}")
        
        # Verify position closed
        time.sleep(2)
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            if pos['symbol'] == symbol:
                remaining = float(pos.get('contracts', 0))
                if remaining == 0:
                    print(f"\n✅✅✅ POSITION FULLY CLOSED! ✅✅✅")
                    return True
                else:
                    print(f"\n⚠️  Remaining position: {remaining}")
                    return False
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("💥 MARKET ORDER Position Closer")
    print("   Instant execution at market price")
    print("=" * 60)
    
    for symbol in SYMBOLS_TO_CLOSE:
        success = close_position_market(symbol)
        print("\n" + "-" * 60)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

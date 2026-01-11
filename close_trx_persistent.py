#!/usr/bin/env python3
"""
Persistent market order closer - keeps trying until position is closed.
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

# Symbol to close
SYMBOL = "TRX/USDT:USDT"
MAX_ATTEMPTS = 50
WAIT_SECONDS = 10


def get_position_info(symbol: str):
    """Get current position info."""
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            if pos['symbol'] == symbol:
                contracts = float(pos.get('contracts', 0) or 0)
                if contracts != 0:
                    return pos
    except Exception as e:
        print(f"   ⚠️  Error fetching position: {e}")
    return None


def cancel_all_orders(symbol: str):
    """Cancel all open orders."""
    try:
        open_orders = exchange.fetch_open_orders(symbol)
        if open_orders:
            print(f"\n🧹 Cancelling {len(open_orders)} open orders...")
            for order in open_orders:
                try:
                    exchange.cancel_order(order['id'], symbol)
                    print(f"   ❌ Cancelled: {order['id'][:20]}...")
                except Exception as e:
                    pass
    except Exception:
        pass


def close_position_persistent(symbol: str):
    """Persistently try to close position with market orders."""
    print(f"\n{'='*70}")
    print(f"🔥 PERSISTENT MARKET CLOSER: {symbol}")
    print(f"   Will keep trying until position is closed")
    print(f"{'='*70}")
    
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        print(f"\n{'─'*70}")
        print(f"🔄 Attempt {attempt}/{MAX_ATTEMPTS}")
        print(f"{'─'*70}")
        
        # Check position
        position = get_position_info(symbol)
        
        if not position:
            print(f"\n✅✅✅ SUCCESS! Position is CLOSED! ✅✅✅")
            return True
        
        # Get position details
        contracts = float(position.get('contracts', 0) or 0)
        side = position.get('side', 'unknown')
        entry_price = float(position.get('entryPrice', 0) or 0)
        unrealized_pnl = float(position.get('unrealizedPnl', 0) or 0)
        
        print(f"\n📊 Current Position:")
        print(f"   Side: {side.upper()}")
        print(f"   Size: {contracts:,.1f}")
        print(f"   Entry Price: {entry_price}")
        print(f"   Unrealized PnL: {unrealized_pnl:+.2f} USDT")
        
        # Cancel existing orders
        cancel_all_orders(symbol)
        time.sleep(1)
        
        # Place market order
        order_side = "buy" if side == "short" else "sell"
        
        print(f"\n💥 Placing MARKET ORDER:")
        print(f"   Side: {order_side.upper()}")
        print(f"   Size: {abs(contracts):,.1f}")
        
        try:
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
            
            order_id = order.get('id', 'N/A')
            print(f"   ✅ Order placed: {order_id}")
            
            # Check order status
            time.sleep(2)
            try:
                order_status = exchange.fetch_order(order_id, symbol)
                status = order_status.get('status', 'unknown')
                filled = float(order_status.get('filled', 0) or 0)
                print(f"   📊 Status: {status} | Filled: {filled:,.1f}")
            except Exception as e:
                print(f"   ⚠️  Could not check order status: {e}")
            
        except Exception as e:
            print(f"   ❌ Order failed: {e}")
        
        # Wait before next attempt
        if attempt < MAX_ATTEMPTS:
            print(f"\n⏳ Waiting {WAIT_SECONDS} seconds before next attempt...")
            time.sleep(WAIT_SECONDS)
    
    # Final check
    position = get_position_info(SYMBOL)
    if not position:
        print(f"\n✅✅✅ SUCCESS! Position CLOSED! ✅✅✅")
        return True
    else:
        contracts = float(position.get('contracts', 0) or 0)
        print(f"\n⚠️  Position still open: {contracts:,.1f} contracts")
        print(f"   Max attempts ({MAX_ATTEMPTS}) reached")
        return False


def main():
    print("=" * 70)
    print("🔥 PERSISTENT MARKET ORDER CLOSER")
    print("=" * 70)
    
    success = close_position_persistent(SYMBOL)
    
    print("\n" + "=" * 70)
    if success:
        print("✅ MISSION ACCOMPLISHED!")
    else:
        print("⚠️  Manual intervention may be needed")
    print("=" * 70)


if __name__ == "__main__":
    main()

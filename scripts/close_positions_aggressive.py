#!/usr/bin/env python3
"""
Aggressive position closer - keeps increasing price until filled.
Continues to cancel and replace orders with higher prices until positions are closed.
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

# Configuration
MAX_ITERATIONS = 20  # Maximum number of attempts
WAIT_SECONDS = 5     # Wait between attempts
INITIAL_PRICE_MULTIPLIER = 1.10  # Start at +10%
PRICE_INCREMENT = 0.05  # Increase by 5% each iteration


def cancel_open_orders(symbol: str):
    """Cancel all open orders for a symbol."""
    try:
        open_orders = exchange.fetch_open_orders(symbol)
        for order in open_orders:
            try:
                exchange.cancel_order(order['id'], symbol)
                print(f"   ❌ Cancelled order {order['id']}")
            except Exception as e:
                print(f"   ⚠️  Could not cancel {order['id']}: {e}")
    except Exception as e:
        print(f"   ⚠️  Error fetching/cancelling orders: {e}")


def get_position_size(symbol: str) -> float:
    """Get current position size (0 if no position)."""
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            if pos['symbol'] == symbol:
                contracts = float(pos.get('contracts', 0))
                if contracts != 0:
                    return contracts
    except Exception as e:
        print(f"   ⚠️  Error fetching position: {e}")
    return 0.0


def close_position_aggressive(symbol: str):
    """Aggressively close position by continuously increasing price."""
    print(f"\n{'='*60}")
    print(f"🎯 Aggressively closing: {symbol}")
    print(f"{'='*60}")
    
    iteration = 0
    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n🔄 Iteration {iteration}/{MAX_ITERATIONS}")
        
        # Check if position still exists
        try:
            positions = exchange.fetch_positions([symbol])
            position = None
            for pos in positions:
                if pos['symbol'] == symbol and float(pos.get('contracts', 0)) != 0:
                    position = pos
                    break
            
            if not position:
                print(f"✅ {symbol}: Position closed successfully!")
                return True
            
            contracts = float(position['contracts'])
            side = position['side']
            entry_price = float(position.get('entryPrice', 0))
            unrealized_pnl = float(position.get('unrealizedPnl', 0))
            
            print(f"📊 Current Position:")
            print(f"   Side: {side.upper()}")
            print(f"   Size: {contracts}")
            print(f"   Entry Price: {entry_price}")
            print(f"   Unrealized PnL: {unrealized_pnl:.2f} USDT")
            
        except Exception as e:
            print(f"❌ Error checking position: {e}")
            time.sleep(WAIT_SECONDS)
            continue
        
        # Cancel existing open orders
        print(f"\n🧹 Cancelling existing orders...")
        cancel_open_orders(symbol)
        time.sleep(1)  # Give exchange time to process cancellation
        
        # Get current market price
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = float(ticker['last'])
            
            # Determine order side (opposite of position)
            order_side = "buy" if side == "short" else "sell"
            
            # Calculate increasingly aggressive price
            price_multiplier = INITIAL_PRICE_MULTIPLIER + (PRICE_INCREMENT * (iteration - 1))
            
            if order_side == "buy":
                limit_price = round(current_price * price_multiplier, 6)
            else:
                limit_price = round(current_price * (2 - price_multiplier), 6)
            
            price_diff_pct = ((limit_price / current_price - 1) * 100)
            
            print(f"\n🚀 Placing Order (Attempt #{iteration}):")
            print(f"   Side: {order_side.upper()}")
            print(f"   Size: {abs(contracts)}")
            print(f"   Current Price: {current_price}")
            print(f"   Limit Price: {limit_price}")
            print(f"   Price Multiplier: {price_multiplier:.2f}x")
            print(f"   Price Diff: {price_diff_pct:+.2f}%")
            
            # Place limit order
            try:
                order = exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=order_side,
                    amount=abs(contracts),
                    price=limit_price,
                    params={
                        'reduceOnly': True,
                        'timeInForce': 'PostOnly',
                        'positionIdx': 0
                    }
                )
                print(f"   ✅ Order placed: {order.get('id', 'N/A')}")
            except Exception as e:
                print(f"   ❌ Order failed: {e}")
                # If PostOnly fails, try GoodTillCancel
                try:
                    print(f"   🔄 Retrying with GTC...")
                    order = exchange.create_order(
                        symbol=symbol,
                        type='limit',
                        side=order_side,
                        amount=abs(contracts),
                        price=limit_price,
                        params={
                            'reduceOnly': True,
                            'timeInForce': 'GTC',
                            'positionIdx': 0
                        }
                    )
                    print(f"   ✅ Order placed (GTC): {order.get('id', 'N/A')}")
                except Exception as e2:
                    print(f"   ❌ GTC order also failed: {e2}")
            
        except Exception as e:
            print(f"❌ Error placing order: {e}")
        
        # Wait before next iteration
        if iteration < MAX_ITERATIONS:
            print(f"\n⏳ Waiting {WAIT_SECONDS} seconds before next check...")
            time.sleep(WAIT_SECONDS)
    
    # Check final status
    final_size = get_position_size(symbol)
    if final_size == 0:
        print(f"\n✅ {symbol}: Position closed!")
        return True
    else:
        print(f"\n⚠️  {symbol}: Position still open (size: {final_size})")
        print(f"   Max iterations reached. Consider manual intervention.")
        return False


def main():
    print("=" * 60)
    print("🔥 AGGRESSIVE Position Closer")
    print("   Continuously increases price until filled")
    print("=" * 60)
    
    for symbol in SYMBOLS_TO_CLOSE:
        success = close_position_aggressive(symbol)
        if not success:
            print(f"\n⚠️  Manual intervention may be needed for {symbol}")
    
    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

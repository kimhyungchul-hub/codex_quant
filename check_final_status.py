import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

import config
from core.orchestrator import build_exchange

async def main():
    exchange = await build_exchange()
    await exchange.load_markets()
    bal = await exchange.fetch_balance()
    usdt = bal.get('USDT', {})
    print(f"USDT Total: {usdt.get('total')}")
    print(f"USDT Free: {usdt.get('free')}")
    
    positions = await exchange.fetch_positions()
    active = [p for p in positions if float(p.get('contracts') or 0.0) != 0]
    print(f"Active positions: {len(active)}")
    for p in active:
        print(f"  {p['symbol']} {p['side']} {p['contracts']} UnrealPnL: {p.get('unrealizedPnl')}")
        
    await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())

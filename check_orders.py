import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

import config
from core.orchestrator import build_exchange

async def main():
    print("Initializing exchange...")
    exchange = await build_exchange()
    await exchange.load_markets()
    
    # Check recent orders for APT
    print("Fetching recent orders for APT/USDT:USDT...")
    orders = await exchange.fetch_orders('APT/USDT:USDT', limit=10)
    for o in orders:
        print(f"Order: {o['id']} {o['side']} {o['type']} {o['status']} {o['filled']}/{o['amount']} fee={o.get('fee')} info={o.get('info')}")
        
    await exchange.close()
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())

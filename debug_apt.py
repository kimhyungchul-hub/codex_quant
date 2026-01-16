import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

import config
from core.orchestrator import LiveOrchestrator, build_exchange

async def main():
    print("Initializing exchange...")
    exchange = await build_exchange()
    await exchange.load_markets()
    
    print("Fetching positions for APT/USDT:USDT...")
    positions = await exchange.fetch_positions(['APT/USDT:USDT'])
    for p in positions:
        print(f"Full position info: {p['info']}")
        sym = p.get('symbol')
        contracts = float(p.get('contracts', 0) or 0)
        if abs(contracts) > 0:
            side = p.get('side', '')
            print(f"Found active position: {sym} {side} {contracts}")
            
    await exchange.close()
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())

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
    
    print("Checking Position Mode...")
    # Bybit V5: getPositionInfo or similar? No, it's in the position response.
    # Actually, we can check the 'positionIdx' value.
    # 0 = One-way, 1/2 = Hedge.
    
    res = await exchange.privateGetV5PositionList({'category': 'linear', 'symbol': 'APTUSDT'})
    print(f"Full Position List response: {res}")
    
    await exchange.close()
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())

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
    
    print("Initializing orchestrator...")
    # We only care about liquidating, so we don't need real data feed here.
    orch = LiveOrchestrator(exchange, symbols=list(config.SYMBOLS))
    orch.enable_orders = True
    
    print("Fetching positions...")
    positions = await exchange.fetch_positions()
    for p in positions:
        sym = p.get('symbol')
        contracts = float(p.get('contracts', 0) or 0)
        if abs(contracts) > 0:
            side = p.get('side', '')
            print(f"Found position: {sym} {side} {contracts}")
            # Try to close
            try:
                side_to_send = 'sell' if side.upper() == 'LONG' else 'buy'
                params = {'reduceOnly': True}
                info = p.get('info', {})
                if 'positionIdx' in info:
                    params['positionIdx'] = int(info['positionIdx'])
                
                print(f"Closing {sym} with {side_to_send} amount {abs(contracts)} params {params}")
                res = await exchange.create_order(
                    symbol=sym,
                    type='market',
                    side=side_to_send,
                    amount=abs(contracts),
                    params=params
                )
                print(f"Result for {sym}: {res.get('id') or res}")
            except Exception as e:
                print(f"Failed to close {sym}: {e}")

    await exchange.close()
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Manual OHLCV preload helper.

Usage:

    python scripts/preload_ohlcv.py --limit 60

This runs `LiveOrchestrator.preload_all_ohlcv` with the provided limit before the full engine starts,
so the MC horizon meta is available within seconds instead of waiting for the normal bootstrap.
"""

import argparse
import asyncio

import aiohttp
import ccxt.async_support as ccxt

from main_engine_mc_v2_final import LiveOrchestrator, CCXT_TIMEOUT_MS, OHLCV_PRELOAD_LIMIT


async def _run(limit: int) -> None:
    session = aiohttp.ClientSession()
    exchange = ccxt.bybit(
        {
            "enableRateLimit": True,
            "timeout": CCXT_TIMEOUT_MS,
            "session": session,
        }
    )
    orchestrator = LiveOrchestrator(exchange)
    try:
        await orchestrator.preload_all_ohlcv(limit=limit)
    finally:
        try:
            await exchange.close()
        except Exception:
            pass
        try:
            await session.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Preload OHLCV so the orchestrator has enough horizons.")
    parser.add_argument(
        "--limit",
        type=int,
        default=OHLCV_PRELOAD_LIMIT,
        help="Number of 1m candles to fetch per symbol before the full engine runs.",
    )
    args = parser.parse_args()
    asyncio.run(_run(limit=args.limit))


if __name__ == "__main__":
    main()

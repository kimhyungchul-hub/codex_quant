import asyncio
import ccxt.async_support as ccxt

async def main():
    exchange = ccxt.bybit()
    try:
        await exchange.load_markets()
        symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        tickers = await exchange.fetch_tickers(symbols)
        for s in symbols:
            t = tickers.get(s, {})
            print(f"{s}: last={t.get('last')}, bid={t.get('bid')}, ask={t.get('ask')}")
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())

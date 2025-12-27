import asyncio
import time
import numpy as np
import os
from collections import deque
from config import *
from utils.helpers import now_ms, _env_float

class DataManager:
    def __init__(self, orchestrator, symbols=None):
        self.orch = orchestrator
        self.exchange = orchestrator.exchange
        self.symbols = symbols if symbols is not None else SYMBOLS
        
        self.market = {s: {"price": None, "bid": None, "ask": None, "ts": 0} for s in self.symbols}
        self.ohlcv_buffer = {s: deque(maxlen=OHLCV_PRELOAD_LIMIT) for s in self.symbols}
        self.orderbook = {s: {"ts": 0, "ready": False, "bids": [], "asks": []} for s in self.symbols}
        
        self._last_kline_ts = {s: 0 for s in self.symbols}
        self._last_kline_ok_ms = {s: 0 for s in self.symbols}
        self._preloaded = {s: False for s in self.symbols}
        self._last_feed_ok_ms = 0
        
        self.data_updated_event = asyncio.Event()

    async def fetch_prices_loop(self):
        while True:
            try:
                tickers = await self.orch._ccxt_call("fetch_tickers", self.exchange.fetch_tickers, self.symbols)
                ts = now_ms()
                ok_any = False
                for s in self.symbols:
                    t = tickers.get(s) or {}
                    last = t.get("last")
                    bid = t.get("bid")
                    ask = t.get("ask")
                    if last is not None:
                        self.market[s]["price"] = float(last)
                        ok_any = True
                    if bid is not None:
                        self.market[s]["bid"] = float(bid)
                        ok_any = True
                    if ask is not None:
                        self.market[s]["ask"] = float(ask)
                        ok_any = True
                    if (last is not None) or (bid is not None) or (ask is not None):
                        self.market[s]["ts"] = ts
                if ok_any:
                    self._last_feed_ok_ms = ts
                    self.data_updated_event.set()
            except Exception as e:
                self.orch._log_err(f"[ERR] fetch_tickers: {e}")
            ticker_sleep = _env_float("TICKER_SLEEP_SEC", 1.0)
            await asyncio.sleep(ticker_sleep)

    async def preload_all_ohlcv(self, limit: int = OHLCV_PRELOAD_LIMIT):
        for sym in self.symbols:
            try:
                print(f"[PRELOAD] Fetching {sym}...")
                ohlcv = await self.exchange.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=limit)
                if not ohlcv:
                    print(f"[PRELOAD] {sym} - no data received")
                    continue
                self.ohlcv_buffer[sym].clear()
                last_ts = 0
                for c in ohlcv:
                    ts_ms = int(c[0])
                    close_price = float(c[4])
                    self.ohlcv_buffer[sym].append(close_price)
                    last_ts = ts_ms
                self._last_kline_ts[sym] = last_ts
                self._last_kline_ok_ms[sym] = now_ms()
                self._preloaded[sym] = True
                msg = f"[PRELOAD] {sym} candles={len(self.ohlcv_buffer[sym])}"
                print(msg)
                self.orch._log(msg)
            except Exception as e:
                err_msg = f"[ERR] preload_ohlcv {sym}: {e}"
                print(err_msg)
                self.orch._log_err(err_msg)


    async def fetch_ohlcv_loop(self):
        while True:
            start = now_ms()
            try:
                for sym in self.symbols:
                    try:
                        ohlcv = await self.orch._ccxt_call(
                            f"fetch_ohlcv {sym}",
                            self.exchange.fetch_ohlcv,
                            sym, timeframe=TIMEFRAME, limit=OHLCV_REFRESH_LIMIT
                        )
                        if not ohlcv:
                            continue
                        last = ohlcv[-1]
                        ts_ms = int(last[0])
                        close_price = float(last[4])

                        if ts_ms != self._last_kline_ts[sym]:
                            self.ohlcv_buffer[sym].append(close_price)
                            self._last_kline_ts[sym] = ts_ms
                            self._last_kline_ok_ms[sym] = now_ms()
                            self.data_updated_event.set()
                    except Exception as e_sym:
                        self.orch._log_err(f"[ERR] fetch_ohlcv {sym}: {e_sym}")
            except Exception as e:
                self.orch._log_err(f"[ERR] fetch_ohlcv(loop): {e}")

            elapsed = (now_ms() - start) / 1000.0
            sleep_left = max(1.0, OHLCV_SLEEP_SEC - elapsed)
            await asyncio.sleep(sleep_left)

    async def fetch_orderbook_loop(self):
        while True:
            start = now_ms()
            for sym in self.symbols:
                try:
                    ob = await self.orch._ccxt_call(
                        f"fetch_orderbook {sym}",
                        self.exchange.fetch_order_book,
                        sym, limit=ORDERBOOK_DEPTH
                    )
                    bids = (ob.get("bids") or [])[:ORDERBOOK_DEPTH]
                    asks = (ob.get("asks") or [])[:ORDERBOOK_DEPTH]
                    ready = bool(bids) and bool(asks)
                    self.orderbook[sym]["bids"] = bids
                    self.orderbook[sym]["asks"] = asks
                    self.orderbook[sym]["ready"] = ready
                    self.orderbook[sym]["ts"] = now_ms()
                    if ready:
                        self.orch._log(f"[ORDERBOOK] {sym} | fetched successfully bids={len(bids)} asks={len(asks)}")
                    else:
                        self.orch._log_err(f"[ORDERBOOK] {sym} | fetched but empty bids={len(bids)} asks={len(asks)}")
                except Exception as e_sym:
                    self.orderbook[sym]["ready"] = False
                    extra = ""
                    try:
                        status = getattr(e_sym, "status", None)
                        body = getattr(e_sym, "body", None)
                        exc_type = type(e_sym).__name__
                        if status is not None: extra += f" status={status}"
                        if body: extra += f" body={str(body)[:300]}"
                        extra += f" type={exc_type}"
                    except Exception: pass
                    self.orch._log_err(f"[ERR] fetch_orderbook {sym}: {repr(e_sym)}{extra}")
                await asyncio.sleep(ORDERBOOK_SYMBOL_INTERVAL_SEC)

            self.data_updated_event.set()
            elapsed = (now_ms() - start) / 1000.0
            sleep_left = max(0.0, ORDERBOOK_SLEEP_SEC - elapsed)
            await asyncio.sleep(sleep_left)

    def get_btc_corr(self, sym: str, window: int = 60) -> float:
        """
        BTC와의 최근 수익률 상관관계를 계산 (최근 window개 캔들 기준)
        """
        if sym.startswith("BTC"):
            return 1.0
        
        btc_sym = self.symbols[0] if self.symbols else "BTC/USDT:USDT"
        if not btc_sym.startswith("BTC"):
            # SYMBOLS[0]이 BTC가 아니면 명시적으로 찾기
            for s in self.symbols:
                if s.startswith("BTC"):
                    btc_sym = s
                    break
        
        btc_closes = list(self.ohlcv_buffer.get(btc_sym) or [])
        sym_closes = list(self.ohlcv_buffer.get(sym) or [])
        
        if len(btc_closes) < window + 1 or len(sym_closes) < window + 1:
            return 0.0
            
        # 길이 맞추기
        min_len = min(len(btc_closes), len(sym_closes))
        btc_closes = btc_closes[-min_len:]
        sym_closes = sym_closes[-min_len:]
        
        try:
            btc_ret = np.diff(np.log(np.array(btc_closes[-window-1:], dtype=np.float64)))
            sym_ret = np.diff(np.log(np.array(sym_closes[-window-1:], dtype=np.float64)))
            
            if len(btc_ret) < window or len(sym_ret) < window:
                return 0.0
                
            corr = np.corrcoef(btc_ret, sym_ret)[0, 1]
            if np.isnan(corr):
                return 0.0
            return float(corr)
        except Exception:
            return 0.0

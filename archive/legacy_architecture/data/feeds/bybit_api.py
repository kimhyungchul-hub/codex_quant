from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass
class BybitCredentials:
    api_key: str
    api_secret: str
    testnet: bool = True


class BybitRESTClient:
    """Minimal async REST wrapper for Bybit v5 endpoints using stdlib HTTP."""

    def __init__(self, creds: BybitCredentials, timeout: float = 5.0) -> None:
        self.creds = creds
        self.timeout = timeout
        self.base_url = (
            "https://api-testnet.bybit.com" if creds.testnet else "https://api.bybit.com"
        )
        self.logger = logging.getLogger("bybit_rest")

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = int(time.time() * 1000)
        params["api_key"] = self.creds.api_key
        params["timestamp"] = timestamp
        # v5 signing: sort keys alphabetically
        ordered = "".join(f"{k}={params[k]}" for k in sorted(params))
        signature = hmac.new(
            self.creds.api_secret.encode(), ordered.encode(), hashlib.sha256
        ).hexdigest()
        params["sign"] = signature
        return params

    async def _request(self, method: str, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        def _do_request() -> Dict[str, Any]:
            signed = self._sign(params.copy())
            encoded = urllib.parse.urlencode(signed).encode()
            url = f"{self.base_url}{path}"
            if method.upper() == "GET":
                url = f"{url}?{encoded.decode()}"
                data = None
            else:
                data = encoded
            req = urllib.request.Request(url=url, data=data, method=method.upper())
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode()
                return json.loads(body)

        return await asyncio.to_thread(_do_request)

    async def create_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "Market",
        price: Optional[float] = None,
        category: str = "linear",
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.capitalize(),
            "qty": qty,
            "orderType": order_type.capitalize(),
            "category": category,
            "timeInForce": "GTC",
        }
        if price is not None:
            payload["price"] = price
        return await self._request("POST", "/v5/order/create", payload)

    async def fetch_positions(self, category: str = "linear") -> Dict[str, Any]:
        payload = {"category": category}
        return await self._request("GET", "/v5/position/list", payload)


class BybitWebSocketClient:
    """Lightweight WS wrapper that streams public trades. Uses websockets if available."""

    def __init__(self, creds: Optional[BybitCredentials] = None, testnet: bool = True) -> None:
        self.creds = creds
        self.testnet = testnet
        self.logger = logging.getLogger("bybit_ws")

    async def stream_public_trades(self, symbols: List[str]) -> AsyncIterator[Dict[str, Any]]:
        try:
            import websockets
        except ImportError:
            self.logger.warning(
                "websockets package not installed; emitting heartbeat-only trade stubs"
            )
            while True:
                await asyncio.sleep(1)
                for symbol in symbols:
                    yield {
                        "symbol": symbol,
                        "price": 0.0,
                        "size": 0.0,
                        "ts": int(time.time() * 1000),
                    }
            return

        endpoint = (
            "wss://stream-testnet.bybit.com/v5/public/linear"
            if self.testnet
            else "wss://stream.bybit.com/v5/public/linear"
        )
        params = {"op": "subscribe", "args": [f"publicTrade.{s}" for s in symbols]}
        async with websockets.connect(endpoint, ping_interval=20) as ws:  # type: ignore
            await ws.send(json.dumps(params))
            async for msg in ws:
                data = json.loads(msg)
                if data.get("topic", "").startswith("publicTrade"):
                    for trade in data.get("data", []):
                        yield trade

import json
import asyncio
import os
import time
from aiohttp import web
from config import PORT, DASHBOARD_FILE, DASHBOARD_HISTORY_MAX, DASHBOARD_TRADE_TAPE_MAX, EXEC_MODE, MAKER_TIMEOUT_MS, MAKER_RETRIES, MAKER_POLL_MS, MAX_NOTIONAL_EXPOSURE
from utils.helpers import now_ms, _sanitize_for_json

def _exec_stats_snapshot(orch):
    # Logic from LiveOrchestrator._exec_stats_snapshot
    res = {}
    for sym, st in orch.exec_stats.items():
        res[sym] = {
            "maker_fills": st.get("maker_limit_filled", 0),
            "taker_fills": st.get("taker_market_filled", 0),
            # ... more stats ...
        }
    return res

def _compute_portfolio(orch):
    # Logic from LiveOrchestrator._compute_portfolio
    equity = orch.balance
    unreal = 0.0
    pos_list = []
    for sym, pos in orch.positions.items():
        pos_size = float(pos.get("size", pos.get("quantity", 0.0)) or 0.0)
        if pos_size != 0:
            unreal += pos.get("unrealized_pnl", 0.0)
            pos_list.append(pos)
    equity += unreal
    util = orch._total_open_notional() / max(orch.balance, 1.0)
    return equity, unreal, util, pos_list

def _compute_eval_metrics(orch):
    # Compute eval metrics from equity history
    try:
        if len(orch._equity_history) >= 2:
            eq_list = list(orch._equity_history)
            initial = eq_list[0]
            current = eq_list[-1]
            if initial > 0:
                total_return = (current - initial) / initial
            else:
                total_return = 0.0
            return {"total_return": total_return}
        return {"total_return": 0.0}
    except Exception:
        return {"total_return": 0.0}

def _build_payload(orch, rows, include_history, include_trade_tape):
    equity, unreal, util, pos_list = _compute_portfolio(orch)
    eval_metrics = _compute_eval_metrics(orch)
    ts = now_ms()

    feed_connected = (orch.data._last_feed_ok_ms > 0) and (ts - orch.data._last_feed_ok_ms < 10_000)
    feed = {
        "connected": bool(feed_connected),
        "last_msg_age": (ts - orch.data._last_feed_ok_ms) if orch.data._last_feed_ok_ms else None
    }

    history = []
    if include_history:
        history = list(orch._equity_history)[-int(DASHBOARD_HISTORY_MAX):]

    trade_tape = []
    if include_trade_tape:
        trade_tape = list(orch.trade_tape)[-int(DASHBOARD_TRADE_TAPE_MAX):]

    return {
        "type": "full_update",
        "server_time": ts,
        "engine": {
            "modules_ok": True,
            "ws_clients": len(orch.clients),
            "loop_ms": getattr(orch, "_loop_ms", None),
            "mc_ready": bool(getattr(orch, "_mc_ready", False)),
            "enable_orders": bool(orch.enable_orders),
            "exec_mode": str(os.environ.get("EXEC_MODE", EXEC_MODE)).strip().lower(),
            "maker_timeout_ms": int(os.environ.get("MAKER_TIMEOUT_MS", str(MAKER_TIMEOUT_MS))),
            "maker_retries": int(os.environ.get("MAKER_RETRIES", str(MAKER_RETRIES))),
            "maker_poll_ms": int(os.environ.get("MAKER_POLL_MS", str(MAKER_POLL_MS))),
            "exec_stats": _exec_stats_snapshot(orch),
            "pmaker": orch.pmaker.status_dict(),
        },
        "feed": feed,
        "market": rows,
        "portfolio": {
            "balance": float(orch.balance),
            "equity": float(equity),
            "unrealized_pnl": float(unreal),
            "utilization": util,
            "utilization_cap": MAX_NOTIONAL_EXPOSURE if orch.exposure_cap_enabled else None,
            "positions": pos_list,
            "history": history,
        },
        "eval_metrics": eval_metrics,
        "logs": list(orch.logs),
        "trade_tape": trade_tape,
    }

async def index_handler(request):
    return web.FileResponse(str(DASHBOARD_FILE))

async def debug_payload_handler(request):
    orch = request.app["orchestrator"]
    ts = now_ms()
    rows = getattr(orch, "_last_rows", None) or orch._rows_snapshot(ts)
    payload = _build_payload(orch, rows, include_history=False, include_trade_tape=False)
    payload = _sanitize_for_json(payload)
    return web.json_response(payload, dumps=lambda x: json.dumps(x, ensure_ascii=False, separators=(",", ":")))

async def ws_handler(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    orch = request.app["orchestrator"]
    orch.clients.add(ws)
    
    try:
        snap_rows0 = getattr(orch, "_last_rows", None) or orch._rows_snapshot(now_ms())
        snap0 = _build_payload(orch, snap_rows0, include_history=False, include_trade_tape=False)
        snap0 = _sanitize_for_json(snap0)
        await ws.send_str(json.dumps(snap0, ensure_ascii=False, separators=(",", ":")))
    except Exception as e:
        print(f"[ERR] ws_handler initial snapshot: {e}")

    await ws.send_str(json.dumps({"type": "init", "msg": "connected"}, ensure_ascii=False, separators=(",", ":")))

    try:
        snap_rows = getattr(orch, "_last_rows", None) or orch._rows_snapshot(now_ms())
        snap_payload = _build_payload(orch, snap_rows, include_history=True, include_trade_tape=True)
        snap_payload = _sanitize_for_json(snap_payload)
        await ws.send_str(json.dumps(snap_payload, ensure_ascii=False, separators=(",", ":")))
    except Exception as e:
        print(f"[ERR] ws_handler full snapshot: {e}")

    async for _ in ws:
        pass

    orch.clients.discard(ws)
    return ws

class DashboardServer:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.app = web.Application()
        self.app["orchestrator"] = orchestrator
        self.app.add_routes([
            web.get("/", index_handler),
            web.get("/ws", ws_handler),
            web.get("/debug/payload", debug_payload_handler)
        ])
        self.runner = None
        self.site = None

    async def start(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "0.0.0.0", PORT)
        await self.site.start()
        print(f"🚀 Dashboard: http://localhost:{PORT}")

    async def stop(self):
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

    async def broadcast(self, rows):
        if not self.orchestrator.clients:
            return
        payload = _build_payload(self.orchestrator, rows, include_history=False, include_trade_tape=False)
        payload = _sanitize_for_json(payload)
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        
        for ws in list(self.orchestrator.clients):
            try:
                await ws.send_str(data)
            except Exception:
                self.orchestrator.clients.discard(ws)

class DummyExchange:
    async def close(self):
        return None


def test_orchestrator_has_expected_public_api():
    from core.orchestrator import LiveOrchestrator

    orch = LiveOrchestrator(DummyExchange(), symbols=["BTC/USDT:USDT"])

    # Dashboard-facing methods
    assert hasattr(orch, "runtime_config")
    assert hasattr(orch, "set_enable_orders")
    assert hasattr(orch, "score_debug_for_symbol")
    assert hasattr(orch, "liquidate_all_positions")
    assert hasattr(orch, "liquidate_all_positions_live")

    # main.py expects this when enable_orders is True
    assert hasattr(orch, "live_sync_loop")

    # Auxiliary modules assume these exist
    assert isinstance(getattr(orch, "_group_info"), dict)
    assert isinstance(getattr(orch, "_latest_rankings"), list)
    assert isinstance(getattr(orch, "_symbol_t_star"), dict)
    assert isinstance(getattr(orch, "_symbol_scores"), dict)

    # Should return a dict even without decisions
    out = orch.score_debug_for_symbol("BTC/USDT:USDT")
    assert isinstance(out, dict)
    assert out.get("symbol") == "BTC/USDT:USDT"


def test_orchestrator_close_position_paper_no_crash():
    from core.orchestrator import LiveOrchestrator

    orch = LiveOrchestrator(DummyExchange(), symbols=["BTC/USDT:USDT"])

    # Seed a minimal paper position and close via compatibility API.
    orch.positions["BTC/USDT:USDT"] = {
        "symbol": "BTC/USDT:USDT",
        "side": "LONG",
        "entry_price": 100.0,
        "size": 1.0,
        "margin": 100.0,
        "notional": 100.0,
        "fee_roundtrip": 0.0,
    }

    import asyncio
    asyncio.run(orch._close_position(sym="BTC/USDT:USDT", exit_price=101.0, ts_ms=1, reason="test"))
    assert "BTC/USDT:USDT" not in orch.positions

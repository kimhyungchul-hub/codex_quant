from __future__ import annotations

import json
from pathlib import Path

from collections import deque

from core.state_store import StateStore


def test_state_store_persist_and_load(tmp_path: Path) -> None:
    store = StateStore(state_dir=tmp_path)

    balance = 1234.5
    positions = {"BTC/USDT": {"qty": 1.0, "entry": 100.0}}
    trade_tape = deque(
        [{"t": 1, "symbol": "BTC/USDT", "side": "buy", "qty": 1.0, "px": 100.0}],
        maxlen=store.trade_tape_maxlen,
    )

    store.persist(balance=balance, positions=positions, trade_tape=trade_tape, force=True)

    loaded_balance, loaded_positions, loaded_trade_tape = store.load()
    assert loaded_balance == balance
    assert loaded_positions == positions
    assert list(loaded_trade_tape) == list(trade_tape)


def test_state_store_reset(tmp_path: Path) -> None:
    store = StateStore(state_dir=tmp_path)

    store.reset(initial_balance=10000.0)

    assert store.files.balance.exists()
    assert store.files.positions.exists()
    assert store.files.trade.exists()
    assert store.files.equity.exists()

    assert float(store.files.balance.read_text(encoding="utf-8").strip()) == 10000.0
    assert json.loads(store.files.positions.read_text(encoding="utf-8")) == {}
    assert json.loads(store.files.trade.read_text(encoding="utf-8")) == []

    equity_rows = json.loads(store.files.equity.read_text(encoding="utf-8"))
    assert isinstance(equity_rows, list) and len(equity_rows) == 1
    assert equity_rows[0]["equity"] == 10000.0


def test_state_store_load_filters_stale_positions(tmp_path: Path) -> None:
    store = StateStore(state_dir=tmp_path)

    # Mock now_ms to return a fixed time
    import time
    from unittest.mock import patch

    fixed_now = 1000 * 3600 * 1000  # 1000 hours in ms
    stale_threshold = store.stale_position_threshold_ms  # 24h

    # Create positions: one fresh, one stale
    positions_data = {
        "BTC/USDT": {"time": fixed_now - 1 * 3600 * 1000, "qty": 1.0},  # fresh (1h ago)
        "ETH/USDT": {"time": fixed_now - 25 * 3600 * 1000, "qty": 2.0},  # stale (25h ago)
    }

    # Write to file
    store.files.positions.write_text(json.dumps(positions_data), encoding="utf-8")

    with patch("core.state_store.now_ms", return_value=fixed_now):
        loaded_balance, loaded_positions, loaded_trade_tape = store.load()

    # Only fresh position should remain
    assert "BTC/USDT" in loaded_positions
    assert "ETH/USDT" not in loaded_positions
    assert loaded_positions["BTC/USDT"]["qty"] == 1.0

"""audit-r3 R5: strategy_server /signal must expose trading_mode so the
MQL5 EA (AISMCReceiver.mq5) can choose between ranging (300s) and
trending (1800s) cooldown.

The field was added in Round 2 but the EA didn't consume it; R5 adds
the EA-side read (mq5 code — not auto-testable).  These tests lock the
Python-side contract so the field cannot regress to missing without
breaking a unit test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    """Import strategy_server with DATA_ROOT pointing at tmp_path.

    scripts/strategy_server.py hardcodes DATA_ROOT = Path("data") at
    import time.  Patch it after import so each test has an isolated
    per-symbol dir.
    """
    scripts_dir = Path(__file__).resolve().parents[4] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import strategy_server  # noqa: WPS433 — test-only import
    finally:
        sys.path.pop(0)

    monkeypatch.setattr(strategy_server, "DATA_ROOT", tmp_path)
    return TestClient(strategy_server.app)


def _write_state(tmp_path: Path, symbol: str, state: dict) -> None:
    sym_dir = tmp_path / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    (sym_dir / "live_state.json").write_text(json.dumps(state))


class TestTradingModePropagation:
    """R5 contract: /signal response MUST carry trading_mode."""

    def test_ranging_mode_propagates(self, client, tmp_path):
        _write_state(tmp_path, "XAUUSD", {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 42,
            "action": "RANGE BUY",
            "trading_mode": "ranging",
            "best_setup": {
                "direction": "long",
                "entry": 2350.0,
                "sl": 2348.0,
                "tp1": 2352.0,
                "position_size_lots": 0.5,
                "confluence": 0.7,
            },
        })
        resp = client.get("/signal?symbol=XAUUSD")
        assert resp.status_code == 200
        body = resp.json()
        assert body["trading_mode"] == "ranging"

    def test_trending_mode_propagates(self, client, tmp_path):
        _write_state(tmp_path, "XAUUSD", {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 42,
            "action": "BUY",
            "trading_mode": "trending",
            "best_setup": {
                "direction": "long",
                "entry": 2350.0,
                "sl": 2348.0,
                "tp1": 2354.0,
                "position_size_lots": 0.3,
                "confluence": 0.8,
            },
        })
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        assert body["trading_mode"] == "trending"

    def test_missing_mode_returns_null(self, client, tmp_path):
        """Legacy state.json without trading_mode → null; EA falls back to trending."""
        _write_state(tmp_path, "XAUUSD", {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 42,
            "action": "HOLD",
            # trading_mode intentionally absent
        })
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        assert body["trading_mode"] is None

    def test_v1_passthrough_mode_propagates(self, client, tmp_path):
        """Any string value should pass through untouched — EA chooses fallback
        for modes it doesn't recognise (defensive client-side)."""
        _write_state(tmp_path, "XAUUSD", {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 42,
            "action": "HOLD",
            "trading_mode": "v1_passthrough",
        })
        resp = client.get("/signal?symbol=XAUUSD")
        body = resp.json()
        assert body["trading_mode"] == "v1_passthrough"


class TestBtcStateIsolation:
    """R5 is symbol-agnostic; BTC state must produce BTC-tagged signal."""

    def test_btc_ranging_mode(self, client, tmp_path):
        _write_state(tmp_path, "BTCUSD", {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "cycle": 10,
            "action": "RANGE SELL",
            "trading_mode": "ranging",
            "best_setup": {
                "direction": "short",
                "entry": 65000.0,
                "sl": 65200.0,
                "tp1": 64900.0,
                "position_size_lots": 1.0,
                "confluence": 0.6,
            },
        })
        resp = client.get("/signal?symbol=BTCUSD")
        body = resp.json()
        assert body["symbol"] == "BTCUSD"
        assert body["trading_mode"] == "ranging"

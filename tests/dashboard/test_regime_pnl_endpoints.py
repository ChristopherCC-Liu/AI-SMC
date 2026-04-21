"""HTTP-level tests for dashboard_server /api/regime + /api/pnl (O1/O2)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent.parent
_SERVER_PATH = _ROOT / "scripts" / "dashboard_server.py"


def _load_server():
    src = str(_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    spec = importlib.util.spec_from_file_location("dashboard_server_rpl", _SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_srv = _load_server()


@pytest.fixture(autouse=True)
def _reset_cache():
    from smc.monitor.dashboard_feeds import reset_pnl_cache
    reset_pnl_cache()
    yield
    reset_pnl_cache()


def _write_structured(path: Path, rows: list[tuple[str, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for severity, row in rows:
            f.write(f"[{severity}] {json.dumps(row)}\n")


# ---------------------------------------------------------------------------
# /api/regime
# ---------------------------------------------------------------------------


def test_api_regime_returns_structured_events(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    _write_structured(logs_dir / "structured.jsonl", [
        ("INFO", {"ts": "2026-04-20T01:00:00+00:00", "event": "ai_regime_classified",
                  "source": "default", "regime": "TRANSITION", "confidence": 0.3}),
        ("INFO", {"ts": "2026-04-20T02:00:00+00:00", "event": "ai_regime_classified",
                  "source": "claude_debate", "regime": "TRENDING", "confidence": 0.85,
                  "direction": "bullish", "reasoning": "SMA up"}),
    ])

    with patch.object(_srv, "ROOT", tmp_path):
        client = TestClient(_srv.app)
        r = client.get("/api/regime?limit=5")
    assert r.status_code == 200
    body = r.json()
    assert "events" in body
    assert "server_time" in body
    assert len(body["events"]) == 1  # default filtered
    assert body["events"][0]["regime"] == "TRENDING"
    assert body["events"][0]["direction"] == "bullish"


def test_api_regime_validates_limit_bounds() -> None:
    client = TestClient(_srv.app)
    assert client.get("/api/regime?limit=0").status_code == 422
    assert client.get("/api/regime?limit=9999").status_code == 422


def test_api_regime_empty_when_log_missing(tmp_path: Path) -> None:
    with patch.object(_srv, "ROOT", tmp_path):
        client = TestClient(_srv.app)
        r = client.get("/api/regime")
    assert r.status_code == 200
    assert r.json()["events"] == []


# ---------------------------------------------------------------------------
# /api/pnl
# ---------------------------------------------------------------------------


class _FakeDeal:
    def __init__(self, magic, symbol="XAUUSD", profit=0.0, entry=1):
        self.magic = magic
        self.symbol = symbol
        self.profit = profit
        self.commission = 0.0
        self.swap = 0.0
        self.entry = entry


class _FakePos:
    def __init__(self, symbol, magic, profit=0.0):
        self.symbol = symbol
        self.magic = magic
        self.profit = profit


class _FakeMT5:
    DEAL_ENTRY_OUT = 1
    DEAL_ENTRY_INOUT = 2

    def __init__(self, deals=None, positions=None):
        self._deals = deals or []
        self._positions = positions or []

    def history_deals_get(self, *_a, **_kw):
        return self._deals

    def positions_get(self):
        return self._positions


def test_api_pnl_returns_three_legs(tmp_path: Path) -> None:
    fake_mt5 = _FakeMT5(
        deals=[
            _FakeDeal(magic=19760418, symbol="XAUUSD", profit=10.0),
            _FakeDeal(magic=19760428, symbol="XAUUSD", profit=-2.0),
        ],
        positions=[
            _FakePos(symbol="XAUUSD", magic=19760418, profit=4.2),
        ],
    )
    with patch.object(_srv, "DATA", tmp_path), \
         patch.dict(sys.modules, {"MetaTrader5": fake_mt5}):
        client = TestClient(_srv.app)
        r = client.get("/api/pnl")

    assert r.status_code == 200
    body = r.json()
    assert set(body["legs"].keys()) == {"control_xau", "treatment_xau", "control_btc"}
    assert body["legs"]["control_xau"]["realized"] == 10.0
    assert body["legs"]["control_xau"]["floating"] == 4.2
    assert body["legs"]["treatment_xau"]["realized"] == -2.0
    assert body["legs"]["control_btc"]["realized"] == 0.0
    assert body["total"]["realized"] == 8.0
    # There must be no BTC-treatment leg ever.
    assert "treatment_btc" not in body["legs"]


def test_api_pnl_graceful_without_mt5(tmp_path: Path) -> None:
    with patch.object(_srv, "DATA", tmp_path), \
         patch.dict(sys.modules, {"MetaTrader5": None}):
        client = TestClient(_srv.app)
        r = client.get("/api/pnl")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "mt5_unavailable"
    assert body["legs"]["control_xau"]["realized"] == 0.0


def test_api_pnl_second_call_hits_cache(tmp_path: Path) -> None:
    """Dashboard polls /api/pnl every 5s; cache TTL should cover repeat calls."""
    calls = {"deals": 0}

    class _CountingMT5(_FakeMT5):
        def history_deals_get(self, *a, **kw):
            calls["deals"] += 1
            return super().history_deals_get(*a, **kw)

    fake = _CountingMT5(deals=[_FakeDeal(magic=19760418, symbol="XAUUSD", profit=5.0)])
    with patch.object(_srv, "DATA", tmp_path), \
         patch.dict(sys.modules, {"MetaTrader5": fake}):
        client = TestClient(_srv.app)
        first = client.get("/api/pnl").json()
        second = client.get("/api/pnl").json()

    assert calls["deals"] == 1
    assert first["cached"] is False
    assert second["cached"] is True
    assert first["legs"]["control_xau"]["realized"] == second["legs"]["control_xau"]["realized"]

"""Round 3 Sprint 2 integration: /api/guards + /healthz endpoint wiring.

Exercises FastAPI wire-up + data-source composition against synthetic
data/ + logs/ trees.  Pairs with test_daily_digest_endpoint.py fixture.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _mk_tree(tmp_path: Path, symbol: str = "XAUUSD") -> Path:
    data_root = tmp_path / "data" / symbol
    (data_root / "journal").mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    return data_root


@pytest.fixture
def client(tmp_path, monkeypatch):
    import sys
    scripts_dir = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))
    if "dashboard_server" in sys.modules:
        del sys.modules["dashboard_server"]
    monkeypatch.chdir(tmp_path)
    import dashboard_server as ds
    ds.ROOT = tmp_path
    ds.DATA = tmp_path / "data"
    return TestClient(ds.app)


class TestHealthz:
    def test_healthz_returns_200_when_data_root_exists(self, client, tmp_path):
        _mk_tree(tmp_path)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["service"] == "AI-SMC Dashboard"
        assert body["probes"]["json_parser"] == "ok"
        assert body["probes"]["data_root_exists"] is True

    def test_healthz_returns_503_when_data_root_missing(self, client, tmp_path):
        # Don't create data/ directory → probe fails
        resp = client.get("/healthz")
        assert resp.status_code == 503
        body = resp.json()
        assert body["ok"] is False
        assert body["probes"]["data_root_exists"] is False


class TestGuardsEndpoint:
    def test_empty_state_returns_green_all(self, client, tmp_path):
        _mk_tree(tmp_path)
        resp = client.get("/api/guards?symbol=XAUUSD")
        assert resp.status_code == 200
        body = resp.json()
        assert body["symbol"] == "XAUUSD"
        assert body["can_trade"] is True
        assert body["consec_halt"]["status"] == "green"
        assert body["phase1a_breaker"]["status"] == "green"
        assert body["asian_range_quota"]["status"] == "green"
        assert body["drawdown_guard"]["status"] == "green"
        # warnings document missing state files so dashboard can render "unknown"
        assert len(body["warnings"]) > 0

    def test_tripped_consec_halt_flips_can_trade(self, client, tmp_path):
        data_root = _mk_tree(tmp_path)
        (data_root / "consec_loss_state.json").write_text(json.dumps({
            "consec_losses": 3, "tripped": True,
            "tripped_at": "2026-04-18T10:00:00+00:00",
        }))
        resp = client.get("/api/guards?symbol=XAUUSD")
        assert resp.status_code == 200
        body = resp.json()
        assert body["consec_halt"]["status"] == "red"
        assert body["can_trade"] is False

    def test_btc_uses_btc_consec_limit(self, client, tmp_path):
        """R4 integration: BTC cfg.consec_loss_limit should flow to snapshot."""
        data_root = _mk_tree(tmp_path, symbol="BTCUSD")
        (data_root / "consec_loss_state.json").write_text(json.dumps({
            "consec_losses": 2, "tripped": False,
        }))
        resp = client.get("/api/guards?symbol=BTCUSD")
        assert resp.status_code == 200
        body = resp.json()
        # BTC default consec_loss_limit is 3 (per btcusd.py)
        assert body["consec_halt"]["max_losses"] == 3

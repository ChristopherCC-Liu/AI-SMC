"""Round 3 Sprint 1 integration: GET /api/daily_digest endpoint.

Validates wire-up of FastAPI endpoint + daily_digest builder against
synthetic data/ + logs/ trees. Exercises happy path + invalid date.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _mk_tree(tmp_path: Path, symbol: str = "XAUUSD") -> tuple[Path, Path]:
    data_root = tmp_path / "data" / symbol
    (data_root / "journal").mkdir(parents=True)
    log_root = tmp_path / "logs"
    log_root.mkdir()
    return data_root, log_root


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Spin up dashboard_server with ROOT pointed at tmp_path."""
    import sys
    scripts_dir = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))
    # Force ROOT to tmp_path BEFORE import so the endpoint resolves to fixtures.
    import importlib
    if "dashboard_server" in sys.modules:
        del sys.modules["dashboard_server"]
    monkeypatch.chdir(tmp_path)
    import dashboard_server as ds
    ds.ROOT = tmp_path
    ds.DATA = tmp_path / "data"
    return TestClient(ds.app)


class TestDailyDigestEndpoint:
    def test_empty_day_returns_200_with_zeros(self, client, tmp_path):
        _mk_tree(tmp_path)
        resp = client.get("/api/daily_digest?symbol=XAUUSD&date=2026-04-18")
        assert resp.status_code == 200
        body = resp.json()
        assert body["symbol"] == "XAUUSD"
        assert body["date"] == "2026-04-18"
        assert body["trades_opened"] == 0
        assert body["trades_closed"] == 0
        assert "journal_missing" in body["warnings"]

    def test_invalid_date_returns_400(self, client):
        resp = client.get("/api/daily_digest?symbol=XAUUSD&date=not-a-date")
        assert resp.status_code == 400
        assert "Invalid date" in resp.json().get("detail", "")

    def test_default_date_is_today_utc(self, client, tmp_path):
        _mk_tree(tmp_path)
        resp = client.get("/api/daily_digest?symbol=XAUUSD")
        assert resp.status_code == 200
        today = datetime.now(timezone.utc).date().isoformat()
        assert resp.json()["date"] == today

    def test_data_wired_through_builder(self, client, tmp_path):
        data_root, log_root = _mk_tree(tmp_path)
        # Write a PAPER journal entry today
        today = datetime.now(timezone.utc).date().isoformat()
        (data_root / "journal" / "live_trades.jsonl").write_text(
            json.dumps({"time": f"{today}T10:00:00+00:00", "mode": "PAPER"}) + "\n",
            encoding="utf-8",
        )
        resp = client.get(f"/api/daily_digest?symbol=XAUUSD&date={today}")
        assert resp.status_code == 200
        assert resp.json()["trades_opened"] == 1

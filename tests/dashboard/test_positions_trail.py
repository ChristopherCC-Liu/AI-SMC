"""Round 6 B4 tests: /api/positions enriches each row with trail params.

Two layers of coverage:
  1. Pure function ``attach_trail_params_to_positions`` — mocks nothing,
     just writes a fake ``structured.jsonl`` + passes a list of position
     dicts. Validates the core regime-lookup → TRAIL_PRESETS → pill math.
  2. HTTP endpoint ``/api/positions`` — monkey-patches ROOT and DATA to
     point into a tmp_path, writes a synthetic ``mt5_positions.json``,
     asserts the JSON shape end-to-end.

The HTTP layer also validates the defensive fallback: if attachment
throws (e.g. malformed structured.jsonl), the endpoint still returns
raw positions instead of 500-ing.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
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
    spec = importlib.util.spec_from_file_location("dashboard_server_b4", _SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_srv = _load_server()


def _write_structured(path: Path, rows: list[tuple[str, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for severity, row in rows:
            f.write(f"[{severity}] {json.dumps(row)}\n")


# ---------------------------------------------------------------------------
# Pure function: attach_trail_params_to_positions
# ---------------------------------------------------------------------------


class TestAttachTrailParams:
    """Unit tests for smc.monitor.dashboard_feeds.attach_trail_params_to_positions."""

    def test_trend_up_position_gets_030_050_preset(self, tmp_path: Path):
        from smc.monitor.dashboard_feeds import attach_trail_params_to_positions

        structured_log = tmp_path / "logs" / "structured.jsonl"
        _write_structured(structured_log, [
            ("INFO", {
                "ts": "2026-04-20T14:00:00+00:00",
                "event": "ai_regime_classified",
                "source": "ai_debate",
                "regime": "TREND_UP",
                "confidence": 0.82,
            }),
        ])
        positions = [{
            "ticket": 100001,
            "symbol": "XAUUSD",
            "magic": 19760418,
            "direction": "long",
            "lots": 0.10,
            "open_price": 2340.50,
            "open_time": "2026-04-20T14:05:00+00:00",
            "pnl_usd": 3.20,
        }]
        enriched = attach_trail_params_to_positions(
            positions, structured_log_path=structured_log,
        )
        assert len(enriched) == 1
        pos = enriched[0]
        assert pos["regime_at_open"] == "TREND_UP"
        assert pos["trail_activate_r"] == pytest.approx(0.3)
        assert pos["trail_distance_r"] == pytest.approx(0.5)
        # Original caller dict must not be mutated.
        assert "trail_activate_r" not in positions[0]

    def test_consolidation_position_gets_050_030_preset(self, tmp_path: Path):
        from smc.monitor.dashboard_feeds import attach_trail_params_to_positions

        structured_log = tmp_path / "logs" / "structured.jsonl"
        _write_structured(structured_log, [
            ("INFO", {
                "ts": "2026-04-20T10:00:00+00:00",
                "event": "ai_regime_classified",
                "source": "atr_fallback",
                "regime": "CONSOLIDATION",
            }),
        ])
        positions = [{
            "ticket": 100002,
            "open_time": "2026-04-20T10:30:00+00:00",
        }]
        enriched = attach_trail_params_to_positions(
            positions, structured_log_path=structured_log,
        )
        assert enriched[0]["trail_activate_r"] == pytest.approx(0.5)
        assert enriched[0]["trail_distance_r"] == pytest.approx(0.3)

    def test_ath_breakout_position_gets_080_070_preset(self, tmp_path: Path):
        from smc.monitor.dashboard_feeds import attach_trail_params_to_positions

        structured_log = tmp_path / "logs" / "structured.jsonl"
        _write_structured(structured_log, [
            ("INFO", {
                "ts": "2026-04-20T09:00:00+00:00",
                "event": "ai_regime_classified",
                "source": "ai_debate",
                "regime": "ATH_BREAKOUT",
            }),
        ])
        positions = [{"ticket": 1, "open_time": "2026-04-20T09:15:00+00:00"}]
        enriched = attach_trail_params_to_positions(
            positions, structured_log_path=structured_log,
        )
        assert enriched[0]["trail_activate_r"] == pytest.approx(0.8)
        assert enriched[0]["trail_distance_r"] == pytest.approx(0.7)

    def test_default_source_regime_is_skipped(self, tmp_path: Path):
        """A default-source (cold-start) regime must not drive the pill."""
        from smc.monitor.dashboard_feeds import attach_trail_params_to_positions

        structured_log = tmp_path / "logs" / "structured.jsonl"
        _write_structured(structured_log, [
            # A useful TREND_DOWN event earlier…
            ("INFO", {
                "ts": "2026-04-20T05:00:00+00:00",
                "event": "ai_regime_classified",
                "source": "ai_debate",
                "regime": "TREND_DOWN",
            }),
            # …then a noisy default row that the UI should ignore.
            ("INFO", {
                "ts": "2026-04-20T09:00:00+00:00",
                "event": "ai_regime_classified",
                "source": "default",
                "regime": "TRANSITION",
            }),
        ])
        positions = [{"ticket": 2, "open_time": "2026-04-20T09:15:00+00:00"}]
        enriched = attach_trail_params_to_positions(
            positions, structured_log_path=structured_log,
        )
        # Should pick TREND_DOWN (last non-default ≤ open_time), not TRANSITION.
        assert enriched[0]["regime_at_open"] == "TREND_DOWN"
        assert enriched[0]["trail_activate_r"] == pytest.approx(0.3)
        assert enriched[0]["trail_distance_r"] == pytest.approx(0.5)

    def test_events_after_open_time_are_ignored(self, tmp_path: Path):
        """Only regime events at or before open_time count."""
        from smc.monitor.dashboard_feeds import attach_trail_params_to_positions

        structured_log = tmp_path / "logs" / "structured.jsonl"
        _write_structured(structured_log, [
            # TREND_UP at T-30min
            ("INFO", {
                "ts": "2026-04-20T09:30:00+00:00",
                "event": "ai_regime_classified",
                "source": "ai_debate",
                "regime": "TREND_UP",
            }),
            # CONSOLIDATION at T+30min — must be ignored for a T-anchored position
            ("INFO", {
                "ts": "2026-04-20T10:30:00+00:00",
                "event": "ai_regime_classified",
                "source": "ai_debate",
                "regime": "CONSOLIDATION",
            }),
        ])
        positions = [{"ticket": 3, "open_time": "2026-04-20T10:00:00+00:00"}]
        enriched = attach_trail_params_to_positions(
            positions, structured_log_path=structured_log,
        )
        assert enriched[0]["regime_at_open"] == "TREND_UP"

    def test_no_matching_event_yields_null_pill(self, tmp_path: Path):
        """Empty log or all events after open_time → trail fields are None."""
        from smc.monitor.dashboard_feeds import attach_trail_params_to_positions

        structured_log = tmp_path / "logs" / "structured.jsonl"
        _write_structured(structured_log, [])
        positions = [{"ticket": 4, "open_time": "2026-04-20T12:00:00+00:00"}]
        enriched = attach_trail_params_to_positions(
            positions, structured_log_path=structured_log,
        )
        assert enriched[0]["trail_activate_r"] is None
        assert enriched[0]["trail_distance_r"] is None

    def test_missing_log_file_yields_null_pill(self, tmp_path: Path):
        from smc.monitor.dashboard_feeds import attach_trail_params_to_positions

        nonexistent = tmp_path / "logs" / "does-not-exist.jsonl"
        positions = [{"ticket": 5, "open_time": "2026-04-20T12:00:00+00:00"}]
        enriched = attach_trail_params_to_positions(
            positions, structured_log_path=nonexistent,
        )
        assert enriched[0]["trail_activate_r"] is None
        assert enriched[0]["trail_distance_r"] is None

    def test_unknown_regime_string_yields_null_pill(self, tmp_path: Path):
        """An unrecognised regime (e.g. legacy 'TRENDING') must not crash."""
        from smc.monitor.dashboard_feeds import attach_trail_params_to_positions

        structured_log = tmp_path / "logs" / "structured.jsonl"
        _write_structured(structured_log, [
            ("INFO", {
                "ts": "2026-04-20T14:00:00+00:00",
                "event": "ai_regime_classified",
                "source": "claude_debate",
                "regime": "TRENDING",  # legacy / unknown — not in TRAIL_PRESETS
            }),
        ])
        positions = [{"ticket": 6, "open_time": "2026-04-20T14:10:00+00:00"}]
        enriched = attach_trail_params_to_positions(
            positions, structured_log_path=structured_log,
        )
        assert enriched[0]["trail_activate_r"] is None
        assert enriched[0]["trail_distance_r"] is None
        # regime_at_open still reports the raw string — useful for debugging.
        assert enriched[0]["regime_at_open"] == "TRENDING"

    def test_missing_open_time_falls_back_to_now(self, tmp_path: Path):
        """Cold-start / legacy position without open_time must still resolve."""
        from smc.monitor.dashboard_feeds import attach_trail_params_to_positions

        structured_log = tmp_path / "logs" / "structured.jsonl"
        _write_structured(structured_log, [
            ("INFO", {
                "ts": "2026-04-20T14:00:00+00:00",
                "event": "ai_regime_classified",
                "source": "ai_debate",
                "regime": "TRANSITION",
            }),
        ])
        # Pin "now" to 2026-04-20 15:00Z so the event is in the past.
        now = datetime(2026, 4, 20, 15, 0, tzinfo=timezone.utc)
        positions = [{"ticket": 7}]  # no open_time
        enriched = attach_trail_params_to_positions(
            positions, structured_log_path=structured_log, now=now,
        )
        assert enriched[0]["regime_at_open"] == "TRANSITION"
        assert enriched[0]["trail_activate_r"] == pytest.approx(0.5)
        assert enriched[0]["trail_distance_r"] == pytest.approx(0.5)

    def test_immutability_returns_new_dicts(self, tmp_path: Path):
        """Caller's list + dicts must not be mutated in-place."""
        from smc.monitor.dashboard_feeds import attach_trail_params_to_positions

        structured_log = tmp_path / "logs" / "structured.jsonl"
        _write_structured(structured_log, [])
        positions = [{"ticket": 8, "open_time": "2026-04-20T10:00:00+00:00"}]
        before_snapshot = [dict(p) for p in positions]
        _ = attach_trail_params_to_positions(
            positions, structured_log_path=structured_log,
        )
        # Original dicts unchanged
        for p, snap in zip(positions, before_snapshot):
            assert p == snap


# ---------------------------------------------------------------------------
# HTTP endpoint: /api/positions
# ---------------------------------------------------------------------------


class TestPositionsEndpoint:
    """End-to-end tests for GET /api/positions with trail enrichment."""

    def _setup_tmp_root(self, tmp_path: Path, symbol: str = "XAUUSD"):
        symbol_dir = tmp_path / "data" / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir

    def test_positions_endpoint_enriches_with_trail_pill(self, tmp_path: Path):
        symbol_dir = self._setup_tmp_root(tmp_path)
        # Write a synthetic mt5_positions.json like live_demo would.
        (symbol_dir / "mt5_positions.json").write_text(json.dumps({
            "ts": "2026-04-20T14:10:00+00:00",
            "positions": [{
                "ticket": 500001,
                "symbol": "XAUUSD",
                "magic": 19760418,
                "direction": "long",
                "lots": 0.10,
                "open_price": 2345.20,
                "open_time": "2026-04-20T14:05:00+00:00",
                "pnl_usd": 1.80,
            }],
        }))
        # Write structured.jsonl with a TREND_DOWN at T-15m.
        _write_structured(tmp_path / "logs" / "structured.jsonl", [
            ("INFO", {
                "ts": "2026-04-20T13:50:00+00:00",
                "event": "ai_regime_classified",
                "source": "ai_debate",
                "regime": "TREND_DOWN",
            }),
        ])
        with patch.object(_srv, "ROOT", tmp_path), \
             patch.object(_srv, "DATA", tmp_path / "data"):
            client = TestClient(_srv.app)
            r = client.get("/api/positions?symbol=XAUUSD")
        assert r.status_code == 200
        body = r.json()
        assert body["ts"] == "2026-04-20T14:10:00+00:00"
        assert len(body["positions"]) == 1
        pos = body["positions"][0]
        assert pos["ticket"] == 500001
        assert pos["regime_at_open"] == "TREND_DOWN"
        assert pos["trail_activate_r"] == pytest.approx(0.3)
        assert pos["trail_distance_r"] == pytest.approx(0.5)

    def test_positions_endpoint_empty_when_no_positions_file(self, tmp_path: Path):
        self._setup_tmp_root(tmp_path)
        with patch.object(_srv, "ROOT", tmp_path), \
             patch.object(_srv, "DATA", tmp_path / "data"):
            client = TestClient(_srv.app)
            r = client.get("/api/positions?symbol=XAUUSD")
        assert r.status_code == 200
        assert r.json()["positions"] == []

    def test_positions_endpoint_null_trail_when_no_regime_event(self, tmp_path: Path):
        """With no structured.jsonl, the pill comes back null — still 200 OK."""
        symbol_dir = self._setup_tmp_root(tmp_path)
        (symbol_dir / "mt5_positions.json").write_text(json.dumps({
            "ts": "2026-04-20T14:10:00+00:00",
            "positions": [{
                "ticket": 500002,
                "symbol": "XAUUSD",
                "magic": 19760418,
                "direction": "long",
                "lots": 0.10,
                "open_price": 2345.00,
                "open_time": "2026-04-20T14:05:00+00:00",
                "pnl_usd": 0.0,
            }],
        }))
        # No structured.jsonl — attach should return None for pill fields.
        with patch.object(_srv, "ROOT", tmp_path), \
             patch.object(_srv, "DATA", tmp_path / "data"):
            client = TestClient(_srv.app)
            r = client.get("/api/positions?symbol=XAUUSD")
        assert r.status_code == 200
        pos = r.json()["positions"][0]
        assert pos["trail_activate_r"] is None
        assert pos["trail_distance_r"] is None

    def test_positions_endpoint_preserves_halt_field(self, tmp_path: Path):
        """The /api/positions halt snapshot is untouched by B4 enrichment."""
        symbol_dir = self._setup_tmp_root(tmp_path)
        (symbol_dir / "mt5_positions.json").write_text(json.dumps({
            "ts": "2026-04-20T14:10:00+00:00",
            "positions": [],
        }))
        (symbol_dir / "consec_loss_state.json").write_text(json.dumps({
            "tripped": True,
            "consec_losses": 3,
            "tripped_at": "2026-04-20T12:00:00+00:00",
        }))
        with patch.object(_srv, "ROOT", tmp_path), \
             patch.object(_srv, "DATA", tmp_path / "data"):
            client = TestClient(_srv.app)
            r = client.get("/api/positions?symbol=XAUUSD")
        body = r.json()
        assert body["halt"]["tripped"] is True
        assert body["halt"]["consec_losses"] == 3

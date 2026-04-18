"""Tests for Round 3 Sprint 2 guards snapshot builder.

Covers green / amber / red transitions for each guard + missing-file
fallback + drawdown soft-dep (live_state.drawdown_snapshot).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from smc.monitor.guards_snapshot import build_guards_snapshot


NOW_UTC = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)


def _root(tmp_path: Path, symbol: str = "XAUUSD") -> Path:
    r = tmp_path / "data" / symbol
    r.mkdir(parents=True)
    return r


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Defaults + missing-state fallback
# ---------------------------------------------------------------------------


class TestEmptyState:
    def test_all_defaults_return_green(self, tmp_path: Path):
        snap = build_guards_snapshot("XAUUSD", data_root=_root(tmp_path), now=NOW_UTC)
        assert snap["can_trade"] is True
        assert snap["consec_halt"]["status"] == "green"
        assert snap["phase1a_breaker"]["status"] == "green"
        assert snap["asian_range_quota"]["status"] == "green"
        assert snap["drawdown_guard"]["status"] == "green"
        assert snap["drawdown_guard"]["source"] == "unavailable"
        # Warnings list every missing source
        assert "consec_loss_state_missing" in snap["warnings"]
        assert "phase1a_breaker_state_missing" in snap["warnings"]
        assert "asian_range_quota_state_missing" in snap["warnings"]


# ---------------------------------------------------------------------------
# Consec halt: green / amber / red transitions
# ---------------------------------------------------------------------------


class TestConsecHalt:
    def test_zero_losses_green(self, tmp_path: Path):
        root = _root(tmp_path)
        _write_json(root / "consec_loss_state.json", {"consec_losses": 0, "tripped": False})
        snap = build_guards_snapshot("XAUUSD", data_root=root, now=NOW_UTC)
        assert snap["consec_halt"]["status"] == "green"
        assert snap["consec_halt"]["losses"] == 0
        assert snap["consec_halt"]["max_losses"] == 3

    def test_two_of_three_losses_amber(self, tmp_path: Path):
        root = _root(tmp_path)
        _write_json(root / "consec_loss_state.json", {"consec_losses": 2, "tripped": False})
        snap = build_guards_snapshot("XAUUSD", data_root=root, now=NOW_UTC)
        assert snap["consec_halt"]["status"] == "amber"

    def test_tripped_red(self, tmp_path: Path):
        root = _root(tmp_path)
        _write_json(root / "consec_loss_state.json", {
            "consec_losses": 3, "tripped": True,
            "tripped_at": "2026-04-18T10:00:00+00:00",
        })
        snap = build_guards_snapshot("XAUUSD", data_root=root, now=NOW_UTC)
        assert snap["consec_halt"]["status"] == "red"
        assert snap["consec_halt"]["tripped"] is True
        assert snap["can_trade"] is False

    def test_custom_limit_from_cfg(self, tmp_path: Path):
        """R4 integration: per-symbol consec_loss_limit override."""
        root = _root(tmp_path)
        _write_json(root / "consec_loss_state.json", {"consec_losses": 3, "tripped": False})
        # Higher limit means 3 losses is under the 2/3 of limit = amber
        snap = build_guards_snapshot("BTCUSD", data_root=root, consec_loss_limit=5, now=NOW_UTC)
        # 3/5 = 60% → below 66% amber threshold → green
        assert snap["consec_halt"]["status"] == "green"
        assert snap["consec_halt"]["max_losses"] == 5


# ---------------------------------------------------------------------------
# Phase1a breaker
# ---------------------------------------------------------------------------


class TestPhase1aBreaker:
    def test_fresh_state_green(self, tmp_path: Path):
        root = _root(tmp_path)
        _write_json(root / "phase1a_breaker_state.json", {"losses": 0, "pnl_usd": 0.0})
        snap = build_guards_snapshot("XAUUSD", data_root=root, now=NOW_UTC)
        assert snap["phase1a_breaker"]["status"] == "green"

    def test_tripped_by_loss_count(self, tmp_path: Path):
        root = _root(tmp_path)
        _write_json(root / "phase1a_breaker_state.json", {
            "losses": 3, "pnl_usd": -15.0, "tripped": True,
        })
        snap = build_guards_snapshot("XAUUSD", data_root=root, now=NOW_UTC)
        assert snap["phase1a_breaker"]["status"] == "red"

    def test_amber_when_loss_count_approaching(self, tmp_path: Path):
        root = _root(tmp_path)
        _write_json(root / "phase1a_breaker_state.json", {"losses": 2, "pnl_usd": -8.0, "tripped": False})
        snap = build_guards_snapshot("XAUUSD", data_root=root, now=NOW_UTC)
        # 2/3 losses = 66% → amber
        assert snap["phase1a_breaker"]["status"] == "amber"

    def test_amber_when_pnl_approaching(self, tmp_path: Path):
        root = _root(tmp_path)
        # 1 loss but pnl near the -$20 limit
        _write_json(root / "phase1a_breaker_state.json", {"losses": 1, "pnl_usd": -14.0, "tripped": False})
        snap = build_guards_snapshot("XAUUSD", data_root=root, now=NOW_UTC)
        # -14 / -20 = 70% → amber
        assert snap["phase1a_breaker"]["status"] == "amber"


# ---------------------------------------------------------------------------
# Asian range quota
# ---------------------------------------------------------------------------


class TestAsianQuota:
    def test_never_opened_green(self, tmp_path: Path):
        root = _root(tmp_path)
        snap = build_guards_snapshot("XAUUSD", data_root=root, now=NOW_UTC)
        assert snap["asian_range_quota"]["status"] == "green"
        assert snap["asian_range_quota"]["exhausted"] is False

    def test_opened_yesterday_green(self, tmp_path: Path):
        root = _root(tmp_path)
        _write_json(root / "asian_range_quota_state.json", {"last_open_date": "2026-04-17"})
        snap = build_guards_snapshot("XAUUSD", data_root=root, now=NOW_UTC)
        assert snap["asian_range_quota"]["status"] == "green"
        assert snap["asian_range_quota"]["exhausted"] is False

    def test_opened_today_red(self, tmp_path: Path):
        root = _root(tmp_path)
        _write_json(root / "asian_range_quota_state.json", {"last_open_date": "2026-04-18"})
        snap = build_guards_snapshot("XAUUSD", data_root=root, now=NOW_UTC)
        assert snap["asian_range_quota"]["status"] == "red"
        assert snap["asian_range_quota"]["exhausted"] is True
        # Note: asian_quota tripped doesn't block other symbols, but for THIS symbol's snapshot
        # we report can_trade=False because this is a per-symbol view.
        assert snap["can_trade"] is False


# ---------------------------------------------------------------------------
# Drawdown guard — live_state.drawdown_snapshot soft-dep
# ---------------------------------------------------------------------------


class TestDrawdownGuard:
    def test_live_state_missing_snapshot_returns_source_unavailable(self, tmp_path: Path):
        root = _root(tmp_path)
        _write_json(root / "live_state.json", {"cycle": 10, "timestamp": "2026-04-18T11:00:00+00:00"})
        snap = build_guards_snapshot("XAUUSD", data_root=root, now=NOW_UTC)
        assert snap["drawdown_guard"]["source"] == "unavailable"
        assert snap["drawdown_guard"]["status"] == "green"  # optimistic when we have no data

    def test_snapshot_green_when_all_clear(self, tmp_path: Path):
        root = _root(tmp_path)
        _write_json(root / "live_state.json", {
            "drawdown_snapshot": {
                "can_trade": True,
                "daily_pnl_usd": -5.0,
                "daily_loss_pct": 0.5,
                "peak_balance": 1000.0,
                "balance": 995.0,
                "total_drawdown_pct": 0.5,
            },
        })
        snap = build_guards_snapshot("XAUUSD", data_root=root, now=NOW_UTC)
        assert snap["drawdown_guard"]["status"] == "green"
        assert snap["drawdown_guard"]["source"] == "live_state"
        assert snap["drawdown_guard"]["daily_pnl_usd"] == -5.0

    def test_snapshot_amber_when_daily_loss_approaching(self, tmp_path: Path):
        root = _root(tmp_path)
        # 2.0 / 3.0 = 66.7% → amber
        _write_json(root / "live_state.json", {
            "drawdown_snapshot": {
                "can_trade": True,
                "daily_pnl_usd": -20.0,
                "daily_loss_pct": 2.0,
                "peak_balance": 1000.0,
                "balance": 980.0,
                "total_drawdown_pct": 2.0,
            },
        })
        snap = build_guards_snapshot("XAUUSD", data_root=root, now=NOW_UTC)
        assert snap["drawdown_guard"]["status"] == "amber"

    def test_snapshot_red_when_cannot_trade(self, tmp_path: Path):
        root = _root(tmp_path)
        _write_json(root / "live_state.json", {
            "drawdown_snapshot": {
                "can_trade": False,
                "daily_pnl_usd": -35.0,
                "daily_loss_pct": 3.5,
                "peak_balance": 1000.0,
                "balance": 965.0,
                "total_drawdown_pct": 3.5,
                "rejection_reason": "Daily loss limit breached",
            },
        })
        snap = build_guards_snapshot("XAUUSD", data_root=root, now=NOW_UTC)
        assert snap["drawdown_guard"]["status"] == "red"
        assert snap["drawdown_guard"]["rejection_reason"] == "Daily loss limit breached"
        assert snap["can_trade"] is False


# ---------------------------------------------------------------------------
# can_trade aggregation + corrupt state tolerance
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_any_red_flips_can_trade_to_false(self, tmp_path: Path):
        root = _root(tmp_path)
        _write_json(root / "consec_loss_state.json", {"consec_losses": 3, "tripped": True})
        # other guards untouched → still green
        snap = build_guards_snapshot("XAUUSD", data_root=root, now=NOW_UTC)
        assert snap["consec_halt"]["status"] == "red"
        assert snap["can_trade"] is False

    def test_amber_does_not_block_trading(self, tmp_path: Path):
        root = _root(tmp_path)
        _write_json(root / "consec_loss_state.json", {"consec_losses": 2, "tripped": False})
        snap = build_guards_snapshot("XAUUSD", data_root=root, now=NOW_UTC)
        assert snap["consec_halt"]["status"] == "amber"
        assert snap["can_trade"] is True  # amber is informational only

    def test_corrupt_state_file_tolerated(self, tmp_path: Path):
        root = _root(tmp_path)
        (root / "consec_loss_state.json").write_text("{not valid json")
        snap = build_guards_snapshot("XAUUSD", data_root=root, now=NOW_UTC)
        assert "consec_loss_state_unreadable" in snap["warnings"]
        # Defaults through → green
        assert snap["consec_halt"]["status"] == "green"

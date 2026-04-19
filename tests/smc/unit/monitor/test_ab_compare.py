"""Unit tests for scripts/ab_compare.py — Round 4 Alt-B W3.

Tests the pure-function math:
  - compute_stats returns correct PF, win rate, avg PnL
  - load_journal correctly filters HOLD rows and parses trade rows
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Add scripts/ to path so we can import ab_compare directly
_SCRIPTS_DIR = Path(__file__).parents[4] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from ab_compare import (
    TradeRow,
    LegStats,
    compute_stats,
    load_journal,
    _is_trade_action,
    _parse_pnl,
)


# ---------------------------------------------------------------------------
# _is_trade_action
# ---------------------------------------------------------------------------

class TestIsTradeAction:
    def test_hold_excluded(self) -> None:
        assert _is_trade_action("HOLD") is False
        assert _is_trade_action("hold") is False

    def test_buy_included(self) -> None:
        assert _is_trade_action("BUY") is True

    def test_range_buy_included(self) -> None:
        assert _is_trade_action("RANGE BUY") is True

    def test_empty_excluded(self) -> None:
        assert _is_trade_action("") is False


# ---------------------------------------------------------------------------
# _parse_pnl
# ---------------------------------------------------------------------------

class TestParsePnl:
    def test_result_field_win(self) -> None:
        pnl, is_win = _parse_pnl({"result": 50.0})
        assert pnl == pytest.approx(50.0)
        assert is_win is True

    def test_result_field_loss(self) -> None:
        pnl, is_win = _parse_pnl({"result": -30.0})
        assert pnl == pytest.approx(-30.0)
        assert is_win is False

    def test_rr_ratio_fallback_win(self) -> None:
        pnl, is_win = _parse_pnl({"rr_ratio": 2.5})
        assert pnl == pytest.approx(2.5)
        assert is_win is True

    def test_rr_ratio_zero_is_loss(self) -> None:
        pnl, is_win = _parse_pnl({"rr_ratio": 0.0})
        assert is_win is False

    def test_no_fields_is_loss(self) -> None:
        pnl, is_win = _parse_pnl({})
        assert is_win is False


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------

def _make_row(is_win: bool, pnl: float, ts: datetime | None = None) -> TradeRow:
    if ts is None:
        ts = datetime(2026, 4, 19, tzinfo=timezone.utc)
    return TradeRow(time=ts, action="BUY", direction="long", pnl_units=pnl, is_win=is_win)


class TestComputeStats:
    def test_empty_returns_zeros(self) -> None:
        stats = compute_stats([])
        assert stats.trade_count == 0
        assert stats.profit_factor == pytest.approx(0.0)
        assert stats.win_rate == pytest.approx(0.0)

    def test_all_wins(self) -> None:
        rows = [_make_row(True, 2.0), _make_row(True, 1.5), _make_row(True, 3.0)]
        stats = compute_stats(rows)
        assert stats.trade_count == 3
        assert stats.win_count == 3
        assert stats.loss_count == 0
        assert stats.win_rate == pytest.approx(1.0)
        assert stats.profit_factor == float("inf")
        assert stats.gross_wins == pytest.approx(6.5)
        assert stats.gross_losses == pytest.approx(0.0)

    def test_mixed_trades_profit_factor(self) -> None:
        # 2 wins of 2.0R each, 1 loss of -1.0R → PF = 4.0 / 1.0 = 4.0
        rows = [
            _make_row(True, 2.0),
            _make_row(True, 2.0),
            _make_row(False, -1.0),
        ]
        stats = compute_stats(rows)
        assert stats.trade_count == 3
        assert stats.win_count == 2
        assert stats.loss_count == 1
        assert stats.win_rate == pytest.approx(2 / 3)
        assert stats.profit_factor == pytest.approx(4.0)
        assert stats.avg_pnl == pytest.approx(1.0)

    def test_all_losses_pf_zero(self) -> None:
        rows = [_make_row(False, -1.0), _make_row(False, -2.0)]
        stats = compute_stats(rows)
        assert stats.profit_factor == pytest.approx(0.0)
        assert stats.win_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# load_journal
# ---------------------------------------------------------------------------

def _write_journal(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


class TestLoadJournal:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        rows = load_journal(tmp_path / "nonexistent.jsonl")
        assert rows == []

    def test_hold_rows_excluded(self, tmp_path: Path) -> None:
        path = tmp_path / "trades.jsonl"
        _write_journal(path, [
            {"time": "2026-04-19T10:00:00+00:00", "action": "HOLD", "rr_ratio": 0.0},
            {"time": "2026-04-19T11:00:00+00:00", "action": "BUY", "rr_ratio": 2.0},
        ])
        rows = load_journal(path)
        assert len(rows) == 1
        assert rows[0].action == "BUY"

    def test_range_buy_included(self, tmp_path: Path) -> None:
        path = tmp_path / "trades.jsonl"
        _write_journal(path, [
            {"time": "2026-04-19T12:00:00+00:00", "action": "RANGE BUY", "rr_ratio": 1.5},
        ])
        rows = load_journal(path)
        assert len(rows) == 1
        assert rows[0].is_win is True

    def test_cutoff_days_filters_old_trades(self, tmp_path: Path) -> None:
        path = tmp_path / "trades.jsonl"
        _write_journal(path, [
            # Very old trade — 100 days ago
            {"time": "2026-01-09T10:00:00+00:00", "action": "BUY", "rr_ratio": 2.0},
            # Recent trade
            {"time": "2026-04-19T10:00:00+00:00", "action": "BUY", "rr_ratio": 2.0},
        ])
        rows = load_journal(path, cutoff_days=7)
        # Only the recent trade should pass the 7-day filter
        assert len(rows) == 1

    def test_malformed_line_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "trades.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.write("{not valid json\n")
            fh.write(json.dumps({"time": "2026-04-19T10:00:00+00:00", "action": "BUY", "rr_ratio": 1.0}) + "\n")
        rows = load_journal(path)
        assert len(rows) == 1

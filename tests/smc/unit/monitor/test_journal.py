"""Tests for smc.monitor.journal.TradeJournal.

Covers:
- Append a single entry and read it back
- Multiple entries on the same day
- Daily summary computation
- Empty day returns zero summary
- Cross-day partitioning
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from smc.monitor.journal import TradeJournal
from smc.monitor.types import JournalEntry


@pytest.fixture()
def journal(tmp_path: Path) -> TradeJournal:
    """TradeJournal writing to a temporary directory."""
    return TradeJournal(journal_dir=tmp_path / "journal")


def _make_entry(
    *,
    ts: datetime | None = None,
    action: str = "close",
    ticket: int = 1001,
    direction: str = "long",
    lots: float = 0.01,
    price: float = 2350.0,
    sl: float = 2340.0,
    tp: float = 2370.0,
    pnl: float = 20.0,
    balance_after: float = 10_020.0,
    confluence: float = 0.65,
    trigger: str = "choch_in_zone",
    regime: str = "trending",
) -> JournalEntry:
    if ts is None:
        ts = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
    return JournalEntry(
        ts=ts,
        action=action,
        ticket=ticket,
        instrument="XAUUSD",
        direction=direction,
        lots=lots,
        price=price,
        sl=sl,
        tp=tp,
        pnl=pnl,
        balance_after=balance_after,
        setup_confluence=confluence,
        trigger_type=trigger,
        regime=regime,
    )


# ---------------------------------------------------------------------------
# Basic append and read
# ---------------------------------------------------------------------------


class TestAppendAndRead:
    def test_log_single_entry(self, journal: TradeJournal) -> None:
        entry = _make_entry()
        journal.log_action(entry)

        df = journal.read_day(date(2024, 6, 15))
        assert len(df) == 1
        assert df["ticket"][0] == 1001
        assert df["action"][0] == "close"

    def test_append_multiple_entries(self, journal: TradeJournal) -> None:
        for i in range(3):
            entry = _make_entry(ticket=1000 + i)
            journal.log_action(entry)

        df = journal.read_day(date(2024, 6, 15))
        assert len(df) == 3

    def test_empty_day_returns_empty_df(self, journal: TradeJournal) -> None:
        df = journal.read_day(date(2024, 1, 1))
        assert len(df) == 0
        assert "ts" in df.columns

    def test_creates_journal_dir(self, tmp_path: Path) -> None:
        j = TradeJournal(journal_dir=tmp_path / "new" / "journal")
        entry = _make_entry()
        j.log_action(entry)
        assert (tmp_path / "new" / "journal").exists()


# ---------------------------------------------------------------------------
# Cross-day partitioning
# ---------------------------------------------------------------------------


class TestCrossDayPartition:
    def test_different_days_different_files(self, journal: TradeJournal) -> None:
        entry_1 = _make_entry(ts=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc))
        entry_2 = _make_entry(ts=datetime(2024, 6, 16, 14, 0, tzinfo=timezone.utc))

        journal.log_action(entry_1)
        journal.log_action(entry_2)

        df_15 = journal.read_day(date(2024, 6, 15))
        df_16 = journal.read_day(date(2024, 6, 16))
        assert len(df_15) == 1
        assert len(df_16) == 1


# ---------------------------------------------------------------------------
# Cycle logging
# ---------------------------------------------------------------------------


class TestCycleLogging:
    def test_log_cycle(self, journal: TradeJournal) -> None:
        journal.log_cycle(
            bar_close_ts=datetime(2024, 6, 15, 10, 15, 0, tzinfo=timezone.utc),
            instrument="XAUUSD",
            regime="trending",
            balance=10_000.0,
        )
        df = journal.read_day(date(2024, 6, 15))
        assert len(df) == 1
        assert df["action"][0] == "cycle"
        assert df["direction"][0] == "none"


# ---------------------------------------------------------------------------
# Daily summary
# ---------------------------------------------------------------------------


class TestDailySummary:
    def test_empty_day_summary(self, journal: TradeJournal) -> None:
        summary = journal.daily_summary(date(2024, 1, 1))
        assert summary.total_trades == 0
        assert summary.win_rate == 0.0

    def test_mixed_trades_summary(self, journal: TradeJournal) -> None:
        ts_base = datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc)
        # Two winners, one loser
        journal.log_action(_make_entry(
            ts=ts_base,
            action="close",
            ticket=1001,
            pnl=50.0,
            balance_after=10_050.0,
        ))
        journal.log_action(_make_entry(
            ts=ts_base,
            action="tp_hit",
            ticket=1002,
            pnl=30.0,
            balance_after=10_080.0,
        ))
        journal.log_action(_make_entry(
            ts=ts_base,
            action="sl_hit",
            ticket=1003,
            pnl=-25.0,
            balance_after=10_055.0,
        ))

        summary = journal.daily_summary(date(2024, 6, 15))
        assert summary.total_trades == 3
        assert summary.winning_trades == 2
        assert summary.losing_trades == 1
        assert summary.gross_pnl == 80.0
        assert summary.net_pnl == 55.0
        assert summary.win_rate == pytest.approx(2 / 3, rel=1e-3)

    def test_cycle_entries_excluded_from_summary(self, journal: TradeJournal) -> None:
        ts_base = datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc)
        journal.log_cycle(
            bar_close_ts=ts_base,
            instrument="XAUUSD",
            regime="trending",
            balance=10_000.0,
        )
        journal.log_action(_make_entry(
            ts=ts_base,
            action="close",
            ticket=1001,
            pnl=50.0,
            balance_after=10_050.0,
        ))
        summary = journal.daily_summary(date(2024, 6, 15))
        assert summary.total_trades == 1  # Cycle excluded

    def test_all_losers_summary(self, journal: TradeJournal) -> None:
        ts_base = datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc)
        journal.log_action(_make_entry(
            ts=ts_base,
            action="sl_hit",
            ticket=1001,
            pnl=-30.0,
            balance_after=9_970.0,
        ))
        journal.log_action(_make_entry(
            ts=ts_base,
            action="sl_hit",
            ticket=1002,
            pnl=-20.0,
            balance_after=9_950.0,
        ))
        summary = journal.daily_summary(date(2024, 6, 15))
        assert summary.total_trades == 2
        assert summary.winning_trades == 0
        assert summary.losing_trades == 2
        assert summary.net_pnl == -50.0
        assert summary.win_rate == 0.0

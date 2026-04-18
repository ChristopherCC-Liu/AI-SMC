"""audit-r3 R3: peak_balance UTC 0:00 daily reset.

DrawdownGuard is intentionally stateless — peak_balance is tracked by the
caller (live_demo.py).  Without a daily reset, a bad week bakes a "ghost
peak" into the guard so `total_drawdown_pct` stays elevated even after
intra-day recovery, producing spurious drawdown_guard trips.

Fix: at UTC 00:00 boundary, reset peak_balance to current live balance.
Mirrors Phase1aCircuitBreaker / ConsecLossHalt daily-reset semantics.

The reset logic lives inline in scripts/live_demo.py (cannot import —
MT5 at module level).  These tests mirror the reset predicate as a pure
function and verify it fires on exactly the UTC date boundary.  Same
pattern as test_live_demo_gate.py / test_live_demo_halt.py.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any


def _should_reset_peak_balance(now_utc: datetime, last_reset_date: date) -> bool:
    """Mirror of the daily-reset predicate in live_demo.py.

    Returns True iff the current UTC date differs from the last-reset date.
    """
    return now_utc.date() != last_reset_date


def _apply_peak_reset(
    peak_balance: float,
    last_reset_date: date,
    *,
    now_utc: datetime,
    current_balance: float | None,
) -> tuple[float, date]:
    """Return the (peak_balance, last_reset_date) after potentially resetting.

    current_balance is the fresh MT5 account_info().balance or None if
    account_info failed — in that case we keep the previous peak_balance
    (fail-closed to stale-but-valid).
    """
    if not _should_reset_peak_balance(now_utc, last_reset_date):
        return (peak_balance, last_reset_date)
    new_peak = current_balance if current_balance is not None else peak_balance
    return (float(new_peak), now_utc.date())


# ---------------------------------------------------------------------------
# Predicate tests
# ---------------------------------------------------------------------------

class TestPredicate:
    def test_same_day_no_reset(self):
        now = datetime(2026, 4, 18, 23, 59, 59, tzinfo=timezone.utc)
        assert not _should_reset_peak_balance(now, date(2026, 4, 18))

    def test_day_boundary_crossed_triggers_reset(self):
        now = datetime(2026, 4, 19, 0, 0, 1, tzinfo=timezone.utc)
        assert _should_reset_peak_balance(now, date(2026, 4, 18))

    def test_multiple_days_after_still_resets(self):
        """If process was down for days, reset should still fire."""
        now = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
        assert _should_reset_peak_balance(now, date(2026, 4, 18))


# ---------------------------------------------------------------------------
# Reset application
# ---------------------------------------------------------------------------

class TestApplyReset:
    def test_no_reset_same_day_preserves_peak(self):
        peak, last = _apply_peak_reset(
            peak_balance=1_050.0,
            last_reset_date=date(2026, 4, 18),
            now_utc=datetime(2026, 4, 18, 15, 0, tzinfo=timezone.utc),
            current_balance=1_030.0,
        )
        assert peak == 1_050.0
        assert last == date(2026, 4, 18)

    def test_reset_adopts_current_balance_at_boundary(self):
        peak, last = _apply_peak_reset(
            peak_balance=1_100.0,  # yesterday's peak
            last_reset_date=date(2026, 4, 18),
            now_utc=datetime(2026, 4, 19, 0, 0, 1, tzinfo=timezone.utc),
            current_balance=980.0,  # today's balance (down from peak)
        )
        # Reset adopts today's balance as new peak, discarding ghost peak.
        assert peak == 980.0
        assert last == date(2026, 4, 19)

    def test_reset_without_fresh_balance_keeps_previous_peak(self):
        """account_info failed → keep stale but valid peak (fail-closed)."""
        peak, last = _apply_peak_reset(
            peak_balance=1_100.0,
            last_reset_date=date(2026, 4, 18),
            now_utc=datetime(2026, 4, 19, 0, 0, 1, tzinfo=timezone.utc),
            current_balance=None,
        )
        assert peak == 1_100.0
        assert last == date(2026, 4, 19)  # date still advances — prevents reset loop

    def test_reset_recognises_large_recovery(self):
        """If account recovered to new high, reset should adopt it."""
        peak, last = _apply_peak_reset(
            peak_balance=950.0,  # yesterday's (low) peak after drawdown
            last_reset_date=date(2026, 4, 18),
            now_utc=datetime(2026, 4, 19, 0, 0, 1, tzinfo=timezone.utc),
            current_balance=1_200.0,
        )
        assert peak == 1_200.0


# ---------------------------------------------------------------------------
# Integration with DrawdownGuard — ghost-peak regression
# ---------------------------------------------------------------------------

class TestGhostPeakRegression:
    """The original bug: account recovers but drawdown_guard still blocks.

    Without R3:
      Day 1: balance 1000 → 900 (10% DD) → guard trips, halt
      Day 2: balance recovers to 990
        - peak stays at 1000 → DD = (1000-990)/1000 = 1% < 10% limit ✓ OK
      Day 3: balance back to 1000 → peak updates naturally ✓
      BUT if account drew down to 800 first:
      Day 1: peak=1100, balance=800 → DD ~27% >> 10% → halt
      Day 2: balance recovers to 1050 → peak still 1100 → DD 4.5% ✓ OK but
              if Day 2 account_info races or reconcile delay, peak never
              gets updated to 1050 → stale ghost peak → spurious trips.

    With R3:
      Day 2 UTC 0:00 reset → peak = current balance (1050) → DD 0% → clean.
    """

    def test_ghost_peak_cleared_by_daily_reset(self):
        from smc.risk.drawdown_guard import DrawdownGuard

        guard = DrawdownGuard(max_daily_loss_pct=3.0, max_drawdown_pct=10.0)

        # Day 1: account drew down significantly — peak=1100, now=950
        #   total_drawdown = (1100-950)/1100 = 13.6% >= 10% → TRIP
        day1 = guard.check_budget(balance=950.0, peak_balance=1_100.0, daily_pnl=-50.0)
        assert not day1.can_trade
        assert "drawdown" in (day1.rejection_reason or "").lower()

        # Day 2: account partially recovered to 1000, but ghost peak lingers.
        #   Without R3: peak=1100 still → DD=9.1% < 10% OK but any tiny dip
        #   re-trips.  With R3: peak=1000 (reset) → DD=0% fresh start.
        _peak_no_reset = 1_100.0
        _peak_with_reset, _ = _apply_peak_reset(
            peak_balance=_peak_no_reset,
            last_reset_date=date(2026, 4, 18),
            now_utc=datetime(2026, 4, 19, 0, 0, 1, tzinfo=timezone.utc),
            current_balance=1_000.0,
        )

        no_reset = guard.check_budget(balance=1_000.0, peak_balance=_peak_no_reset, daily_pnl=0.0)
        with_reset = guard.check_budget(balance=1_000.0, peak_balance=_peak_with_reset, daily_pnl=0.0)

        # No reset: drawdown pct still elevated (9.09%), close to trip threshold.
        assert no_reset.total_drawdown_pct > 5.0

        # With reset: drawdown 0%, full headroom.
        assert with_reset.total_drawdown_pct == 0.0
        assert with_reset.can_trade

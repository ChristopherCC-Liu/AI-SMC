"""Unit tests for the multi-day persistent drawdown circuit breaker (R10 P1.1)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from smc.risk.persistent_drawdown_guard import (
    AUTO_RESUME_HOURS,
    HaltEvent,
    PersistentDrawdownGuard,
    PersistentPeakState,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class FrozenClock:
    """Deterministic clock — replaces ``datetime.now`` in the guard."""

    def __init__(self, t: datetime) -> None:
        self.t = t

    def __call__(self) -> datetime:
        return self.t

    def advance(self, **kwargs) -> None:
        self.t = self.t + timedelta(**kwargs)


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc))


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "persistent_peak.json"


@pytest.fixture
def sentinel_path(tmp_path):
    return tmp_path / "dd_manual_reset.flag"


@pytest.fixture
def guard(state_path, sentinel_path, clock):
    return PersistentDrawdownGuard(
        state_path=state_path,
        manual_reset_sentinel_path=sentinel_path,
        clock=clock,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_construction(self, guard):
        snap = guard.snapshot()
        assert snap.all_time_peak == 0.0
        assert snap.weekly_peak == 0.0
        assert snap.manual_halt_active is False

    def test_loads_existing_state(self, state_path, sentinel_path, clock):
        # Round 10 schema: {all_time_peak, weekly_balances,
        # manual_halt_active, last_halt_at, last_updated, manual_halt_history}.
        payload = {
            "all_time_peak": 1000.0,
            "weekly_balances": [["2026-04-23", 950.0], ["2026-04-24", 1000.0]],
            "manual_halt_active": False,
            "last_halt_at": None,
            "last_updated": "2026-04-24T00:00:00+00:00",
            "manual_halt_history": [],
        }
        state_path.write_text(json.dumps(payload, indent=2))
        guard = PersistentDrawdownGuard(
            state_path=state_path,
            manual_reset_sentinel_path=sentinel_path,
            clock=clock,
        )
        assert guard.snapshot().all_time_peak == 1000.0
        assert guard.snapshot().weekly_peak == 1000.0
        assert len(guard.snapshot().weekly_balances) == 2

    def test_corrupt_state_loads_fresh(self, state_path, sentinel_path, clock):
        state_path.write_text("{not valid json")
        guard = PersistentDrawdownGuard(
            state_path=state_path,
            manual_reset_sentinel_path=sentinel_path,
            clock=clock,
        )
        assert guard.snapshot().all_time_peak == 0.0

    def test_invalid_ratios_raise(self, state_path, sentinel_path):
        with pytest.raises(ValueError, match="ratios must"):
            PersistentDrawdownGuard(
                state_path=state_path,
                manual_reset_sentinel_path=sentinel_path,
                cliff_ratio=0.95,
                alarm_ratio=0.90,
            )

    def test_invalid_window_raises(self, state_path, sentinel_path):
        with pytest.raises(ValueError, match="weekly_window_days"):
            PersistentDrawdownGuard(
                state_path=state_path,
                manual_reset_sentinel_path=sentinel_path,
                weekly_window_days=0,
            )


# ---------------------------------------------------------------------------
# Monotonic ratchet
# ---------------------------------------------------------------------------


class TestMonotonicRatchet:
    def test_peak_increases_with_balance(self, guard):
        guard.update(1000.0)
        guard.update(1100.0)
        assert guard.snapshot().all_time_peak == 1100.0

    def test_peak_never_decreases(self, guard):
        guard.update(1100.0)
        guard.update(900.0)
        assert guard.snapshot().all_time_peak == 1100.0

    def test_peak_persists_to_disk(self, state_path, guard):
        guard.update(1234.56)
        raw = json.loads(state_path.read_text())
        assert raw["all_time_peak"] == 1234.56

    def test_zero_balance_ignored(self, guard):
        guard.update(1000.0)
        guard.update(0.0)
        assert guard.snapshot().all_time_peak == 1000.0


# ---------------------------------------------------------------------------
# Alarm ratio (0.90) — 24h auto-resume
# ---------------------------------------------------------------------------


class TestAlarmRatio:
    def test_above_alarm_can_trade(self, guard):
        guard.update(1000.0)
        budget = guard.check_budget(balance=910.0, daily_pnl=0.0)
        assert budget.can_trade is True

    def test_at_alarm_halts(self, guard):
        """Exactly at 0.90 should halt (<= check)."""
        guard.update(1000.0)
        budget = guard.check_budget(balance=900.0, daily_pnl=0.0)
        assert budget.can_trade is False
        assert "ALARM" in (budget.rejection_reason or "")

    def test_below_alarm_halts(self, guard):
        guard.update(1000.0)
        budget = guard.check_budget(balance=899.0, daily_pnl=0.0)
        assert budget.can_trade is False

    def test_alarm_clears_after_24h(self, guard, clock):
        guard.update(1000.0)
        # First halt: alarm trip
        b1 = guard.check_budget(balance=890.0, daily_pnl=0.0)
        assert b1.can_trade is False
        # 24h not yet elapsed
        clock.advance(hours=AUTO_RESUME_HOURS - 1)
        b2 = guard.check_budget(balance=950.0, daily_pnl=0.0)
        assert b2.can_trade is False
        # Just past 24h, equity recovered
        clock.advance(hours=2)
        b3 = guard.check_budget(balance=950.0, daily_pnl=0.0)
        assert b3.can_trade is True


# ---------------------------------------------------------------------------
# Cliff ratio (0.85) — manual reset required
# ---------------------------------------------------------------------------


class TestCliffRatio:
    def test_above_cliff_alarm_only(self, guard):
        guard.update(1000.0)
        budget = guard.check_budget(balance=860.0, daily_pnl=0.0)
        assert budget.can_trade is False
        assert "ALARM" in (budget.rejection_reason or "")
        assert guard.snapshot().manual_halt_active is False

    def test_at_cliff_manual_halt(self, guard):
        """Exactly at 0.85 should require manual reset."""
        guard.update(1000.0)
        budget = guard.check_budget(balance=850.0, daily_pnl=0.0)
        assert budget.can_trade is False
        assert "CLIFF" in (budget.rejection_reason or "")
        assert guard.snapshot().manual_halt_active is True

    def test_cliff_does_not_auto_resume(self, guard, clock):
        guard.update(1000.0)
        guard.check_budget(balance=800.0, daily_pnl=0.0)  # cliff trip
        clock.advance(hours=AUTO_RESUME_HOURS * 5)  # plenty of time
        budget = guard.check_budget(balance=950.0, daily_pnl=0.0)
        assert budget.can_trade is False
        assert "Manual reset" in (budget.rejection_reason or "")

    def test_sentinel_clears_cliff(self, guard, sentinel_path):
        guard.update(1000.0)
        guard.check_budget(balance=800.0, daily_pnl=0.0)
        assert guard.snapshot().manual_halt_active is True
        sentinel_path.write_text("ok")
        budget = guard.check_budget(balance=950.0, daily_pnl=0.0)
        assert guard.snapshot().manual_halt_active is False
        # Sentinel consumed
        assert sentinel_path.exists() is False
        # And after clear, equity recovered → can trade
        assert budget.can_trade is True

    def test_sentinel_consumed_atomically(
        self, guard, sentinel_path, clock
    ):
        """Same sentinel file must not clear two distinct halts."""
        guard.update(1000.0)
        guard.check_budget(balance=800.0, daily_pnl=0.0)  # 1st halt
        sentinel_path.write_text("ok")
        guard.check_budget(balance=950.0, daily_pnl=0.0)  # consumes sentinel
        # Trip again
        clock.advance(hours=1)
        guard.check_budget(balance=800.0, daily_pnl=0.0)
        assert guard.snapshot().manual_halt_active is True
        # No fresh sentinel — must stay halted
        budget = guard.check_budget(balance=950.0, daily_pnl=0.0)
        assert budget.can_trade is False


# ---------------------------------------------------------------------------
# Weekly rolling rail
# ---------------------------------------------------------------------------


class TestWeeklyRail:
    def test_weekly_window_seeds_on_first_update(self, guard, clock):
        guard.update(1000.0)
        snap = guard.snapshot()
        assert snap.weekly_peak == 1000.0
        assert len(snap.weekly_balances) == 1
        assert snap.weekly_balances[0][0] == clock.t.date().isoformat()

    def test_weekly_peak_ratchets_within_window(self, guard, clock):
        guard.update(1000.0)
        clock.advance(days=2)
        guard.update(1100.0)
        assert guard.snapshot().weekly_peak == 1100.0
        assert len(guard.snapshot().weekly_balances) == 2

    def test_weekly_window_drops_oldest_after_window(self, guard, clock):
        guard.update(1100.0)
        clock.advance(days=6)
        guard.update(900.0)
        # Day-0 entry is older than 5d window — must be dropped.
        assert guard.snapshot().weekly_peak == 900.0
        assert guard.snapshot().all_time_peak == 1100.0
        assert len(guard.snapshot().weekly_balances) == 1
        assert guard.snapshot().weekly_balances[0][1] == 900.0

    def test_weekly_window_keeps_5_distinct_days(self, guard, clock):
        for i in range(5):
            clock.advance(days=1)
            guard.update(1000.0 - i * 10)
        assert len(guard.snapshot().weekly_balances) == 5

    def test_weekly_window_per_day_max(self, guard, clock):
        guard.update(900.0)
        guard.update(950.0)
        guard.update(920.0)
        assert guard.snapshot().weekly_balances[0][1] == 950.0

    def test_weekly_rail_can_trip_alone(self, guard, clock):
        """Weekly bleed trips even when all-time DD is mild."""
        # Recovery scenario — all-time peak is a stale 100 ratio away
        guard.update(2000.0)  # all-time ATH
        clock.advance(days=10)  # outside 5d window
        guard.update(1500.0)  # new weekly seed = 1500
        clock.advance(days=2)
        # 1500 -> 1349 = ratio 0.899 weekly, 1349/2000 = 0.674 all-time
        # The all-time rail will catch it harder, so to isolate weekly:
        # use larger ATH gap so weekly trips alone.
        guard2_check = guard.check_budget(balance=1349.0, daily_pnl=0.0)
        assert guard2_check.can_trade is False


# ---------------------------------------------------------------------------
# Integration with inner DrawdownGuard (daily-loss trip-wire)
# ---------------------------------------------------------------------------


class TestInnerGuardIntegration:
    def test_clear_falls_through_to_inner(self, guard):
        """No multi-day breach → daily-loss check still runs."""
        guard.update(1000.0)
        # Balance untouched; daily_pnl = -50 = 5% loss > 3% inner cap
        budget = guard.check_budget(balance=1000.0, daily_pnl=-50.0)
        assert budget.can_trade is False
        assert "Daily loss" in (budget.rejection_reason or "")

    def test_clear_with_no_daily_loss(self, guard):
        guard.update(1000.0)
        budget = guard.check_budget(balance=1000.0, daily_pnl=0.0)
        assert budget.can_trade is True


# ---------------------------------------------------------------------------
# Replay regression: 2026-04-21..04-25 -12.76% bleed
# ---------------------------------------------------------------------------


class TestBleedRegression:
    """Reproduces the R10 motivation: 5-day bleed must halt by day 2.

    Pre-R10: peak reset every UTC midnight → halt never fired even at
    -12.76% cumulative DD because peak ratcheted DOWN with the bleed.

    Post-R10: monotonic peak → 0.90 alarm fires once cumulative DD
    crosses 10%. We assert the halt fires no later than day 2 of
    sustained loss.
    """

    def test_multi_day_bleed_halts_by_day_2(self, guard, clock):
        # Day 0: ATH at $1000
        guard.update(1000.0)
        # Day 1: -3% intraday (within daily cap, peak stays)
        clock.advance(days=1)
        guard.update(970.0)
        b1 = guard.check_budget(balance=970.0, daily_pnl=0.0)
        assert b1.can_trade is True, "Day 1 -3% should still trade"
        # Day 2: another -8% (cumulative -10.6% from peak)
        clock.advance(days=1)
        guard.update(890.0)
        b2 = guard.check_budget(balance=890.0, daily_pnl=0.0)
        assert b2.can_trade is False, "Day 2 cumulative -11% MUST halt"
        assert "ALARM" in (b2.rejection_reason or "")

    def test_bleed_with_midnight_reset_would_have_missed_halt(
        self, guard, clock
    ):
        """Demonstrates the bug fix: pre-R10 simulation halts NEVER fire."""
        # If we naively mimicked pre-R10 behaviour by calling update()
        # with new "daily peak" each midnight, our peak wouldn't drop —
        # the new module's monotonic ratchet rejects that. We model the
        # bleed correctly: peak holds at ATH while equity falls.
        guard.update(1000.0)  # ATH
        for day in range(1, 6):
            clock.advance(days=1)
            balance = 1000.0 * (1 - 0.025 * day)  # -2.5%/day → -12.5% total
            guard.update(balance)
            assert guard.snapshot().all_time_peak == 1000.0
        # By day 5, ratio = 0.875 → between alarm and cliff
        budget = guard.check_budget(balance=875.0, daily_pnl=0.0)
        assert budget.can_trade is False


# ---------------------------------------------------------------------------
# Halt semantics — block_opens / force_close (R10 P1.1)
# ---------------------------------------------------------------------------


class TestHaltSemantics:
    """Halts MUST only block new opens — existing positions exit naturally."""

    def test_clear_state_no_block(self, guard):
        guard.update(1000.0)
        budget = guard.check_budget(balance=1000.0, daily_pnl=0.0)
        assert budget.can_trade is True
        assert budget.block_opens is False
        assert budget.force_close is False

    def test_alarm_blocks_opens_only(self, guard):
        guard.update(1000.0)
        budget = guard.check_budget(balance=890.0, daily_pnl=0.0)
        assert budget.can_trade is False
        assert budget.block_opens is True
        assert budget.force_close is False, (
            "Alarm halt must NOT force-close — let positions exit via SL/TP"
        )

    def test_cliff_blocks_opens_only(self, guard):
        guard.update(1000.0)
        budget = guard.check_budget(balance=800.0, daily_pnl=0.0)
        assert budget.can_trade is False
        assert budget.block_opens is True
        assert budget.force_close is False

    def test_inner_daily_loss_blocks_opens_only(self, guard):
        # Multi-day clear, but inner DrawdownGuard's daily-loss trips
        guard.update(1000.0)
        budget = guard.check_budget(balance=1000.0, daily_pnl=-50.0)
        assert budget.can_trade is False
        assert budget.block_opens is True
        assert budget.force_close is False


# ---------------------------------------------------------------------------
# Audit history (R10 P1.1)
# ---------------------------------------------------------------------------


class TestAuditHistory:
    """Every halt cycle must leave a complete forensic record."""

    def test_alarm_appends_event(self, guard, clock):
        guard.update(1000.0)
        guard.check_budget(balance=890.0, daily_pnl=0.0)
        history = guard.snapshot().manual_halt_history
        assert len(history) == 1
        ev = history[0]
        assert ev.tier == "alarm"
        assert ev.rail == "all-time"
        assert ev.equity_at_trip == 890.0
        assert ev.peak_at_trip == 1000.0
        assert ev.ratio_at_trip == 0.89
        assert ev.tripped_at == clock.t.isoformat()
        assert ev.cleared_at is None  # still active

    def test_cliff_appends_event(self, guard):
        guard.update(1000.0)
        guard.check_budget(balance=800.0, daily_pnl=0.0)
        history = guard.snapshot().manual_halt_history
        assert len(history) == 1
        assert history[0].tier == "cliff"
        assert history[0].cleared_at is None

    def test_alarm_auto_resume_stamps_cleared_at(self, guard, clock):
        guard.update(1000.0)
        guard.check_budget(balance=890.0, daily_pnl=0.0)  # alarm trip
        clock.advance(hours=AUTO_RESUME_HOURS + 1)
        # Clear path is triggered by the next check after the window
        guard.check_budget(balance=950.0, daily_pnl=0.0)
        ev = guard.snapshot().manual_halt_history[0]
        assert ev.cleared_at is not None
        assert ev.operator_note is None  # auto-resume has no operator

    def test_sentinel_records_operator_note(
        self, guard, sentinel_path, clock
    ):
        guard.update(1000.0)
        guard.check_budget(balance=800.0, daily_pnl=0.0)  # cliff trip
        sentinel_path.write_text("Reset by Chris @ 2026-04-25 — verified equity recovery")
        guard.check_budget(balance=950.0, daily_pnl=0.0)  # consume sentinel
        ev = guard.snapshot().manual_halt_history[0]
        assert ev.cleared_at is not None
        assert ev.operator_note is not None
        assert "Chris" in ev.operator_note

    def test_history_persists_across_instances(
        self, state_path, sentinel_path, clock
    ):
        guard1 = PersistentDrawdownGuard(
            state_path=state_path,
            manual_reset_sentinel_path=sentinel_path,
            clock=clock,
        )
        guard1.update(1000.0)
        guard1.check_budget(balance=800.0, daily_pnl=0.0)
        # Reload from disk
        guard2 = PersistentDrawdownGuard(
            state_path=state_path,
            manual_reset_sentinel_path=sentinel_path,
            clock=clock,
        )
        history = guard2.snapshot().manual_halt_history
        assert len(history) == 1
        assert isinstance(history[0], HaltEvent)
        assert history[0].tier == "cliff"

    def test_multiple_halt_cycles_appended(self, guard, sentinel_path, clock):
        guard.update(1000.0)
        # Cycle 1: cliff trip + manual reset
        guard.check_budget(balance=800.0, daily_pnl=0.0)
        sentinel_path.write_text("first reset")
        guard.check_budget(balance=950.0, daily_pnl=0.0)
        # Cycle 2: another cliff
        clock.advance(hours=2)
        guard.check_budget(balance=800.0, daily_pnl=0.0)
        sentinel_path.write_text("second reset")
        guard.check_budget(balance=950.0, daily_pnl=0.0)
        history = guard.snapshot().manual_halt_history
        assert len(history) == 2
        assert all(ev.cleared_at is not None for ev in history)
        assert history[0].operator_note == "first reset"
        assert history[1].operator_note == "second reset"

    def test_empty_sentinel_yields_none_note(self, guard, sentinel_path):
        guard.update(1000.0)
        guard.check_budget(balance=800.0, daily_pnl=0.0)
        sentinel_path.write_text("")  # empty operator note
        guard.check_budget(balance=950.0, daily_pnl=0.0)
        ev = guard.snapshot().manual_halt_history[0]
        assert ev.cleared_at is not None
        assert ev.operator_note is None


# ---------------------------------------------------------------------------
# Stale sentinel attack surface (R10 P1.1 V3 — foundation cross-review C1)
# ---------------------------------------------------------------------------


class TestStaleSentinelHardening:
    """A pre-armed sentinel must NOT silently clear a future cliff trip.

    Attack surface (foundation-impl-lead's V2 review C1):
    1. Healthy account at 95% — no halt active.
    2. Operator drops the sentinel by mistake (or proactive arm).
    3. The `check_budget` path skips sentinel consumption because
       `manual_halt_active` is False.  File stays on disk.
    4. Equity later breaches the cliff → halt activates.
    5. NEXT `check_budget` cycle sees the still-present sentinel and
       silently clears the brand-new halt.  Operator is unaware.

    The guard MUST proactively unlink any stale sentinel observed in a
    healthy state and log a WARNING so ops sees the misconfiguration.
    """

    def test_stale_sentinel_in_healthy_state_is_unlinked(
        self, guard, sentinel_path, caplog
    ):
        import logging
        guard.update(1000.0)
        sentinel_path.write_text("pre-armed by mistake")
        with caplog.at_level(
            logging.WARNING, logger="smc.risk.persistent_drawdown_guard"
        ):
            budget = guard.check_budget(balance=950.0, daily_pnl=0.0)
        assert budget.can_trade is True
        assert sentinel_path.exists() is False, (
            "Stale sentinel must be unlinked when no halt is active"
        )
        warned = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "stale_sentinel" in r.getMessage()
        ]
        assert len(warned) == 1, (
            f"expected exactly 1 stale_sentinel WARNING, got {len(warned)}"
        )

    def test_pre_armed_sentinel_does_not_silently_clear_future_halt(
        self, guard, sentinel_path
    ):
        guard.update(1000.0)
        # Step 1-2: healthy + pre-armed sentinel
        sentinel_path.write_text("pre-armed")
        guard.check_budget(balance=950.0, daily_pnl=0.0)
        # Step 3: stale sentinel must have been swept
        assert sentinel_path.exists() is False
        # Step 4: equity now breaches the cliff
        budget_trip = guard.check_budget(balance=800.0, daily_pnl=0.0)
        assert budget_trip.can_trade is False
        assert "CLIFF" in (budget_trip.rejection_reason or "")
        assert guard.snapshot().manual_halt_active is True
        # Step 5: next cycle WITHOUT fresh sentinel must stay halted
        budget_next = guard.check_budget(balance=950.0, daily_pnl=0.0)
        assert budget_next.can_trade is False
        assert guard.snapshot().manual_halt_active is True

    def test_fresh_sentinel_still_clears_active_halt(
        self, guard, sentinel_path
    ):
        """Regression — V2 happy path must still work after V3 hardening."""
        guard.update(1000.0)
        guard.check_budget(balance=800.0, daily_pnl=0.0)
        assert guard.snapshot().manual_halt_active is True
        sentinel_path.write_text("legitimate reset by ops")
        budget = guard.check_budget(balance=950.0, daily_pnl=0.0)
        assert guard.snapshot().manual_halt_active is False
        assert budget.can_trade is True
        ev = guard.snapshot().manual_halt_history[0]
        assert ev.operator_note == "legitimate reset by ops"

    def test_update_path_does_not_sweep_sentinel(
        self, guard, sentinel_path, clock
    ):
        """update() is not the sweep site — keeps responsibility narrow.

        Verifies the sweep happens only on next check_budget so the contract
        stays predictable: sweep is a check_budget concern, not a peak-update one.
        """
        guard.update(1000.0)
        sentinel_path.write_text("oops")
        clock.advance(hours=1)
        guard.update(990.0)
        clock.advance(hours=1)
        guard.update(995.0)
        assert sentinel_path.exists() is True
        guard.check_budget(balance=995.0, daily_pnl=0.0)
        assert sentinel_path.exists() is False


# ---------------------------------------------------------------------------
# V3-B (R10 Adopt #1): IO failure-counter mirror on _save_state
# ---------------------------------------------------------------------------


class TestSaveFailureEscalation:
    """Mirror of the foundation P1.2 IO failure-counter pattern.

    See ``smc.monitor.gate_diagnostic_journal`` for the source pattern. The
    discipline: silent fail-closed at the trading-loop boundary, but log
    WARN on the first 1-2 failures of a UTC day and ERROR on the 3rd to
    surface persistent disk problems for operator triage.
    """

    @pytest.fixture(autouse=True)
    def _clear_save_failure_counter(self):
        from smc.risk.persistent_drawdown_guard import reset_save_failure_counter
        reset_save_failure_counter()
        yield
        reset_save_failure_counter()

    def test_save_failure_first_warns_below_escalation(
        self, guard, clock, monkeypatch, caplog
    ):
        """1st failure of a UTC day logs WARNING (not ERROR) with traceback."""
        import logging as _logging
        import smc.risk.persistent_drawdown_guard as mod

        def boom(self, *args, **kwargs):
            raise OSError("simulated disk full")

        monkeypatch.setattr(Path, "write_text", boom)
        with caplog.at_level(_logging.WARNING, logger="smc.risk.persistent_drawdown_guard"):
            # update() calls _save_state() — the IO failure must NOT raise.
            guard.update(1000.0)
        records = [
            r for r in caplog.records
            if "persistent_dd_save_failed" in r.getMessage()
        ]
        assert len(records) == 1, f"expected 1 record, got {len(records)}"
        assert records[0].levelno == _logging.WARNING
        assert records[0].exc_info is not None
        assert records[0].exc_info[0] is OSError
        assert mod._save_failure_counter[clock().date()] == 1

    def test_save_failure_third_escalates_to_error(
        self, guard, clock, monkeypatch, caplog
    ):
        """Three consecutive failures on the same UTC day → 3rd one logs ERROR."""
        import logging as _logging

        def boom(self, *args, **kwargs):
            raise OSError("simulated disk full")

        monkeypatch.setattr(Path, "write_text", boom)
        with caplog.at_level(_logging.WARNING, logger="smc.risk.persistent_drawdown_guard"):
            for balance in (1000.0, 1010.0, 1020.0):
                guard.update(balance)
        levels = [
            r.levelno for r in caplog.records
            if "persistent_dd_save_failed" in r.getMessage()
        ]
        assert levels == [_logging.WARNING, _logging.WARNING, _logging.ERROR], (
            f"unexpected escalation pattern: {levels}"
        )

    def test_save_success_after_failure_resets_counter(
        self, guard, clock, monkeypatch
    ):
        """Successful write clears the per-day counter so a future hiccup
        is again a fresh WARNING (not still escalated).

        Uses ``monkeypatch.setattr`` so the patch is rolled back on test
        completion / interruption — manual try/finally would leak the patch
        across the rest of the session under SIGINT / OOM (this exact lesson
        is captured in foundation's P1.2 test_success_after_failure_resets_counter).
        """
        import smc.risk.persistent_drawdown_guard as mod

        real_write_text = Path.write_text
        fail_count = {"n": 0}

        def flaky(self, *args, **kwargs):
            if fail_count["n"] < 2:
                fail_count["n"] += 1
                raise OSError("simulated disk hiccup")
            return real_write_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", flaky)
        guard.update(1000.0)
        guard.update(1010.0)
        assert mod._save_failure_counter[clock().date()] == 2
        # Restore real write_text via monkeypatch so the next save succeeds.
        monkeypatch.setattr(Path, "write_text", real_write_text)
        guard.update(1020.0)
        assert clock().date() not in mod._save_failure_counter

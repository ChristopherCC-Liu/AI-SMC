import json
from datetime import date, datetime, timedelta, timezone

import pytest

from smc.strategy.phase1a_circuit_breaker import Phase1aCircuitBreaker, BreakerState


class TestPhase1aCircuitBreaker:
    def test_initial_not_tripped(self, tmp_path):
        b = Phase1aCircuitBreaker(state_path=tmp_path / "state.json")
        assert not b.is_tripped()

    def test_two_losses_not_trip(self, tmp_path):
        b = Phase1aCircuitBreaker(state_path=tmp_path / "state.json")
        b.record_trade_close(-5.0)
        b.record_trade_close(-3.0)
        # 2 losses, pnl=-8 — neither limit crossed
        assert not b.is_tripped()

    def test_three_losses_trips(self, tmp_path):
        b = Phase1aCircuitBreaker(state_path=tmp_path / "state.json")
        b.record_trade_close(-5.0)
        b.record_trade_close(-3.0)
        b.record_trade_close(-2.0)
        # 3 losses triggers trip even though pnl=-10 (above -20)
        assert b.is_tripped()

    def test_pnl_limit_trips(self, tmp_path):
        b = Phase1aCircuitBreaker(state_path=tmp_path / "state.json")
        b.record_trade_close(-15.0)
        b.record_trade_close(-6.0)
        # pnl=-21 crosses limit even with only 2 losses
        assert b.is_tripped()

    def test_wins_offset_pnl(self, tmp_path):
        b = Phase1aCircuitBreaker(state_path=tmp_path / "state.json")
        b.record_trade_close(-10.0)
        b.record_trade_close(15.0)
        b.record_trade_close(-5.0)
        # pnl=0, 2 losses, no trip
        assert not b.is_tripped()

    def test_state_persists_across_instances(self, tmp_path):
        path = tmp_path / "state.json"
        b1 = Phase1aCircuitBreaker(state_path=path)
        b1.record_trade_close(-10.0)
        b1.record_trade_close(-12.0)
        # pnl=-22 trips
        assert b1.is_tripped()

        b2 = Phase1aCircuitBreaker(state_path=path)
        assert b2.is_tripped()  # loaded from disk

    def test_reset_restores_fresh_state(self, tmp_path):
        b = Phase1aCircuitBreaker(state_path=tmp_path / "state.json")
        b.record_trade_close(-25.0)
        assert b.is_tripped()
        b.reset()
        assert not b.is_tripped()
        assert b.snapshot().losses == 0
        assert b.snapshot().pnl_usd == 0.0

    def test_tripped_breaker_ignores_new_closes(self, tmp_path):
        b = Phase1aCircuitBreaker(state_path=tmp_path / "state.json")
        b.record_trade_close(-25.0)
        assert b.is_tripped()
        # Any subsequent record is no-op
        b.record_trade_close(-100.0)
        state = b.snapshot()
        assert state.pnl_usd == -25.0  # unchanged

    def test_tripped_at_is_set(self, tmp_path):
        b = Phase1aCircuitBreaker(state_path=tmp_path / "state.json")
        b.record_trade_close(-25.0)
        state = b.snapshot()
        assert state.tripped_at is not None
        # Should be ISO 8601
        datetime.fromisoformat(state.tripped_at)


class TestDailyReset:
    """Round 4.5 — breaker auto-resets at UTC 00:00 rollover."""

    def test_stale_tripped_state_resets_on_new_day(self, tmp_path):
        path = tmp_path / "state.json"
        yesterday = (datetime.now(tz=timezone.utc).date() - timedelta(days=1)).isoformat()
        # Simulate yesterday's tripped state persisted to disk
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "losses": 3,
                    "pnl_usd": -25.0,
                    "tripped": True,
                    "tripped_at": f"{yesterday}T23:00:00+00:00",
                    "last_updated": f"{yesterday}T23:00:00+00:00",
                    "last_reset_date": yesterday,
                }
            )
        )
        b = Phase1aCircuitBreaker(state_path=path)
        # is_tripped() triggers the daily reset check
        assert not b.is_tripped()
        state = b.snapshot()
        assert state.losses == 0
        assert state.pnl_usd == 0.0
        assert state.tripped_at is None
        assert state.last_reset_date == datetime.now(tz=timezone.utc).date().isoformat()

    def test_same_day_tripped_state_preserved(self, tmp_path):
        path = tmp_path / "state.json"
        b1 = Phase1aCircuitBreaker(state_path=path)
        b1.record_trade_close(-25.0)
        assert b1.is_tripped()

        # Reload same instance simulating process restart same day
        b2 = Phase1aCircuitBreaker(state_path=path)
        assert b2.is_tripped()  # NOT reset because same UTC day

    def test_record_trade_close_applies_daily_reset(self, tmp_path):
        path = tmp_path / "state.json"
        yesterday = (datetime.now(tz=timezone.utc).date() - timedelta(days=1)).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "losses": 2,
                    "pnl_usd": -10.0,
                    "tripped": False,
                    "tripped_at": None,
                    "last_updated": f"{yesterday}T23:00:00+00:00",
                    "last_reset_date": yesterday,
                }
            )
        )
        b = Phase1aCircuitBreaker(state_path=path)
        # First record on new day resets stale counters before applying
        b.record_trade_close(-5.0)
        state = b.snapshot()
        assert state.losses == 1  # NOT 3 (2 stale + 1 new); reset wiped stale
        assert state.pnl_usd == -5.0

    def test_fresh_state_has_today_reset_date(self, tmp_path):
        b = Phase1aCircuitBreaker(state_path=tmp_path / "state.json")
        today = datetime.now(tz=timezone.utc).date().isoformat()
        assert b.snapshot().last_reset_date == today

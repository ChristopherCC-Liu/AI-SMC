import pytest
from datetime import datetime

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

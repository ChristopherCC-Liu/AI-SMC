"""Unit tests for ConsecLossHalt — Round 5 T1 F3."""
from __future__ import annotations

from datetime import datetime

import pytest

from smc.risk.consec_loss_halt import ConsecLossHalt, ConsecLossState


class TestConsecLossHalt:
    def test_initial_not_tripped(self, tmp_path):
        h = ConsecLossHalt(state_path=tmp_path / "s.json")
        assert not h.is_tripped()
        assert h.snapshot().consec_losses == 0

    def test_two_losses_not_trip(self, tmp_path):
        h = ConsecLossHalt(state_path=tmp_path / "s.json")
        h.record(-5.0)
        h.record(-3.0)
        assert not h.is_tripped()
        assert h.snapshot().consec_losses == 2

    def test_three_consecutive_losses_trip(self, tmp_path):
        h = ConsecLossHalt(state_path=tmp_path / "s.json")
        h.record(-5.0)
        h.record(-3.0)
        h.record(-1.0)
        assert h.is_tripped()
        assert h.snapshot().consec_losses == 3
        assert h.snapshot().tripped_at is not None
        # ISO 8601 parseable
        datetime.fromisoformat(h.snapshot().tripped_at)

    def test_win_resets_streak(self, tmp_path):
        h = ConsecLossHalt(state_path=tmp_path / "s.json")
        h.record(-5.0)
        h.record(-3.0)
        h.record(10.0)  # WIN
        assert not h.is_tripped()
        assert h.snapshot().consec_losses == 0
        # Now 3 more losses DO trip
        h.record(-1.0)
        h.record(-2.0)
        h.record(-3.0)
        assert h.is_tripped()

    def test_break_even_resets_streak(self, tmp_path):
        h = ConsecLossHalt(state_path=tmp_path / "s.json")
        h.record(-5.0)
        h.record(0.0)  # break-even counts as non-loss
        assert h.snapshot().consec_losses == 0

    def test_state_persists_across_instances(self, tmp_path):
        path = tmp_path / "s.json"
        h1 = ConsecLossHalt(state_path=path)
        h1.record(-5.0)
        h1.record(-3.0)
        h1.record(-1.0)
        assert h1.is_tripped()

        h2 = ConsecLossHalt(state_path=path)
        assert h2.is_tripped()
        assert h2.snapshot().consec_losses == 3

    def test_tripped_halt_ignores_subsequent_records(self, tmp_path):
        h = ConsecLossHalt(state_path=tmp_path / "s.json")
        h.record(-5.0)
        h.record(-3.0)
        h.record(-1.0)
        assert h.is_tripped()
        # Any subsequent record is no-op; streak frozen
        h.record(50.0)  # would reset streak but halt is tripped
        assert h.is_tripped()
        assert h.snapshot().consec_losses == 3

    def test_reset_clears_state(self, tmp_path):
        h = ConsecLossHalt(state_path=tmp_path / "s.json")
        h.record(-5.0)
        h.record(-3.0)
        h.record(-1.0)
        assert h.is_tripped()
        h.reset()
        assert not h.is_tripped()
        assert h.snapshot().consec_losses == 0

    def test_daily_reset_clears_trip(self, tmp_path, monkeypatch):
        import smc.risk.consec_loss_halt as module
        path = tmp_path / "s.json"

        # Trip today
        h = ConsecLossHalt(state_path=path)
        h.record(-5.0)
        h.record(-3.0)
        h.record(-1.0)
        assert h.is_tripped()

        # Simulate UTC date rollover by monkey-patching _today_utc_iso
        real_today = module._today_utc_iso()
        fake_tomorrow = _advance_date_iso(real_today, 1)
        monkeypatch.setattr(module, "_today_utc_iso", lambda: fake_tomorrow)

        h2 = ConsecLossHalt(state_path=path)
        # Reading state on new day should auto-reset
        assert not h2.is_tripped()
        assert h2.snapshot().consec_losses == 0

    def test_record_applies_daily_reset_first(self, tmp_path, monkeypatch):
        """A record received on a new UTC day should start a fresh streak."""
        import smc.risk.consec_loss_halt as module
        path = tmp_path / "s.json"

        h = ConsecLossHalt(state_path=path)
        h.record(-5.0)
        h.record(-3.0)
        assert h.snapshot().consec_losses == 2

        real_today = module._today_utc_iso()
        fake_tomorrow = _advance_date_iso(real_today, 1)
        monkeypatch.setattr(module, "_today_utc_iso", lambda: fake_tomorrow)

        h2 = ConsecLossHalt(state_path=path)
        h2.record(-1.0)
        # On new day the streak should be 1 (not 3)
        assert h2.snapshot().consec_losses == 1
        assert not h2.is_tripped()


def _advance_date_iso(iso: str, days: int) -> str:
    from datetime import date, timedelta
    d = date.fromisoformat(iso)
    return (d + timedelta(days=days)).isoformat()

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


class TestConfigurableLimit:
    """audit-r3 R4: consec_limit driven by cfg.consec_loss_limit (per-symbol)."""

    def test_default_limit_is_3(self, tmp_path):
        """Backward compat: no kwarg → 3 losses trip (previous hardcode)."""
        h = ConsecLossHalt(state_path=tmp_path / "s.json")
        h.record(-1.0)
        h.record(-1.0)
        assert not h.is_tripped()
        h.record(-1.0)
        assert h.is_tripped()

    def test_custom_limit_2_trips_earlier(self, tmp_path):
        h = ConsecLossHalt(state_path=tmp_path / "s.json", consec_limit=2)
        h.record(-1.0)
        assert not h.is_tripped()
        h.record(-1.0)
        assert h.is_tripped()

    def test_custom_limit_5_trips_later(self, tmp_path):
        h = ConsecLossHalt(state_path=tmp_path / "s.json", consec_limit=5)
        for _ in range(4):
            h.record(-1.0)
        assert not h.is_tripped()
        h.record(-1.0)
        assert h.is_tripped()

    def test_limit_one_trips_on_first_loss(self, tmp_path):
        """Degenerate: limit=1 → any loss trips immediately."""
        h = ConsecLossHalt(state_path=tmp_path / "s.json", consec_limit=1)
        h.record(-0.01)
        assert h.is_tripped()

    def test_limit_zero_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="consec_limit must be >= 1"):
            ConsecLossHalt(state_path=tmp_path / "s.json", consec_limit=0)

    def test_limit_negative_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="consec_limit must be >= 1"):
            ConsecLossHalt(state_path=tmp_path / "s.json", consec_limit=-1)

    def test_win_between_losses_still_resets_regardless_of_limit(self, tmp_path):
        """Streak semantics unchanged by limit — WIN resets."""
        h = ConsecLossHalt(state_path=tmp_path / "s.json", consec_limit=5)
        h.record(-1.0)
        h.record(-1.0)
        h.record(5.0)  # WIN — streak reset
        h.record(-1.0)
        assert not h.is_tripped()
        assert h.snapshot().consec_losses == 1

    def test_limit_read_from_instrument_config(self, tmp_path):
        """Integration: cfg.consec_loss_limit drives the halt limit."""
        from smc.instruments import get_instrument_config
        cfg = get_instrument_config("XAUUSD")
        h = ConsecLossHalt(state_path=tmp_path / "s.json", consec_limit=cfg.consec_loss_limit)
        for _ in range(cfg.consec_loss_limit - 1):
            h.record(-1.0)
        assert not h.is_tripped()
        h.record(-1.0)
        assert h.is_tripped()


class TestPerSuffixStateFileIsolation:
    """audit-r4 v5 Option B: control + treatment legs share the same TMGM Demo
    account but need independent halt state.  ConsecLossHalt is already
    state_path-driven, so per-leg isolation is achieved purely by the caller
    passing distinct paths.  These tests pin that behaviour.
    """

    def test_two_instances_with_distinct_paths_are_isolated(self, tmp_path):
        """Control path and treatment path maintain independent streaks."""
        control_path = tmp_path / "consec_loss_state.json"
        treatment_path = tmp_path / "consec_loss_state_macro.json"

        control = ConsecLossHalt(state_path=control_path)
        treatment = ConsecLossHalt(state_path=treatment_path)

        # Trip control with 3 losses
        control.record(-1.0)
        control.record(-1.0)
        control.record(-1.0)
        assert control.is_tripped()

        # Treatment is unaffected — still fresh, still tradeable
        assert not treatment.is_tripped()
        assert treatment.snapshot().consec_losses == 0

        # Treatment can take its own losses independently
        treatment.record(-5.0)
        assert treatment.snapshot().consec_losses == 1
        assert not treatment.is_tripped()

    def test_distinct_suffix_paths_write_distinct_files(self, tmp_path):
        """Naming pattern: data/<sym>/consec_loss_state{suffix}.json."""
        control_path = tmp_path / "consec_loss_state.json"
        treatment_path = tmp_path / "consec_loss_state_macro.json"

        control = ConsecLossHalt(state_path=control_path)
        treatment = ConsecLossHalt(state_path=treatment_path)
        control.record(-1.0)
        treatment.record(-2.0)

        assert control_path.exists()
        assert treatment_path.exists()
        # Different files → different content
        c_content = control_path.read_text()
        t_content = treatment_path.read_text()
        assert c_content != t_content

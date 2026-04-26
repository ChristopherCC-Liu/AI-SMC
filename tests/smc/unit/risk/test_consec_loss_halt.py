"""Unit tests for ConsecLossHalt — Round 5 T1 F3 + R10 P3.2 hygiene rewrite.

R10 P3.2 changed the halt semantics from "3 consecutive losses, daily UTC
reset" to "3 losses inside the most recent 6 trades, no automatic reset".
Tests below cover both the new rolling-window behaviour and the
backward-compat surface (limit configurability, per-leg state isolation,
``snapshot().consec_losses`` property).
"""
from __future__ import annotations

import json

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
        from datetime import datetime
        h = ConsecLossHalt(state_path=tmp_path / "s.json")
        h.record(-5.0)
        h.record(-3.0)
        h.record(-1.0)
        assert h.is_tripped()
        assert h.snapshot().consec_losses == 3
        assert h.snapshot().tripped_at is not None
        datetime.fromisoformat(h.snapshot().tripped_at)

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
        # Once tripped, the rolling window is frozen until reset() — wins
        # do NOT clear the trip retroactively. Operator must reset.
        h.record(50.0)
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


class TestRollingWindow:
    """R10 P3.2 hygiene: 3-in-6 rolling window semantics."""

    def test_rolling_window_3_losses_in_6_trips(self, tmp_path):
        """3 losses interleaved with wins (still within 6-window) trips."""
        h = ConsecLossHalt(state_path=tmp_path / "s.json")
        # L W L W L  → 3 losses in 5 trades, all inside window=6
        h.record(-1.0)
        h.record(2.0)
        h.record(-1.0)
        h.record(2.0)
        h.record(-1.0)
        assert h.is_tripped()

    def test_3_losses_after_3_wins_trips(self, tmp_path):
        """W W W L L L → window holds [W,W,W,L,L,L]; loss count = 3 → trip."""
        h = ConsecLossHalt(state_path=tmp_path / "s.json")
        for _ in range(3):
            h.record(5.0)
        h.record(-1.0)
        h.record(-1.0)
        h.record(-1.0)
        assert h.is_tripped()
        assert h.snapshot().consec_losses == 3

    def test_2_losses_in_window_with_3_wins_does_not_trip(self, tmp_path):
        h = ConsecLossHalt(state_path=tmp_path / "s.json")
        h.record(-1.0)
        h.record(5.0)
        h.record(-1.0)
        h.record(5.0)
        h.record(5.0)
        assert not h.is_tripped()
        # Window = (L,W,L,W,W). loss_count_in_window=2; trailing consec=0.
        assert h.snapshot().loss_count_in_window == 2
        assert h.snapshot().consec_losses == 0

    def test_old_loss_evicted_after_window_size_trades(self, tmp_path):
        """L then 6 wins → window holds [W,W,W,W,W,W]; original loss evicted."""
        h = ConsecLossHalt(state_path=tmp_path / "s.json")
        h.record(-1.0)
        for _ in range(6):
            h.record(5.0)
        # Loss has aged out of the 6-window
        assert h.snapshot().consec_losses == 0
        assert not h.is_tripped()

    def test_2_losses_then_4_wins_then_loss_does_not_trip(self, tmp_path):
        """L L W W W W L → window=[L,L,W,W,W,W,L][-6:]=[L,W,W,W,W,L];
        loss_count_in_window=2 → no trip; trailing consec_losses=1."""
        h = ConsecLossHalt(state_path=tmp_path / "s.json")
        h.record(-1.0)
        h.record(-1.0)
        for _ in range(4):
            h.record(5.0)
        h.record(-1.0)
        assert h.snapshot().loss_count_in_window == 2
        assert h.snapshot().consec_losses == 1  # trailing: most-recent is loss
        assert not h.is_tripped()

    def test_window_truncates_to_window_size(self, tmp_path):
        """After many trades, rolling window length stays bounded."""
        h = ConsecLossHalt(state_path=tmp_path / "s.json")
        for _ in range(50):
            h.record(5.0)
        snap = h.snapshot()
        assert len(snap.recent_outcomes) <= 6


class TestConsecLossesProperty:
    """R10 P3.2 backward-compat: ``consec_losses`` is the TRAILING streak
    (most recent loss backward), NOT total losses inside the window.

    Pre-R10 callers (e.g. ``live_demo.py:1041`` halt-reason string,
    monitor/guards_snapshot.py) read ``snap.consec_losses`` as "how many
    losses in a row right now?" — preserving that semantic prevents
    cosmetic surprises in operator-facing log lines.
    """

    def test_consec_losses_trailing_zero_when_last_is_win(self, tmp_path):
        h = ConsecLossHalt(state_path=tmp_path / "s.json")
        h.record(-1.0)
        h.record(-1.0)
        h.record(5.0)  # most-recent is WIN
        assert h.snapshot().consec_losses == 0
        assert h.snapshot().loss_count_in_window == 2

    def test_consec_losses_trailing_counts_from_tail(self, tmp_path):
        """Window=(L,W,L,L,L). Trailing=3, total in window=4."""
        # High consec_limit + matching window_size avoid trip-noop.
        h = ConsecLossHalt(
            state_path=tmp_path / "s.json", consec_limit=99, window_size=99
        )
        h.record(-1.0)
        h.record(5.0)
        h.record(-1.0)
        h.record(-1.0)
        h.record(-1.0)
        assert h.snapshot().consec_losses == 3
        assert h.snapshot().loss_count_in_window == 4

    def test_loss_count_in_window_is_total(self, tmp_path):
        """``loss_count_in_window`` reports total losses inside the window —
        the value that drives the trip flag."""
        h = ConsecLossHalt(
            state_path=tmp_path / "s.json", consec_limit=99, window_size=99
        )
        h.record(-1.0)
        h.record(5.0)
        h.record(-1.0)
        h.record(5.0)
        h.record(-1.0)
        assert h.snapshot().loss_count_in_window == 3
        assert h.snapshot().consec_losses == 1  # trailing only the last L


class TestConfigurableWindowSize:
    """R10 P3.2 refinement: window_size sourced from cfg.consec_loss_window_size.
    Allows control vs treatment legs to A/B different window widths."""

    def test_window_size_default_is_6(self, tmp_path):
        # consec_limit=99 so the halt never trips and records aren't
        # silently no-op'd (window_size validator forces window>=limit).
        h = ConsecLossHalt(
            state_path=tmp_path / "s.json", consec_limit=99, window_size=99
        )
        # The default-window-size assertion needs a fresh instance using
        # the default window_size; do that with all-wins so no trip risk.
        h2 = ConsecLossHalt(state_path=tmp_path / "s2.json")
        for _ in range(7):
            h2.record(5.0)
        assert len(h2.snapshot().recent_outcomes) == 6

    def test_custom_window_size_10(self, tmp_path):
        # consec_limit kept high so the halt doesn't trip mid-loop and
        # silently noop subsequent records (window_size==consec_limit OK).
        h = ConsecLossHalt(
            state_path=tmp_path / "s.json", consec_limit=10, window_size=10
        )
        # All wins → never trip. Verifies window truncation only.
        for _ in range(15):
            h.record(5.0)
        assert len(h.snapshot().recent_outcomes) == 10

    def test_window_size_read_from_instrument_config(self, tmp_path):
        """Integration: cfg.consec_loss_window_size drives the window."""
        from smc.instruments import get_instrument_config
        cfg = get_instrument_config("XAUUSD")
        h = ConsecLossHalt(
            state_path=tmp_path / "s.json",
            consec_limit=cfg.consec_loss_limit,
            window_size=cfg.consec_loss_window_size,
        )
        for _ in range(cfg.consec_loss_window_size + 3):
            h.record(5.0)
        assert (
            len(h.snapshot().recent_outcomes) == cfg.consec_loss_window_size
        )


class TestNoDailyReset:
    """R10 P3.2: daily UTC reset removed. State persists across midnights."""

    def test_no_daily_reset_module_function_removed(self):
        """The pre-R10 ``_today_utc_iso`` helper was removed alongside the
        daily-reset semantic. Asserting absence prevents accidental reintro."""
        import smc.risk.consec_loss_halt as module
        assert not hasattr(module, "_apply_daily_reset_if_needed")
        assert not hasattr(module, "_today_utc_iso")

    def test_state_does_not_carry_last_reset_date_field(self, tmp_path):
        """Persisted state file omits the legacy ``last_reset_date`` field."""
        h = ConsecLossHalt(state_path=tmp_path / "s.json")
        h.record(-1.0)
        payload = json.loads((tmp_path / "s.json").read_text())
        assert "last_reset_date" not in payload
        assert "recent_outcomes" in payload


class TestLegacyMigration:
    """Operators with pre-R10 state files must not lose halt state on upgrade."""

    def test_legacy_tripped_state_migrated(self, tmp_path, caplog):
        import logging
        path = tmp_path / "s.json"
        legacy = {
            "consec_losses": 3,
            "last_pnl_usd": -2.5,
            "tripped": True,
            "tripped_at": "2026-04-25T22:00:00+00:00",
            "last_updated": "2026-04-25T22:00:00+00:00",
            "last_reset_date": "2026-04-25",
        }
        path.write_text(json.dumps(legacy))
        with caplog.at_level(
            logging.WARNING, logger="smc.risk.consec_loss_halt"
        ):
            h = ConsecLossHalt(state_path=path)
        snap = h.snapshot()
        assert h.is_tripped() is True
        assert snap.consec_losses == 3
        assert len(snap.recent_outcomes) == 3
        assert all(snap.recent_outcomes)
        warns = [r for r in caplog.records if "legacy_migrated" in r.getMessage()]
        assert len(warns) == 1

    def test_legacy_untripped_state_migrated(self, tmp_path):
        path = tmp_path / "s.json"
        legacy = {
            "consec_losses": 1,
            "last_pnl_usd": -1.0,
            "tripped": False,
            "tripped_at": None,
            "last_updated": "2026-04-25T20:00:00+00:00",
            "last_reset_date": "2026-04-25",
        }
        path.write_text(json.dumps(legacy))
        h = ConsecLossHalt(state_path=path)
        assert h.is_tripped() is False
        assert h.snapshot().consec_losses == 1


class TestConfigurableLimit:
    """audit-r3 R4: consec_limit driven by cfg.consec_loss_limit (per-symbol)."""

    def test_default_limit_is_3(self, tmp_path):
        h = ConsecLossHalt(state_path=tmp_path / "s.json")
        h.record(-1.0)
        h.record(-1.0)
        assert not h.is_tripped()
        h.record(-1.0)
        assert h.is_tripped()

    def test_custom_limit_2_trips_earlier(self, tmp_path):
        h = ConsecLossHalt(
            state_path=tmp_path / "s.json", consec_limit=2, window_size=6
        )
        h.record(-1.0)
        assert not h.is_tripped()
        h.record(-1.0)
        assert h.is_tripped()

    def test_custom_limit_5_trips_later(self, tmp_path):
        h = ConsecLossHalt(
            state_path=tmp_path / "s.json", consec_limit=5, window_size=10
        )
        for _ in range(4):
            h.record(-1.0)
        assert not h.is_tripped()
        h.record(-1.0)
        assert h.is_tripped()

    def test_limit_one_trips_on_first_loss(self, tmp_path):
        h = ConsecLossHalt(state_path=tmp_path / "s.json", consec_limit=1)
        h.record(-0.01)
        assert h.is_tripped()

    def test_limit_zero_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="consec_limit must be >= 1"):
            ConsecLossHalt(state_path=tmp_path / "s.json", consec_limit=0)

    def test_limit_negative_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="consec_limit must be >= 1"):
            ConsecLossHalt(state_path=tmp_path / "s.json", consec_limit=-1)

    def test_window_smaller_than_limit_rejected(self, tmp_path):
        """A window smaller than the limit produces a halt that can never
        trip — refuse it loudly to surface configuration errors."""
        with pytest.raises(ValueError, match="window_size"):
            ConsecLossHalt(
                state_path=tmp_path / "s.json",
                consec_limit=3,
                window_size=2,
            )

    def test_limit_read_from_instrument_config(self, tmp_path):
        from smc.instruments import get_instrument_config
        cfg = get_instrument_config("XAUUSD")
        h = ConsecLossHalt(
            state_path=tmp_path / "s.json", consec_limit=cfg.consec_loss_limit
        )
        for _ in range(cfg.consec_loss_limit - 1):
            h.record(-1.0)
        assert not h.is_tripped()
        h.record(-1.0)
        assert h.is_tripped()


class TestPerSuffixStateFileIsolation:
    """audit-r4 v5 Option B: control + treatment legs share the same TMGM Demo
    account but need independent halt state. ConsecLossHalt is already
    state_path-driven, so per-leg isolation is achieved purely by the caller
    passing distinct paths. These tests pin that behaviour.
    """

    def test_two_instances_with_distinct_paths_are_isolated(self, tmp_path):
        control_path = tmp_path / "consec_loss_state.json"
        treatment_path = tmp_path / "consec_loss_state_macro.json"
        control = ConsecLossHalt(state_path=control_path)
        treatment = ConsecLossHalt(state_path=treatment_path)
        control.record(-1.0)
        control.record(-1.0)
        control.record(-1.0)
        assert control.is_tripped()
        assert not treatment.is_tripped()
        assert treatment.snapshot().consec_losses == 0
        treatment.record(-5.0)
        assert treatment.snapshot().consec_losses == 1
        assert not treatment.is_tripped()

    def test_distinct_suffix_paths_write_distinct_files(self, tmp_path):
        control_path = tmp_path / "consec_loss_state.json"
        treatment_path = tmp_path / "consec_loss_state_macro.json"
        control = ConsecLossHalt(state_path=control_path)
        treatment = ConsecLossHalt(state_path=treatment_path)
        control.record(-1.0)
        treatment.record(-2.0)
        assert control_path.exists()
        assert treatment_path.exists()
        assert control_path.read_text() != treatment_path.read_text()

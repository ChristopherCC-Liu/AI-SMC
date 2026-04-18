"""Unit tests for smc.strategy.hysteresis.HysteresisState.

Audit R3 S5: asymmetric latency. Enter (v1_passthrough → trending/ranging)
commits on 1 bar; all other transitions commit on 2 bars.

Covers:
- Initial state defaults
- Same-mode no-op
- Enter transitions (1-bar commit)
- Exit transitions (2-bar commit, pending reset)
- Alternating / cycling sequences
- Multi-hop transitions (v1 → ranging → trending)
"""

from __future__ import annotations

from smc.strategy.hysteresis import (
    _ENTER_BAR_THRESHOLD,
    _EXIT_BAR_THRESHOLD,
    HysteresisState,
    _required_bars,
)


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


class TestConstants:
    def test_enter_threshold_is_one(self) -> None:
        """S5: enter is 1-bar."""
        assert _ENTER_BAR_THRESHOLD == 1

    def test_exit_threshold_is_two(self) -> None:
        """S5: exit preserves 2-bar whipsaw guard."""
        assert _EXIT_BAR_THRESHOLD == 2

    def test_required_bars_enter_path(self) -> None:
        assert _required_bars("v1_passthrough", "trending") == 1
        assert _required_bars("v1_passthrough", "ranging") == 1

    def test_required_bars_exit_path(self) -> None:
        """Exit to neutral."""
        assert _required_bars("trending", "v1_passthrough") == 2
        assert _required_bars("ranging", "v1_passthrough") == 2

    def test_required_bars_swap_treated_as_exit(self) -> None:
        """trending↔ranging uses exit window — same noise risk as leaving
        a committed mode.
        """
        assert _required_bars("trending", "ranging") == 2
        assert _required_bars("ranging", "trending") == 2


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_default_mode_is_v1_passthrough(self) -> None:
        state = HysteresisState()
        assert state.current_mode == "v1_passthrough"
        assert state.pending_mode is None
        assert state.pending_bar_count == 0

    def test_custom_initial_mode(self) -> None:
        state = HysteresisState(current_mode="ranging")
        assert state.current_mode == "ranging"


# ---------------------------------------------------------------------------
# Same mode proposed → no-op
# ---------------------------------------------------------------------------


class TestSameModeProposed:
    def test_same_mode_returns_current(self) -> None:
        state = HysteresisState()
        result = state.update("v1_passthrough")
        assert result == "v1_passthrough"

    def test_same_mode_clears_pending(self) -> None:
        """After a 1-bar enter commit to ranging, revert to ranging
        proposal (now same as current) should leave pending cleared.
        """
        state = HysteresisState()
        state.update("ranging")  # S5: enter commits on bar 1
        assert state.current_mode == "ranging"
        state.update("ranging")  # same as current → no-op
        assert state.pending_mode is None
        assert state.pending_bar_count == 0


# ---------------------------------------------------------------------------
# Audit R3 S5 — enter transitions (1-bar commit)
# ---------------------------------------------------------------------------


class TestEnterTransitions:
    def test_enter_trending_from_v1_in_one_bar(self) -> None:
        """S5: v1 → trending commits immediately after one bar."""
        state = HysteresisState()
        result = state.update("trending")
        assert result == "trending"
        assert state.current_mode == "trending"
        assert state.pending_mode is None
        assert state.pending_bar_count == 0

    def test_enter_ranging_from_v1_in_one_bar(self) -> None:
        """S5: v1 → ranging commits immediately (mirror of trending)."""
        state = HysteresisState()
        result = state.update("ranging")
        assert result == "ranging"
        assert state.current_mode == "ranging"

    def test_enter_then_revert_same_bar_is_same_mode(self) -> None:
        """Enter flicker protection: after enter commit, proposing v1
        becomes a valid exit proposal requiring the 2-bar path.
        """
        state = HysteresisState()
        state.update("ranging")  # enter: bar 1 commit → current=ranging
        result = state.update("v1_passthrough")  # exit bar 1: pending
        assert result == "ranging"
        assert state.pending_mode == "v1_passthrough"
        assert state.pending_bar_count == 1


# ---------------------------------------------------------------------------
# Audit R3 S5 — exit transitions (2-bar commit preserved)
# ---------------------------------------------------------------------------


class TestExitTransitions:
    def test_exit_trending_still_requires_two_bars(self) -> None:
        """S5 preserves exit 2-bar: trending → v1 needs 2 consecutive bars."""
        state = HysteresisState(current_mode="trending")
        result_bar1 = state.update("v1_passthrough")
        assert result_bar1 == "trending"  # bar 1: pending
        assert state.pending_mode == "v1_passthrough"
        result_bar2 = state.update("v1_passthrough")
        assert result_bar2 == "v1_passthrough"  # bar 2: commit
        assert state.current_mode == "v1_passthrough"

    def test_exit_ranging_still_requires_two_bars(self) -> None:
        """Mirror: ranging → v1 needs 2 bars."""
        state = HysteresisState(current_mode="ranging")
        state.update("v1_passthrough")
        assert state.current_mode == "ranging"
        state.update("v1_passthrough")
        assert state.current_mode == "v1_passthrough"

    def test_exit_one_bar_then_break_resets_pending(self) -> None:
        """After 1 exit bar, proposal back to current resets pending
        to prevent slow bleed-out under 1-bar-on 1-bar-off noise.
        """
        state = HysteresisState(current_mode="ranging")
        state.update("v1_passthrough")  # exit bar 1: pending
        result = state.update("ranging")  # back to current → reset
        assert result == "ranging"
        assert state.pending_mode is None
        assert state.pending_bar_count == 0

    def test_trending_to_ranging_requires_two_bars(self) -> None:
        """Swap between non-neutral modes uses exit path (2 bars) not enter."""
        state = HysteresisState(current_mode="trending")
        result_bar1 = state.update("ranging")
        assert result_bar1 == "trending"  # bar 1: pending only
        result_bar2 = state.update("ranging")
        assert result_bar2 == "ranging"  # bar 2: commit

    def test_ranging_to_trending_requires_two_bars(self) -> None:
        """Mirror of trending→ranging swap."""
        state = HysteresisState(current_mode="ranging")
        state.update("trending")
        assert state.current_mode == "ranging"
        state.update("trending")
        assert state.current_mode == "trending"


# ---------------------------------------------------------------------------
# Alternating / cycling — flicker protection intact
# ---------------------------------------------------------------------------


class TestAlternatingProposals:
    def test_alternating_ranging_v1_commits_enter_but_not_exit(self) -> None:
        """S5: alternating v1/ranging commits ranging each time (enter 1-bar)
        but exit to v1 needs 2 consecutive bars which alternation never gives.

        Sequence ranging,v1,ranging,v1,...:
        - bar0 ranging → commit ranging (enter)
        - bar1 v1 → pending v1, count=1 (exit in flight)
        - bar2 ranging → back to current ranging → pending reset
        - bar3 v1 → pending v1, count=1
        - ...
        Result: current_mode oscillates between ranging (after even bars)
        but never settles at v1. Final state: ranging with exit pending.
        """
        state = HysteresisState()
        results = []
        for i in range(8):
            mode = "ranging" if i % 2 == 0 else "v1_passthrough"
            results.append(state.update(mode))
        # Under S5 the first enter commits immediately; exits never complete.
        assert results[0] == "ranging"
        assert state.current_mode == "ranging"
        # v1 proposals never get 2 consecutive bars → no exit commit.
        # Committed mode after seq is still ranging.

    def test_three_different_modes_cycling(self) -> None:
        """v1 → ranging → trending → ranging → trending → ranging.

        - bar0 ranging → enter commit → current=ranging
        - bar1 trending → pending (exit path, 2 bars needed), count=1
        - bar2 ranging → back to current → pending reset
        - bar3 trending → pending, count=1
        - bar4 ranging → back to current → pending reset

        Final: ranging (first enter), never flips to trending because
        trending proposals never land twice consecutively.
        """
        state = HysteresisState()
        sequence = ["ranging", "trending", "ranging", "trending", "ranging"]
        results = [state.update(m) for m in sequence]
        assert results[0] == "ranging"  # enter commit
        assert state.current_mode == "ranging"
        # trending never gets 2 consecutive bars under this cycle.
        assert all(r == "ranging" for r in results[1:])

    def test_flicker_protection_intact_for_exits(self) -> None:
        """Exit whipsaw scenario: ranging,v1,ranging,v1 — enters commit,
        exits never do. This is the protective behavior we want.
        """
        state = HysteresisState()
        state.update("ranging")  # enter → current=ranging
        state.update("v1_passthrough")  # exit pending 1
        state.update("ranging")  # back to current → reset
        state.update("v1_passthrough")  # exit pending 1
        assert state.current_mode == "ranging"
        assert state.pending_mode == "v1_passthrough"


# ---------------------------------------------------------------------------
# Multi-hop transitions
# ---------------------------------------------------------------------------


class TestMultiHopTransitions:
    def test_v1_to_ranging_one_bar_to_trending_two_bars(self) -> None:
        """Full multi-hop under S5:
        - v1 → ranging: 1 bar (enter)
        - ranging → trending: 2 bars (swap treated as exit)
        """
        state = HysteresisState()

        # Enter ranging (1 bar).
        assert state.update("ranging") == "ranging"
        assert state.current_mode == "ranging"

        # Swap to trending (2 bars).
        assert state.update("trending") == "ranging"  # pending
        assert state.current_mode == "ranging"
        assert state.update("trending") == "trending"  # commit
        assert state.current_mode == "trending"

    def test_full_round_trip_asymmetric(self) -> None:
        """v1 → ranging (1 bar) → v1 (2 bars)."""
        state = HysteresisState()

        # Enter: 1 bar.
        state.update("ranging")
        assert state.current_mode == "ranging"

        # Exit: 2 bars.
        state.update("v1_passthrough")
        assert state.current_mode == "ranging"  # still ranging
        state.update("v1_passthrough")
        assert state.current_mode == "v1_passthrough"

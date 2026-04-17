"""Unit tests for smc.strategy.hysteresis.HysteresisState.

Covers all state transitions:
- Same mode proposed → no change, pending cleared
- First proposal of new mode → pending but not committed
- Two consecutive proposals → committed flip
- Alternating proposals → never flip
- Multi-step flip then flip back
"""

from __future__ import annotations

from smc.strategy.hysteresis import HysteresisState


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
        state = HysteresisState()
        state.update("ranging")  # set pending_mode="ranging", count=1
        state.update("v1_passthrough")  # back to current → clears pending
        assert state.pending_mode is None
        assert state.pending_bar_count == 0


# ---------------------------------------------------------------------------
# Flip from v1_passthrough → ranging
# ---------------------------------------------------------------------------


class TestFlipToRanging:
    def test_bar1_propose_ranging_still_v1(self) -> None:
        """First new proposal sets pending but does not commit."""
        state = HysteresisState()
        result = state.update("ranging")
        assert result == "v1_passthrough"
        assert state.pending_mode == "ranging"
        assert state.pending_bar_count == 1

    def test_bar1_ranging_bar2_break_no_flip(self) -> None:
        """Second bar breaks back to current mode → pending is cleared."""
        state = HysteresisState()
        state.update("ranging")  # bar 1
        result = state.update("v1_passthrough")  # bar 2: same as current → reset
        assert result == "v1_passthrough"
        assert state.pending_mode is None
        assert state.pending_bar_count == 0

    def test_bar1_bar2_ranging_commits(self) -> None:
        """Two consecutive identical proposals commit the mode change."""
        state = HysteresisState()
        state.update("ranging")  # bar 1: pending
        result = state.update("ranging")  # bar 2: commit
        assert result == "ranging"
        assert state.current_mode == "ranging"
        assert state.pending_mode is None
        assert state.pending_bar_count == 0

    def test_bar1_bar2_trending_commits(self) -> None:
        """Flip to trending also requires 2 consecutive proposals."""
        state = HysteresisState()
        state.update("trending")
        result = state.update("trending")
        assert result == "trending"


# ---------------------------------------------------------------------------
# Flip back from ranging → v1_passthrough
# ---------------------------------------------------------------------------


class TestFlipBack:
    def test_one_break_bar_not_enough(self) -> None:
        state = HysteresisState(current_mode="ranging")
        result = state.update("v1_passthrough")  # bar 1
        assert result == "ranging"
        assert state.pending_mode == "v1_passthrough"

    def test_two_break_bars_flip_back(self) -> None:
        state = HysteresisState(current_mode="ranging")
        state.update("v1_passthrough")  # bar 1
        result = state.update("v1_passthrough")  # bar 2
        assert result == "v1_passthrough"
        assert state.current_mode == "v1_passthrough"

    def test_break_then_same_resets_pending(self) -> None:
        """After one break bar, if ranging is proposed again, pending resets."""
        state = HysteresisState(current_mode="ranging")
        state.update("v1_passthrough")  # pending v1, count=1
        result = state.update("ranging")  # back to current → reset
        assert result == "ranging"
        assert state.pending_mode is None


# ---------------------------------------------------------------------------
# Alternating proposals never flip
# ---------------------------------------------------------------------------


class TestAlternatingProposals:
    def test_alternating_ranging_v1_never_commits(self) -> None:
        state = HysteresisState()
        results = []
        for i in range(8):
            mode = "ranging" if i % 2 == 0 else "v1_passthrough"
            results.append(state.update(mode))
        assert all(r == "v1_passthrough" for r in results)

    def test_three_different_modes_cycling(self) -> None:
        """v1 → ranging → trending → ranging → ... never commits."""
        state = HysteresisState()
        sequence = ["ranging", "trending", "ranging", "trending", "ranging"]
        results = [state.update(m) for m in sequence]
        assert all(r == "v1_passthrough" for r in results)


# ---------------------------------------------------------------------------
# Multi-hop transitions
# ---------------------------------------------------------------------------


class TestMultiHopTransitions:
    def test_v1_to_ranging_to_trending(self) -> None:
        state = HysteresisState()

        # Flip to ranging (2 bars)
        state.update("ranging")
        assert state.update("ranging") == "ranging"

        # Flip to trending (2 bars)
        state.update("trending")
        assert state.current_mode == "ranging"  # pending, not committed yet
        assert state.update("trending") == "trending"
        assert state.current_mode == "trending"

    def test_full_round_trip(self) -> None:
        """v1 → ranging → v1, using 2 bars each leg."""
        state = HysteresisState()

        state.update("ranging")
        state.update("ranging")
        assert state.current_mode == "ranging"

        state.update("v1_passthrough")
        state.update("v1_passthrough")
        assert state.current_mode == "v1_passthrough"

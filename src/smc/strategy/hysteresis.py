"""Hysteresis state tracker for mode router — prevents rapid mode flipping.

Audit R3 S5: asymmetric latency.
  - Enter (v1_passthrough → trending/ranging): 1 bar (seize early trending).
  - Exit (trending/ranging → anything else): 2 bars (anti-whipsaw).

Rationale: a trending H1 window often lasts only 2-3 bars in practice
(LON breakout, NY reversal). Requiring 2 consecutive bars to enter costs
half the window as pure lag. Exits keep 2-bar confirmation because a
single contrary bar at an active-session boundary is frequently noise.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["HysteresisState"]

# Audit R3 S5 asymmetric thresholds.
_ENTER_BAR_THRESHOLD: int = 1
_EXIT_BAR_THRESHOLD: int = 2

# "Neutral" mode from which any other proposal counts as an `enter`.
_NEUTRAL_MODE: str = "v1_passthrough"


def _required_bars(current_mode: str, proposed_mode: str) -> int:
    """Return the bar-count threshold for committing `current → proposed`.

    Enter transitions (leaving v1_passthrough) use the shorter window so
    the router can pick up early trending/ranging signals without losing
    a bar. All other transitions (including trending↔ranging swaps) keep
    the exit window as a whipsaw guard.
    """
    if current_mode == _NEUTRAL_MODE and proposed_mode != _NEUTRAL_MODE:
        return _ENTER_BAR_THRESHOLD
    return _EXIT_BAR_THRESHOLD


@dataclass
class HysteresisState:
    """Mutable state — bar-count hysteresis for mode commits.

    Hold a singleton of this in the live_demo aggregator and call
    ``update()`` once per H1 bar close with the proposed mode from
    ``route_trading_mode()``.
    """

    current_mode: str = _NEUTRAL_MODE
    pending_mode: str | None = None
    pending_bar_count: int = 0

    def update(self, proposed_mode: str) -> str:
        """Return the committed mode after hysteresis filtering."""
        if proposed_mode == self.current_mode:
            self.pending_mode = None
            self.pending_bar_count = 0
            return self.current_mode
        if self.pending_mode == proposed_mode:
            self.pending_bar_count += 1
        else:
            self.pending_mode = proposed_mode
            self.pending_bar_count = 1
        threshold = _required_bars(self.current_mode, proposed_mode)
        if self.pending_bar_count >= threshold:
            self.current_mode = proposed_mode
            self.pending_mode = None
            self.pending_bar_count = 0
        return self.current_mode

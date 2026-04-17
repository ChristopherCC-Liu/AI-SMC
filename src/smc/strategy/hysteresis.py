"""Hysteresis state tracker for mode router — prevents rapid mode flipping.

2+ consecutive H1 bar proposals required to commit a mode change.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["HysteresisState"]


@dataclass
class HysteresisState:
    """Mutable state — 2+ consecutive H1 bars required to flip mode on/off.

    Hold a singleton of this in the live_demo aggregator and call
    ``update()`` once per H1 bar close with the proposed mode from
    ``route_trading_mode()``.
    """

    current_mode: str = "v1_passthrough"
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
        if self.pending_bar_count >= 2:
            self.current_mode = proposed_mode
            self.pending_mode = None
            self.pending_bar_count = 0
        return self.current_mode

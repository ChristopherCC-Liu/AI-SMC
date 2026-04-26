"""Entry gate based on rolling spread baseline — Round 10 P3.2.

Blocks new opens when the current spread exceeds a multiplier of the
recent rolling-window median spread. The motivation is to suppress
entries during news-spike or low-liquidity windows where the broker's
quoted spread balloons and any opened position immediately bleeds the
multiplier.

Composition with R10 P1.1's ``PersistentDrawdownGuard``:
- The DD guard answers "can I take any new risk at all?" (multi-day
  bleed protection).
- This spread gate answers "is the broker's quote currently sane?"
  (per-cycle market-microstructure protection).
Both must pass before an order is sent.

Public API:
- :func:`compute_spread_baseline` — pure helper, ``list[float] -> float | None``
- :func:`check_spread_gate` — decision combining current + baseline
- :class:`SpreadGateDecision` — frozen result with diagnostic fields

Bootstrap path: while we have fewer than ``DEFAULT_BASELINE_WINDOW``
non-zero samples, ``compute_spread_baseline`` returns ``None`` and
``check_spread_gate`` passes the cycle through (we trust the broker
until we have evidence to compare against). This avoids freezing the
account at startup when the spread history is empty.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "DEFAULT_BASELINE_WINDOW",
    "DEFAULT_THRESHOLD_MULTIPLIER",
    "SpreadGateDecision",
    "compute_spread_baseline",
    "check_spread_gate",
]


DEFAULT_BASELINE_WINDOW = 20
DEFAULT_THRESHOLD_MULTIPLIER = 1.5


@dataclass(frozen=True)
class SpreadGateDecision:
    """Immutable result of a spread-gate check."""

    can_open: bool
    current_spread: float
    baseline_median: Optional[float]
    threshold_multiplier: float
    rejection_reason: Optional[str] = None


def compute_spread_baseline(
    spreads: list[float],
    *,
    window: int = DEFAULT_BASELINE_WINDOW,
) -> Optional[float]:
    """Return the median of the last ``window`` non-zero positive spreads.

    Filters zero and negative samples (defence against MT5 returning
    placeholder ticks). Returns ``None`` when fewer than ``window`` valid
    samples exist — the caller should treat that as "bootstrap, allow
    pass-through" rather than as a halt.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    valid = [s for s in spreads if s > 0]
    tail = valid[-window:]
    if len(tail) < window:
        return None
    return statistics.median(tail)


def check_spread_gate(
    current_spread: float,
    baseline: Optional[float],
    *,
    multiplier: float = DEFAULT_THRESHOLD_MULTIPLIER,
) -> SpreadGateDecision:
    """Block when ``current_spread >= baseline * multiplier``.

    Pass-through when ``baseline`` is ``None`` or non-positive (bootstrap
    or defensive path — we cannot assess sanity yet).
    """
    if baseline is None or baseline <= 0:
        return SpreadGateDecision(
            can_open=True,
            current_spread=current_spread,
            baseline_median=baseline,
            threshold_multiplier=multiplier,
            rejection_reason=None,
        )

    threshold = baseline * multiplier
    if current_spread >= threshold:
        reason = (
            f"spread {current_spread:.2f} >= baseline {baseline:.2f} * "
            f"{multiplier:.2f} = {threshold:.2f}"
        )
        return SpreadGateDecision(
            can_open=False,
            current_spread=current_spread,
            baseline_median=baseline,
            threshold_multiplier=multiplier,
            rejection_reason=reason,
        )

    return SpreadGateDecision(
        can_open=True,
        current_spread=current_spread,
        baseline_median=baseline,
        threshold_multiplier=multiplier,
        rejection_reason=None,
    )

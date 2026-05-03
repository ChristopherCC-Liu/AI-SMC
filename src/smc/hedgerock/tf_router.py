"""Timeframe router — picks the single TF the EA may trade right now.

The user's core constraint is "**one timeframe at a time, one strategy
per timeframe, never let multiple TF strategies fight each other**".
This module produces that single `active_timeframe` decision based on
volatility regime + trend strength.

Decision tree (XAUUSD-tuned, revisit for other instruments):

    +-------------------------------+--------------------+
    | Condition                     | active_timeframe   |
    +-------------------------------+--------------------+
    | vol_rank ≥ 0.85 (extreme)     | M5  (fast scalp)   |
    | vol_rank ≤ 0.15 (dead)        | H4  (avoid noise)  |
    | strong directional trend      | H1  (trend follow) |
    | otherwise (mid-vol, no trend) | M15 (consolidation)|
    +-------------------------------+--------------------+

A "strong directional trend" is:
  - |hh_count - ll_count| ≥ 5  (clear bias in last N swings)
  - AND h4_trend_bars ≥ 4      (price holding direction vs SMA50)

The router returns a `TimeframeRoute` with `confidence` (0..1) and a
human-readable `reason` so the journal/dashboard can explain the
choice. Confidence is a coarse heuristic, not a calibrated probability.
"""

from __future__ import annotations

from dataclasses import dataclass

from smc.hedgerock.schemas import Timeframe


__all__ = [
    "DEAD_VOL_THRESHOLD",
    "EXTREME_VOL_THRESHOLD",
    "TREND_BAR_FLOOR",
    "TREND_SWING_DELTA",
    "TimeframeRoute",
    "route_timeframe",
]


# Thresholds — exposed as module constants so tests can sanity-check
# them and downstream tuning can adjust without touching logic.

EXTREME_VOL_THRESHOLD: float = 0.85
"""ATR percentile rank at/above which we consider volatility extreme."""

DEAD_VOL_THRESHOLD: float = 0.15
"""ATR percentile rank at/below which volatility is considered dead."""

TREND_SWING_DELTA: int = 5
"""|hh - ll| at/above which we treat the swing pattern as directional."""

TREND_BAR_FLOOR: int = 4
"""h4_trend_bars at/above which the trend has held long enough to act on."""


@dataclass(frozen=True)
class TimeframeRoute:
    """Result of a routing decision."""

    timeframe: Timeframe
    confidence: float  # 0..1, higher = more decisive
    reason: str  # human-readable explanation


def route_timeframe(
    *,
    volatility_rank: float,
    hh_count: int,
    ll_count: int,
    h4_trend_bars: int,
) -> TimeframeRoute:
    """Pick the active timeframe given current market features.

    All keyword-only to avoid call-site ambiguity (every arg has units
    that look interchangeable as positional ints/floats).

    Args:
        volatility_rank: ATR percentile vs recent history. Range [0, 1].
        hh_count: count of higher-highs in the recent swing window.
        ll_count: count of lower-lows in the recent swing window.
        h4_trend_bars: consecutive H4 bars closing in same direction
            relative to SMA50.

    Raises:
        ValueError: if `volatility_rank` is outside [0, 1] or counts
            are negative.
    """
    if not 0.0 <= volatility_rank <= 1.0:
        raise ValueError(
            f"volatility_rank must be in [0, 1], got {volatility_rank}"
        )
    if hh_count < 0 or ll_count < 0:
        raise ValueError(f"hh_count/ll_count must be >= 0, got {hh_count}/{ll_count}")
    if h4_trend_bars < 0:
        raise ValueError(f"h4_trend_bars must be >= 0, got {h4_trend_bars}")

    # 1. Extreme volatility → fast TF (catch the move, scalp out fast).
    if volatility_rank >= EXTREME_VOL_THRESHOLD:
        return TimeframeRoute(
            timeframe="M5",
            confidence=min(1.0, 0.6 + (volatility_rank - EXTREME_VOL_THRESHOLD) * 2),
            reason=(
                f"extreme volatility (rank={volatility_rank:.2f} ≥ "
                f"{EXTREME_VOL_THRESHOLD}) → M5 fast scalp"
            ),
        )

    # 2. Dead volatility → slow TF (avoid noise grinding the account).
    if volatility_rank <= DEAD_VOL_THRESHOLD:
        return TimeframeRoute(
            timeframe="H4",
            confidence=min(1.0, 0.6 + (DEAD_VOL_THRESHOLD - volatility_rank) * 2),
            reason=(
                f"dead volatility (rank={volatility_rank:.2f} ≤ "
                f"{DEAD_VOL_THRESHOLD}) → H4 to avoid noise"
            ),
        )

    # 3. Strong directional trend → H1 trend-follow.
    swing_delta = abs(hh_count - ll_count)
    if swing_delta >= TREND_SWING_DELTA and h4_trend_bars >= TREND_BAR_FLOOR:
        # Confidence scales with how much the trend exceeds the floor.
        excess = (swing_delta - TREND_SWING_DELTA) + (h4_trend_bars - TREND_BAR_FLOOR)
        confidence = min(1.0, 0.65 + 0.05 * excess)
        return TimeframeRoute(
            timeframe="H1",
            confidence=confidence,
            reason=(
                f"directional trend (|hh-ll|={swing_delta}, "
                f"h4_trend_bars={h4_trend_bars}) → H1 trend-follow"
            ),
        )

    # 4. Default — mid-volatility, no trend → M15 consolidation play.
    return TimeframeRoute(
        timeframe="M15",
        confidence=0.55,
        reason=(
            f"mid volatility (rank={volatility_rank:.2f}), "
            f"no clear trend (|hh-ll|={swing_delta}, "
            f"h4_trend_bars={h4_trend_bars}) → M15 consolidation"
        ),
    )

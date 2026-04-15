"""Liquidity level detection and sweep tracking for XAUUSD SMC analysis.

Equal highs and equal lows represent pooled buy/sell-side liquidity that
smart money targets for stop-hunt runs.  This module groups swing points
within a price tolerance band and tracks when the level is swept.

XAUUSD point convention: 1 point = $0.01.  Default tolerance is 5 points = $0.05.
"""

from __future__ import annotations

from datetime import datetime
from itertools import combinations

import polars as pl

from smc.smc_core._utils import to_aware_utc
from smc.smc_core.constants import XAUUSD_POINT_SIZE
from smc.smc_core.types import LiquidityLevel, SwingPoint

__all__ = [
    "detect_liquidity_levels",
    "detect_liquidity_sweep",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _group_swings_by_price(
    swings: list[SwingPoint],
    tolerance: float,
) -> list[list[SwingPoint]]:
    """Group swings into clusters where every member is within *tolerance* of each other.

    Uses a single-pass greedy clustering: swings are sorted by price and
    consecutive swings within tolerance are grouped.
    """
    if not swings:
        return []

    sorted_swings = sorted(swings, key=lambda s: s.price)
    groups: list[list[SwingPoint]] = [[sorted_swings[0]]]

    for swing in sorted_swings[1:]:
        # Check if this swing fits within the current group's price range
        group_min = min(s.price for s in groups[-1])
        group_max = max(s.price for s in groups[-1])
        if swing.price - group_min <= tolerance and swing.price - group_max <= tolerance:
            groups[-1].append(swing)
        else:
            groups.append([swing])

    return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_liquidity_levels(
    df: pl.DataFrame,
    swings: tuple[SwingPoint, ...],
    *,
    tolerance_points: float = 5.0,
) -> tuple[LiquidityLevel, ...]:
    """Identify equal-highs and equal-lows liquidity clusters.

    Two or more swing points of the same type (high or low) within
    ``tolerance_points`` of each other form a liquidity level.  The
    representative price of the level is the average of all clustered swing
    prices.

    Parameters
    ----------
    df:
        Polars OHLCV DataFrame (reserved for future trendline detection;
        not used for equal-high/low detection).
    swings:
        Detected swing points, typically from ``detect_swings``.
    tolerance_points:
        Maximum price distance (in points) for two swings to be considered
        "equal".  For XAUUSD: 5 points = $0.05.  Defaults to 5.0.

    Returns
    -------
    tuple[LiquidityLevel, ...]
        Immutable tuple of detected liquidity levels, ordered by price
        ascending.
    """
    if not swings:
        return ()

    tolerance = tolerance_points * XAUUSD_POINT_SIZE

    swing_highs = [s for s in swings if s.swing_type == "high"]
    swing_lows = [s for s in swings if s.swing_type == "low"]

    levels: list[LiquidityLevel] = []

    for swing_group, level_type_str in [
        (swing_highs, "equal_highs"),
        (swing_lows, "equal_lows"),
    ]:
        if len(swing_group) < 2:
            continue

        clusters = _group_swings_by_price(swing_group, tolerance)
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            avg_price = sum(s.price for s in cluster) / len(cluster)
            levels.append(
                LiquidityLevel(
                    price=avg_price,
                    level_type=level_type_str,  # type: ignore[arg-type]
                    touches=len(cluster),
                    swept=False,
                    swept_at=None,
                )
            )

    return tuple(sorted(levels, key=lambda lv: lv.price))


def detect_liquidity_sweep(
    levels: tuple[LiquidityLevel, ...],
    current_bar_high: float,
    current_bar_low: float,
    current_ts: datetime,
) -> tuple[LiquidityLevel, ...]:
    """Return updated levels where price has swept through the level.

    A liquidity sweep occurs when:
    - For **equal_highs**: ``current_bar_high > level.price`` (price pierces
      above the cluster of equal highs, triggering stop-loss orders above).
    - For **equal_lows**: ``current_bar_low < level.price`` (price pierces
      below the cluster of equal lows, triggering stop-loss orders below).

    Already-swept levels are left unchanged.

    Parameters
    ----------
    levels:
        Current tuple of liquidity levels.
    current_bar_high:
        The high of the bar being evaluated.
    current_bar_low:
        The low of the bar being evaluated.
    current_ts:
        Timestamp of the current bar.

    Returns
    -------
    tuple[LiquidityLevel, ...]
        New immutable tuple with ``swept`` and ``swept_at`` fields updated
        where applicable.
    """
    if not levels:
        return levels

    ts_aware = to_aware_utc(current_ts)

    updated: list[LiquidityLevel] = []
    for level in levels:
        if level.swept:
            updated.append(level)
            continue

        swept = False
        if level.level_type == "equal_highs" and current_bar_high > level.price:
            swept = True
        elif level.level_type == "equal_lows" and current_bar_low < level.price:
            swept = True
        elif level.level_type == "trendline":
            # Trendline: swept if price closes on either side; use midpoint convention.
            if current_bar_high > level.price or current_bar_low < level.price:
                swept = True

        if swept:
            updated.append(level.model_copy(update={"swept": True, "swept_at": ts_aware}))
        else:
            updated.append(level)

    return tuple(updated)

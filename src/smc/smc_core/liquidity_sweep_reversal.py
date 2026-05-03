"""Liquidity-sweep + SMC-confluence reversal detector.

Phase 5 议题 1 (cross-debate consensus). Builds on the existing
:func:`smc.smc_core.liquidity.detect_liquidity_sweep` (which only
flags the moment a liquidity zone is pierced) by looking for the
**reversal pattern** Aldo describes — sweep takes out stops, then
price reverses back through the sweep zone, leaving a candle pattern
plus SMC confluence (FVG / OB nearby).

This is a deterministic rule engine (per cross-system-lessons.md A3
"reverse_pf<1.0 = real edge"). No ML. The shape is intentionally
simple so it can be:

- Unit tested on hand-crafted bar fixtures.
- Tuned by changing scalar thresholds (no model retraining).
- Audited live: every component of ``confidence`` traces back to a
  named bar / pattern.

Signal flow per call:

    1. Caller has already run swing detection + ``detect_liquidity_levels``
       + ``detect_liquidity_sweep`` on the recent bars.
    2. We look at the **most-recently swept** zones (within
       ``lookback_bars`` of the current bar).
    3. For each swept zone, scan forward bars for a reversal pattern
       (engulfing / pin bar / structure-break opposite to the sweep).
    4. If ``confluence_required``, additionally require an FVG or OB
       within ``confluence_distance_pts`` of the reversal bar's price.
    5. Pick the highest-confidence candidate and return it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import polars as pl

from smc.smc_core.constants import XAUUSD_POINT_SIZE
from smc.smc_core.types import FairValueGap, LiquidityLevel, OrderBlock


__all__ = [
    "DEFAULT_LOOKBACK_BARS",
    "DEFAULT_MIN_REVERSAL_ATR",
    "DEFAULT_CONFLUENCE_DISTANCE_PTS",
    "LiquiditySweepReversal",
    "detect_liquidity_sweep_reversal",
    "is_engulfing_bar",
    "is_pin_bar",
]


DEFAULT_LOOKBACK_BARS: int = 10
"""Bars after the sweep to scan for the reversal pattern.

Aldo describes the SL hunt as "the run after the pierce" — typically
within 5-15 M5 bars. 10 is the lead default and balances false
positives (longer windows) vs missed reversals (shorter).
"""

DEFAULT_MIN_REVERSAL_ATR: float = 0.5
"""Minimum reversal-bar body / wick height as multiples of recent ATR.

Filters out tiny indecision bars that happen to engulf-by-pixel; we
only count a reversal when the bar carries real momentum.
"""

DEFAULT_CONFLUENCE_DISTANCE_PTS: float = 50.0
"""Maximum distance (in instrument points) between reversal bar low/high
and an FVG or OB to count as SMC confluence.

XAUUSD: 50 pts = $0.50 — roughly one M5 bar's range in normal vol.
"""


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiquiditySweepReversal:
    """Detector output. Field set verbatim from lead [GO]."""

    active: bool
    direction: Literal["bullish_reversal", "bearish_reversal"] | None
    distance_pts: float | None
    confidence: float | None
    sweep_event: LiquidityLevel | None
    reversal_bar_idx: int | None


# ---------------------------------------------------------------------------
# Reversal pattern primitives (pure, easy to test)
# ---------------------------------------------------------------------------


def is_engulfing_bar(
    *,
    prev_open: float,
    prev_close: float,
    curr_open: float,
    curr_close: float,
    direction: Literal["bullish", "bearish"],
) -> bool:
    """True if ``curr`` engulfs ``prev`` in the given direction.

    Bullish engulfing: prev is bearish (close < open) and curr is
    bullish (close > open) with curr's body fully covering prev's body.

    The classical definition allows wicks outside the body — we keep
    that to maximise recall.
    """
    if direction == "bullish":
        prev_bearish = prev_close < prev_open
        curr_bullish = curr_close > curr_open
        body_engulfs = curr_open <= prev_close and curr_close >= prev_open
        return prev_bearish and curr_bullish and body_engulfs
    # bearish engulfing
    prev_bullish = prev_close > prev_open
    curr_bearish = curr_close < curr_open
    body_engulfs = curr_open >= prev_close and curr_close <= prev_open
    return prev_bullish and curr_bearish and body_engulfs


def is_pin_bar(
    *,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    direction: Literal["bullish", "bearish"],
    wick_to_body_ratio: float = 2.0,
) -> bool:
    """True if the bar is a pin (long wick, small body) pointing the right way.

    Bullish pin: long lower wick (rejected lower prices, expecting
    bounce up). Bearish pin: long upper wick.
    """
    body = abs(bar_close - bar_open)
    if body <= 0:
        # Doji-class — treat as ambiguous, not a pin signal.
        return False

    upper_wick = bar_high - max(bar_open, bar_close)
    lower_wick = min(bar_open, bar_close) - bar_low

    if direction == "bullish":
        return lower_wick >= wick_to_body_ratio * body and lower_wick > upper_wick
    return upper_wick >= wick_to_body_ratio * body and upper_wick > lower_wick


def _bar_atr(bars: pl.DataFrame, *, lookback: int = 14) -> float:
    """Crude ATR over the last ``lookback`` bars. Fallback to range mean."""
    n = bars.height
    if n == 0:
        return 0.0
    take = min(lookback, n)
    recent = bars.tail(take)
    highs = recent["high"].to_list()
    lows = recent["low"].to_list()
    return sum(h - l for h, l in zip(highs, lows)) / take


# ---------------------------------------------------------------------------
# Confluence
# ---------------------------------------------------------------------------


def _has_confluence(
    *,
    reversal_price: float,
    direction: Literal["bullish_reversal", "bearish_reversal"],
    fvgs: tuple[FairValueGap, ...],
    order_blocks: tuple[OrderBlock, ...],
    distance_pts: float,
) -> tuple[bool, float]:
    """Check if a same-direction FVG or OB sits within distance of price.

    Returns ``(has_confluence, score_0_to_1)`` where the score is 1.0
    if either pattern sits exactly at the reversal price and decays
    with distance.
    """
    target_type = "bullish" if direction == "bullish_reversal" else "bearish"
    distance_threshold = distance_pts * XAUUSD_POINT_SIZE

    best_dist = float("inf")

    for fvg in fvgs:
        if fvg.fvg_type != target_type or fvg.fully_filled:
            continue
        # Distance: nearest edge of the FVG band to the reversal price.
        dist = max(0.0, min(abs(reversal_price - fvg.high), abs(reversal_price - fvg.low)))
        if dist < best_dist:
            best_dist = dist

    for ob in order_blocks:
        if ob.ob_type != target_type or ob.mitigated:
            continue
        dist = max(0.0, min(abs(reversal_price - ob.high), abs(reversal_price - ob.low)))
        if dist < best_dist:
            best_dist = dist

    if best_dist > distance_threshold:
        return False, 0.0
    # Linear decay: 1.0 at dist=0, 0.0 at dist=threshold.
    score = max(0.0, 1.0 - best_dist / distance_threshold)
    return True, score


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


def detect_liquidity_sweep_reversal(
    bars: pl.DataFrame,
    swept_zones: tuple[LiquidityLevel, ...],
    *,
    fvgs: tuple[FairValueGap, ...] = (),
    order_blocks: tuple[OrderBlock, ...] = (),
    lookback_bars: int = DEFAULT_LOOKBACK_BARS,
    min_reversal_atr: float = DEFAULT_MIN_REVERSAL_ATR,
    confluence_required: bool = True,
    confluence_distance_pts: float = DEFAULT_CONFLUENCE_DISTANCE_PTS,
) -> LiquiditySweepReversal | None:
    """Look for a sweep+reversal+confluence pattern in the recent window.

    Args:
        bars: Polars OHLCV bars (asc by ts). Must include ``open``,
            ``high``, ``low``, ``close`` columns. Need at least 2 bars
            for engulfing detection.
        swept_zones: Output of ``detect_liquidity_sweep`` — only zones
            with ``swept=True`` and ``swept_at`` populated are
            considered.
        fvgs / order_blocks: SMC confluence candidates from the same
            snapshot. Empty tuples are valid inputs (no confluence).
        lookback_bars: How many bars after each sweep to scan for
            reversal. Default 10.
        min_reversal_atr: Minimum reversal bar body relative to ATR.
        confluence_required: If True, return None when no SMC pattern
            is within ``confluence_distance_pts`` of the reversal bar.
        confluence_distance_pts: Max distance to count an FVG/OB.

    Returns:
        Best ``LiquiditySweepReversal`` (highest confidence) or None
        when no candidate qualifies.
    """
    if bars.height < 2:
        return None
    if not swept_zones:
        return None

    swept_active = tuple(
        z for z in swept_zones if z.swept and z.swept_at is not None
    )
    if not swept_active:
        return None

    atr = _bar_atr(bars)
    bar_count = bars.height
    ts_col = bars["ts"].to_list()
    open_col = bars["open"].to_list()
    high_col = bars["high"].to_list()
    low_col = bars["low"].to_list()
    close_col = bars["close"].to_list()
    last_close = close_col[-1]

    candidates: list[LiquiditySweepReversal] = []

    for zone in swept_active:
        # Direction implied by zone type:
        # equal_highs swept → price spiked up & may reverse down → bearish_reversal
        # equal_lows  swept → price spiked down & may reverse up → bullish_reversal
        if zone.level_type == "equal_highs":
            direction: Literal["bullish_reversal", "bearish_reversal"] = "bearish_reversal"
            reversal_pattern_dir: Literal["bullish", "bearish"] = "bearish"
        elif zone.level_type == "equal_lows":
            direction = "bullish_reversal"
            reversal_pattern_dir = "bullish"
        else:
            # trendline: ambiguous direction; skip.
            continue

        # Find the swept_at index in bars.
        try:
            sweep_idx = next(
                i for i, ts in enumerate(ts_col) if _to_aware(ts) >= _to_aware(zone.swept_at)
            )
        except (StopIteration, ValueError):
            continue

        scan_end = min(bar_count, sweep_idx + lookback_bars + 1)

        for j in range(sweep_idx + 1, scan_end):
            # Reversal pattern check: engulfing or pin bar.
            engulfing = j >= 1 and is_engulfing_bar(
                prev_open=open_col[j - 1],
                prev_close=close_col[j - 1],
                curr_open=open_col[j],
                curr_close=close_col[j],
                direction=reversal_pattern_dir,
            )
            pin = is_pin_bar(
                bar_open=open_col[j],
                bar_high=high_col[j],
                bar_low=low_col[j],
                bar_close=close_col[j],
                direction=reversal_pattern_dir,
            )
            if not (engulfing or pin):
                continue

            # ATR strength filter.
            body = abs(close_col[j] - open_col[j])
            if atr > 0 and body < min_reversal_atr * atr:
                continue

            reversal_price = close_col[j]

            # Confluence (FVG / OB).
            has_conf, conf_score = _has_confluence(
                reversal_price=reversal_price,
                direction=direction,
                fvgs=fvgs,
                order_blocks=order_blocks,
                distance_pts=confluence_distance_pts,
            )
            if confluence_required and not has_conf:
                continue

            # Strength score: how much of an ATR the body covers, capped at 1.0.
            strength_score = min(1.0, body / atr) if atr > 0 else 0.5
            confidence = (strength_score + conf_score) / 2

            distance_pts = abs(last_close - zone.price) / XAUUSD_POINT_SIZE

            candidates.append(
                LiquiditySweepReversal(
                    active=True,
                    direction=direction,
                    distance_pts=distance_pts,
                    confidence=confidence,
                    sweep_event=zone,
                    reversal_bar_idx=j,
                )
            )

    if not candidates:
        return None
    # Highest confidence wins; tie-break on most recent reversal_bar_idx.
    candidates.sort(
        key=lambda c: (c.confidence or 0.0, c.reversal_bar_idx or 0),
        reverse=True,
    )
    return candidates[0]


def _to_aware(value):
    """Coerce a polars timestamp / datetime to a comparable form.

    Polars ts column values come back as ``datetime`` (already
    tz-aware UTC because the lake schema fixes that); ``LiquidityLevel.swept_at``
    is the same. We return them as-is — comparison works directly.
    """
    return value

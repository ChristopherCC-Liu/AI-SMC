"""Shared fixtures for strategy unit tests.

Provides pre-built SMCSnapshot objects for D1, H4, H1, and M15 timeframes
with deterministic structure breaks, order blocks, FVGs, and liquidity levels.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from smc.data.schemas import Timeframe
from smc.smc_core.types import (
    FairValueGap,
    LiquidityLevel,
    OrderBlock,
    SMCSnapshot,
    StructureBreak,
    SwingPoint,
)
from smc.strategy.types import BiasDirection, EntrySignal, TradeZone

# ---------------------------------------------------------------------------
# Base timestamp
# ---------------------------------------------------------------------------

_BASE_TS = datetime(2024, 6, 10, 0, 0, 0, tzinfo=timezone.utc)


def _ts(hours: int) -> datetime:
    return _BASE_TS + timedelta(hours=hours)


# ---------------------------------------------------------------------------
# Reusable swing points
# ---------------------------------------------------------------------------

_BULLISH_SWINGS = (
    SwingPoint(ts=_ts(0), price=2340.00, swing_type="low", strength=10),
    SwingPoint(ts=_ts(4), price=2365.00, swing_type="high", strength=10),
    SwingPoint(ts=_ts(8), price=2345.00, swing_type="low", strength=10),
    SwingPoint(ts=_ts(12), price=2375.00, swing_type="high", strength=10),
    SwingPoint(ts=_ts(16), price=2355.00, swing_type="low", strength=10),
    SwingPoint(ts=_ts(20), price=2385.00, swing_type="high", strength=10),
)

_BEARISH_SWINGS = (
    SwingPoint(ts=_ts(0), price=2390.00, swing_type="high", strength=10),
    SwingPoint(ts=_ts(4), price=2365.00, swing_type="low", strength=10),
    SwingPoint(ts=_ts(8), price=2380.00, swing_type="high", strength=10),
    SwingPoint(ts=_ts(12), price=2350.00, swing_type="low", strength=10),
    SwingPoint(ts=_ts(16), price=2370.00, swing_type="high", strength=10),
    SwingPoint(ts=_ts(20), price=2340.00, swing_type="low", strength=10),
)

_NEUTRAL_SWINGS = (
    SwingPoint(ts=_ts(0), price=2360.00, swing_type="low", strength=10),
    SwingPoint(ts=_ts(4), price=2370.00, swing_type="high", strength=10),
    SwingPoint(ts=_ts(8), price=2362.00, swing_type="low", strength=10),
    SwingPoint(ts=_ts(12), price=2368.00, swing_type="high", strength=10),
)


# ---------------------------------------------------------------------------
# Snapshot builders
# ---------------------------------------------------------------------------


def _make_snapshot(
    *,
    timeframe: Timeframe,
    trend: str = "bullish",
    swing_points: tuple[SwingPoint, ...] = (),
    order_blocks: tuple[OrderBlock, ...] = (),
    fvgs: tuple[FairValueGap, ...] = (),
    structure_breaks: tuple[StructureBreak, ...] = (),
    liquidity_levels: tuple[LiquidityLevel, ...] = (),
) -> SMCSnapshot:
    return SMCSnapshot(
        ts=_ts(24),
        timeframe=timeframe,
        swing_points=swing_points,
        order_blocks=order_blocks,
        fvgs=fvgs,
        structure_breaks=structure_breaks,
        liquidity_levels=liquidity_levels,
        trend_direction=trend,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# D1 Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def d1_bullish_snapshot() -> SMCSnapshot:
    """D1 snapshot with clear bullish trend: consecutive bullish BOS."""
    return _make_snapshot(
        timeframe=Timeframe.D1,
        trend="bullish",
        swing_points=_BULLISH_SWINGS,
        structure_breaks=(
            StructureBreak(ts=_ts(6), price=2365.00, break_type="bos", direction="bullish", timeframe=Timeframe.D1),
            StructureBreak(ts=_ts(14), price=2375.00, break_type="bos", direction="bullish", timeframe=Timeframe.D1),
            StructureBreak(ts=_ts(22), price=2385.00, break_type="bos", direction="bullish", timeframe=Timeframe.D1),
        ),
        order_blocks=(
            OrderBlock(ts_start=_ts(7), ts_end=_ts(8), high=2360.00, low=2355.00, ob_type="bullish", timeframe=Timeframe.D1, mitigated=False),
            OrderBlock(ts_start=_ts(15), ts_end=_ts(16), high=2370.00, low=2365.00, ob_type="bullish", timeframe=Timeframe.D1, mitigated=False),
        ),
        liquidity_levels=(
            LiquidityLevel(price=2340.00, level_type="equal_lows", touches=3, swept=False),
        ),
    )


@pytest.fixture()
def d1_bearish_snapshot() -> SMCSnapshot:
    """D1 snapshot with clear bearish trend: consecutive bearish BOS."""
    return _make_snapshot(
        timeframe=Timeframe.D1,
        trend="bearish",
        swing_points=_BEARISH_SWINGS,
        structure_breaks=(
            StructureBreak(ts=_ts(6), price=2365.00, break_type="bos", direction="bearish", timeframe=Timeframe.D1),
            StructureBreak(ts=_ts(14), price=2350.00, break_type="bos", direction="bearish", timeframe=Timeframe.D1),
            StructureBreak(ts=_ts(22), price=2340.00, break_type="bos", direction="bearish", timeframe=Timeframe.D1),
        ),
        order_blocks=(
            OrderBlock(ts_start=_ts(7), ts_end=_ts(8), high=2385.00, low=2380.00, ob_type="bearish", timeframe=Timeframe.D1, mitigated=False),
        ),
    )


@pytest.fixture()
def d1_neutral_snapshot() -> SMCSnapshot:
    """D1 snapshot with no clear trend."""
    return _make_snapshot(
        timeframe=Timeframe.D1,
        trend="ranging",
        swing_points=_NEUTRAL_SWINGS,
        structure_breaks=(),
    )


# ---------------------------------------------------------------------------
# H4 Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def h4_bullish_snapshot() -> SMCSnapshot:
    """H4 snapshot confirming bullish bias."""
    return _make_snapshot(
        timeframe=Timeframe.H4,
        trend="bullish",
        swing_points=_BULLISH_SWINGS,
        structure_breaks=(
            StructureBreak(ts=_ts(3), price=2360.00, break_type="bos", direction="bullish", timeframe=Timeframe.H4),
            StructureBreak(ts=_ts(11), price=2370.00, break_type="bos", direction="bullish", timeframe=Timeframe.H4),
        ),
        order_blocks=(
            OrderBlock(ts_start=_ts(9), ts_end=_ts(10), high=2355.00, low=2350.00, ob_type="bullish", timeframe=Timeframe.H4, mitigated=False),
        ),
        liquidity_levels=(
            LiquidityLevel(price=2390.00, level_type="equal_highs", touches=2, swept=False),
        ),
    )


@pytest.fixture()
def h4_bearish_snapshot() -> SMCSnapshot:
    """H4 snapshot with bearish bias (conflicts with bullish D1)."""
    return _make_snapshot(
        timeframe=Timeframe.H4,
        trend="bearish",
        swing_points=_BEARISH_SWINGS,
        structure_breaks=(
            StructureBreak(ts=_ts(5), price=2365.00, break_type="bos", direction="bearish", timeframe=Timeframe.H4),
            StructureBreak(ts=_ts(13), price=2350.00, break_type="bos", direction="bearish", timeframe=Timeframe.H4),
        ),
    )


@pytest.fixture()
def h4_neutral_snapshot() -> SMCSnapshot:
    """H4 snapshot with no clear trend."""
    return _make_snapshot(
        timeframe=Timeframe.H4,
        trend="ranging",
        swing_points=_NEUTRAL_SWINGS,
        structure_breaks=(),
    )


# ---------------------------------------------------------------------------
# H1 Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def h1_bullish_snapshot() -> SMCSnapshot:
    """H1 snapshot with bullish OBs and FVGs for zone scanning."""
    return _make_snapshot(
        timeframe=Timeframe.H1,
        trend="bullish",
        swing_points=_BULLISH_SWINGS,
        order_blocks=(
            OrderBlock(ts_start=_ts(2), ts_end=_ts(3), high=2352.00, low=2348.00, ob_type="bullish", timeframe=Timeframe.H1, mitigated=False),
            OrderBlock(ts_start=_ts(10), ts_end=_ts(11), high=2362.00, low=2358.00, ob_type="bullish", timeframe=Timeframe.H1, mitigated=False),
            OrderBlock(ts_start=_ts(18), ts_end=_ts(19), high=2380.00, low=2376.00, ob_type="bearish", timeframe=Timeframe.H1, mitigated=False),
        ),
        fvgs=(
            FairValueGap(ts=_ts(4), high=2356.00, low=2352.00, fvg_type="bullish", timeframe=Timeframe.H1, filled_pct=0.0, fully_filled=False),
            FairValueGap(ts=_ts(12), high=2366.00, low=2362.00, fvg_type="bullish", timeframe=Timeframe.H1, filled_pct=0.3, fully_filled=False),
            FairValueGap(ts=_ts(20), high=2378.00, low=2374.00, fvg_type="bearish", timeframe=Timeframe.H1, filled_pct=0.0, fully_filled=False),
        ),
        structure_breaks=(
            StructureBreak(ts=_ts(6), price=2365.00, break_type="bos", direction="bullish", timeframe=Timeframe.H1),
        ),
        liquidity_levels=(
            LiquidityLevel(price=2340.00, level_type="equal_lows", touches=3, swept=False),
            LiquidityLevel(price=2385.00, level_type="equal_highs", touches=2, swept=False),
        ),
    )


@pytest.fixture()
def h1_empty_snapshot() -> SMCSnapshot:
    """H1 snapshot with no patterns."""
    return _make_snapshot(
        timeframe=Timeframe.H1,
        trend="ranging",
    )


# ---------------------------------------------------------------------------
# M15 Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def m15_choch_in_zone_snapshot() -> SMCSnapshot:
    """M15 snapshot with a bullish CHoCH inside a typical bullish OB zone (2348-2352)."""
    return _make_snapshot(
        timeframe=Timeframe.M15,
        trend="bullish",
        swing_points=(
            SwingPoint(ts=_ts(1), price=2346.00, swing_type="low", strength=5),
            SwingPoint(ts=_ts(2), price=2354.00, swing_type="high", strength=5),
            SwingPoint(ts=_ts(3), price=2349.00, swing_type="low", strength=5),
            SwingPoint(ts=_ts(4), price=2358.00, swing_type="high", strength=5),
        ),
        structure_breaks=(
            StructureBreak(ts=_ts(2), price=2349.50, break_type="choch", direction="bullish", timeframe=Timeframe.M15),
        ),
        fvgs=(
            FairValueGap(ts=_ts(3), high=2353.00, low=2349.00, fvg_type="bullish", timeframe=Timeframe.M15, filled_pct=0.6, fully_filled=False),
        ),
        liquidity_levels=(
            LiquidityLevel(price=2385.00, level_type="equal_highs", touches=2, swept=False),
            LiquidityLevel(price=2340.00, level_type="equal_lows", touches=2, swept=False),
        ),
    )


@pytest.fixture()
def m15_no_trigger_snapshot() -> SMCSnapshot:
    """M15 snapshot with no entry triggers."""
    return _make_snapshot(
        timeframe=Timeframe.M15,
        trend="ranging",
        swing_points=(
            SwingPoint(ts=_ts(1), price=2360.00, swing_type="low", strength=5),
            SwingPoint(ts=_ts(2), price=2365.00, swing_type="high", strength=5),
        ),
    )


# ---------------------------------------------------------------------------
# Pre-built strategy types
# ---------------------------------------------------------------------------


@pytest.fixture()
def bullish_bias() -> BiasDirection:
    return BiasDirection(
        direction="bullish",
        confidence=0.8,
        key_levels=(2340.00, 2355.00, 2365.00, 2375.00, 2385.00, 2390.00),
        rationale="D1 and H4 both bullish — confirmed multi-timeframe bias.",
    )


@pytest.fixture()
def neutral_bias() -> BiasDirection:
    return BiasDirection(
        direction="neutral",
        confidence=0.0,
        key_levels=(2360.00, 2370.00),
        rationale="D1 structure is indeterminate (no clear trend).",
    )


@pytest.fixture()
def bullish_ob_zone() -> TradeZone:
    return TradeZone(
        zone_high=2352.00,
        zone_low=2348.00,
        zone_type="ob",
        direction="long",
        timeframe=Timeframe.H1,
        confidence=0.8,
    )


@pytest.fixture()
def bullish_overlap_zone() -> TradeZone:
    return TradeZone(
        zone_high=2352.00,
        zone_low=2348.00,
        zone_type="ob_fvg_overlap",
        direction="long",
        timeframe=Timeframe.H1,
        confidence=0.95,
    )


@pytest.fixture()
def sample_entry_signal() -> EntrySignal:
    return EntrySignal(
        entry_price=2350.00,
        stop_loss=2347.70,
        take_profit_1=2355.75,
        take_profit_2=2385.00,
        risk_points=230.0,
        reward_points=575.0,
        rr_ratio=2.5,
        trigger_type="choch_in_zone",
        direction="long",
        grade="A",
    )

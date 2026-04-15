"""SMC pattern type definitions for the AI-SMC trading system.

All models are frozen (immutable) Pydantic BaseModel instances.  Timeframe is
imported from the canonical location — smc.data.schemas — to avoid duplication.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from smc.data.schemas import Timeframe

__all__ = [
    "Timeframe",
    "SwingPoint",
    "OrderBlock",
    "FairValueGap",
    "StructureBreak",
    "LiquidityLevel",
    "SMCSnapshot",
]

# ---------------------------------------------------------------------------
# Swing Points
# ---------------------------------------------------------------------------


class SwingPoint(BaseModel):
    """A single confirmed swing high or swing low on the price chart."""

    model_config = ConfigDict(frozen=True)

    ts: datetime
    price: float
    swing_type: Literal["high", "low"]
    strength: int  # number of bars on each side confirming the swing


# ---------------------------------------------------------------------------
# Order Blocks
# ---------------------------------------------------------------------------


class OrderBlock(BaseModel):
    """A rectangular price zone that marks an order block (bullish or bearish)."""

    model_config = ConfigDict(frozen=True)

    ts_start: datetime
    ts_end: datetime
    high: float
    low: float
    ob_type: Literal["bullish", "bearish"]
    timeframe: Timeframe
    mitigated: bool = False
    mitigated_at: datetime | None = None


# ---------------------------------------------------------------------------
# Fair Value Gaps
# ---------------------------------------------------------------------------


class FairValueGap(BaseModel):
    """A price imbalance gap between two non-adjacent candles."""

    model_config = ConfigDict(frozen=True)

    ts: datetime
    high: float
    low: float
    fvg_type: Literal["bullish", "bearish"]
    timeframe: Timeframe
    filled_pct: float = 0.0
    fully_filled: bool = False


# ---------------------------------------------------------------------------
# Structure Breaks
# ---------------------------------------------------------------------------


class StructureBreak(BaseModel):
    """A confirmed Break of Structure (BOS) or Change of Character (CHoCH)."""

    model_config = ConfigDict(frozen=True)

    ts: datetime
    price: float
    break_type: Literal["bos", "choch"]
    direction: Literal["bullish", "bearish"]
    timeframe: Timeframe


# ---------------------------------------------------------------------------
# Liquidity Levels
# ---------------------------------------------------------------------------


class LiquidityLevel(BaseModel):
    """A price level where liquidity is pooled (equal highs/lows or trendline)."""

    model_config = ConfigDict(frozen=True)

    price: float
    level_type: Literal["equal_highs", "equal_lows", "trendline"]
    touches: int
    swept: bool = False
    swept_at: datetime | None = None


# ---------------------------------------------------------------------------
# SMC Snapshot
# ---------------------------------------------------------------------------


class SMCSnapshot(BaseModel):
    """Immutable snapshot of all detected SMC patterns for a single timeframe."""

    model_config = ConfigDict(frozen=True)

    ts: datetime
    timeframe: Timeframe
    swing_points: tuple[SwingPoint, ...]
    order_blocks: tuple[OrderBlock, ...]
    fvgs: tuple[FairValueGap, ...]
    structure_breaks: tuple[StructureBreak, ...]
    liquidity_levels: tuple[LiquidityLevel, ...]
    trend_direction: Literal["bullish", "bearish", "ranging"]

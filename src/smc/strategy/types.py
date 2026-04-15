"""Strategy-layer data types for the AI-SMC multi-timeframe trading system.

All models are frozen (immutable) Pydantic BaseModel instances.
Uses POINTS (not pips) as the base unit — consistent with MT5 internal representation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from smc.data.schemas import Timeframe

__all__ = [
    "BiasDirection",
    "TradeZone",
    "SetupGrade",
    "EntrySignal",
    "TradeSetup",
]

# ---------------------------------------------------------------------------
# HTF Directional Bias
# ---------------------------------------------------------------------------

SetupGrade = Literal["A", "B", "C"]


class BiasDirection(BaseModel):
    """Higher-timeframe directional bias derived from D1 + H4 structure."""

    model_config = ConfigDict(frozen=True)

    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0.0 – 1.0
    key_levels: tuple[float, ...]
    rationale: str


# ---------------------------------------------------------------------------
# H1 Trade Zones
# ---------------------------------------------------------------------------


class TradeZone(BaseModel):
    """A price zone on H1 where institutional footprints overlap."""

    model_config = ConfigDict(frozen=True)

    zone_high: float
    zone_low: float
    zone_type: Literal["ob", "fvg", "ob_fvg_overlap"]
    direction: Literal["long", "short"]
    timeframe: Timeframe
    confidence: float  # 0.0 – 1.0


# ---------------------------------------------------------------------------
# M15 Entry Signal
# ---------------------------------------------------------------------------


class EntrySignal(BaseModel):
    """Precise entry parameters generated from M15 price action inside an H1 zone."""

    model_config = ConfigDict(frozen=True)

    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_points: float
    reward_points: float
    rr_ratio: float
    trigger_type: Literal["choch_in_zone", "fvg_fill_in_zone", "ob_test_rejection"]
    direction: Literal["long", "short"]
    grade: SetupGrade


# ---------------------------------------------------------------------------
# Complete Trade Setup
# ---------------------------------------------------------------------------


class TradeSetup(BaseModel):
    """A fully-scored trade setup ready for execution or filtering."""

    model_config = ConfigDict(frozen=True)

    entry_signal: EntrySignal
    bias: BiasDirection
    zone: TradeZone
    confluence_score: float  # 0.0 – 1.0
    generated_at: datetime

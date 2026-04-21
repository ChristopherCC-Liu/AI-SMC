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
    "EntrySignalV2",
    "TriggerTypeV2",
    "TradeSetup",
    "TradeSetupV2",
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
    """A price zone on H1 where institutional footprints overlap.

    .. versionchanged:: Round 5 A-track Task #9 (+ R-review refinement)
       Added four ``synthetic_*`` zone types (``synthetic_vwap``,
       ``synthetic_session``, ``synthetic_round``, ``synthetic_prev_week``)
       for ATH breakout synthetic zones when historical OB/FVG zones
       are absent in new-high regimes.  Separate sub-types (not a
       single ``"synthetic"``) so downstream journal / dashboard /
       post-mortem analysis can inspect which generator sourced the
       anchor.
    """

    model_config = ConfigDict(frozen=True)

    zone_high: float
    zone_low: float
    zone_type: Literal[
        "ob",
        "fvg",
        "ob_fvg_overlap",
        # Round 5 A-track Task #9: provenance-tagged synthetic zones.
        "synthetic_vwap",         # M15 VWAP ± 1 std dev band
        "synthetic_session",      # Asian/London/NY session high/low
        "synthetic_round",        # Psychological round-number level
        "synthetic_prev_week",    # Prior ISO-week high/low
    ]
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
    trigger_type: Literal["choch_in_zone", "fvg_fill_in_zone", "ob_test_rejection", "bos_in_zone", "fvg_sweep_continuation"]
    direction: Literal["long", "short"]
    grade: SetupGrade


# ---------------------------------------------------------------------------
# M15 Entry Signal V2 (Dual: normal + inverted + new)
# ---------------------------------------------------------------------------

TriggerTypeV2 = Literal[
    "fvg_fill_in_zone",
    "bos_in_zone",
    "choch_in_zone",
    "ob_breakout",
    "choch_continuation",
    "fvg_sweep_continuation",
]


class EntrySignalV2(BaseModel):
    """V2 entry signal supporting normal, inverted, and new trigger types."""

    model_config = ConfigDict(frozen=True)

    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_points: float
    reward_points: float
    rr_ratio: float
    direction: Literal["long", "short"]
    grade: SetupGrade
    trigger_type: TriggerTypeV2
    entry_mode: Literal["normal", "inverted"]
    inversion_confidence: float  # 0.0-1.0 (1.0 for normal entries)


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


# ---------------------------------------------------------------------------
# V2 Trade Setup (uses AIDirection + EntrySignalV2)
# ---------------------------------------------------------------------------


class TradeSetupV2(BaseModel):
    """V2 trade setup using AI direction and dual-mode entry signals.

    Key differences from v1 TradeSetup:
    - ``ai_direction`` replaces ``bias`` (AIDirection from DirectionEngine)
    - ``entry_signal`` uses ``EntrySignalV2`` (supports normal + inverted modes)
    - ``entry_mode`` surfaces the entry mode for quick filtering
    """

    model_config = ConfigDict(frozen=True)

    entry_signal: EntrySignalV2
    ai_direction: str  # "bullish" | "bearish" | "neutral" — from AIDirection
    ai_confidence: float  # 0.0 – 1.0
    zone: TradeZone
    confluence_score: float  # 0.0 – 1.0
    entry_mode: Literal["normal", "inverted"]
    generated_at: datetime

"""Data types for the ranging / mean-reversion trading mode.

All models are frozen (immutable) Pydantic BaseModel instances.
Uses POINTS (not pips) as the base unit -- consistent with MT5 internal
representation and the rest of the AI-SMC codebase.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from smc.strategy.regime import MarketRegime

__all__ = [
    "RangeBounds",
    "RangeSetup",
    "TradingMode",
]

# ---------------------------------------------------------------------------
# Range Boundaries
# ---------------------------------------------------------------------------


class RangeBounds(BaseModel):
    """Detected horizontal range boundaries on H1."""

    model_config = ConfigDict(frozen=True)

    upper: float
    lower: float
    width_points: float  # (upper - lower) / XAUUSD_POINT_SIZE
    midpoint: float
    detected_at: datetime
    source: Literal["ob_boundaries", "swing_extremes", "donchian_channel"]
    confidence: float  # 0.0 – 1.0
    duration_bars: int = 0  # H1 bars the range has been active; 0 = unknown


# ---------------------------------------------------------------------------
# Range Setup (mean-reversion entry)
# ---------------------------------------------------------------------------


class RangeSetup(BaseModel):
    """A mean-reversion trade setup generated at a range boundary."""

    model_config = ConfigDict(frozen=True)

    direction: Literal["long", "short"]
    entry_price: float
    stop_loss: float
    take_profit: float  # conservative: midpoint
    take_profit_ext: float  # aggressive: opposite boundary -10%
    risk_points: float
    reward_points: float
    rr_ratio: float
    range_bounds: RangeBounds
    confidence: float  # 0.0 – 1.0
    trigger: Literal["support_bounce", "resistance_rejection", "midpoint_fade"]
    grade: Literal["A", "B", "C"]


# ---------------------------------------------------------------------------
# Trading Mode (trending vs ranging router output)
# ---------------------------------------------------------------------------


class TradingMode(BaseModel):
    """The active trading mode selected by the mode router."""

    model_config = ConfigDict(frozen=True)

    mode: Literal["trending", "ranging", "v1_passthrough"]
    reason: str
    ai_direction: str  # "bullish" | "bearish" | "neutral"
    ai_confidence: float  # 0.0 – 1.0
    regime: MarketRegime  # "trending" | "transitional" | "ranging"
    range_bounds: RangeBounds | None = None
    # Round 7 P0-1: telemetry tag for downstream observability.
    # None = gate OFF or legacy Priority 1-3 path took the decision.
    # Examples:
    #   "forced_trending_by_TREND_UP_conf_0.85"
    #   "consolidation_to_ranging_conf_0.70"
    #   "consolidation_preconditions_missing_conf_0.70"
    #   "transition_to_v1_conf_0.72"
    #   "transition_momentum_follow_conf_0.70"
    #   "fell_through_low_conf_0.55"
    #   "fell_through_gate_off"  (populated by the caller when gate is OFF)
    ai_regime_decision: str | None = None

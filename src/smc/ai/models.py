"""Frozen Pydantic types for Sprint 6 AI regime classification.

All models are immutable (frozen=True).  These types form the data contract
between the regime classifier, parameter router, debate pipeline, and
the aggregator integration layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

__all__ = [
    "MarketRegimeAI",
    "RegimeParams",
    "AIRegimeAssessment",
    "RegimeState",
    "AnalystView",
    "DebateRound",
    "JudgeVerdict",
    "ExternalContext",
    "AIDirection",
    "H4TechnicalContext",
]

# ---------------------------------------------------------------------------
# Regime enum
# ---------------------------------------------------------------------------

MarketRegimeAI = Literal[
    "TREND_UP",
    "TREND_DOWN",
    "CONSOLIDATION",
    "TRANSITION",
    "ATH_BREAKOUT",
]

# ---------------------------------------------------------------------------
# Strategy parameter preset
# ---------------------------------------------------------------------------


class RegimeParams(BaseModel):
    """Frozen parameter preset for a given market regime.

    Each regime maps to a unique set of strategy parameters that control
    stop-loss width, take-profit targets, confluence thresholds, position
    limits, zone cooldowns, and trigger gating.
    """

    model_config = ConfigDict(frozen=True)

    sl_atr_multiplier: float
    tp1_rr: float
    confluence_floor: float
    allowed_directions: tuple[str, ...]
    allowed_triggers: tuple[str, ...]
    max_concurrent: int
    zone_cooldown_hours: int
    enable_ob_test: bool
    regime_label: str


# ---------------------------------------------------------------------------
# Primary output
# ---------------------------------------------------------------------------


class AIRegimeAssessment(BaseModel):
    """Result of an AI regime classification — the primary output type.

    Contains the classified regime, the frozen parameter preset, the
    classification source (ai_debate / atr_fallback / default), and
    cost tracking metadata.
    """

    model_config = ConfigDict(frozen=True)

    regime: MarketRegimeAI
    trend_direction: Literal["bullish", "bearish", "neutral"]
    confidence: float
    param_preset: RegimeParams
    reasoning: str
    assessed_at: datetime
    source: Literal["ai_debate", "atr_fallback", "default"]
    cost_usd: float


# ---------------------------------------------------------------------------
# Regime hysteresis state
# ---------------------------------------------------------------------------


class RegimeState(BaseModel):
    """Tracks current regime with hysteresis to prevent flip-flopping.

    Frozen — the aggregator creates a new instance on every
    ``generate_setups()`` call (incrementing ``bars_in_regime`` or
    transitioning to a new regime).
    """

    model_config = ConfigDict(frozen=True)

    current_regime: MarketRegimeAI
    regime_since: datetime
    bars_in_regime: int
    previous_regime: MarketRegimeAI | None
    transition_confidence: float
    consecutive_different_count: int = 0


# ---------------------------------------------------------------------------
# Debate pipeline types
# ---------------------------------------------------------------------------


class AnalystView(BaseModel):
    """Single-domain analysis from one of the 4 analyst agents."""

    model_config = ConfigDict(frozen=True)

    domain: Literal["trend", "zone", "macro", "risk"]
    regime_vote: MarketRegimeAI
    confidence: float
    reasoning: str


class DebateRound(BaseModel):
    """One round of bull vs bear debate."""

    model_config = ConfigDict(frozen=True)

    round_num: int
    bull_argument: str
    bear_argument: str


class JudgeVerdict(BaseModel):
    """Final regime verdict from the judge agent."""

    model_config = ConfigDict(frozen=True)

    regime: MarketRegimeAI
    confidence: float
    decisive_factors: tuple[str, ...]
    reasoning: str


# ---------------------------------------------------------------------------
# External macro context
# ---------------------------------------------------------------------------


class ExternalContext(BaseModel):
    """Snapshot of macro data for gold regime classification.

    All fields are optional — the system degrades gracefully when
    external data sources are unavailable.
    """

    model_config = ConfigDict(frozen=True)

    dxy_direction: Literal["strengthening", "weakening", "flat"]
    dxy_value: float | None = None
    vix_level: float | None = None
    vix_regime: Literal["low", "normal", "elevated", "extreme"] | None = None
    real_rate_10y: float | None = None
    cot_net_spec: float | None = None
    central_bank_stance: Literal["hawkish", "neutral", "dovish"] | None = None
    fetched_at: datetime
    source_quality: Literal["live", "cached", "unavailable"]


# ---------------------------------------------------------------------------
# Direction Engine types (Sprint 7)
# ---------------------------------------------------------------------------


class H4TechnicalContext(BaseModel):
    """Pre-computed H4 technical features for the direction debate.

    Extracted deterministically from H4 OHLCV data. Used by the Judge
    to apply technical confirmation bonuses/penalties.
    """

    model_config = ConfigDict(frozen=True)

    sma50_direction: Literal["up", "down", "flat"]
    sma50_slope: float  # normalized % slope per bar
    higher_highs: int
    lower_lows: int
    bar_count: int  # number of H4 bars used for extraction


class AIDirection(BaseModel):
    """Result of an AI direction assessment — bullish/bearish/neutral.

    This is the primary output of the Direction Engine. The strategy
    uses ``direction`` + ``confidence`` to filter trade entries:
    only take long setups when bullish, shorts when bearish.
    """

    model_config = ConfigDict(frozen=True)

    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float
    key_drivers: tuple[str, ...]
    reasoning: str
    assessed_at: datetime
    source: Literal["ai_debate", "sma_fallback", "cache", "neutral_default"]
    cost_usd: float = 0.0
    reasoning_tag: str | None = None  # e.g. "macro_free_capped", "analyst_disagreement"

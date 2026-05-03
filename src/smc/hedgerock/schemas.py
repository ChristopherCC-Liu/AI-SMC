"""Frozen Pydantic schemas for HedgeRock decision envelopes — v2.0.0.

Contract between the Python Decision Center and the HedgeRock_v2 MQL5 EA.

v2.0.0 (dynamic Decision Center):
- All trading parameters are dynamic — TP, SL, grid spacing, recovery
  multiplier, max-lot, max-orders. The EA no longer treats inputs as
  source of truth; inputs become safety bounds (`MaxAllowed*`) and the
  EA clamps every Decision-Center value to those bounds.
- New fields: mode, hedgerock_enabled, cooldown_until, max_next_lot,
  takeprofit_points, stoploss_points, recovery_multiplier,
  max_orders_buy, max_orders_sell, risk_tier, reason.
- Regime taxonomy expanded to include news/breakout/crisis/unknown.
- v1.x fields retained verbatim (envelope is forward-compatible — adding
  fields is OK, removing them is the next major bump).

Design rules:
- All models are frozen (immutable copies via .model_copy(update=...)).
- `MarketRegimeAI` legacy enum (UPPERCASE) is kept for the internal
  features pipeline; the envelope's `regime` field uses the v2 lowercase
  `RegimeV2` enum which is what the EA reads from JSON.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from smc.ai.models import MarketRegimeAI

__all__ = [
    "EXIT_DIRECTIVES",
    "ExitDirective",
    "MODES",
    "Mode",
    "NEWS_DIRECTIONS",
    "NEWS_INTENSITIES",
    "NewsDirection",
    "NewsIntensity",
    "REGIMES_V2",
    "RISK_TIERS",
    "RegimeV2",
    "RiskTier",
    "SCHEMA_VERSION",
    "SUPPORTED_TIMEFRAMES",
    "SignalEnvelope",
    "Timeframe",
    "regime_v1_to_v2",
]

# ---------------------------------------------------------------------------
# Schema version — ticks when EA-visible JSON shape changes.
# v2.0.0 = dynamic Decision Center release (cf. dynamic-decision-center-rfc.md).
# ---------------------------------------------------------------------------

SCHEMA_VERSION: str = "v2.0.0"

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

Timeframe = Literal["M5", "M15", "H1", "H4"]
SUPPORTED_TIMEFRAMES: tuple[Timeframe, ...] = ("M5", "M15", "H1", "H4")

# Strategy mode — what the EA is currently doing.
# - hedgerock: full grid + recovery + hedge (legacy v1 behaviour)
# - momentum: directional momentum (Phase 2+; EA falls back to observe)
# - halt:     close all + block new entries
# - observe:  manage existing positions only, do NOT open new entries
Mode = Literal["hedgerock", "momentum", "halt", "observe"]
MODES: tuple[Mode, ...] = ("hedgerock", "momentum", "halt", "observe")

# Risk tier — semantic label for logging / reason. Actual sizing is driven by
# the numeric fields (lot_factor, max_next_lot, max_orders_*).
RiskTier = Literal["observe", "normal", "aggressive", "attack"]
RISK_TIERS: tuple[RiskTier, ...] = ("observe", "normal", "aggressive", "attack")

# v2 regime taxonomy — lowercase, expanded with news/breakout/crisis/unknown.
RegimeV2 = Literal[
    "range",
    "trend_up",
    "trend_down",
    "news",
    "breakout",
    "crisis",
    "unknown",
]
REGIMES_V2: tuple[RegimeV2, ...] = (
    "range",
    "trend_up",
    "trend_down",
    "news",
    "breakout",
    "crisis",
    "unknown",
)

ExitDirective = Literal[
    "none",
    "urgent_take_profit",
    "halt_and_close_all",
    "news_trade_window",
]
EXIT_DIRECTIVES: tuple[ExitDirective, ...] = (
    "none",
    "urgent_take_profit",
    "halt_and_close_all",
    "news_trade_window",
)

NewsIntensity = Literal["none", "low", "medium", "high"]
NEWS_INTENSITIES: tuple[NewsIntensity, ...] = ("none", "low", "medium", "high")

NewsDirection = Literal["with", "against", "neutral"]
NEWS_DIRECTIONS: tuple[NewsDirection, ...] = ("with", "against", "neutral")


# ---------------------------------------------------------------------------
# Legacy regime mapper
# ---------------------------------------------------------------------------

_LEGACY_TO_V2: dict[str, RegimeV2] = {
    "TREND_UP": "trend_up",
    "TREND_DOWN": "trend_down",
    "CONSOLIDATION": "range",
    "TRANSITION": "unknown",
    "ATH_BREAKOUT": "breakout",
}


def regime_v1_to_v2(legacy: MarketRegimeAI | None) -> RegimeV2 | None:
    """Convert legacy UPPERCASE regime to v2.0.0 lowercase enum.

    Returns None for None input. Unknown legacy values map to ``"unknown"``.
    """
    if legacy is None:
        return None
    return _LEGACY_TO_V2.get(legacy, "unknown")


# ---------------------------------------------------------------------------
# Primary contract
# ---------------------------------------------------------------------------


class SignalEnvelope(BaseModel):
    """Single source of truth the MQL5 EA reads on each OnTimer tick.

    v2.0.0 fields drive every dynamic trading parameter. The EA inputs
    are interpreted as **safety bounds only** — Decision Center values
    must be clamped on the EA side before use.

    Forward-compat: future minor versions may add fields; the EA's JSON
    parser falls back to safe defaults for any missing field.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- identity ----------------------------------------------------------
    symbol: str = Field(..., min_length=1)
    schema_version: str = Field(default=SCHEMA_VERSION)
    generated_at: datetime = Field(...)

    # --- routing -----------------------------------------------------------
    active_timeframe: Timeframe = Field(...)
    active_strategy_id: str = Field(..., min_length=1)

    # --- regime + state ----------------------------------------------------
    regime: RegimeV2 = Field(...)
    prev_regime: RegimeV2 | None = Field(default=None)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # --- mode / risk-tier / hedgerock toggle -------------------------------
    # Mode controls EA behaviour; risk_tier is a label for logging.
    mode: Mode = Field(default="observe")
    hedgerock_enabled: bool = Field(default=False)
    risk_tier: RiskTier = Field(default="observe")

    # --- transition + cooldown locks (parallel concepts) -------------------
    # transition_lock_until_ts: regime just changed → short anti-thrash lock.
    # cooldown_until:           risk control (DD/spread/news/loss-streak) → longer lock.
    # The EA takes max(transition_lock_until_ts, cooldown_until) as the
    # final entry-block time.
    transition_lock_until_ts: datetime | None = Field(default=None)
    cooldown_until: datetime | None = Field(default=None)

    # --- exit directive ----------------------------------------------------
    exit_directive: ExitDirective = Field(default="none")

    # --- dynamic numerical parameters --------------------------------------
    # All clamped on the EA side to MaxAllowed* inputs.
    lot_factor: float = Field(default=1.0, ge=0.0, le=5.0)
    grid_multiplier: float = Field(default=1.0, gt=0.0, le=10.0)
    max_next_lot: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description="Per-symbol next-lot ceiling before EA's own MaxAllowedNextLot clamp",
    )
    takeprofit_points: int = Field(
        default=600,
        gt=0,
        description="Dynamic TP in points; EA clamps to [MinDynamicTakeProfitPoints, MaxAllowedTakeProfitPoints]",
    )
    stoploss_points: int = Field(
        default=3900,
        gt=0,
        description="Dynamic SL in points; EA clamps to [MinDynamicStopLossPoints, MaxAllowedStopLossPoints]",
    )
    recovery_multiplier: float = Field(
        default=1.2,
        gt=0.0,
        le=3.0,
        description="Replaces the GearRH stage-1 ratio at runtime",
    )
    max_orders_buy: int = Field(default=2, ge=0, le=20)
    max_orders_sell: int = Field(default=2, ge=0, le=20)

    # --- news context (legacy, kept) ---------------------------------------
    news_intensity: NewsIntensity = Field(default="none")
    news_direction: NewsDirection | None = Field(default=None)
    news_event_name: str | None = Field(default=None, max_length=200)

    # --- liquidity sweep (legacy, kept) ------------------------------------
    liquidity_sweep_active: bool | None = Field(default=None)
    liquidity_sweep_direction: Literal["bullish_reversal", "bearish_reversal"] | None = Field(
        default=None,
    )
    liquidity_sweep_distance_pts: float | None = Field(default=None)
    liquidity_sweep_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    # --- diagnostic --------------------------------------------------------
    reason: str = Field(
        default="",
        max_length=500,
        description="Free-text justification — logged by EA on every change",
    )

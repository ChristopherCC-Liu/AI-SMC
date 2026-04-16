"""Parameter router — maps AI regime enum to frozen strategy parameter presets.

Pure function, no side effects, no LLM calls.  Presets are module-level
constants derived from the architect's v1.1 design spec.

Usage::

    from smc.ai.param_router import route
    params = route("TREND_UP")
    assert params.allowed_directions == ("long",)
"""

from __future__ import annotations

from smc.ai.models import MarketRegimeAI, RegimeParams

__all__ = ["route", "PRESETS"]

# ---------------------------------------------------------------------------
# Regime parameter presets (v1.1 spec)
# ---------------------------------------------------------------------------

# Sprint 5 validated base parameters — DO NOT change these without backtest evidence.
# Regime routing only varies: allowed_directions, allowed_triggers, confluence_floor.
_ALL_TRIGGERS = ("choch_in_zone", "fvg_fill_in_zone", "ob_test_rejection", "bos_in_zone", "fvg_sweep_continuation")
_BASE_SL = 0.75
_BASE_TP1 = 2.5
_BASE_FLOOR = 0.45
_BASE_CONCURRENT = 3
_BASE_COOLDOWN = 24

PRESETS: dict[MarketRegimeAI, RegimeParams] = {
    "TREND_UP": RegimeParams(
        sl_atr_multiplier=_BASE_SL,
        tp1_rr=_BASE_TP1,
        confluence_floor=_BASE_FLOOR,
        allowed_directions=("long",),
        allowed_triggers=_ALL_TRIGGERS,
        max_concurrent=_BASE_CONCURRENT,
        zone_cooldown_hours=_BASE_COOLDOWN,
        enable_ob_test=True,
        regime_label="Trend Up",
    ),
    "TREND_DOWN": RegimeParams(
        sl_atr_multiplier=_BASE_SL,
        tp1_rr=_BASE_TP1,
        confluence_floor=_BASE_FLOOR,
        allowed_directions=("short",),
        allowed_triggers=_ALL_TRIGGERS,
        max_concurrent=_BASE_CONCURRENT,
        zone_cooldown_hours=_BASE_COOLDOWN,
        enable_ob_test=True,
        regime_label="Trend Down",
    ),
    "CONSOLIDATION": RegimeParams(
        sl_atr_multiplier=_BASE_SL,
        tp1_rr=_BASE_TP1,
        confluence_floor=_BASE_FLOOR,
        allowed_directions=("long", "short"),
        allowed_triggers=_ALL_TRIGGERS,
        max_concurrent=_BASE_CONCURRENT,
        zone_cooldown_hours=_BASE_COOLDOWN,
        enable_ob_test=False,
        regime_label="Consolidation",
    ),
    "TRANSITION": RegimeParams(
        sl_atr_multiplier=_BASE_SL,
        tp1_rr=_BASE_TP1,
        confluence_floor=0.55,
        allowed_directions=("long", "short"),
        allowed_triggers=("fvg_fill_in_zone", "bos_in_zone", "fvg_sweep_continuation"),
        max_concurrent=_BASE_CONCURRENT,
        zone_cooldown_hours=_BASE_COOLDOWN,
        enable_ob_test=False,
        regime_label="Transition",
    ),
    "ATH_BREAKOUT": RegimeParams(
        sl_atr_multiplier=_BASE_SL,
        tp1_rr=_BASE_TP1,
        confluence_floor=_BASE_FLOOR,
        allowed_directions=("long",),
        allowed_triggers=_ALL_TRIGGERS,
        max_concurrent=_BASE_CONCURRENT,
        zone_cooldown_hours=_BASE_COOLDOWN,
        enable_ob_test=False,
        regime_label="ATH Breakout",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route(regime: MarketRegimeAI) -> RegimeParams:
    """Return the frozen parameter preset for the given regime.

    Raises ``KeyError`` if the regime is not recognised — this should
    never happen in practice since ``MarketRegimeAI`` is a Literal type.
    """
    return PRESETS[regime]

"""Parameter router — maps AI regime enum to frozen strategy parameter presets.

Pure function, no side effects, no LLM calls.  Presets are module-level
constants derived from the architect's v1.1 design spec.

Usage::

    from smc.ai.param_router import route
    params = route("TREND_UP")
    assert params.allowed_directions == ("long",)
"""

from __future__ import annotations

from typing import NamedTuple

from smc.ai.models import MarketRegimeAI, RegimeParams

__all__ = ["route", "PRESETS", "get_trail_params", "TrailParams", "TRAIL_PRESETS"]


class TrailParams(NamedTuple):
    """Immutable pair of (activate_r, distance_r) for regime-aware trailing SL.

    Round 5 A-track Task #8: the EA reads these per-ticket from the
    /signal response (instead of the global ``TrailActivateR`` /
    ``TrailDistanceR`` inputs) so trend regimes lock profit early
    (0.3R activate) while ATH breakouts give the move room to run
    (0.8R activate, 0.7R trail distance).
    """

    activate_r: float  # start trailing at N × initial_risk profit
    distance_r: float  # trail N × initial_risk behind current price

# ---------------------------------------------------------------------------
# Regime parameter presets (v1.1 spec)
# ---------------------------------------------------------------------------

# Sprint 5 validated base parameters.
# Round 4 v5 regime-aware SL/TP: Sprint 6 previously kept all regimes uniform
# pending backtest evidence. Now differentiated conservatively (±20% deltas)
# with Sprint 5 TRANSITION as the neutral baseline. Rationale per regime:
#   - TREND: wider TP to ride momentum (SL unchanged — trend reversals
#     still break the same ATR buffer as ranging false starts).
#   - CONSOLIDATION: tighter both — range-bound mean-reversion scalps.
#   - ATH_BREAKOUT: slightly wider both — post-breakout vol expansion
#     demands room; TP extended to capture the breakout impulse.
#   - TRANSITION: baseline (Sprint 5) — confluence floor already raised
#     to 0.55, no need to also perturb SL/TP.
_ALL_TRIGGERS = ("choch_in_zone", "fvg_fill_in_zone", "ob_test_rejection", "bos_in_zone", "fvg_sweep_continuation")
_BASE_SL = 0.75
_BASE_TP1 = 2.5
_BASE_FLOOR = 0.45
_BASE_CONCURRENT = 3
_BASE_COOLDOWN = 24

PRESETS: dict[MarketRegimeAI, RegimeParams] = {
    "TREND_UP": RegimeParams(
        sl_atr_multiplier=_BASE_SL,      # 0.75 — trend reversals break at same ATR
        tp1_rr=3.0,                      # +20% vs base — ride the trend
        confluence_floor=_BASE_FLOOR,
        allowed_directions=("long",),
        allowed_triggers=_ALL_TRIGGERS,
        max_concurrent=_BASE_CONCURRENT,
        zone_cooldown_hours=_BASE_COOLDOWN,
        enable_ob_test=True,
        regime_label="Trend Up",
    ),
    "TREND_DOWN": RegimeParams(
        sl_atr_multiplier=_BASE_SL,      # 0.75 (symmetric with TREND_UP)
        tp1_rr=3.0,                      # +20%
        confluence_floor=_BASE_FLOOR,
        allowed_directions=("short",),
        allowed_triggers=_ALL_TRIGGERS,
        max_concurrent=_BASE_CONCURRENT,
        zone_cooldown_hours=_BASE_COOLDOWN,
        enable_ob_test=True,
        regime_label="Trend Down",
    ),
    "CONSOLIDATION": RegimeParams(
        sl_atr_multiplier=0.6,           # -20% — tight scalp in range
        tp1_rr=2.0,                      # -20% — quick mean-reversion exits
        confluence_floor=_BASE_FLOOR,
        allowed_directions=("long", "short"),
        allowed_triggers=_ALL_TRIGGERS,
        max_concurrent=_BASE_CONCURRENT,
        zone_cooldown_hours=_BASE_COOLDOWN,
        enable_ob_test=False,
        regime_label="Consolidation",
    ),
    "TRANSITION": RegimeParams(
        sl_atr_multiplier=_BASE_SL,      # 0.75 — neutral baseline
        tp1_rr=_BASE_TP1,                # 2.5 — neutral baseline
        confluence_floor=0.55,
        allowed_directions=("long", "short"),
        allowed_triggers=("fvg_fill_in_zone", "bos_in_zone", "fvg_sweep_continuation"),
        max_concurrent=_BASE_CONCURRENT,
        zone_cooldown_hours=_BASE_COOLDOWN,
        enable_ob_test=False,
        regime_label="Transition",
    ),
    "ATH_BREAKOUT": RegimeParams(
        sl_atr_multiplier=0.9,           # +20% — survive post-breakout vol
        tp1_rr=3.5,                      # +40% — capture breakout impulse
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


# ---------------------------------------------------------------------------
# Trailing SL presets (Round 5 A-track Task #8)
# ---------------------------------------------------------------------------
#
# 2024 PF collapse (W13 deep dive: avg win $6.98 vs avg loss $19.20) was
# driven by a fixed 50-pt trail behaving as 1/27 of SL in high-ATR periods —
# winners locked in $0.50 while losers gave up the full 1343 pts.  The
# R-proportional trail (Round 4 v5 hotfix) fixed scaling, but still used
# one global preset.  A2 makes the trail regime-aware:
#
#   TREND_UP / TREND_DOWN — lock profit fast (0.3R activate, 0.5R trail).
#       Trends reverse suddenly; once price runs 0.3R in our favour we want
#       to protect that + give 0.5R of breathing room.
#   CONSOLIDATION — tight scalp (0.5R activate, 0.3R trail).  Range-bound
#       mean-reversion exits need to be fast; a wider trail in a choppy
#       market just hands profit back.
#   TRANSITION — baseline (0.5R / 0.5R).  Symmetric, ambiguous regime.
#   ATH_BREAKOUT — give room (0.8R activate, 0.7R trail).  Post-breakout
#       impulse often runs 2–4R; stopping out at 0.3R would strand a
#       winner on the first pullback.

TRAIL_PRESETS: dict[MarketRegimeAI, TrailParams] = {
    "TREND_UP":      TrailParams(activate_r=0.3, distance_r=0.5),
    "TREND_DOWN":    TrailParams(activate_r=0.3, distance_r=0.5),
    "CONSOLIDATION": TrailParams(activate_r=0.5, distance_r=0.3),
    "TRANSITION":    TrailParams(activate_r=0.5, distance_r=0.5),
    "ATH_BREAKOUT":  TrailParams(activate_r=0.8, distance_r=0.7),
}


def get_trail_params(regime: MarketRegimeAI) -> TrailParams:
    """Return the regime-specific (activate_r, distance_r) for trailing SL.

    The strategy_server emits these in the ``/signal`` response per-leg
    so the EA can set per-ticket activate/distance at position open.
    Downstream backward-compat: if the EA sees missing fields (old
    server), it falls back to its input defaults.
    """
    return TRAIL_PRESETS[regime]

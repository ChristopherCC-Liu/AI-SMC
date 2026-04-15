"""Multi-factor confluence scoring for trade setups.

Evaluates the confluence of multiple SMC factors to produce a final
quality score. Only setups scoring >= 0.45 are considered tradeable.

Weight distribution:
- HTF alignment:     0.25 (D1+H4 bias strength)
- Zone quality:      0.25 (zone type, unmitigated status)
- Entry trigger:     0.20 (trigger type quality)
- RR ratio:          0.15 (risk-reward attractiveness)
- Liquidity context: 0.15 (proximity to liquidity pools)
"""

from __future__ import annotations

from smc.strategy.types import BiasDirection, EntrySignal, TradeZone

__all__ = ["score_confluence"]

# ---------------------------------------------------------------------------
# Weight constants
# ---------------------------------------------------------------------------

_W_HTF_ALIGNMENT = 0.25
_W_ZONE_QUALITY = 0.25
_W_ENTRY_TRIGGER = 0.20
_W_RR_RATIO = 0.15
_W_LIQUIDITY = 0.15

# Minimum tradeable score — lowered from 0.6 to 0.45 to increase
# trade frequency while still filtering low-quality setups.
TRADEABLE_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------


def _score_htf_alignment(bias: BiasDirection) -> float:
    """Score the HTF alignment component (0.0 – 1.0).

    Directly uses the bias confidence, which already encodes D1+H4 agreement.
    """
    if bias.direction == "neutral":
        return 0.0
    return bias.confidence


def _score_zone_quality(zone: TradeZone) -> float:
    """Score the zone quality component (0.0 – 1.0).

    OB+FVG overlap > OB > FVG. The zone's own confidence is factored in.
    """
    type_scores = {
        "ob_fvg_overlap": 1.0,
        "ob": 0.7,
        "fvg": 0.4,
    }
    type_score = type_scores.get(zone.zone_type, 0.3)

    # Combine type quality with zone confidence
    return type_score * 0.5 + zone.confidence * 0.5


def _score_entry_trigger(entry: EntrySignal) -> float:
    """Score the entry trigger quality (0.0 – 1.0).

    CHoCH in zone is the strongest confirmation, followed by FVG fill, then OB rejection.
    Grade is also factored in.
    """
    trigger_scores = {
        "choch_in_zone": 0.9,
        "fvg_fill_in_zone": 0.7,
        "bos_in_zone": 0.55,
        "ob_test_rejection": 0.5,
    }
    trigger_score = trigger_scores.get(entry.trigger_type, 0.3)

    grade_scores = {"A": 1.0, "B": 0.7, "C": 0.4}
    grade_score = grade_scores.get(entry.grade, 0.3)

    return trigger_score * 0.6 + grade_score * 0.4


def _score_rr_ratio(entry: EntrySignal) -> float:
    """Score the risk-reward ratio (0.0 – 1.0).

    RR >= 3.0 → 1.0, RR >= 2.0 → 0.7, RR >= 1.5 → 0.4, else → 0.2
    """
    if entry.rr_ratio >= 3.0:
        return 1.0
    if entry.rr_ratio >= 2.0:
        return 0.7
    if entry.rr_ratio >= 1.5:
        return 0.4
    return 0.2


def _score_liquidity_context(
    bias: BiasDirection,
    zone: TradeZone,
    entry: EntrySignal,
) -> float:
    """Score the liquidity context (0.0 – 1.0).

    Factors:
    - Number of key levels from bias (more levels → better market structure)
    - TP2 reaching a liquidity level (entry_trigger sets TP2 to liq level if found)
    - Zone type overlap (OB+FVG overlap suggests institutional intent)
    """
    score = 0.0

    # Key levels density: more levels = better mapped structure
    num_levels = len(bias.key_levels)
    if num_levels >= 6:
        score += 0.4
    elif num_levels >= 3:
        score += 0.25
    elif num_levels >= 1:
        score += 0.1

    # TP2 vs TP1 spread: if TP2 >> TP1 it likely hit a liquidity level
    if entry.take_profit_2 != entry.take_profit_1:
        tp2_distance = abs(entry.take_profit_2 - entry.entry_price)
        tp1_distance = abs(entry.take_profit_1 - entry.entry_price)
        if tp1_distance > 0 and tp2_distance / tp1_distance >= 1.3:
            score += 0.3

    # Zone type bonus
    if zone.zone_type == "ob_fvg_overlap":
        score += 0.3
    elif zone.zone_type == "ob":
        score += 0.15

    return min(1.0, score)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_confluence(
    bias: BiasDirection,
    zone: TradeZone,
    entry: EntrySignal,
) -> float:
    """Compute a multi-factor confluence score for a trade setup.

    Parameters
    ----------
    bias:
        HTF directional bias from ``compute_htf_bias``.
    zone:
        H1 trade zone from ``scan_zones``.
    entry:
        M15 entry signal from ``check_entry``.

    Returns
    -------
    float
        Confluence score in [0.0, 1.0]. Only setups >= 0.45 are tradeable.
    """
    htf_score = _score_htf_alignment(bias)
    zone_score = _score_zone_quality(zone)
    trigger_score = _score_entry_trigger(entry)
    rr_score = _score_rr_ratio(entry)
    liq_score = _score_liquidity_context(bias, zone, entry)

    raw = (
        _W_HTF_ALIGNMENT * htf_score
        + _W_ZONE_QUALITY * zone_score
        + _W_ENTRY_TRIGGER * trigger_score
        + _W_RR_RATIO * rr_score
        + _W_LIQUIDITY * liq_score
    )

    return round(min(1.0, max(0.0, raw)), 3)

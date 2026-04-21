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
_W_ENTRY_TRIGGER = 0.25  # Sprint 4: raised from 0.20 (low variance, needs more influence)
_W_RR_RATIO = 0.15
_W_LIQUIDITY = 0.10      # Sprint 4: lowered from 0.15 (redundant with RR, r=0.534)

# Minimum tradeable score — lowered from 0.6 to 0.45 to increase
# trade frequency while still filtering low-quality setups.
TRADEABLE_THRESHOLD = 0.45

# Tier-gated confluence floors: Tier 2 (H4-only) requires higher
# confluence since it lacks D1 confirmation.  Tier 3 (D1-only) uses
# an even higher floor since it lacks the more responsive H4 signal.
TIER2_CONFLUENCE_FLOOR = 0.55
TIER3_CONFLUENCE_FLOOR = 0.55

# Sprint 5: Transitional regime confluence floor — raised from implicit 0.45
# to 0.60 to filter low-quality setups in ambiguous volatility conditions.
# v5 data: transitional had 34 trades at 32.4% WR, -$202.
TRANSITIONAL_CONFLUENCE_FLOOR = 0.60


def effective_threshold(bias_rationale: str) -> float:
    """Return the tier-based confluence threshold.

    The bias rationale string from ``compute_htf_bias`` starts with
    "Tier N:" which encodes the tier level.

    .. versionchanged:: Sprint 6
       Removed ``regime`` parameter.  Regime-based confluence floor is
       now provided by ``RegimeParams.confluence_floor`` from the AI
       regime classifier.  The aggregator takes ``max(tier_floor, regime_floor)``.
    """
    if bias_rationale.startswith("Tier 2:"):
        return TIER2_CONFLUENCE_FLOOR
    if bias_rationale.startswith("Tier 3:"):
        return TIER3_CONFLUENCE_FLOOR
    return TRADEABLE_THRESHOLD


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
        # Round 5 A-track Task #9: ATH synthetic zones (VWAP / session H/L
        # / round numbers / prev week H/L).  Weaker than historical OB/FVG
        # — these are derived, not traded-through.  Kept above the default
        # fallback so they still clear the tradeable threshold when
        # confluence_score is otherwise strong.
        "synthetic": 0.35,
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
        "fvg_sweep_continuation": 0.6,
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
            score += 0.6

    # Sprint 4: Removed zone_type bonus — already captured by _score_zone_quality
    # (decorrelation: zone_type in both scorers caused r=0.346 correlation)

    return min(1.0, score)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_confluence(
    bias: BiasDirection,
    zone: TradeZone,
    entry: EntrySignal,
    macro_bias: float = 0.0,
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
    macro_bias:
        Optional macro overlay contribution in [-0.3, +0.3].  Added to the
        weighted confluence sum in a direction-aware manner before clamping.
        When the SMC bias direction and macro bias sign agree the score is
        boosted; when they disagree the score is penalised.  Default ``0.0``
        means no overlay — backward-compatible no-op.

    Returns
    -------
    float
        Confluence score in [0.0, 1.0]. Only setups >= 0.45 are tradeable.

    Notes
    -----
    Direction-aware bonus logic (Alt-B Round 4 W2):

    * ``bias_sign = +1`` if SMC bias is bullish, ``-1`` if bearish, ``0`` if neutral.
    * ``directional_bonus = bias_sign * macro_bias``
    * Aligned (same sign):  adds positive bonus → higher score.
    * Opposed (opposite sign): subtracts → lower score.
    * Neutral SMC bias: ``bias_sign = 0`` → macro_bias has no effect (no phantom signal).
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

    # Directional macro bonus: applied only when SMC bias has a direction.
    # bias_sign encodes agreement/disagreement with the macro signal.
    if macro_bias != 0.0 and bias.direction != "neutral":
        bias_sign = 1.0 if bias.direction == "bullish" else -1.0
        raw += bias_sign * macro_bias

    return round(min(1.0, max(0.0, raw)), 3)

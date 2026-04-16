"""V2 multi-factor confluence scoring with AI direction weight + mode bonus.

Replaces HTF alignment (from compute_htf_bias) with AI direction scoring
and adds a mode bonus for inverted entries.

Weight distribution (v2):
- AI direction:      0.30 (was HTF 0.25) — higher weight for AI signal
- Zone quality:      0.20 (was 0.25)
- Entry trigger:     0.25 (same)
- RR ratio:          0.10 (was 0.15)
- Mode bonus:        0.15 (NEW — replaces liquidity)

Threshold: 0.40 (was 0.45) — more permissive for higher trade frequency.

V1 module ``confluence.py`` is preserved untouched.
"""

from __future__ import annotations

from smc.strategy.types import EntrySignalV2, TradeZone

__all__ = ["score_confluence_v2", "TRADEABLE_THRESHOLD_V2"]

# ---------------------------------------------------------------------------
# Weight constants (v2)
# ---------------------------------------------------------------------------

_W_AI_DIRECTION = 0.30
_W_ZONE_QUALITY = 0.20
_W_ENTRY_TRIGGER = 0.25
_W_RR_RATIO = 0.10
_W_MODE_BONUS = 0.15

TRADEABLE_THRESHOLD_V2 = 0.40


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------


def _score_ai_direction(ai_confidence: float, entry_mode: str) -> float:
    """Score the AI direction component (0.0 - 1.0).

    For normal entries: directly uses AI confidence (higher = better alignment).
    For inverted entries: uses inversion logic — moderate AI confidence
    (0.3-0.5 range) is actually ideal for inversions, very high or very low
    confidence penalizes inversions.
    """
    if entry_mode == "normal":
        return min(1.0, max(0.0, ai_confidence))

    # Inverted: sweet spot is moderate confidence (0.3-0.5)
    # Map 0.3-0.5 to 0.8-1.0, penalize extremes
    if ai_confidence < 0.2:
        return 0.2  # too uncertain even for inversion
    if ai_confidence > 0.7:
        return 0.3  # too confident = bad for counter-trend
    if 0.3 <= ai_confidence <= 0.5:
        return 0.8 + (0.5 - abs(ai_confidence - 0.4)) * 2.0
    # transition zones
    return 0.5


def _score_zone_quality(zone: TradeZone) -> float:
    """Score the zone quality component (0.0 - 1.0).

    Same logic as v1: OB+FVG overlap > OB > FVG.
    """
    type_scores = {
        "ob_fvg_overlap": 1.0,
        "ob": 0.7,
        "fvg": 0.4,
    }
    type_score = type_scores.get(zone.zone_type, 0.3)
    return type_score * 0.5 + zone.confidence * 0.5


def _score_entry_trigger(entry: EntrySignalV2) -> float:
    """Score the entry trigger quality (0.0 - 1.0).

    V2 trigger scores include the two new trigger types:
    - ob_breakout: institutional breakout through zone (0.65)
    - choch_continuation: CHoCH confirming existing trend (0.75)
    - fvg_sweep_continuation: FVG sweep + continuation (0.60)
    """
    trigger_scores = {
        "choch_in_zone": 0.9,
        "choch_continuation": 0.75,
        "fvg_fill_in_zone": 0.7,
        "ob_breakout": 0.65,
        "fvg_sweep_continuation": 0.60,
        "bos_in_zone": 0.55,
    }
    trigger_score = trigger_scores.get(entry.trigger_type, 0.3)

    grade_scores = {"A": 1.0, "B": 0.7, "C": 0.4}
    grade_score = grade_scores.get(entry.grade, 0.3)

    return trigger_score * 0.6 + grade_score * 0.4


def _score_rr_ratio(entry: EntrySignalV2) -> float:
    """Score the risk-reward ratio (0.0 - 1.0). Same as v1."""
    if entry.rr_ratio >= 3.0:
        return 1.0
    if entry.rr_ratio >= 2.0:
        return 0.7
    if entry.rr_ratio >= 1.5:
        return 0.4
    return 0.2


def _score_mode_bonus(entry: EntrySignalV2) -> float:
    """Score the entry mode bonus (0.0 - 1.0).

    Normal entries: flat 0.5 (neutral — no bonus or penalty).
    Inverted entries: scored by inversion_confidence.
    High inversion confidence means strong counter-trend pattern detected.
    """
    if entry.entry_mode == "normal":
        return 0.5

    # Inverted mode: inversion_confidence directly maps to bonus
    # Higher inversion_confidence = more reliable inversion signal
    return min(1.0, max(0.0, entry.inversion_confidence))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_confluence_v2(
    ai_confidence: float,
    zone: TradeZone,
    entry: EntrySignalV2,
) -> float:
    """Compute a v2 multi-factor confluence score for a trade setup.

    Parameters
    ----------
    ai_confidence:
        Confidence from AI DirectionEngine (0.0 - 1.0).
    zone:
        H1 trade zone from ``scan_zones_v2``.
    entry:
        M15 v2 entry signal from ``check_entry_v2``.

    Returns
    -------
    float
        Confluence score in [0.0, 1.0]. Only setups >= 0.40 are tradeable.
    """
    ai_score = _score_ai_direction(ai_confidence, entry.entry_mode)
    zone_score = _score_zone_quality(zone)
    trigger_score = _score_entry_trigger(entry)
    rr_score = _score_rr_ratio(entry)
    mode_score = _score_mode_bonus(entry)

    raw = (
        _W_AI_DIRECTION * ai_score
        + _W_ZONE_QUALITY * zone_score
        + _W_ENTRY_TRIGGER * trigger_score
        + _W_RR_RATIO * rr_score
        + _W_MODE_BONUS * mode_score
    )

    return round(min(1.0, max(0.0, raw)), 3)

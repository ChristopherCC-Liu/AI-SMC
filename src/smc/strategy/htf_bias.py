"""Higher-timeframe directional bias from D1 and H4 structure analysis.

The HTF bias is the first stage of the multi-timeframe strategy pipeline.
It uses a tiered system to produce bias from D1 and/or H4 snapshots:

  Tier 1 — D1 + H4 aligned:  confidence 0.7–1.0 (strongest)
  Tier 2 — H4-only:          confidence 0.4–0.7 (medium)
  Tier 3 — D1-only:          confidence 0.3–0.5 (weakest)

Either snapshot may be None, enabling trade generation even when one
HTF timeframe is unavailable or neutral.

Confluence scoring considers:
- Number and recency of confirming BOS/CHoCH breaks
- Proximity to unmitigated HTF order blocks
"""

from __future__ import annotations

from smc.smc_core.constants import XAUUSD_POINT_SIZE
from smc.smc_core.types import SMCSnapshot
from smc.strategy.types import BiasDirection

__all__ = ["compute_htf_bias", "htf_bias_tier"]


# ---------------------------------------------------------------------------
# Tier classification helper (audit-r2 ops #18 journal monitor)
# ---------------------------------------------------------------------------

def htf_bias_tier(confidence: float) -> str:
    """Map HTF bias confidence to tier label for monitoring / UI / journal.

    Thresholds mirror compute_htf_bias docstring:
      Tier 1 — D1 + H4 aligned:  conf 0.7 – 1.0
      Tier 2 — H4-only:          conf 0.4 – 0.7
      Tier 3 — D1-only:          conf 0.3 – 0.5
      neutral                  — conf < 0.3 (includes explicit 0.0 for disagreement)

    Note: Tier 2 / Tier 3 ranges overlap at 0.4-0.5; we classify by the
    floor (>=0.4 → tier_2) because tier_2 has the higher floor and is
    the more likely producer in that band.  Tier 3 is only reached when
    conf is strictly in [0.3, 0.4).

    audit-r2 ops #18: used by decision-reviewer's Guard 6 debate to size
    Round 3 S2 two-stage soft-multiplier decision.  Round 3 S2 may reuse
    this helper directly if sizing by tier bucket.
    """
    if confidence >= 0.7:
        return "tier_1"
    if confidence >= 0.4:
        return "tier_2"
    if confidence >= 0.3:
        return "tier_3"
    return "neutral"

# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------

_MAX_RECENT_BREAKS = 5


def _direction_from_breaks(
    snapshot: SMCSnapshot,
) -> str:
    """Derive direction from the most recent structure breaks in a snapshot.

    Returns "bullish", "bearish", or "neutral".
    """
    breaks = snapshot.structure_breaks
    if not breaks:
        return "neutral"

    latest = breaks[-1]

    # CHoCH is a strong reversal signal
    if latest.break_type == "choch":
        return latest.direction

    # Count recent BOS directions
    recent = breaks[-_MAX_RECENT_BREAKS:]
    bullish_count = sum(1 for b in recent if b.direction == "bullish")
    bearish_count = sum(1 for b in recent if b.direction == "bearish")

    if bullish_count > bearish_count:
        return "bullish"
    if bearish_count > bullish_count:
        return "bearish"
    return "neutral"


def _break_confidence(snapshot: SMCSnapshot) -> float:
    """Score confidence from the structure break sequence (0.0 – 1.0).

    More confirming breaks in the same direction → higher confidence.
    """
    breaks = snapshot.structure_breaks
    if not breaks:
        return 0.0

    latest_dir = breaks[-1].direction
    recent = breaks[-_MAX_RECENT_BREAKS:]
    confirming = sum(1 for b in recent if b.direction == latest_dir)

    return min(1.0, confirming / _MAX_RECENT_BREAKS)


def _ob_proximity_bonus(snapshot: SMCSnapshot, direction: str) -> float:
    """Award bonus confidence when price is near unmitigated HTF OBs aligned with bias.

    Returns a bonus in [0.0, 0.2].
    """
    if direction == "neutral":
        return 0.0

    aligned_type = "bullish" if direction == "bullish" else "bearish"
    unmitigated_aligned = [
        ob for ob in snapshot.order_blocks
        if ob.ob_type == aligned_type and not ob.mitigated
    ]

    if not unmitigated_aligned:
        return 0.0

    # More unmitigated aligned OBs → stronger institutional footprint
    count = len(unmitigated_aligned)
    return min(0.2, count * 0.05)


def _collect_key_levels(
    d1: SMCSnapshot | None,
    h4: SMCSnapshot | None,
) -> tuple[float, ...]:
    """Gather significant price levels from available timeframes for the bias output."""
    levels: list[float] = []

    for snap in (d1, h4):
        if snap is None:
            continue
        # Recent structure break prices
        for brk in snap.structure_breaks[-3:]:
            levels.append(brk.price)
        # Unmitigated OB boundaries
        for ob in snap.order_blocks:
            if not ob.mitigated:
                levels.extend([ob.high, ob.low])
        # Unswept liquidity
        for liq in snap.liquidity_levels:
            if not liq.swept:
                levels.append(liq.price)

    # Deduplicate and sort
    unique = sorted(set(round(lv, 2) for lv in levels))
    return tuple(unique)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_htf_bias(
    d1_snapshot: SMCSnapshot | None,
    h4_snapshot: SMCSnapshot | None,
) -> BiasDirection:
    """Compute the higher-timeframe directional bias using a tiered system.

    Tiers
    -----
    Tier 1 — D1 + H4 aligned:  confidence 0.7–1.0 (strongest signal)
    Tier 2 — H4-only:          confidence 0.4–0.7 (D1 missing or neutral)
    Tier 3 — D1-only:          confidence 0.3–0.5 (H4 missing or neutral)

    If D1 and H4 actively disagree (both non-neutral, opposite directions),
    the result is neutral with confidence 0.0.

    Parameters
    ----------
    d1_snapshot:
        SMCSnapshot for the D1 timeframe, or None if unavailable.
    h4_snapshot:
        SMCSnapshot for the H4 timeframe, or None if unavailable.

    Returns
    -------
    BiasDirection
        Frozen model with direction, confidence, key_levels, and rationale.
    """
    d1_dir = _direction_from_breaks(d1_snapshot) if d1_snapshot is not None else "neutral"
    h4_dir = _direction_from_breaks(h4_snapshot) if h4_snapshot is not None else "neutral"

    key_levels = _collect_key_levels(d1_snapshot, h4_snapshot)

    # Both neutral or both missing → overall neutral
    if d1_dir == "neutral" and h4_dir == "neutral":
        return BiasDirection(
            direction="neutral",
            confidence=0.0,
            key_levels=key_levels,
            rationale="Both D1 and H4 structure indeterminate (no clear trend).",
        )

    # Active disagreement → neutral
    if (
        d1_dir != "neutral"
        and h4_dir != "neutral"
        and d1_dir != h4_dir
    ):
        return BiasDirection(
            direction="neutral",
            confidence=0.0,
            key_levels=key_levels,
            rationale=f"D1 is {d1_dir} but H4 is {h4_dir} — conflicting bias.",
        )

    # --- Tier 1: D1 + H4 aligned (both non-neutral, same direction) ---
    if d1_dir != "neutral" and h4_dir == d1_dir:
        d1_conf = _break_confidence(d1_snapshot)  # type: ignore[arg-type]
        h4_conf = _break_confidence(h4_snapshot)  # type: ignore[arg-type]
        ob_bonus = (
            _ob_proximity_bonus(d1_snapshot, d1_dir)  # type: ignore[arg-type]
            + _ob_proximity_bonus(h4_snapshot, d1_dir)  # type: ignore[arg-type]
        )
        # Weighted average: D1 has more weight.  Floor at 0.7 for Tier 1.
        raw_conf = d1_conf * 0.5 + h4_conf * 0.3 + ob_bonus
        confidence = min(1.0, max(0.7, raw_conf))
        return BiasDirection(
            direction=d1_dir,  # type: ignore[arg-type]
            confidence=round(confidence, 3),
            key_levels=key_levels,
            rationale=f"Tier 1: D1 and H4 both {d1_dir} — confirmed multi-timeframe bias.",
        )

    # --- Tier 2: H4-only (D1 neutral or missing, H4 has direction) ---
    if h4_dir != "neutral":
        h4_conf = _break_confidence(h4_snapshot)  # type: ignore[arg-type]
        ob_bonus = _ob_proximity_bonus(h4_snapshot, h4_dir)  # type: ignore[arg-type]
        # Confidence range 0.4–0.7 for Tier 2
        raw_conf = h4_conf * 0.6 + ob_bonus
        confidence = min(0.7, max(0.4, raw_conf))
        return BiasDirection(
            direction=h4_dir,  # type: ignore[arg-type]
            confidence=round(confidence, 3),
            key_levels=key_levels,
            rationale=f"Tier 2: H4 {h4_dir} bias (D1 indeterminate or unavailable).",
        )

    # --- Tier 3: D1-only (H4 neutral or missing, D1 has direction) ---
    d1_conf = _break_confidence(d1_snapshot)  # type: ignore[arg-type]
    ob_bonus = _ob_proximity_bonus(d1_snapshot, d1_dir)  # type: ignore[arg-type]
    # Confidence range 0.3–0.5 for Tier 3
    raw_conf = (d1_conf + ob_bonus) * 0.5
    confidence = min(0.5, max(0.3, raw_conf))
    return BiasDirection(
        direction=d1_dir,  # type: ignore[arg-type]
        confidence=round(confidence, 3),
        key_levels=key_levels,
        rationale=f"Tier 3: D1 {d1_dir} trend (H4 indeterminate or unavailable).",
    )

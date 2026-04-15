"""Higher-timeframe directional bias from D1 and H4 structure analysis.

The HTF bias is the first stage of the multi-timeframe strategy pipeline.
It examines D1 structure breaks to establish the primary trend, then
requires H4 confirmation before producing a tradeable bias.

Confluence scoring considers:
- Number and recency of confirming BOS/CHoCH breaks
- Proximity to unmitigated HTF order blocks
"""

from __future__ import annotations

from smc.smc_core.constants import XAUUSD_POINT_SIZE
from smc.smc_core.types import SMCSnapshot
from smc.strategy.types import BiasDirection

__all__ = ["compute_htf_bias"]

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


def _collect_key_levels(d1: SMCSnapshot, h4: SMCSnapshot) -> tuple[float, ...]:
    """Gather significant price levels from both timeframes for the bias output."""
    levels: list[float] = []

    # Recent structure break prices
    for brk in d1.structure_breaks[-3:]:
        levels.append(brk.price)
    for brk in h4.structure_breaks[-3:]:
        levels.append(brk.price)

    # Unmitigated OB boundaries
    for ob in d1.order_blocks:
        if not ob.mitigated:
            levels.extend([ob.high, ob.low])
    for ob in h4.order_blocks:
        if not ob.mitigated:
            levels.extend([ob.high, ob.low])

    # Unswept liquidity
    for liq in d1.liquidity_levels:
        if not liq.swept:
            levels.append(liq.price)
    for liq in h4.liquidity_levels:
        if not liq.swept:
            levels.append(liq.price)

    # Deduplicate and sort
    unique = sorted(set(round(lv, 2) for lv in levels))
    return tuple(unique)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_htf_bias(
    d1_snapshot: SMCSnapshot,
    h4_snapshot: SMCSnapshot,
) -> BiasDirection:
    """Compute the higher-timeframe directional bias from D1 and H4 snapshots.

    Rules
    -----
    1. D1 trend is derived from the latest BOS/CHoCH sequence.
    2. H4 must confirm D1 direction for a non-neutral bias.
    3. If D1 and H4 disagree → neutral.
    4. Confidence scales with confirming breaks + proximity to unmitigated HTF OBs.

    Parameters
    ----------
    d1_snapshot:
        SMCSnapshot for the D1 timeframe.
    h4_snapshot:
        SMCSnapshot for the H4 timeframe.

    Returns
    -------
    BiasDirection
        Frozen model with direction, confidence, key_levels, and rationale.
    """
    d1_dir = _direction_from_breaks(d1_snapshot)
    h4_dir = _direction_from_breaks(h4_snapshot)

    key_levels = _collect_key_levels(d1_snapshot, h4_snapshot)

    # D1 neutral → overall neutral regardless of H4
    if d1_dir == "neutral":
        return BiasDirection(
            direction="neutral",
            confidence=0.0,
            key_levels=key_levels,
            rationale="D1 structure is indeterminate (no clear trend).",
        )

    # D1 and H4 disagree → neutral
    if h4_dir != d1_dir and h4_dir != "neutral":
        return BiasDirection(
            direction="neutral",
            confidence=0.0,
            key_levels=key_levels,
            rationale=f"D1 is {d1_dir} but H4 is {h4_dir} — conflicting bias.",
        )

    # H4 neutral while D1 has direction → weak bias (halved confidence)
    if h4_dir == "neutral":
        d1_conf = _break_confidence(d1_snapshot)
        ob_bonus = _ob_proximity_bonus(d1_snapshot, d1_dir)
        confidence = min(1.0, (d1_conf + ob_bonus) * 0.5)
        return BiasDirection(
            direction=d1_dir,  # type: ignore[arg-type]
            confidence=round(confidence, 3),
            key_levels=key_levels,
            rationale=f"D1 {d1_dir} trend with weak H4 confirmation (H4 neutral).",
        )

    # D1 and H4 aligned → full bias
    d1_conf = _break_confidence(d1_snapshot)
    h4_conf = _break_confidence(h4_snapshot)
    ob_bonus = _ob_proximity_bonus(d1_snapshot, d1_dir) + _ob_proximity_bonus(h4_snapshot, d1_dir)

    # Weighted average: D1 has more weight
    confidence = min(1.0, d1_conf * 0.5 + h4_conf * 0.3 + ob_bonus)

    return BiasDirection(
        direction=d1_dir,  # type: ignore[arg-type]
        confidence=round(confidence, 3),
        key_levels=key_levels,
        rationale=f"D1 and H4 both {d1_dir} — confirmed multi-timeframe bias.",
    )

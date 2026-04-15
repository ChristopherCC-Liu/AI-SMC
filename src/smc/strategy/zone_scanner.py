"""H1 trade zone identification aligned with higher-timeframe bias.

The zone scanner is the second stage of the multi-timeframe pipeline.
It examines H1 order blocks and fair value gaps to identify zones where
institutional footprints exist, filtered by the HTF bias direction.

Zone ranking:
1. Unmitigated Order Blocks (strongest institutional footprint)
2. Unfilled FVGs
3. Partially-filled FVGs (weakest)
4. OB + FVG overlap gets a confidence bonus

Only returns zones aligned with HTF bias direction.  Maximum 3 active zones.
"""

from __future__ import annotations

from smc.smc_core.types import FairValueGap, OrderBlock, SMCSnapshot
from smc.strategy.types import BiasDirection, TradeZone

__all__ = ["scan_zones"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_ZONES = 3
_BASE_CONFIDENCE_OB = 0.8
_BASE_CONFIDENCE_FVG_UNFILLED = 0.6
_BASE_CONFIDENCE_FVG_PARTIAL = 0.4
_OVERLAP_BONUS = 0.15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ob_aligned(ob: OrderBlock, bias_direction: str) -> bool:
    """Check if an order block aligns with the bias direction."""
    if bias_direction == "bullish":
        return ob.ob_type == "bullish"
    if bias_direction == "bearish":
        return ob.ob_type == "bearish"
    return False


def _fvg_aligned(fvg: FairValueGap, bias_direction: str) -> bool:
    """Check if an FVG aligns with the bias direction."""
    if bias_direction == "bullish":
        return fvg.fvg_type == "bullish"
    if bias_direction == "bearish":
        return fvg.fvg_type == "bearish"
    return False


def _zones_overlap(
    zone_high_a: float, zone_low_a: float,
    zone_high_b: float, zone_low_b: float,
) -> bool:
    """Return True if two zones have any price overlap."""
    return zone_low_a < zone_high_b and zone_low_b < zone_high_a


def _direction_from_bias(bias_direction: str) -> str:
    """Convert bias direction to trade direction literal."""
    if bias_direction == "bullish":
        return "long"
    return "short"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_zones(
    h1_snapshot: SMCSnapshot,
    bias: BiasDirection,
) -> tuple[TradeZone, ...]:
    """Identify H1 trade zones aligned with the HTF bias.

    Parameters
    ----------
    h1_snapshot:
        SMCSnapshot for the H1 timeframe.
    bias:
        Higher-timeframe directional bias from ``compute_htf_bias``.

    Returns
    -------
    tuple[TradeZone, ...]
        Up to 3 trade zones, sorted by confidence descending.
        Returns empty tuple if bias is neutral.
    """
    if bias.direction == "neutral":
        return ()

    direction = _direction_from_bias(bias.direction)
    candidates: list[TradeZone] = []

    # Collect aligned, unmitigated order blocks
    ob_zones: list[tuple[float, float]] = []
    for ob in h1_snapshot.order_blocks:
        if not _ob_aligned(ob, bias.direction):
            continue
        if ob.mitigated:
            continue

        ob_zones.append((ob.high, ob.low))
        candidates.append(
            TradeZone(
                zone_high=ob.high,
                zone_low=ob.low,
                zone_type="ob",
                direction=direction,  # type: ignore[arg-type]
                timeframe=h1_snapshot.timeframe,
                confidence=_BASE_CONFIDENCE_OB,
            )
        )

    # Collect aligned, unfilled/partially-filled FVGs
    fvg_zones: list[tuple[float, float]] = []
    for fvg in h1_snapshot.fvgs:
        if not _fvg_aligned(fvg, bias.direction):
            continue
        if fvg.fully_filled:
            continue

        fvg_zones.append((fvg.high, fvg.low))
        base_conf = (
            _BASE_CONFIDENCE_FVG_UNFILLED
            if fvg.filled_pct == 0.0
            else _BASE_CONFIDENCE_FVG_PARTIAL
        )
        candidates.append(
            TradeZone(
                zone_high=fvg.high,
                zone_low=fvg.low,
                zone_type="fvg",
                direction=direction,  # type: ignore[arg-type]
                timeframe=h1_snapshot.timeframe,
                confidence=base_conf,
            )
        )

    # Check for OB + FVG overlap and upgrade those zones
    upgraded: list[TradeZone] = []
    for zone in candidates:
        is_ob = zone.zone_type == "ob"
        is_fvg = zone.zone_type == "fvg"

        has_overlap = False
        if is_ob:
            # Check if any FVG overlaps with this OB
            has_overlap = any(
                _zones_overlap(zone.zone_high, zone.zone_low, fh, fl)
                for fh, fl in fvg_zones
            )
        elif is_fvg:
            # Check if any OB overlaps with this FVG
            has_overlap = any(
                _zones_overlap(zone.zone_high, zone.zone_low, oh, ol)
                for oh, ol in ob_zones
            )

        if has_overlap:
            new_conf = min(1.0, zone.confidence + _OVERLAP_BONUS)
            upgraded.append(
                zone.model_copy(
                    update={"zone_type": "ob_fvg_overlap", "confidence": new_conf}
                )
            )
        else:
            upgraded.append(zone)

    # Deduplicate overlapping zones: keep the one with highest confidence
    deduplicated = _deduplicate_zones(upgraded)

    # Sort by confidence descending and take top N
    sorted_zones = sorted(deduplicated, key=lambda z: z.confidence, reverse=True)
    return tuple(sorted_zones[:_MAX_ZONES])


def _deduplicate_zones(zones: list[TradeZone]) -> list[TradeZone]:
    """Remove overlapping zones, keeping the one with higher confidence."""
    if len(zones) <= 1:
        return zones

    # Sort by confidence descending so we keep the best one
    sorted_z = sorted(zones, key=lambda z: z.confidence, reverse=True)
    kept: list[TradeZone] = []

    for zone in sorted_z:
        overlaps_existing = any(
            _zones_overlap(zone.zone_high, zone.zone_low, k.zone_high, k.zone_low)
            for k in kept
        )
        if not overlaps_existing:
            kept.append(zone)

    return kept

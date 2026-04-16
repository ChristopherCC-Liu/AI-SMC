"""V2 H1 trade zone scanner with relaxed limits for higher trade frequency.

Same core logic as ``zone_scanner.scan_zones`` but with:
  - ``_MAX_ZONES = 5`` (was 3) — more candidate zones per scan
  - ``_ZONE_EXPANSION = 0.40`` (was 0.25 implicit) — wider proximity check

V1 module is preserved untouched.  Import ``scan_zones_v2`` for v2 pipeline.
"""

from __future__ import annotations

from smc.smc_core.types import FairValueGap, OrderBlock, SMCSnapshot
from smc.strategy.types import TradeZone

__all__ = ["scan_zones_v2"]

# ---------------------------------------------------------------------------
# Constants (v2 — relaxed)
# ---------------------------------------------------------------------------

_MAX_ZONES = 5
_ZONE_EXPANSION = 0.40  # fraction of zone range for proximity expansion
_BASE_CONFIDENCE_OB = 0.8
_BASE_CONFIDENCE_FVG_UNFILLED = 0.6
_BASE_CONFIDENCE_FVG_PARTIAL = 0.4
_OVERLAP_BONUS = 0.15


# ---------------------------------------------------------------------------
# Internal helpers (same logic as v1)
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


def _deduplicate_zones(zones: list[TradeZone]) -> list[TradeZone]:
    """Remove overlapping zones, keeping the one with higher confidence."""
    if len(zones) <= 1:
        return zones

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_zones_v2(
    h1_snapshot: SMCSnapshot,
    bias_compat: str,
) -> tuple[TradeZone, ...]:
    """Identify H1 trade zones aligned with the directional bias.

    V2 differences from ``scan_zones``:
    - Accepts ``bias_compat`` as a plain direction string ("bullish"/"bearish")
      instead of requiring a full ``BiasDirection`` object.  This decouples
      from ``compute_htf_bias`` and lets the v2 aggregator pass AI direction.
    - Returns up to 5 zones (was 3).
    - Zone expansion factor is 0.40 (was 0.25).

    Parameters
    ----------
    h1_snapshot:
        SMCSnapshot for the H1 timeframe.
    bias_compat:
        Direction string: "bullish", "bearish", or "neutral".

    Returns
    -------
    tuple[TradeZone, ...]
        Up to 5 trade zones, sorted by confidence descending.
        Returns empty tuple if bias_compat is "neutral".
    """
    if bias_compat == "neutral":
        return ()

    direction = _direction_from_bias(bias_compat)
    candidates: list[TradeZone] = []

    # Collect aligned, unmitigated order blocks
    ob_zones: list[tuple[float, float]] = []
    for ob in h1_snapshot.order_blocks:
        if not _ob_aligned(ob, bias_compat):
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
        if not _fvg_aligned(fvg, bias_compat):
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
            has_overlap = any(
                _zones_overlap(zone.zone_high, zone.zone_low, fh, fl)
                for fh, fl in fvg_zones
            )
        elif is_fvg:
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

    # Sort by confidence descending and take top N (v2: 5)
    sorted_zones = sorted(deduplicated, key=lambda z: z.confidence, reverse=True)
    return tuple(sorted_zones[:_MAX_ZONES])

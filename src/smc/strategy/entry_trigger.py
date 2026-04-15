"""M15 precise entry trigger logic for the AI-SMC strategy.

The entry trigger is the third stage of the multi-timeframe pipeline.
It monitors M15 price action inside an H1 trade zone and fires when
specific SMC confirmation patterns appear.

Trigger conditions (any one is sufficient):
1. **CHoCH in zone**: Price enters zone + M15 CHoCH in bias direction inside zone
2. **FVG fill in zone**: M15 FVG fill inside an HTF OB zone
3. **OB test rejection**: Price tests the zone boundary and shows rejection

Stop-loss is placed beyond the zone boundary + buffer.
TP1 at 1:2 RR, TP2 at next HTF liquidity level.
"""

from __future__ import annotations

from smc.smc_core.constants import XAUUSD_POINT_SIZE
from smc.smc_core.types import SMCSnapshot
from smc.strategy.types import EntrySignal, TradeZone

__all__ = ["check_entry"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Buffer beyond zone boundary for stop-loss placement (in points)
_SL_BUFFER_POINTS = 30.0  # 30 points = $0.30 for XAUUSD
_TP1_RR_RATIO = 2.0
_TP2_RR_RATIO = 3.0  # Fallback if no liquidity level found


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _price_in_zone(price: float, zone: TradeZone) -> bool:
    """Check if price is within the zone boundaries."""
    return zone.zone_low <= price <= zone.zone_high


def _find_choch_in_zone(
    m15_snapshot: SMCSnapshot,
    zone: TradeZone,
) -> bool:
    """Check if there is a M15 CHoCH in the bias direction inside the zone."""
    target_direction = "bullish" if zone.direction == "long" else "bearish"

    for brk in reversed(m15_snapshot.structure_breaks):
        if brk.break_type != "choch":
            continue
        if brk.direction != target_direction:
            continue
        # Check if the break price is within the zone
        if _price_in_zone(brk.price, zone):
            return True

    return False


def _find_fvg_fill_in_zone(
    m15_snapshot: SMCSnapshot,
    zone: TradeZone,
) -> bool:
    """Check if there is a M15 FVG fill event inside the zone (OB overlap)."""
    if zone.zone_type not in ("ob", "ob_fvg_overlap"):
        return False

    target_type = "bullish" if zone.direction == "long" else "bearish"

    for fvg in m15_snapshot.fvgs:
        if fvg.fvg_type != target_type:
            continue
        if fvg.filled_pct < 0.5:
            continue
        # Check if the FVG midpoint is within the zone
        fvg_mid = (fvg.high + fvg.low) / 2.0
        if _price_in_zone(fvg_mid, zone):
            return True

    return False


def _find_ob_rejection(
    m15_snapshot: SMCSnapshot,
    zone: TradeZone,
    current_price: float,
) -> bool:
    """Check for OB test + rejection pattern at zone boundary."""
    if zone.zone_type not in ("ob", "ob_fvg_overlap"):
        return False

    # Need at least some swing points to assess rejection
    if len(m15_snapshot.swing_points) < 2:
        return False

    recent_swings = m15_snapshot.swing_points[-4:]

    if zone.direction == "long":
        # Look for a swing low near or inside the zone followed by price moving up
        for sw in recent_swings:
            if sw.swing_type != "low":
                continue
            if _price_in_zone(sw.price, zone) and current_price > sw.price:
                return True
    else:
        # Look for a swing high near or inside the zone followed by price moving down
        for sw in recent_swings:
            if sw.swing_type != "high":
                continue
            if _price_in_zone(sw.price, zone) and current_price < sw.price:
                return True

    return False


def _compute_sl(zone: TradeZone) -> float:
    """Compute stop-loss price beyond zone boundary + buffer."""
    buffer = _SL_BUFFER_POINTS * XAUUSD_POINT_SIZE
    if zone.direction == "long":
        return zone.zone_low - buffer
    return zone.zone_high + buffer


def _find_next_liquidity_level(
    m15_snapshot: SMCSnapshot,
    zone: TradeZone,
    current_price: float,
) -> float | None:
    """Find the next relevant liquidity level for TP2."""
    if not m15_snapshot.liquidity_levels:
        return None

    unswept = [lv for lv in m15_snapshot.liquidity_levels if not lv.swept]
    if not unswept:
        return None

    if zone.direction == "long":
        # Look for liquidity above current price (equal highs to target)
        above = [lv for lv in unswept if lv.price > current_price]
        if above:
            return min(above, key=lambda lv: lv.price).price
    else:
        # Look for liquidity below current price (equal lows to target)
        below = [lv for lv in unswept if lv.price < current_price]
        if below:
            return max(below, key=lambda lv: lv.price).price

    return None


def _grade_entry(
    trigger_type: str,
    zone: TradeZone,
    rr_ratio: float,
) -> str:
    """Assign a setup grade based on trigger quality and zone type."""
    score = 0.0

    # Trigger type scoring
    if trigger_type == "choch_in_zone":
        score += 0.4
    elif trigger_type == "fvg_fill_in_zone":
        score += 0.35
    elif trigger_type == "ob_test_rejection":
        score += 0.25

    # Zone type scoring
    if zone.zone_type == "ob_fvg_overlap":
        score += 0.3
    elif zone.zone_type == "ob":
        score += 0.2
    elif zone.zone_type == "fvg":
        score += 0.1

    # RR scoring
    if rr_ratio >= 3.0:
        score += 0.3
    elif rr_ratio >= 2.0:
        score += 0.2
    else:
        score += 0.1

    if score >= 0.7:
        return "A"
    if score >= 0.5:
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_entry(
    m15_snapshot: SMCSnapshot,
    zone: TradeZone,
    current_price: float,
) -> EntrySignal | None:
    """Check for a valid M15 entry trigger inside an H1 trade zone.

    Parameters
    ----------
    m15_snapshot:
        SMCSnapshot for the M15 timeframe.
    zone:
        An H1 trade zone from ``scan_zones``.
    current_price:
        The current market price.

    Returns
    -------
    EntrySignal | None
        A frozen entry signal if a valid trigger is found, otherwise None.
    """
    # Price must be in or very near the zone
    zone_range = zone.zone_high - zone.zone_low
    expanded_high = zone.zone_high + zone_range * 0.1
    expanded_low = zone.zone_low - zone_range * 0.1

    if not (expanded_low <= current_price <= expanded_high):
        return None

    # Check triggers in priority order
    trigger_type: str | None = None
    if _find_choch_in_zone(m15_snapshot, zone):
        trigger_type = "choch_in_zone"
    elif _find_fvg_fill_in_zone(m15_snapshot, zone):
        trigger_type = "fvg_fill_in_zone"
    elif _find_ob_rejection(m15_snapshot, zone, current_price):
        trigger_type = "ob_test_rejection"

    if trigger_type is None:
        return None

    # Compute entry parameters
    entry_price = current_price
    stop_loss = _compute_sl(zone)
    risk_points = abs(entry_price - stop_loss) / XAUUSD_POINT_SIZE

    if risk_points == 0:
        return None

    # TP1 at 1:2 RR
    reward_1 = risk_points * _TP1_RR_RATIO
    if zone.direction == "long":
        tp1 = entry_price + reward_1 * XAUUSD_POINT_SIZE
    else:
        tp1 = entry_price - reward_1 * XAUUSD_POINT_SIZE

    # TP2 at next liquidity level or fallback to 1:3 RR
    tp2_liq = _find_next_liquidity_level(m15_snapshot, zone, current_price)
    if tp2_liq is not None:
        tp2 = tp2_liq
        reward_2_points = abs(tp2 - entry_price) / XAUUSD_POINT_SIZE
    else:
        reward_2_points = risk_points * _TP2_RR_RATIO
        if zone.direction == "long":
            tp2 = entry_price + reward_2_points * XAUUSD_POINT_SIZE
        else:
            tp2 = entry_price - reward_2_points * XAUUSD_POINT_SIZE

    rr_ratio = reward_1 / risk_points if risk_points > 0 else 0.0
    grade = _grade_entry(trigger_type, zone, rr_ratio)

    return EntrySignal(
        entry_price=round(entry_price, 2),
        stop_loss=round(stop_loss, 2),
        take_profit_1=round(tp1, 2),
        take_profit_2=round(tp2, 2),
        risk_points=round(risk_points, 1),
        reward_points=round(reward_1, 1),
        rr_ratio=round(rr_ratio, 2),
        trigger_type=trigger_type,  # type: ignore[arg-type]
        direction=zone.direction,
        grade=grade,  # type: ignore[arg-type]
    )

"""M15 dual entry trigger logic V2 for the AI-SMC strategy.

Extends v1 entry_trigger with three categories of signals:

**Normal (proven from v1):**
1. fvg_fill_in_zone  -- FVG filled inside OB zone -> reversal entry (45% WR anchor)
2. bos_in_zone       -- BOS in zone direction -> continuation entry
3. choch_in_zone     -- CHoCH inside zone -> reversal entry (regime-dependent)

**Inverted (NEW -- from quant-analyst data):**
4. ob_breakout       -- OB test fails, price breaks through -> trade BREAKOUT direction
   - Inverted WR 76.9% from 26 historical trades
   - Inverts in ALL regimes (even ranging 25% WR -> 75% inverted)
5. choch_continuation -- CHoCH fires but retraces >61.8% -> false reversal -> continue trend
   - Inverted WR 90.9% in transitional regime
   - Only invert in TRANSITIONAL regime

**New:**
6. fvg_sweep_continuation -- FVG fully filled + price continues through -> continuation

Priority order: fvg_fill > bos > choch > ob_breakout > choch_continuation > fvg_sweep
"""

from __future__ import annotations

from typing import Literal

from smc.smc_core.constants import XAUUSD_POINT_SIZE
from smc.smc_core.types import SMCSnapshot
from smc.strategy.entry_trigger import (
    _find_bos_in_zone,
    _find_choch_in_zone,
    _find_fvg_fill_in_zone,
    _price_in_zone,
)
from smc.strategy.regime import MarketRegime
from smc.strategy.types import (
    EntrySignalV2,
    SetupGrade,
    TradeZone,
    TriggerTypeV2,
)

__all__ = ["check_entry_v2"]

# ---------------------------------------------------------------------------
# Constants -- Normal vs Inverted parameters
# ---------------------------------------------------------------------------

_SL_ATR_MULTIPLIER_NORMAL = 0.75
_SL_ATR_MULTIPLIER_INVERTED = 1.0
_SL_MIN_BUFFER = 200.0  # points floor ($2.00)

_TP1_RR_NORMAL = 2.5
_TP1_RR_INVERTED = 2.0
_TP2_RR_NORMAL = 4.0
_TP2_RR_INVERTED = 3.0

_CONFLUENCE_FLOOR_NORMAL = 0.40
_CONFLUENCE_FLOOR_INVERTED = 0.50

# Inverted signal confidence scores (from quant-analyst backtest data)
_OB_BREAKOUT_CONFIDENCE = 0.769  # 76.9% inverted WR across 26 trades
_CHOCH_CONTINUATION_CONFIDENCE = 0.909  # 90.9% inverted WR in transitional

# OB breakout detection: bars to wait for rejection failure
_OB_FAILURE_BARS_DEFAULT = 6

# CHoCH continuation: retrace threshold for false reversal detection
_CHOCH_RETRACE_PCT_DEFAULT = 0.618


# ---------------------------------------------------------------------------
# Inverted detection helpers (NEW -- not in v1)
# ---------------------------------------------------------------------------


def _find_ob_breakout(
    m15_snapshot: SMCSnapshot,
    zone: TradeZone,
    current_price: float,
    ob_failure_bars: int = _OB_FAILURE_BARS_DEFAULT,
) -> bool:
    """Detect OB breakout: price tested zone boundary, no rejection formed,
    price closed beyond opposite boundary.

    This is an INVERTED signal -- the OB that should have held has failed.
    A bullish OB breakout (price falls through) = SHORT signal.
    """
    if zone.zone_type not in ("ob", "ob_fvg_overlap"):
        return False

    # Need enough swing points to assess rejection failure
    if len(m15_snapshot.swing_points) < 2:
        return False

    # Count recent swings that could indicate rejection -- if fewer than
    # expected within the lookback window, it means rejection failed
    recent_swings = m15_snapshot.swing_points[-ob_failure_bars:]

    if zone.direction == "long":
        # Bullish OB: expected to hold as support. Breakout = price falls below zone_low
        # No rejection swing (swing low with bounce) formed
        rejection_swings = [
            sw for sw in recent_swings
            if sw.swing_type == "low" and _price_in_zone(sw.price, zone)
        ]
        if len(rejection_swings) == 0 and current_price < zone.zone_low:
            return True
    else:
        # Bearish OB: expected to hold as resistance. Breakout = price rises above zone_high
        rejection_swings = [
            sw for sw in recent_swings
            if sw.swing_type == "high" and _price_in_zone(sw.price, zone)
        ]
        if len(rejection_swings) == 0 and current_price > zone.zone_high:
            return True

    return False


def _find_choch_continuation(
    m15_snapshot: SMCSnapshot,
    zone: TradeZone,
    current_price: float,
    choch_retrace_pct: float = _CHOCH_RETRACE_PCT_DEFAULT,
) -> bool:
    """Detect CHoCH continuation: CHoCH fired but price retraced >61.8%
    of the impulse move, indicating a false reversal.

    Only meaningful in TRANSITIONAL regime (caller must gate this).
    """
    # Find the most recent CHoCH
    choch_break = None
    for brk in reversed(m15_snapshot.structure_breaks):
        if brk.break_type == "choch":
            choch_break = brk
            break

    if choch_break is None:
        return False

    # Find the impulse move: swing before CHoCH to CHoCH price
    # We need at least one swing point before the CHoCH to measure the impulse
    swings_before = [
        sw for sw in m15_snapshot.swing_points
        if sw.ts <= choch_break.ts
    ]
    if not swings_before:
        return False

    # The impulse start is the last swing point before the CHoCH
    impulse_start = swings_before[-1]
    impulse_size = abs(choch_break.price - impulse_start.price)

    if impulse_size == 0:
        return False

    # Calculate retrace from the CHoCH price
    retrace = abs(current_price - choch_break.price)
    retrace_ratio = retrace / impulse_size

    if retrace_ratio <= choch_retrace_pct:
        return False

    # Verify direction: if bullish CHoCH retraces >61.8%, it's a failed bullish
    # reversal -> continue SHORT (original trend direction)
    if choch_break.direction == "bullish":
        # Failed bullish reversal: price should be dropping back
        return current_price < choch_break.price
    else:
        # Failed bearish reversal: price should be rising back
        return current_price > choch_break.price


def _find_fvg_sweep_continuation(
    m15_snapshot: SMCSnapshot,
    zone: TradeZone,
    current_price: float,
) -> bool:
    """Detect FVG sweep continuation: FVG fully filled (100%) then
    2+ bars continue in the original sweep direction.

    This is a NEW signal type -- the FVG fill completed and price
    didn't reverse as expected, instead continuing through.
    """
    if not m15_snapshot.fvgs:
        return False

    for fvg in m15_snapshot.fvgs:
        if not fvg.fully_filled:
            continue

        # The FVG was fully filled. Check if price continues through.
        fvg_mid = (fvg.high + fvg.low) / 2.0
        if not _price_in_zone(fvg_mid, zone):
            continue

        # Determine sweep direction: a bullish FVG getting fully filled
        # means price swept down through it -> continuation = SHORT
        if fvg.fvg_type == "bullish":
            # Sweep was downward, continuation = price below FVG low
            if current_price < fvg.low:
                return True
        else:
            # Bearish FVG swept upward, continuation = price above FVG high
            if current_price > fvg.high:
                return True

    return False


# ---------------------------------------------------------------------------
# SL / TP computation
# ---------------------------------------------------------------------------


def _compute_sl_buffer_v2(
    h1_atr: float,
    sl_atr_multiplier: float,
) -> float:
    """Compute adaptive SL buffer from H1 ATR(14) in points."""
    return max(h1_atr * sl_atr_multiplier, _SL_MIN_BUFFER)


def _compute_sl_v2(
    zone: TradeZone,
    h1_atr: float,
    sl_atr_multiplier: float,
    *,
    is_inverted: bool = False,
) -> float:
    """Compute stop-loss price beyond zone boundary + ATR-adaptive buffer.

    For inverted signals, we flip the SL side: ob_breakout trades in the
    breakout direction, so SL goes back inside the zone.
    """
    buffer = _compute_sl_buffer_v2(h1_atr, sl_atr_multiplier) * XAUUSD_POINT_SIZE

    if is_inverted:
        # Inverted: direction already flipped by caller, SL placed on the
        # "wrong" side of the original zone
        if zone.direction == "long":
            # Original zone was bullish, inverted = SHORT, SL above zone
            return zone.zone_high + buffer
        return zone.zone_low - buffer

    # Normal: same as v1
    if zone.direction == "long":
        return zone.zone_low - buffer
    return zone.zone_high + buffer


def _find_next_liquidity_level_v2(
    m15_snapshot: SMCSnapshot,
    direction: Literal["long", "short"],
    current_price: float,
) -> float | None:
    """Find the next relevant liquidity level for TP2."""
    if not m15_snapshot.liquidity_levels:
        return None

    unswept = [lv for lv in m15_snapshot.liquidity_levels if not lv.swept]
    if not unswept:
        return None

    if direction == "long":
        above = [lv for lv in unswept if lv.price > current_price]
        if above:
            return min(above, key=lambda lv: lv.price).price
    else:
        below = [lv for lv in unswept if lv.price < current_price]
        if below:
            return max(below, key=lambda lv: lv.price).price

    return None


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def _grade_entry_v2(
    trigger_type: TriggerTypeV2,
    zone: TradeZone,
    rr_ratio: float,
    entry_mode: Literal["normal", "inverted"],
) -> SetupGrade:
    """Assign a setup grade based on trigger quality, zone type, and mode."""
    score = 0.0

    # Trigger type scoring (v2 priority order)
    trigger_scores: dict[str, float] = {
        "fvg_fill_in_zone": 0.35,
        "bos_in_zone": 0.20,
        "choch_in_zone": 0.30,
        "ob_breakout": 0.30,
        "choch_continuation": 0.35,
        "fvg_sweep_continuation": 0.25,
    }
    score += trigger_scores.get(trigger_type, 0.15)

    # Zone type scoring
    if zone.zone_type == "ob_fvg_overlap":
        score += 0.30
    elif zone.zone_type == "ob":
        score += 0.20
    elif zone.zone_type == "fvg":
        score += 0.10

    # RR scoring
    if rr_ratio >= 3.0:
        score += 0.30
    elif rr_ratio >= 2.0:
        score += 0.20
    else:
        score += 0.10

    # Inverted signals get a small penalty for higher uncertainty
    if entry_mode == "inverted":
        score -= 0.05

    if score >= 0.70:
        return "A"
    if score >= 0.50:
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# Inversion direction resolver
# ---------------------------------------------------------------------------


def _resolve_inverted_direction(
    zone: TradeZone,
    trigger_type: TriggerTypeV2,
) -> Literal["long", "short"]:
    """For inverted signals, direction is OPPOSITE of what zone suggests.

    ob_breakout: bullish OB breakout = SHORT (zone.direction was "long")
    choch_continuation: bullish CHoCH that fails = SHORT (original trend)
    """
    if trigger_type in ("ob_breakout", "choch_continuation"):
        return "short" if zone.direction == "long" else "long"
    return zone.direction


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_entry_v2(
    m15_snapshot: SMCSnapshot,
    zone: TradeZone,
    current_price: float,
    h1_atr: float = 0.0,
    *,
    regime: MarketRegime = "transitional",
    enable_inverted: bool = True,
    enable_fvg_sweep: bool = True,
    sl_atr_multiplier: float = _SL_ATR_MULTIPLIER_NORMAL,
    tp1_rr_normal: float = _TP1_RR_NORMAL,
    tp1_rr_inverted: float = _TP1_RR_INVERTED,
    ob_failure_bars: int = _OB_FAILURE_BARS_DEFAULT,
    choch_retrace_pct: float = _CHOCH_RETRACE_PCT_DEFAULT,
) -> EntrySignalV2 | None:
    """Check for a valid M15 entry trigger inside an H1 trade zone (V2).

    Supports normal, inverted, and new signal types with regime-dependent
    inversion gating.

    Parameters
    ----------
    m15_snapshot:
        SMCSnapshot for the M15 timeframe.
    zone:
        An H1 trade zone from ``scan_zones``.
    current_price:
        The current market price.
    h1_atr:
        H1 ATR(14) in points for adaptive SL buffer computation.
    regime:
        Current market regime from ``classify_regime``.
    enable_inverted:
        Enable inverted signals (ob_breakout, choch_continuation).
    enable_fvg_sweep:
        Enable fvg_sweep_continuation signal.
    sl_atr_multiplier:
        ATR multiplier for SL buffer (overridden per entry_mode).
    tp1_rr_normal:
        TP1 risk-reward ratio for normal entries.
    tp1_rr_inverted:
        TP1 risk-reward ratio for inverted entries.
    ob_failure_bars:
        Number of M15 bars to check for OB rejection failure.
    choch_retrace_pct:
        Fibonacci retrace threshold for CHoCH continuation detection.

    Returns
    -------
    EntrySignalV2 | None
        A frozen v2 entry signal if a valid trigger is found, otherwise None.
    """
    # Price must be in or near the zone (same expansion as v1)
    zone_range = zone.zone_high - zone.zone_low
    expanded_high = zone.zone_high + zone_range * 0.25
    expanded_low = zone.zone_low - zone_range * 0.25

    # For inverted signals, we need a wider check since price may have
    # broken through the zone
    if enable_inverted:
        expanded_high = zone.zone_high + zone_range * 1.0
        expanded_low = zone.zone_low - zone_range * 1.0

    if not (expanded_low <= current_price <= expanded_high):
        return None

    # Check triggers in priority order
    trigger_type: TriggerTypeV2 | None = None
    entry_mode: Literal["normal", "inverted"] = "normal"
    inversion_confidence: float = 1.0

    # --- Normal signals (priority 1-3) ---
    if _find_fvg_fill_in_zone(m15_snapshot, zone):
        trigger_type = "fvg_fill_in_zone"
    elif _find_bos_in_zone(m15_snapshot, zone):
        trigger_type = "bos_in_zone"
    elif _find_choch_in_zone(m15_snapshot, zone):
        trigger_type = "choch_in_zone"

    # --- Inverted signals (priority 4-5) ---
    if trigger_type is None and enable_inverted:
        if _find_ob_breakout(m15_snapshot, zone, current_price, ob_failure_bars):
            trigger_type = "ob_breakout"
            entry_mode = "inverted"
            inversion_confidence = _OB_BREAKOUT_CONFIDENCE

        elif (
            regime == "transitional"
            and _find_choch_continuation(
                m15_snapshot, zone, current_price, choch_retrace_pct
            )
        ):
            trigger_type = "choch_continuation"
            entry_mode = "inverted"
            inversion_confidence = _CHOCH_CONTINUATION_CONFIDENCE

    # --- New signal (priority 6) ---
    if trigger_type is None and enable_fvg_sweep:
        if _find_fvg_sweep_continuation(m15_snapshot, zone, current_price):
            trigger_type = "fvg_sweep_continuation"

    if trigger_type is None:
        return None

    # Resolve direction
    if entry_mode == "inverted":
        direction = _resolve_inverted_direction(zone, trigger_type)
    else:
        direction = zone.direction

    # Select SL/TP parameters based on entry mode
    if entry_mode == "inverted":
        sl_mult = _SL_ATR_MULTIPLIER_INVERTED
        tp1_rr = tp1_rr_inverted
        tp2_rr = _TP2_RR_INVERTED
    else:
        sl_mult = sl_atr_multiplier
        tp1_rr = tp1_rr_normal
        tp2_rr = _TP2_RR_NORMAL

    # Compute entry parameters
    entry_price = current_price
    is_inverted = entry_mode == "inverted"
    stop_loss = _compute_sl_v2(zone, h1_atr, sl_mult, is_inverted=is_inverted)
    risk_points = abs(entry_price - stop_loss) / XAUUSD_POINT_SIZE

    if risk_points == 0:
        return None

    # TP1 at configured RR
    reward_1 = risk_points * tp1_rr
    if direction == "long":
        tp1 = entry_price + reward_1 * XAUUSD_POINT_SIZE
    else:
        tp1 = entry_price - reward_1 * XAUUSD_POINT_SIZE

    # TP2 at next liquidity level or fallback RR
    tp2_liq = _find_next_liquidity_level_v2(m15_snapshot, direction, current_price)
    if tp2_liq is not None:
        tp2 = tp2_liq
    else:
        reward_2 = risk_points * tp2_rr
        if direction == "long":
            tp2 = entry_price + reward_2 * XAUUSD_POINT_SIZE
        else:
            tp2 = entry_price - reward_2 * XAUUSD_POINT_SIZE

    rr_ratio = reward_1 / risk_points if risk_points > 0 else 0.0
    grade = _grade_entry_v2(trigger_type, zone, rr_ratio, entry_mode)

    return EntrySignalV2(
        entry_price=round(entry_price, 2),
        stop_loss=round(stop_loss, 2),
        take_profit_1=round(tp1, 2),
        take_profit_2=round(tp2, 2),
        risk_points=round(risk_points, 1),
        reward_points=round(reward_1, 1),
        rr_ratio=round(rr_ratio, 2),
        direction=direction,
        grade=grade,
        trigger_type=trigger_type,
        entry_mode=entry_mode,
        inversion_confidence=round(inversion_confidence, 3),
    )

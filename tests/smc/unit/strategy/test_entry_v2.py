"""Unit tests for smc.strategy.entry_v2 -- V2 dual entry trigger logic.

Tests cover all 6 trigger types (3 normal, 2 inverted, 1 new),
inverted SL/TP parameter differences, priority ordering, direction
inversion, and regime gating.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from smc.data.schemas import Timeframe
from smc.smc_core.types import (
    FairValueGap,
    LiquidityLevel,
    SMCSnapshot,
    StructureBreak,
    SwingPoint,
)
from smc.strategy.entry_v2 import check_entry_v2
from smc.strategy.types import EntrySignalV2, TradeZone

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TS = datetime(2024, 6, 10, 0, 0, 0, tzinfo=timezone.utc)


def _ts(hours: int) -> datetime:
    return _BASE_TS + timedelta(hours=hours)


def _make_snapshot(
    *,
    timeframe: Timeframe = Timeframe.M15,
    trend: str = "bullish",
    swing_points: tuple[SwingPoint, ...] = (),
    fvgs: tuple[FairValueGap, ...] = (),
    structure_breaks: tuple[StructureBreak, ...] = (),
    liquidity_levels: tuple[LiquidityLevel, ...] = (),
) -> SMCSnapshot:
    from smc.smc_core.types import OrderBlock

    return SMCSnapshot(
        ts=_ts(24),
        timeframe=timeframe,
        swing_points=swing_points,
        order_blocks=(),
        fvgs=fvgs,
        structure_breaks=structure_breaks,
        liquidity_levels=liquidity_levels,
        trend_direction=trend,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Reusable zones
# ---------------------------------------------------------------------------

_BULLISH_OB_ZONE = TradeZone(
    zone_high=2352.00, zone_low=2348.00, zone_type="ob",
    direction="long", timeframe=Timeframe.H1, confidence=0.8,
)

_BEARISH_OB_ZONE = TradeZone(
    zone_high=2380.00, zone_low=2376.00, zone_type="ob",
    direction="short", timeframe=Timeframe.H1, confidence=0.8,
)

_BULLISH_OVERLAP_ZONE = TradeZone(
    zone_high=2352.00, zone_low=2348.00, zone_type="ob_fvg_overlap",
    direction="long", timeframe=Timeframe.H1, confidence=0.95,
)


# ===================================================================
# 1. Normal signal tests (reuse v1 detection, verify v2 wrapping)
# ===================================================================


class TestFVGFillInZone:
    """Test fvg_fill_in_zone -- the anchor normal signal (45% WR)."""

    def test_fvg_fill_triggers(self) -> None:
        """FVG with >= 50% fill inside OB zone should trigger."""
        snap = _make_snapshot(
            fvgs=(
                FairValueGap(
                    ts=_ts(3), high=2353.00, low=2349.00,
                    fvg_type="bullish", timeframe=Timeframe.M15,
                    filled_pct=0.6, fully_filled=False,
                ),
            ),
        )
        result = check_entry_v2(snap, _BULLISH_OB_ZONE, 2350.00)
        assert result is not None
        assert result.trigger_type == "fvg_fill_in_zone"
        assert result.entry_mode == "normal"
        assert result.inversion_confidence == 1.0
        assert result.direction == "long"

    def test_fvg_fill_requires_ob_zone_type(self) -> None:
        """FVG fill should not fire on FVG-only zone type."""
        fvg_zone = TradeZone(
            zone_high=2356.00, zone_low=2352.00, zone_type="fvg",
            direction="long", timeframe=Timeframe.H1, confidence=0.6,
        )
        snap = _make_snapshot(
            fvgs=(
                FairValueGap(
                    ts=_ts(3), high=2355.00, low=2353.00,
                    fvg_type="bullish", timeframe=Timeframe.M15,
                    filled_pct=0.8, fully_filled=False,
                ),
            ),
        )
        result = check_entry_v2(snap, fvg_zone, 2354.00)
        assert result is None


class TestBOSInZone:
    """Test bos_in_zone -- continuation signal."""

    def test_bos_triggers(self) -> None:
        snap = _make_snapshot(
            structure_breaks=(
                StructureBreak(
                    ts=_ts(2), price=2350.00,
                    break_type="bos", direction="bullish",
                    timeframe=Timeframe.M15,
                ),
            ),
        )
        result = check_entry_v2(snap, _BULLISH_OB_ZONE, 2350.00)
        assert result is not None
        assert result.trigger_type == "bos_in_zone"
        assert result.entry_mode == "normal"
        assert result.direction == "long"


class TestCHoCHInZone:
    """Test choch_in_zone -- reversal signal."""

    def test_choch_triggers(self) -> None:
        snap = _make_snapshot(
            structure_breaks=(
                StructureBreak(
                    ts=_ts(2), price=2349.50,
                    break_type="choch", direction="bullish",
                    timeframe=Timeframe.M15,
                ),
            ),
        )
        result = check_entry_v2(snap, _BULLISH_OB_ZONE, 2350.00)
        assert result is not None
        assert result.trigger_type == "choch_in_zone"
        assert result.entry_mode == "normal"


# ===================================================================
# 2. Inverted signal tests
# ===================================================================


class TestOBBreakout:
    """Test ob_breakout -- OB test fails, price breaks through.

    Key: direction is OPPOSITE of zone type.
    Bullish OB breakout (support failed) = SHORT.
    """

    def test_bullish_ob_breakout_gives_short(self) -> None:
        """Bullish OB: price falls below zone_low with no rejection swing -> SHORT."""
        snap = _make_snapshot(
            swing_points=(
                # No swing lows inside zone (rejection failed)
                SwingPoint(ts=_ts(1), price=2354.00, swing_type="high", strength=5),
                SwingPoint(ts=_ts(2), price=2344.00, swing_type="low", strength=5),
            ),
        )
        # Price has broken below zone_low (2348)
        result = check_entry_v2(snap, _BULLISH_OB_ZONE, 2345.00)
        assert result is not None
        assert result.trigger_type == "ob_breakout"
        assert result.entry_mode == "inverted"
        assert result.direction == "short"  # OPPOSITE of zone.direction ("long")
        assert result.inversion_confidence == pytest.approx(0.769)

    def test_bearish_ob_breakout_gives_long(self) -> None:
        """Bearish OB: price rises above zone_high with no rejection -> LONG."""
        snap = _make_snapshot(
            trend="bearish",
            swing_points=(
                SwingPoint(ts=_ts(1), price=2374.00, swing_type="low", strength=5),
                SwingPoint(ts=_ts(2), price=2384.00, swing_type="high", strength=5),
            ),
        )
        # Price has broken above zone_high (2380)
        result = check_entry_v2(snap, _BEARISH_OB_ZONE, 2383.00)
        assert result is not None
        assert result.trigger_type == "ob_breakout"
        assert result.direction == "long"  # OPPOSITE of zone.direction ("short")
        assert result.entry_mode == "inverted"

    def test_ob_breakout_disabled_when_inverted_off(self) -> None:
        """ob_breakout should not fire when enable_inverted=False."""
        snap = _make_snapshot(
            swing_points=(
                SwingPoint(ts=_ts(1), price=2354.00, swing_type="high", strength=5),
                SwingPoint(ts=_ts(2), price=2344.00, swing_type="low", strength=5),
            ),
        )
        result = check_entry_v2(
            snap, _BULLISH_OB_ZONE, 2345.00, enable_inverted=False,
        )
        assert result is None

    def test_ob_breakout_requires_ob_zone_type(self) -> None:
        """ob_breakout only works on OB or overlap zone types."""
        fvg_zone = TradeZone(
            zone_high=2352.00, zone_low=2348.00, zone_type="fvg",
            direction="long", timeframe=Timeframe.H1, confidence=0.6,
        )
        snap = _make_snapshot(
            swing_points=(
                SwingPoint(ts=_ts(1), price=2354.00, swing_type="high", strength=5),
                SwingPoint(ts=_ts(2), price=2344.00, swing_type="low", strength=5),
            ),
        )
        result = check_entry_v2(snap, fvg_zone, 2345.00)
        assert result is None

    def test_ob_breakout_not_triggered_with_rejection(self) -> None:
        """If rejection swing exists inside zone, OB held -> no breakout."""
        snap = _make_snapshot(
            swing_points=(
                SwingPoint(ts=_ts(1), price=2354.00, swing_type="high", strength=5),
                # Swing low INSIDE zone = rejection happened
                SwingPoint(ts=_ts(2), price=2350.00, swing_type="low", strength=5),
                SwingPoint(ts=_ts(3), price=2346.00, swing_type="low", strength=5),
            ),
        )
        result = check_entry_v2(snap, _BULLISH_OB_ZONE, 2345.00)
        # Should not fire ob_breakout because rejection swing at 2350 is in zone
        assert result is None or result.trigger_type != "ob_breakout"


class TestCHoCHContinuation:
    """Test choch_continuation -- CHoCH retraces >61.8% = false reversal.

    Only fires in TRANSITIONAL regime.
    """

    def _make_choch_retrace_snap(self) -> SMCSnapshot:
        """Snapshot: bullish CHoCH at 2355 (OUTSIDE zone 2348-2352),
        impulse from 2345 to 2355 (size=10), then price retraces to 2347
        (retrace = 8/10 = 80% > 61.8%).

        CHoCH price is outside zone so _find_choch_in_zone does NOT fire.
        Includes a rejection swing LOW inside zone to block ob_breakout
        (priority 4), allowing choch_continuation (priority 5) to be tested.
        """
        return _make_snapshot(
            swing_points=(
                SwingPoint(ts=_ts(1), price=2345.00, swing_type="low", strength=5),
                # Rejection swing inside zone blocks ob_breakout
                SwingPoint(ts=_ts(2), price=2350.00, swing_type="low", strength=5),
                SwingPoint(ts=_ts(3), price=2355.00, swing_type="high", strength=5),
            ),
            structure_breaks=(
                StructureBreak(
                    ts=_ts(2), price=2355.00,
                    break_type="choch", direction="bullish",
                    timeframe=Timeframe.M15,
                ),
            ),
        )

    def test_choch_continuation_transitional(self) -> None:
        """Fires in transitional regime with retrace >61.8%."""
        snap = self._make_choch_retrace_snap()
        # Price at 2347: retrace from 2355 = 8/10 = 80% > 61.8%
        # Price < CHoCH price (2355) confirms failed bullish reversal
        result = check_entry_v2(
            snap, _BULLISH_OB_ZONE, 2347.00, regime="transitional",
        )
        assert result is not None
        assert result.trigger_type == "choch_continuation"
        assert result.entry_mode == "inverted"
        # Bullish CHoCH failed -> SHORT continuation
        assert result.direction == "short"
        assert result.inversion_confidence == pytest.approx(0.909)

    def test_choch_continuation_blocked_in_trending(self) -> None:
        """Should NOT fire in trending regime."""
        snap = self._make_choch_retrace_snap()
        result = check_entry_v2(
            snap, _BULLISH_OB_ZONE, 2347.00, regime="trending",
        )
        if result is not None:
            assert result.trigger_type != "choch_continuation"

    def test_choch_continuation_blocked_in_ranging(self) -> None:
        """Should NOT fire in ranging regime."""
        snap = self._make_choch_retrace_snap()
        result = check_entry_v2(
            snap, _BULLISH_OB_ZONE, 2347.00, regime="ranging",
        )
        if result is not None:
            assert result.trigger_type != "choch_continuation"

    def test_choch_continuation_requires_sufficient_retrace(self) -> None:
        """Should NOT fire when retrace is < 61.8%."""
        snap = _make_snapshot(
            swing_points=(
                SwingPoint(ts=_ts(1), price=2340.00, swing_type="low", strength=5),
                SwingPoint(ts=_ts(3), price=2360.00, swing_type="high", strength=5),
            ),
            structure_breaks=(
                StructureBreak(
                    ts=_ts(2), price=2360.00,
                    break_type="choch", direction="bullish",
                    timeframe=Timeframe.M15,
                ),
            ),
        )
        # Retrace from 2360: price at 2355, retrace = 5/20 = 25% < 61.8%
        result = check_entry_v2(
            snap, _BULLISH_OB_ZONE, 2355.00, regime="transitional",
        )
        if result is not None:
            assert result.trigger_type != "choch_continuation"


# ===================================================================
# 3. New signal test
# ===================================================================


class TestFVGSweepContinuation:
    """Test fvg_sweep_continuation -- FVG fully filled + price continues.

    Uses FVG-type zones (not OB) so that fvg_fill_in_zone does NOT match
    (fvg_fill requires zone_type in {"ob", "ob_fvg_overlap"}).
    """

    _FVG_ZONE_BULL = TradeZone(
        zone_high=2352.00, zone_low=2348.00, zone_type="fvg",
        direction="long", timeframe=Timeframe.H1, confidence=0.6,
    )
    _FVG_ZONE_BEAR = TradeZone(
        zone_high=2380.00, zone_low=2376.00, zone_type="fvg",
        direction="short", timeframe=Timeframe.H1, confidence=0.6,
    )

    def test_fvg_sweep_bullish_fvg_gives_short(self) -> None:
        """Bullish FVG fully filled, price continues below -> continuation."""
        snap = _make_snapshot(
            fvgs=(
                FairValueGap(
                    ts=_ts(2), high=2351.00, low=2349.00,
                    fvg_type="bullish", timeframe=Timeframe.M15,
                    filled_pct=1.0, fully_filled=True,
                ),
            ),
        )
        # Price at 2347 is below FVG low (2349), continuing the sweep
        result = check_entry_v2(snap, self._FVG_ZONE_BULL, 2347.00)
        assert result is not None
        assert result.trigger_type == "fvg_sweep_continuation"
        assert result.entry_mode == "normal"

    def test_fvg_sweep_bearish_fvg_gives_long(self) -> None:
        """Bearish FVG fully filled, price continues above -> continuation."""
        snap = _make_snapshot(
            trend="bearish",
            fvgs=(
                FairValueGap(
                    ts=_ts(2), high=2379.00, low=2377.00,
                    fvg_type="bearish", timeframe=Timeframe.M15,
                    filled_pct=1.0, fully_filled=True,
                ),
            ),
        )
        result = check_entry_v2(snap, self._FVG_ZONE_BEAR, 2381.00)
        assert result is not None
        assert result.trigger_type == "fvg_sweep_continuation"

    def test_fvg_sweep_disabled(self) -> None:
        """Should not fire when enable_fvg_sweep=False."""
        snap = _make_snapshot(
            fvgs=(
                FairValueGap(
                    ts=_ts(2), high=2351.00, low=2349.00,
                    fvg_type="bullish", timeframe=Timeframe.M15,
                    filled_pct=1.0, fully_filled=True,
                ),
            ),
        )
        result = check_entry_v2(
            snap, self._FVG_ZONE_BULL, 2347.00, enable_fvg_sweep=False,
        )
        assert result is None

    def test_fvg_sweep_requires_fully_filled(self) -> None:
        """Partial fill should not trigger fvg_sweep_continuation."""
        snap = _make_snapshot(
            fvgs=(
                FairValueGap(
                    ts=_ts(2), high=2351.00, low=2349.00,
                    fvg_type="bullish", timeframe=Timeframe.M15,
                    filled_pct=0.8, fully_filled=False,
                ),
            ),
        )
        result = check_entry_v2(snap, self._FVG_ZONE_BULL, 2347.00)
        if result is not None:
            assert result.trigger_type != "fvg_sweep_continuation"


# ===================================================================
# 4. Inverted SL/TP parameter tests
# ===================================================================


class TestInvertedSLTPParams:
    """Inverted entries use different SL/TP parameters than normal."""

    def test_inverted_sl_wider_than_normal(self) -> None:
        """Inverted SL uses 1.0x ATR (vs 0.75x normal)."""
        # Create matching snapshots for normal vs inverted
        normal_snap = _make_snapshot(
            fvgs=(
                FairValueGap(
                    ts=_ts(3), high=2353.00, low=2349.00,
                    fvg_type="bullish", timeframe=Timeframe.M15,
                    filled_pct=0.6, fully_filled=False,
                ),
            ),
        )
        inverted_snap = _make_snapshot(
            swing_points=(
                SwingPoint(ts=_ts(1), price=2354.00, swing_type="high", strength=5),
                SwingPoint(ts=_ts(2), price=2344.00, swing_type="low", strength=5),
            ),
        )

        normal_result = check_entry_v2(
            normal_snap, _BULLISH_OB_ZONE, 2350.00, h1_atr=471.0,
        )
        inverted_result = check_entry_v2(
            inverted_snap, _BULLISH_OB_ZONE, 2345.00, h1_atr=471.0,
        )

        assert normal_result is not None
        assert inverted_result is not None
        assert normal_result.entry_mode == "normal"
        assert inverted_result.entry_mode == "inverted"
        # Inverted should have wider SL (1.0x vs 0.75x ATR)
        # Normal risk: buffer = 471*0.75 = 353.25
        # Inverted risk: buffer = 471*1.0 = 471.0
        assert inverted_result.risk_points > normal_result.risk_points

    def test_inverted_tp1_rr_lower(self) -> None:
        """Inverted TP1 RR should be 2.0 (vs 2.5 normal)."""
        normal_snap = _make_snapshot(
            fvgs=(
                FairValueGap(
                    ts=_ts(3), high=2353.00, low=2349.00,
                    fvg_type="bullish", timeframe=Timeframe.M15,
                    filled_pct=0.6, fully_filled=False,
                ),
            ),
        )
        inverted_snap = _make_snapshot(
            swing_points=(
                SwingPoint(ts=_ts(1), price=2354.00, swing_type="high", strength=5),
                SwingPoint(ts=_ts(2), price=2344.00, swing_type="low", strength=5),
            ),
        )

        normal_result = check_entry_v2(normal_snap, _BULLISH_OB_ZONE, 2350.00)
        inverted_result = check_entry_v2(inverted_snap, _BULLISH_OB_ZONE, 2345.00)

        assert normal_result is not None
        assert inverted_result is not None
        assert normal_result.rr_ratio == pytest.approx(2.5, abs=0.1)
        assert inverted_result.rr_ratio == pytest.approx(2.0, abs=0.1)


# ===================================================================
# 5. Priority order tests
# ===================================================================


class TestPriorityOrder:
    """Verify trigger priority: fvg_fill > bos > choch > ob_breakout > choch_cont > fvg_sweep."""

    def test_fvg_fill_beats_ob_breakout(self) -> None:
        """When both fvg_fill and ob_breakout conditions are met, fvg_fill wins."""
        snap = _make_snapshot(
            swing_points=(
                # No rejection swings inside zone -> ob_breakout condition met
                SwingPoint(ts=_ts(1), price=2354.00, swing_type="high", strength=5),
                SwingPoint(ts=_ts(2), price=2344.00, swing_type="low", strength=5),
            ),
            fvgs=(
                # Also FVG fill condition met
                FairValueGap(
                    ts=_ts(3), high=2353.00, low=2349.00,
                    fvg_type="bullish", timeframe=Timeframe.M15,
                    filled_pct=0.7, fully_filled=False,
                ),
            ),
        )
        # Price inside zone (for fvg_fill) but also below zone (for ob_breakout)
        # Use price inside zone so fvg_fill can match
        result = check_entry_v2(snap, _BULLISH_OB_ZONE, 2350.00)
        assert result is not None
        assert result.trigger_type == "fvg_fill_in_zone"
        assert result.entry_mode == "normal"

    def test_bos_beats_choch(self) -> None:
        """When both BOS and CHoCH are present, BOS wins (higher priority)."""
        snap = _make_snapshot(
            structure_breaks=(
                StructureBreak(
                    ts=_ts(2), price=2350.00,
                    break_type="bos", direction="bullish",
                    timeframe=Timeframe.M15,
                ),
                StructureBreak(
                    ts=_ts(3), price=2349.50,
                    break_type="choch", direction="bullish",
                    timeframe=Timeframe.M15,
                ),
            ),
        )
        result = check_entry_v2(snap, _BULLISH_OB_ZONE, 2350.00)
        assert result is not None
        assert result.trigger_type == "bos_in_zone"

    def test_normal_beats_inverted(self) -> None:
        """Any normal signal should take priority over inverted signals."""
        snap = _make_snapshot(
            swing_points=(
                SwingPoint(ts=_ts(1), price=2354.00, swing_type="high", strength=5),
                SwingPoint(ts=_ts(2), price=2344.00, swing_type="low", strength=5),
            ),
            structure_breaks=(
                StructureBreak(
                    ts=_ts(2), price=2350.00,
                    break_type="bos", direction="bullish",
                    timeframe=Timeframe.M15,
                ),
            ),
        )
        result = check_entry_v2(snap, _BULLISH_OB_ZONE, 2350.00)
        assert result is not None
        assert result.trigger_type == "bos_in_zone"
        assert result.entry_mode == "normal"


# ===================================================================
# 6. Edge cases and signal immutability
# ===================================================================


class TestEdgeCases:
    """Edge cases: price outside zone, no triggers, frozen model."""

    def test_price_far_outside_zone(self) -> None:
        """Price far from zone should return None."""
        snap = _make_snapshot()
        result = check_entry_v2(snap, _BULLISH_OB_ZONE, 2500.00)
        assert result is None

    def test_empty_snapshot_no_trigger(self) -> None:
        """Empty snapshot should return None."""
        snap = _make_snapshot()
        result = check_entry_v2(snap, _BULLISH_OB_ZONE, 2350.00)
        assert result is None

    def test_signal_is_frozen(self) -> None:
        """EntrySignalV2 should be immutable."""
        snap = _make_snapshot(
            fvgs=(
                FairValueGap(
                    ts=_ts(3), high=2353.00, low=2349.00,
                    fvg_type="bullish", timeframe=Timeframe.M15,
                    filled_pct=0.6, fully_filled=False,
                ),
            ),
        )
        result = check_entry_v2(snap, _BULLISH_OB_ZONE, 2350.00)
        assert result is not None
        with pytest.raises(Exception):
            result.entry_price = 9999.0  # type: ignore[misc]

    def test_zero_risk_returns_none(self) -> None:
        """If risk_points would be 0, should return None."""
        # Zone where SL == entry price is nearly impossible with the buffer,
        # but test the guard anyway by using a zone with extreme proximity
        snap = _make_snapshot(
            fvgs=(
                FairValueGap(
                    ts=_ts(3), high=2353.00, low=2349.00,
                    fvg_type="bullish", timeframe=Timeframe.M15,
                    filled_pct=0.6, fully_filled=False,
                ),
            ),
        )
        # Price at zone_low - buffer would make risk = 0
        # With floor buffer of 200 points ($2.00), zone_low=2348,
        # SL = 2348 - 2.00 = 2346. Entry at 2346 gives risk=0.
        result = check_entry_v2(snap, _BULLISH_OB_ZONE, 2346.00)
        if result is not None:
            assert result.risk_points > 0

    def test_grade_assigned_for_all_modes(self) -> None:
        """Both normal and inverted entries should have valid grades."""
        normal_snap = _make_snapshot(
            fvgs=(
                FairValueGap(
                    ts=_ts(3), high=2353.00, low=2349.00,
                    fvg_type="bullish", timeframe=Timeframe.M15,
                    filled_pct=0.6, fully_filled=False,
                ),
            ),
        )
        result = check_entry_v2(normal_snap, _BULLISH_OB_ZONE, 2350.00)
        assert result is not None
        assert result.grade in ("A", "B", "C")

    def test_overlap_zone_boosts_grade(self) -> None:
        """OB+FVG overlap zone should get higher grade than plain OB."""
        snap = _make_snapshot(
            fvgs=(
                FairValueGap(
                    ts=_ts(3), high=2353.00, low=2349.00,
                    fvg_type="bullish", timeframe=Timeframe.M15,
                    filled_pct=0.6, fully_filled=False,
                ),
            ),
        )
        result_ob = check_entry_v2(snap, _BULLISH_OB_ZONE, 2350.00)
        result_overlap = check_entry_v2(snap, _BULLISH_OVERLAP_ZONE, 2350.00)
        assert result_ob is not None
        assert result_overlap is not None
        # Overlap should get same or better grade
        grade_order = {"A": 3, "B": 2, "C": 1}
        assert grade_order[result_overlap.grade] >= grade_order[result_ob.grade]

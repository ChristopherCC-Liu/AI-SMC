"""Unit tests for smc.strategy.entry_trigger — M15 precise entry logic."""

from __future__ import annotations

import pytest

from smc.smc_core.types import SMCSnapshot
from smc.strategy.entry_trigger import check_entry
from smc.strategy.types import EntrySignal, TradeZone


class TestCheckEntry:
    def test_choch_in_zone_triggers(
        self,
        m15_choch_in_zone_snapshot: SMCSnapshot,
        bullish_ob_zone: TradeZone,
    ) -> None:
        """M15 CHoCH inside bullish OB zone should trigger entry."""
        # Price inside the zone (2348-2352)
        result = check_entry(m15_choch_in_zone_snapshot, bullish_ob_zone, 2350.00)
        assert result is not None
        assert result.trigger_type == "choch_in_zone"
        assert result.direction == "long"

    def test_price_outside_zone_no_trigger(
        self,
        m15_choch_in_zone_snapshot: SMCSnapshot,
        bullish_ob_zone: TradeZone,
    ) -> None:
        """Price far from zone should not trigger."""
        result = check_entry(m15_choch_in_zone_snapshot, bullish_ob_zone, 2400.00)
        assert result is None

    def test_no_confirmation_no_trigger(
        self,
        m15_no_trigger_snapshot: SMCSnapshot,
        bullish_ob_zone: TradeZone,
    ) -> None:
        """No CHoCH/FVG/rejection in zone → no entry."""
        result = check_entry(m15_no_trigger_snapshot, bullish_ob_zone, 2350.00)
        assert result is None

    def test_stop_loss_beyond_zone(
        self,
        m15_choch_in_zone_snapshot: SMCSnapshot,
        bullish_ob_zone: TradeZone,
    ) -> None:
        """SL should be below zone low + buffer for long entries."""
        result = check_entry(m15_choch_in_zone_snapshot, bullish_ob_zone, 2350.00)
        assert result is not None
        # SL should be below the zone low (2348.00)
        assert result.stop_loss < bullish_ob_zone.zone_low

    def test_tp1_at_2rr(
        self,
        m15_choch_in_zone_snapshot: SMCSnapshot,
        bullish_ob_zone: TradeZone,
    ) -> None:
        """TP1 should be at approximately 1:2 RR."""
        result = check_entry(m15_choch_in_zone_snapshot, bullish_ob_zone, 2350.00)
        assert result is not None
        assert result.rr_ratio == pytest.approx(2.0, abs=0.1)

    def test_tp2_at_liquidity_level(
        self,
        m15_choch_in_zone_snapshot: SMCSnapshot,
        bullish_ob_zone: TradeZone,
    ) -> None:
        """TP2 should target the next liquidity level if available."""
        result = check_entry(m15_choch_in_zone_snapshot, bullish_ob_zone, 2350.00)
        assert result is not None
        # The fixture has a liquidity level at 2385.00 (equal_highs above price)
        assert result.take_profit_2 == 2385.00

    def test_entry_signal_is_frozen(
        self,
        m15_choch_in_zone_snapshot: SMCSnapshot,
        bullish_ob_zone: TradeZone,
    ) -> None:
        result = check_entry(m15_choch_in_zone_snapshot, bullish_ob_zone, 2350.00)
        assert result is not None
        with pytest.raises(Exception):
            result.entry_price = 9999.0  # type: ignore[misc]

    def test_risk_points_positive(
        self,
        m15_choch_in_zone_snapshot: SMCSnapshot,
        bullish_ob_zone: TradeZone,
    ) -> None:
        result = check_entry(m15_choch_in_zone_snapshot, bullish_ob_zone, 2350.00)
        assert result is not None
        assert result.risk_points > 0
        assert result.reward_points > 0

    def test_grade_assigned(
        self,
        m15_choch_in_zone_snapshot: SMCSnapshot,
        bullish_ob_zone: TradeZone,
    ) -> None:
        result = check_entry(m15_choch_in_zone_snapshot, bullish_ob_zone, 2350.00)
        assert result is not None
        assert result.grade in ("A", "B", "C")

    def test_ob_fvg_overlap_zone_fvg_fill_trigger(
        self,
        m15_choch_in_zone_snapshot: SMCSnapshot,
        bullish_overlap_zone: TradeZone,
    ) -> None:
        """OB+FVG overlap zone with FVG fill should also trigger."""
        result = check_entry(m15_choch_in_zone_snapshot, bullish_overlap_zone, 2350.00)
        assert result is not None
        # Should trigger (choch takes priority over fvg_fill)
        assert result.trigger_type in ("choch_in_zone", "fvg_fill_in_zone")

    def test_entry_price_matches_current(
        self,
        m15_choch_in_zone_snapshot: SMCSnapshot,
        bullish_ob_zone: TradeZone,
    ) -> None:
        price = 2350.50
        result = check_entry(m15_choch_in_zone_snapshot, bullish_ob_zone, price)
        assert result is not None
        assert result.entry_price == round(price, 2)


class TestCheckEntryShortDirection:
    """Tests for short/bearish entry scenarios."""

    def test_short_sl_above_zone(self) -> None:
        """For short entries, SL should be above the zone high + buffer."""
        from tests.smc.unit.strategy.conftest import _make_snapshot, _ts

        from smc.data.schemas import Timeframe
        from smc.smc_core.types import (
            FairValueGap,
            LiquidityLevel,
            StructureBreak,
            SwingPoint,
        )

        bearish_zone = TradeZone(
            zone_high=2380.00, zone_low=2376.00, zone_type="ob",
            direction="short", timeframe=Timeframe.H1, confidence=0.8,
        )
        m15_snap = _make_snapshot(
            timeframe=Timeframe.M15,
            trend="bearish",
            swing_points=(
                SwingPoint(ts=_ts(1), price=2382.00, swing_type="high", strength=5),
                SwingPoint(ts=_ts(2), price=2374.00, swing_type="low", strength=5),
                SwingPoint(ts=_ts(3), price=2379.00, swing_type="high", strength=5),
                SwingPoint(ts=_ts(4), price=2370.00, swing_type="low", strength=5),
            ),
            structure_breaks=(
                StructureBreak(ts=_ts(3), price=2378.00, break_type="choch", direction="bearish", timeframe=Timeframe.M15),
            ),
            liquidity_levels=(
                LiquidityLevel(price=2365.00, level_type="equal_lows", touches=2, swept=False),
            ),
        )
        result = check_entry(m15_snap, bearish_zone, 2378.00)
        assert result is not None
        assert result.direction == "short"
        assert result.stop_loss > bearish_zone.zone_high
        assert result.take_profit_1 < result.entry_price
        assert result.take_profit_2 == 2365.00  # Next liquidity level below

    def test_no_liquidity_level_fallback_tp2(self) -> None:
        """When no liquidity level exists, TP2 should use fallback RR."""
        from tests.smc.unit.strategy.conftest import _make_snapshot, _ts

        from smc.data.schemas import Timeframe
        from smc.smc_core.types import StructureBreak, SwingPoint

        zone = TradeZone(
            zone_high=2352.00, zone_low=2348.00, zone_type="ob",
            direction="long", timeframe=Timeframe.H1, confidence=0.8,
        )
        m15_snap = _make_snapshot(
            timeframe=Timeframe.M15,
            trend="bullish",
            swing_points=(
                SwingPoint(ts=_ts(1), price=2346.00, swing_type="low", strength=5),
                SwingPoint(ts=_ts(2), price=2354.00, swing_type="high", strength=5),
            ),
            structure_breaks=(
                StructureBreak(ts=_ts(2), price=2349.00, break_type="choch", direction="bullish", timeframe=Timeframe.M15),
            ),
            # No liquidity levels
        )
        result = check_entry(m15_snap, zone, 2350.00)
        assert result is not None
        # TP2 should be above TP1 (fallback 3:1 RR)
        assert result.take_profit_2 > result.take_profit_1

    def test_fvg_zone_no_fvg_fill_trigger(self) -> None:
        """FVG-only zones should not match the FVG fill trigger (needs OB type)."""
        from tests.smc.unit.strategy.conftest import _make_snapshot, _ts

        from smc.data.schemas import Timeframe
        from smc.smc_core.types import FairValueGap, SwingPoint

        fvg_zone = TradeZone(
            zone_high=2356.00, zone_low=2352.00, zone_type="fvg",
            direction="long", timeframe=Timeframe.H1, confidence=0.6,
        )
        m15_snap = _make_snapshot(
            timeframe=Timeframe.M15,
            trend="bullish",
            swing_points=(
                SwingPoint(ts=_ts(1), price=2350.00, swing_type="low", strength=5),
                SwingPoint(ts=_ts(2), price=2358.00, swing_type="high", strength=5),
            ),
            fvgs=(
                FairValueGap(ts=_ts(3), high=2355.00, low=2353.00, fvg_type="bullish", timeframe=Timeframe.M15, filled_pct=0.8, fully_filled=False),
            ),
        )
        result = check_entry(m15_snap, fvg_zone, 2354.00)
        # Should be None since FVG fill trigger requires OB or overlap zone type
        # and there's no CHoCH or OB rejection either
        assert result is None

    def test_ob_rejection_trigger(self) -> None:
        """Test the OB test + rejection trigger path."""
        from tests.smc.unit.strategy.conftest import _make_snapshot, _ts

        from smc.data.schemas import Timeframe
        from smc.smc_core.types import SwingPoint

        zone = TradeZone(
            zone_high=2352.00, zone_low=2348.00, zone_type="ob",
            direction="long", timeframe=Timeframe.H1, confidence=0.8,
        )
        # No CHoCH, no FVG fill, but swing low inside zone + price above
        m15_snap = _make_snapshot(
            timeframe=Timeframe.M15,
            trend="bullish",
            swing_points=(
                SwingPoint(ts=_ts(1), price=2355.00, swing_type="high", strength=5),
                SwingPoint(ts=_ts(2), price=2349.00, swing_type="low", strength=5),
                SwingPoint(ts=_ts(3), price=2356.00, swing_type="high", strength=5),
                SwingPoint(ts=_ts(4), price=2350.00, swing_type="low", strength=5),
            ),
        )
        result = check_entry(m15_snap, zone, 2351.00)
        assert result is not None
        assert result.trigger_type == "ob_test_rejection"

    def test_bearish_ob_rejection_trigger(self) -> None:
        """Test bearish OB rejection: swing high inside zone + price below."""
        from tests.smc.unit.strategy.conftest import _make_snapshot, _ts

        from smc.data.schemas import Timeframe
        from smc.smc_core.types import SwingPoint

        zone = TradeZone(
            zone_high=2380.00, zone_low=2376.00, zone_type="ob",
            direction="short", timeframe=Timeframe.H1, confidence=0.8,
        )
        m15_snap = _make_snapshot(
            timeframe=Timeframe.M15,
            trend="bearish",
            swing_points=(
                SwingPoint(ts=_ts(1), price=2374.00, swing_type="low", strength=5),
                SwingPoint(ts=_ts(2), price=2379.00, swing_type="high", strength=5),
                SwingPoint(ts=_ts(3), price=2373.00, swing_type="low", strength=5),
                SwingPoint(ts=_ts(4), price=2378.00, swing_type="high", strength=5),
            ),
        )
        result = check_entry(m15_snap, zone, 2377.00)
        assert result is not None
        assert result.trigger_type == "ob_test_rejection"
        assert result.direction == "short"

    def test_grade_scoring_logic(self) -> None:
        """Verify grade assignment: choch + overlap + high RR = A grade."""
        from smc.strategy.entry_trigger import _grade_entry

        grade = _grade_entry(
            "choch_in_zone",
            TradeZone(zone_high=2355.0, zone_low=2350.0, zone_type="ob_fvg_overlap",
                      direction="long", timeframe="H1", confidence=0.95),  # type: ignore[arg-type]
            3.0,
        )
        assert grade == "A"

        grade_c = _grade_entry(
            "ob_test_rejection",
            TradeZone(zone_high=2355.0, zone_low=2350.0, zone_type="fvg",
                      direction="long", timeframe="H1", confidence=0.5),  # type: ignore[arg-type]
            1.2,
        )
        assert grade_c == "C"

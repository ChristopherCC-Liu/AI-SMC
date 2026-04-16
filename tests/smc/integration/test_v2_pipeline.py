"""Integration tests for the v2 strategy pipeline.

Tests full v2 pipeline: direction → detect → zones → entry → confluence → setups.
Also verifies neutral AI direction produces inverted setups, and that v1/v2
can coexist without import conflicts.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from smc.ai.direction_engine import DirectionEngine
from smc.ai.models import AIDirection
from smc.data.schemas import Timeframe
from smc.smc_core.detector import SMCDetector
from smc.smc_core.types import (
    FairValueGap,
    LiquidityLevel,
    OrderBlock,
    SMCSnapshot,
    StructureBreak,
    SwingPoint,
)
from smc.strategy.aggregator_v2 import AggregatorV2
from smc.strategy.confluence_v2 import TRADEABLE_THRESHOLD_V2, score_confluence_v2
from smc.strategy.types import EntrySignalV2, TradeSetupV2, TradeZone
from smc.strategy.zone_scanner_v2 import scan_zones_v2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TS = datetime(2024, 7, 15, 0, 0, 0, tzinfo=timezone.utc)


def _ts(hours: int) -> datetime:
    return _BASE_TS + timedelta(hours=hours)


def _make_snapshot(
    *,
    timeframe: Timeframe,
    trend: str = "bullish",
    swing_points: tuple[SwingPoint, ...] = (),
    order_blocks: tuple[OrderBlock, ...] = (),
    fvgs: tuple[FairValueGap, ...] = (),
    structure_breaks: tuple[StructureBreak, ...] = (),
    liquidity_levels: tuple[LiquidityLevel, ...] = (),
) -> SMCSnapshot:
    return SMCSnapshot(
        ts=_ts(24),
        timeframe=timeframe,
        swing_points=swing_points,
        order_blocks=order_blocks,
        fvgs=fvgs,
        structure_breaks=structure_breaks,
        liquidity_levels=liquidity_levels,
        trend_direction=trend,  # type: ignore[arg-type]
    )


def _make_h1_bullish_snapshot() -> SMCSnapshot:
    """H1 snapshot with bullish OBs and FVGs for zone scanning."""
    return _make_snapshot(
        timeframe=Timeframe.H1,
        trend="bullish",
        order_blocks=(
            OrderBlock(
                ts_start=_ts(2), ts_end=_ts(3),
                high=2352.00, low=2348.00,
                ob_type="bullish", timeframe=Timeframe.H1, mitigated=False,
            ),
            OrderBlock(
                ts_start=_ts(10), ts_end=_ts(11),
                high=2362.00, low=2358.00,
                ob_type="bullish", timeframe=Timeframe.H1, mitigated=False,
            ),
        ),
        fvgs=(
            FairValueGap(
                ts=_ts(4), high=2356.00, low=2352.00,
                fvg_type="bullish", timeframe=Timeframe.H1,
                filled_pct=0.0, fully_filled=False,
            ),
        ),
        structure_breaks=(
            StructureBreak(
                ts=_ts(6), price=2365.00,
                break_type="bos", direction="bullish", timeframe=Timeframe.H1,
            ),
        ),
        liquidity_levels=(
            LiquidityLevel(price=2340.00, level_type="equal_lows", touches=3, swept=False),
            LiquidityLevel(price=2385.00, level_type="equal_highs", touches=2, swept=False),
        ),
    )


def _make_m15_choch_snapshot() -> SMCSnapshot:
    """M15 snapshot with bullish CHoCH inside the OB zone (2348-2352)."""
    return _make_snapshot(
        timeframe=Timeframe.M15,
        trend="bullish",
        swing_points=(
            SwingPoint(ts=_ts(1), price=2346.00, swing_type="low", strength=5),
            SwingPoint(ts=_ts(2), price=2354.00, swing_type="high", strength=5),
            SwingPoint(ts=_ts(3), price=2349.00, swing_type="low", strength=5),
            SwingPoint(ts=_ts(4), price=2358.00, swing_type="high", strength=5),
        ),
        structure_breaks=(
            StructureBreak(
                ts=_ts(2), price=2349.50,
                break_type="choch", direction="bullish", timeframe=Timeframe.M15,
            ),
        ),
        fvgs=(
            FairValueGap(
                ts=_ts(3), high=2353.00, low=2349.00,
                fvg_type="bullish", timeframe=Timeframe.M15,
                filled_pct=0.6, fully_filled=False,
            ),
        ),
        liquidity_levels=(
            LiquidityLevel(price=2385.00, level_type="equal_highs", touches=2, swept=False),
            LiquidityLevel(price=2340.00, level_type="equal_lows", touches=2, swept=False),
        ),
    )


def _make_m15_ob_breakout_snapshot() -> SMCSnapshot:
    """M15 snapshot with OB breakout (no rejection, price below zone)."""
    return _make_snapshot(
        timeframe=Timeframe.M15,
        trend="bearish",
        swing_points=(
            SwingPoint(ts=_ts(1), price=2354.00, swing_type="high", strength=5),
            SwingPoint(ts=_ts(2), price=2346.00, swing_type="low", strength=5),
            SwingPoint(ts=_ts(3), price=2350.00, swing_type="high", strength=5),
            SwingPoint(ts=_ts(4), price=2342.00, swing_type="low", strength=5),
        ),
        structure_breaks=(
            StructureBreak(
                ts=_ts(3), price=2346.00,
                break_type="bos", direction="bearish", timeframe=Timeframe.M15,
            ),
        ),
        liquidity_levels=(
            LiquidityLevel(price=2330.00, level_type="equal_lows", touches=3, swept=False),
        ),
    )


def _make_bullish_ai_direction(confidence: float = 0.7) -> AIDirection:
    return AIDirection(
        direction="bullish",
        confidence=confidence,
        key_drivers=("sma50_up", "dxy_weakening"),
        reasoning="H4 SMA50 trending up, DXY weakening",
        assessed_at=_ts(0),
        source="sma_fallback",
        cost_usd=0.0,
    )


def _make_neutral_ai_direction(confidence: float = 0.35) -> AIDirection:
    return AIDirection(
        direction="neutral",
        confidence=confidence,
        key_drivers=("sma50_flat",),
        reasoning="H4 SMA50 flat — no directional bias",
        assessed_at=_ts(0),
        source="sma_fallback",
        cost_usd=0.0,
    )


def _make_h4_ohlcv_df(n_bars: int = 60) -> pl.DataFrame:
    """Build a minimal H4 OHLCV DataFrame with rising trend."""
    ts_list = [_ts(i * 4) for i in range(n_bars)]
    base = 2300.0
    return pl.DataFrame({
        "ts": ts_list,
        "open": [base + i * 0.5 for i in range(n_bars)],
        "high": [base + i * 0.5 + 3.0 for i in range(n_bars)],
        "low": [base + i * 0.5 - 2.0 for i in range(n_bars)],
        "close": [base + i * 0.5 + 1.0 for i in range(n_bars)],
        "volume": [100] * n_bars,
    })


def _make_h1_ohlcv_df(n_bars: int = 30) -> pl.DataFrame:
    """Build a minimal H1 OHLCV DataFrame."""
    ts_list = [_ts(i) for i in range(n_bars)]
    base = 2340.0
    return pl.DataFrame({
        "ts": ts_list,
        "open": [base + i * 0.3 for i in range(n_bars)],
        "high": [base + i * 0.3 + 2.0 for i in range(n_bars)],
        "low": [base + i * 0.3 - 1.5 for i in range(n_bars)],
        "close": [base + i * 0.3 + 0.5 for i in range(n_bars)],
        "volume": [100] * n_bars,
    })


# ===========================================================================
# Tests: zone_scanner_v2
# ===========================================================================


class TestZoneScannerV2:
    """Test scan_zones_v2 with relaxed limits."""

    def test_max_zones_is_5(self) -> None:
        """V2 returns up to 5 zones (was 3 in v1)."""
        h1 = _make_snapshot(
            timeframe=Timeframe.H1,
            trend="bullish",
            order_blocks=tuple(
                OrderBlock(
                    ts_start=_ts(i * 2), ts_end=_ts(i * 2 + 1),
                    high=2350.00 + i * 20.0, low=2346.00 + i * 20.0,
                    ob_type="bullish", timeframe=Timeframe.H1, mitigated=False,
                )
                for i in range(7)
            ),
        )
        zones = scan_zones_v2(h1, "bullish")
        assert len(zones) == 5  # capped at _MAX_ZONES=5

    def test_neutral_returns_empty(self) -> None:
        """Neutral bias returns no zones."""
        h1 = _make_h1_bullish_snapshot()
        zones = scan_zones_v2(h1, "neutral")
        assert zones == ()

    def test_accepts_string_bias(self) -> None:
        """V2 accepts plain string bias instead of BiasDirection object."""
        h1 = _make_h1_bullish_snapshot()
        zones = scan_zones_v2(h1, "bullish")
        assert len(zones) > 0
        assert all(z.direction == "long" for z in zones)

    def test_sorted_by_confidence(self) -> None:
        """Zones are sorted by confidence descending."""
        h1 = _make_h1_bullish_snapshot()
        zones = scan_zones_v2(h1, "bullish")
        confidences = [z.confidence for z in zones]
        assert confidences == sorted(confidences, reverse=True)


# ===========================================================================
# Tests: confluence_v2
# ===========================================================================


class TestConfluenceV2:
    """Test score_confluence_v2 with AI direction weight + mode bonus."""

    @pytest.fixture()
    def normal_entry(self) -> EntrySignalV2:
        return EntrySignalV2(
            entry_price=2350.00,
            stop_loss=2347.70,
            take_profit_1=2355.75,
            take_profit_2=2385.00,
            risk_points=230.0,
            reward_points=575.0,
            rr_ratio=2.5,
            direction="long",
            grade="A",
            trigger_type="choch_in_zone",
            entry_mode="normal",
            inversion_confidence=1.0,
        )

    @pytest.fixture()
    def inverted_entry(self) -> EntrySignalV2:
        return EntrySignalV2(
            entry_price=2345.00,
            stop_loss=2352.00,
            take_profit_1=2331.00,
            take_profit_2=2324.00,
            risk_points=700.0,
            reward_points=1400.0,
            rr_ratio=2.0,
            direction="short",
            grade="B",
            trigger_type="ob_breakout",
            entry_mode="inverted",
            inversion_confidence=0.769,
        )

    @pytest.fixture()
    def bullish_zone(self) -> TradeZone:
        return TradeZone(
            zone_high=2352.00,
            zone_low=2348.00,
            zone_type="ob",
            direction="long",
            timeframe=Timeframe.H1,
            confidence=0.8,
        )

    def test_threshold_is_0_40(self) -> None:
        assert TRADEABLE_THRESHOLD_V2 == 0.40

    def test_normal_entry_high_confidence(
        self, normal_entry: EntrySignalV2, bullish_zone: TradeZone,
    ) -> None:
        score = score_confluence_v2(0.8, bullish_zone, normal_entry)
        assert score >= TRADEABLE_THRESHOLD_V2
        assert 0.0 <= score <= 1.0

    def test_inverted_entry_scores(
        self, inverted_entry: EntrySignalV2, bullish_zone: TradeZone,
    ) -> None:
        score = score_confluence_v2(0.4, bullish_zone, inverted_entry)
        assert 0.0 <= score <= 1.0

    def test_low_confidence_reduces_score(
        self, normal_entry: EntrySignalV2, bullish_zone: TradeZone,
    ) -> None:
        high = score_confluence_v2(0.9, bullish_zone, normal_entry)
        low = score_confluence_v2(0.2, bullish_zone, normal_entry)
        assert high > low


# ===========================================================================
# Tests: AggregatorV2 full pipeline
# ===========================================================================


class TestAggregatorV2Pipeline:
    """Integration tests for the full v2 pipeline."""

    def _build_aggregator(
        self,
        ai_direction: AIDirection,
    ) -> AggregatorV2:
        """Build AggregatorV2 with a mocked DirectionEngine."""
        detector = SMCDetector(swing_length=10)
        engine = DirectionEngine()
        engine.get_direction = MagicMock(return_value=ai_direction)  # type: ignore[method-assign]
        return AggregatorV2(
            detector=detector,
            direction_engine=engine,
            enable_inverted=True,
            enable_fvg_sweep=True,
        )

    def _build_data(self) -> dict[Timeframe, pl.DataFrame]:
        """Build minimal OHLCV data for H4, H1, M15."""
        return {
            Timeframe.H4: _make_h4_ohlcv_df(60),
            Timeframe.H1: _make_h1_ohlcv_df(30),
            Timeframe.M15: _make_h1_ohlcv_df(30),
        }

    def test_bullish_direction_generates_setups(self) -> None:
        """Bullish AI direction should generate long setups."""
        ai_dir = _make_bullish_ai_direction(0.7)
        agg = self._build_aggregator(ai_dir)

        # Mock the detector to return controlled snapshots
        agg._detector.detect = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda df, tf: {
                Timeframe.H4: _make_snapshot(timeframe=Timeframe.H4, trend="bullish"),
                Timeframe.H1: _make_h1_bullish_snapshot(),
                Timeframe.M15: _make_m15_choch_snapshot(),
            }.get(tf, _make_snapshot(timeframe=tf)),
        )

        data = self._build_data()
        setups = agg.generate_setups(data, current_price=2350.00)

        # Should find at least one setup with the CHoCH trigger in zone
        assert isinstance(setups, tuple)
        for setup in setups:
            assert isinstance(setup, TradeSetupV2)
            assert setup.ai_direction == "bullish"
            assert setup.confluence_score >= TRADEABLE_THRESHOLD_V2

    def test_very_low_confidence_blocked(self) -> None:
        """AI confidence < 0.2 should produce no setups (neutral gate)."""
        ai_dir = AIDirection(
            direction="bullish",
            confidence=0.1,  # below 0.2 gate
            key_drivers=("weak_signal",),
            reasoning="Very weak signal",
            assessed_at=_ts(0),
            source="sma_fallback",
            cost_usd=0.0,
        )
        agg = self._build_aggregator(ai_dir)
        data = self._build_data()
        setups = agg.generate_setups(data, current_price=2350.00)
        assert setups == ()

    def test_neutral_direction_inverted_only(self) -> None:
        """Neutral AI direction should only produce inverted setups."""
        ai_dir = _make_neutral_ai_direction(0.35)
        agg = self._build_aggregator(ai_dir)

        # Mock detector to return snapshots with OB breakout conditions
        agg._detector.detect = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda df, tf: {
                Timeframe.H4: _make_snapshot(timeframe=Timeframe.H4, trend="ranging"),
                Timeframe.H1: _make_h1_bullish_snapshot(),
                Timeframe.M15: _make_m15_ob_breakout_snapshot(),
            }.get(tf, _make_snapshot(timeframe=tf)),
        )

        data = self._build_data()
        setups = agg.generate_setups(data, current_price=2345.00)

        # All setups (if any) should be inverted mode
        for setup in setups:
            assert setup.entry_mode == "inverted"

    def test_zone_cooldown_8h(self) -> None:
        """V2 zone cooldown is 8h (was 24h in v1)."""
        ai_dir = _make_bullish_ai_direction()
        agg = self._build_aggregator(ai_dir)
        assert agg._ZONE_COOLDOWN_HOURS == 8

        # Record a loss
        loss_time = datetime(2024, 7, 15, 10, 0, 0, tzinfo=timezone.utc)
        agg.record_zone_loss(2352.00, 2348.00, "long", loss_time)

        # Check the cooldown
        key = (2352.00, 2348.00, "long")
        assert key in agg._zone_cooldowns
        expected_until = loss_time + timedelta(hours=8)
        assert agg._zone_cooldowns[key] == expected_until

    def test_max_entries_per_zone_is_2(self) -> None:
        """V2 allows 2 entries per zone (was 1 in v1)."""
        ai_dir = _make_bullish_ai_direction()
        agg = self._build_aggregator(ai_dir)
        assert agg._MAX_ENTRIES_PER_ZONE == 2

        # Mark one active — should still allow one more
        agg.mark_zone_active(2352.00, 2348.00, "long")
        key = (2352.00, 2348.00, "long")
        assert agg._active_zones[key] == 1

        # Mark another active — should be at limit
        agg.mark_zone_active(2352.00, 2348.00, "long")
        assert agg._active_zones[key] == 2

    def test_clear_zone_active_decrements(self) -> None:
        """Clearing one active trade decrements the count."""
        ai_dir = _make_bullish_ai_direction()
        agg = self._build_aggregator(ai_dir)

        agg.mark_zone_active(2352.00, 2348.00, "long")
        agg.mark_zone_active(2352.00, 2348.00, "long")

        key = (2352.00, 2348.00, "long")
        assert agg._active_zones[key] == 2

        agg.clear_zone_active(2352.00, 2348.00, "long")
        assert agg._active_zones[key] == 1

        agg.clear_zone_active(2352.00, 2348.00, "long")
        assert key not in agg._active_zones

    def test_max_concurrent_is_5(self) -> None:
        """V2 caps setups at 5 (was 3 in v1 default)."""
        ai_dir = _make_bullish_ai_direction()
        agg = self._build_aggregator(ai_dir)

        # Generate many setups and verify the cap
        # This is a structural test — the 5 cap is in the constants
        from smc.strategy.aggregator_v2 import _MAX_CONCURRENT

        assert _MAX_CONCURRENT == 5

    def test_setups_sorted_by_confluence(self) -> None:
        """Generated setups are sorted by confluence score descending."""
        ai_dir = _make_bullish_ai_direction(0.7)
        agg = self._build_aggregator(ai_dir)

        agg._detector.detect = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda df, tf: {
                Timeframe.H4: _make_snapshot(timeframe=Timeframe.H4, trend="bullish"),
                Timeframe.H1: _make_h1_bullish_snapshot(),
                Timeframe.M15: _make_m15_choch_snapshot(),
            }.get(tf, _make_snapshot(timeframe=tf)),
        )

        data = self._build_data()
        setups = agg.generate_setups(data, current_price=2350.00)
        if len(setups) > 1:
            scores = [s.confluence_score for s in setups]
            assert scores == sorted(scores, reverse=True)


# ===========================================================================
# Tests: V1 and V2 coexistence
# ===========================================================================


class TestV1V2Coexistence:
    """Verify v1 and v2 modules can both import without conflicts."""

    def test_v1_aggregator_import(self) -> None:
        """V1 MultiTimeframeAggregator is still importable."""
        from smc.strategy.aggregator import MultiTimeframeAggregator

        assert MultiTimeframeAggregator._MAX_ENTRIES_PER_ZONE == 1  # v1 value

    def test_v2_aggregator_import(self) -> None:
        """V2 AggregatorV2 is importable alongside v1."""
        from smc.strategy.aggregator_v2 import AggregatorV2

        assert AggregatorV2._MAX_ENTRIES_PER_ZONE == 2  # v2 value

    def test_v1_confluence_import(self) -> None:
        """V1 score_confluence is still importable."""
        from smc.strategy.confluence import TRADEABLE_THRESHOLD, score_confluence

        assert TRADEABLE_THRESHOLD == 0.45

    def test_v2_confluence_import(self) -> None:
        """V2 score_confluence_v2 is importable alongside v1."""
        from smc.strategy.confluence_v2 import (
            TRADEABLE_THRESHOLD_V2,
            score_confluence_v2,
        )

        assert TRADEABLE_THRESHOLD_V2 == 0.40

    def test_v1_zone_scanner_import(self) -> None:
        """V1 scan_zones is still importable."""
        from smc.strategy.zone_scanner import scan_zones

        assert scan_zones is not None

    def test_v2_zone_scanner_import(self) -> None:
        """V2 scan_zones_v2 is importable alongside v1."""
        from smc.strategy.zone_scanner_v2 import scan_zones_v2

        assert scan_zones_v2 is not None

    def test_entry_signals_both_types(self) -> None:
        """Both v1 EntrySignal and v2 EntrySignalV2 coexist."""
        from smc.strategy.types import EntrySignal, EntrySignalV2

        assert "trigger_type" in EntrySignal.model_fields
        assert "entry_mode" in EntrySignalV2.model_fields
        assert "inversion_confidence" in EntrySignalV2.model_fields

    def test_trade_setups_both_types(self) -> None:
        """Both v1 TradeSetup and v2 TradeSetupV2 coexist."""
        from smc.strategy.types import TradeSetup, TradeSetupV2

        assert "bias" in TradeSetup.model_fields  # v1 uses BiasDirection
        assert "ai_direction" in TradeSetupV2.model_fields  # v2 uses string
        assert "entry_mode" in TradeSetupV2.model_fields  # v2 adds mode

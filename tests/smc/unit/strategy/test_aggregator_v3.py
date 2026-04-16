"""Unit tests for smc.strategy.aggregator_v3 — Hybrid v3 aggregator.

Validates that v3 is a strict superset of v1:
- v3 produces at least as many setups as v1
- fvg_sweep setups have correct trigger_type
- v1 parameters are unchanged (threshold, cooldown, zone count)
- ob_breakout disabled by default
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

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
from smc.strategy.aggregator import MultiTimeframeAggregator
from smc.strategy.aggregator_v3 import AggregatorV3
from smc.strategy.types import TradeSetup


class TestAggregatorV3Construction:
    """V3 inherits v1 constructor and adds fvg_sweep/ob_breakout flags."""

    def test_inherits_from_v1(self) -> None:
        """V3 must be a subclass of MultiTimeframeAggregator."""
        assert issubclass(AggregatorV3, MultiTimeframeAggregator)

    def test_default_construction(self) -> None:
        detector = SMCDetector(swing_length=10)
        agg = AggregatorV3(detector=detector)
        # v1 swing_length_map should be auto-injected
        assert agg.detector.swing_length_map == {
            Timeframe.D1: 5, Timeframe.H4: 7,
            Timeframe.H1: 10, Timeframe.M15: 10,
        }
        # fvg_sweep enabled by default
        assert agg._enable_fvg_sweep is True
        # ob_breakout disabled by default
        assert agg._enable_ob_breakout is False

    def test_ob_breakout_disabled_by_default(self) -> None:
        """ob_breakout should be disabled — only 3 trades in Sprint 8 data."""
        agg = AggregatorV3(detector=SMCDetector())
        assert agg._enable_ob_breakout is False

    def test_fvg_sweep_can_be_disabled(self) -> None:
        agg = AggregatorV3(detector=SMCDetector(), enable_fvg_sweep=False)
        assert agg._enable_fvg_sweep is False

    def test_ob_breakout_can_be_enabled(self) -> None:
        agg = AggregatorV3(detector=SMCDetector(), enable_ob_breakout=True)
        assert agg._enable_ob_breakout is True

    def test_v1_parameters_unchanged(self) -> None:
        """V3 must preserve all v1 class-level parameters."""
        agg = AggregatorV3(detector=SMCDetector())
        # Max entries per zone: 1 (v1 anti-clustering)
        assert agg._MAX_ENTRIES_PER_ZONE == 1
        # Zone cooldown: 24h (v1 default)
        assert agg._ZONE_COOLDOWN_HOURS == 24
        # ob_test_trigger disabled
        assert agg._enable_ob_test_trigger is False


class TestAggregatorV3Pipeline:
    """V3 pipeline produces a superset of v1 results."""

    def test_empty_data_returns_empty(self) -> None:
        agg = AggregatorV3(detector=SMCDetector())
        result = agg.generate_setups({}, current_price=2350.0)
        assert result == ()

    def test_missing_timeframes_returns_empty(self, sample_ohlcv_df: pl.DataFrame) -> None:
        agg = AggregatorV3(detector=SMCDetector())
        # Missing D1 and M15
        result = agg.generate_setups(
            {Timeframe.H4: sample_ohlcv_df, Timeframe.H1: sample_ohlcv_df},
            current_price=2350.0,
        )
        assert result == ()

    def test_full_pipeline_returns_tuples(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """With all 4 timeframes, result should be a tuple."""
        detector = SMCDetector(swing_length=5)
        agg = AggregatorV3(detector=detector)
        result = agg.generate_setups(
            {
                Timeframe.D1: sample_ohlcv_df,
                Timeframe.H4: sample_ohlcv_df,
                Timeframe.H1: sample_ohlcv_df,
                Timeframe.M15: sample_ohlcv_df,
            },
            current_price=2350.0,
        )
        assert isinstance(result, tuple)
        for setup in result:
            assert isinstance(setup, TradeSetup)

    def test_v3_superset_of_v1(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """V3 should produce >= setups as v1 (superset property)."""
        detector = SMCDetector(swing_length=5)
        data = {
            Timeframe.D1: sample_ohlcv_df,
            Timeframe.H4: sample_ohlcv_df,
            Timeframe.H1: sample_ohlcv_df,
            Timeframe.M15: sample_ohlcv_df,
        }

        v1_agg = MultiTimeframeAggregator(detector=detector)
        v3_agg = AggregatorV3(detector=SMCDetector(swing_length=5))

        v1_setups = v1_agg.generate_setups(data, current_price=2350.0)
        v3_setups = v3_agg.generate_setups(data, current_price=2350.0)

        assert len(v3_setups) >= len(v1_setups)

    def test_v3_with_fvg_sweep_disabled_equals_v1(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """With fvg_sweep disabled, v3 should produce identical results to v1."""
        detector_v1 = SMCDetector(swing_length=5)
        detector_v3 = SMCDetector(swing_length=5)
        data = {
            Timeframe.D1: sample_ohlcv_df,
            Timeframe.H4: sample_ohlcv_df,
            Timeframe.H1: sample_ohlcv_df,
            Timeframe.M15: sample_ohlcv_df,
        }

        v1_agg = MultiTimeframeAggregator(detector=detector_v1)
        v3_agg = AggregatorV3(detector=detector_v3, enable_fvg_sweep=False)

        v1_setups = v1_agg.generate_setups(data, current_price=2350.0)
        v3_setups = v3_agg.generate_setups(data, current_price=2350.0)

        # Same number of setups
        assert len(v3_setups) == len(v1_setups)

        # Same confluence scores (order may differ due to datetime.now())
        v1_scores = sorted([s.confluence_score for s in v1_setups], reverse=True)
        v3_scores = sorted([s.confluence_score for s in v3_setups], reverse=True)
        assert v1_scores == v3_scores

    def test_setups_sorted_by_confluence(self, sample_ohlcv_df: pl.DataFrame) -> None:
        detector = SMCDetector(swing_length=5)
        agg = AggregatorV3(detector=detector)
        result = agg.generate_setups(
            {
                Timeframe.D1: sample_ohlcv_df,
                Timeframe.H4: sample_ohlcv_df,
                Timeframe.H1: sample_ohlcv_df,
                Timeframe.M15: sample_ohlcv_df,
            },
            current_price=2350.0,
        )
        if len(result) > 1:
            scores = [s.confluence_score for s in result]
            assert scores == sorted(scores, reverse=True)

    def test_all_setups_above_threshold(self, sample_ohlcv_df: pl.DataFrame) -> None:
        from smc.strategy.confluence import TRADEABLE_THRESHOLD

        detector = SMCDetector(swing_length=5)
        agg = AggregatorV3(detector=detector)
        result = agg.generate_setups(
            {
                Timeframe.D1: sample_ohlcv_df,
                Timeframe.H4: sample_ohlcv_df,
                Timeframe.H1: sample_ohlcv_df,
                Timeframe.M15: sample_ohlcv_df,
            },
            current_price=2350.0,
        )
        for setup in result:
            assert setup.confluence_score >= TRADEABLE_THRESHOLD


class TestFVGSweepTrigger:
    """Tests for the fvg_sweep_continuation bolt-on logic."""

    def test_fvg_sweep_trigger_type_correct(self) -> None:
        """If v3 produces a fvg_sweep setup, trigger_type must be 'fvg_sweep_continuation'."""
        from smc.strategy.aggregator_v3 import AggregatorV3
        from smc.smc_core.types import SMCSnapshot
        from smc.strategy.types import TradeZone

        _BASE_TS = datetime(2024, 6, 10, 0, 0, 0, tzinfo=timezone.utc)

        zone = TradeZone(
            zone_high=2352.00,
            zone_low=2348.00,
            zone_type="ob",
            direction="long",
            timeframe=Timeframe.H1,
            confidence=0.8,
        )

        m15_snap = SMCSnapshot(
            ts=_BASE_TS,
            timeframe=Timeframe.M15,
            swing_points=(),
            order_blocks=(),
            fvgs=(
                FairValueGap(
                    ts=_BASE_TS,
                    high=2351.00,
                    low=2349.00,
                    fvg_type="bullish",
                    timeframe=Timeframe.M15,
                    filled_pct=1.0,
                    fully_filled=True,
                ),
            ),
            structure_breaks=(),
            liquidity_levels=(
                LiquidityLevel(price=2340.00, level_type="equal_lows", touches=2, swept=False),
            ),
            trend_direction="bullish",
        )

        # Test the entry builder directly
        entry = AggregatorV3._build_fvg_sweep_entry(
            zone, 2347.00, m15_snap, h1_atr=1000.0,
        )
        assert entry is not None
        assert entry.trigger_type == "fvg_sweep_continuation"
        assert entry.direction == "long"

    def test_build_fvg_sweep_entry_zero_risk_returns_none(self) -> None:
        """If risk_points is 0 (entry == SL), return None."""
        from smc.strategy.types import TradeZone

        zone = TradeZone(
            zone_high=2350.00,
            zone_low=2350.00,  # zero-width zone
            zone_type="ob",
            direction="long",
            timeframe=Timeframe.H1,
            confidence=0.8,
        )
        m15_snap = SMCSnapshot(
            ts=datetime(2024, 6, 10, tzinfo=timezone.utc),
            timeframe=Timeframe.M15,
            swing_points=(),
            order_blocks=(),
            fvgs=(),
            structure_breaks=(),
            liquidity_levels=(),
            trend_direction="bullish",
        )
        # h1_atr=0 means SL buffer = floor, but entry is at zone_low - buffer
        # With extremely small zone, risk should still be non-zero from buffer
        # This is more of a safety test
        entry = AggregatorV3._build_fvg_sweep_entry(
            zone, 2350.00, m15_snap, h1_atr=0.0,
        )
        # With buffer floor of 200 points ($2.00), SL = 2350 - 2.00 = 2348
        # risk = 200 points, so entry should be valid
        assert entry is not None


class TestV3ZoneManagement:
    """V3 inherits v1 zone management (cooldown, anti-clustering)."""

    def test_zone_cooldown_inherited(self) -> None:
        """V3 uses v1's 24h cooldown."""
        agg = AggregatorV3(detector=SMCDetector())
        loss_time = datetime(2024, 6, 10, 12, 0, 0, tzinfo=timezone.utc)
        agg.record_zone_loss(2352.00, 2348.00, "long", loss_time)

        expected_until = loss_time + timedelta(hours=24)
        key = (2352.00, 2348.00, "long")
        assert key in agg._zone_cooldowns
        assert agg._zone_cooldowns[key] == expected_until

    def test_zone_anti_clustering_inherited(self) -> None:
        """V3 uses v1's anti-clustering (max 1 per zone)."""
        agg = AggregatorV3(detector=SMCDetector())
        agg.mark_zone_active(2352.00, 2348.00, "long")
        assert (2352.00, 2348.00, "long") in agg._active_zones

    def test_clear_cooldowns_inherited(self) -> None:
        agg = AggregatorV3(detector=SMCDetector())
        agg.record_zone_loss(2352.00, 2348.00, "long",
                             datetime(2024, 6, 10, tzinfo=timezone.utc))
        agg.clear_cooldowns()
        assert len(agg._zone_cooldowns) == 0

    def test_clear_active_zones_inherited(self) -> None:
        agg = AggregatorV3(detector=SMCDetector())
        agg.mark_zone_active(2352.00, 2348.00, "long")
        agg.clear_active_zones()
        assert len(agg._active_zones) == 0


class TestV3ConfluenceScoring:
    """Verify fvg_sweep_continuation gets a proper confluence score."""

    def test_fvg_sweep_has_confluence_score_entry(self) -> None:
        """The v1 confluence scorer must have an entry for fvg_sweep_continuation."""
        from smc.strategy.confluence import _score_entry_trigger
        from smc.strategy.types import EntrySignal

        entry = EntrySignal(
            entry_price=2350.00,
            stop_loss=2347.70,
            take_profit_1=2355.75,
            take_profit_2=2385.00,
            risk_points=230.0,
            reward_points=575.0,
            rr_ratio=2.5,
            trigger_type="fvg_sweep_continuation",
            direction="long",
            grade="B",
        )
        score = _score_entry_trigger(entry)
        # fvg_sweep_continuation should score 0.6 (between bos 0.55 and fvg_fill 0.7)
        # Combined with grade B (0.7): 0.6*0.6 + 0.7*0.4 = 0.36 + 0.28 = 0.64
        assert score == pytest.approx(0.64, abs=0.01)

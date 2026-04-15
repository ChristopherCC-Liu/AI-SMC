"""Unit tests for smc.strategy.aggregator — multi-timeframe orchestrator."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from smc.data.schemas import Timeframe
from smc.smc_core.detector import SMCDetector
from smc.strategy.aggregator import MultiTimeframeAggregator
from smc.strategy.types import TradeSetup


class TestMultiTimeframeAggregator:
    def test_construction(self) -> None:
        detector = SMCDetector(swing_length=10)
        agg = MultiTimeframeAggregator(detector=detector)
        # Aggregator auto-injects swing_length_map if absent
        assert agg.detector.swing_length == 10
        assert agg.detector.swing_length_map == {
            Timeframe.D1: 5, Timeframe.H4: 7,
            Timeframe.H1: 10, Timeframe.M15: 10,
        }

    def test_construction_preserves_custom_map(self) -> None:
        """If detector already has a swing_length_map, aggregator preserves it."""
        custom_map = {Timeframe.D1: 3, Timeframe.H4: 5}
        detector = SMCDetector(swing_length=10, swing_length_map=custom_map)
        agg = MultiTimeframeAggregator(detector=detector)
        assert agg.detector is detector
        assert agg.detector.swing_length_map == custom_map

    def test_empty_data_returns_empty(self) -> None:
        detector = SMCDetector()
        agg = MultiTimeframeAggregator(detector=detector)
        result = agg.generate_setups({}, current_price=2350.0)
        assert result == ()

    def test_missing_d1_returns_empty(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Missing D1 → tiered bias may produce H4-only, but no M15 → no setups."""
        detector = SMCDetector()
        agg = MultiTimeframeAggregator(detector=detector)
        result = agg.generate_setups(
            {Timeframe.H4: sample_ohlcv_df, Timeframe.H1: sample_ohlcv_df},
            current_price=2350.0,
        )
        assert result == ()

    def test_missing_h4_returns_empty(self, sample_ohlcv_df: pl.DataFrame) -> None:
        detector = SMCDetector()
        agg = MultiTimeframeAggregator(detector=detector)
        result = agg.generate_setups(
            {Timeframe.D1: sample_ohlcv_df, Timeframe.H1: sample_ohlcv_df},
            current_price=2350.0,
        )
        assert result == ()

    def test_missing_h1_returns_empty(self, sample_ohlcv_df: pl.DataFrame) -> None:
        detector = SMCDetector()
        agg = MultiTimeframeAggregator(detector=detector)
        result = agg.generate_setups(
            {Timeframe.D1: sample_ohlcv_df, Timeframe.H4: sample_ohlcv_df},
            current_price=2350.0,
        )
        assert result == ()

    def test_missing_m15_returns_empty(self, sample_ohlcv_df: pl.DataFrame) -> None:
        detector = SMCDetector()
        agg = MultiTimeframeAggregator(detector=detector)
        result = agg.generate_setups(
            {
                Timeframe.D1: sample_ohlcv_df,
                Timeframe.H4: sample_ohlcv_df,
                Timeframe.H1: sample_ohlcv_df,
            },
            current_price=2350.0,
        )
        assert result == ()

    def test_full_pipeline_returns_tuples(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """With all 4 timeframes, result should be a tuple (possibly empty if no setups)."""
        detector = SMCDetector(swing_length=5)
        agg = MultiTimeframeAggregator(detector=detector)
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

    def test_setups_sorted_by_confluence_descending(self, sample_ohlcv_df: pl.DataFrame) -> None:
        detector = SMCDetector(swing_length=5)
        agg = MultiTimeframeAggregator(detector=detector)
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
        agg = MultiTimeframeAggregator(detector=detector)
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

    def test_generated_at_is_set(self, sample_ohlcv_df: pl.DataFrame) -> None:
        detector = SMCDetector(swing_length=5)
        agg = MultiTimeframeAggregator(detector=detector)
        before = datetime.now(tz=timezone.utc)
        result = agg.generate_setups(
            {
                Timeframe.D1: sample_ohlcv_df,
                Timeframe.H4: sample_ohlcv_df,
                Timeframe.H1: sample_ohlcv_df,
                Timeframe.M15: sample_ohlcv_df,
            },
            current_price=2350.0,
        )
        after = datetime.now(tz=timezone.utc)
        for setup in result:
            assert before <= setup.generated_at <= after


class TestZoneAntiClustering:
    """Sprint 5: Tests for zone anti-clustering — max 1 active trade per zone."""

    def test_mark_zone_active(self) -> None:
        agg = MultiTimeframeAggregator(detector=SMCDetector())
        agg.mark_zone_active(2352.00, 2348.00, "long")
        assert (2352.00, 2348.00, "long") in agg._active_zones

    def test_clear_zone_active(self) -> None:
        agg = MultiTimeframeAggregator(detector=SMCDetector())
        agg.mark_zone_active(2352.00, 2348.00, "long")
        agg.clear_zone_active(2352.00, 2348.00, "long")
        assert (2352.00, 2348.00, "long") not in agg._active_zones

    def test_clear_zone_active_nonexistent_is_safe(self) -> None:
        """Clearing a zone that was never active should not raise."""
        agg = MultiTimeframeAggregator(detector=SMCDetector())
        agg.clear_zone_active(2352.00, 2348.00, "long")  # no error

    def test_clear_all_active_zones(self) -> None:
        agg = MultiTimeframeAggregator(detector=SMCDetector())
        agg.mark_zone_active(2352.00, 2348.00, "long")
        agg.mark_zone_active(2380.00, 2376.00, "short")
        agg.clear_active_zones()
        assert len(agg._active_zones) == 0

    def test_ob_test_trigger_disabled_by_default(self) -> None:
        """Aggregator constructed with default should have ob_test disabled."""
        agg = MultiTimeframeAggregator(detector=SMCDetector())
        assert agg._enable_ob_test_trigger is False

    def test_ob_test_trigger_can_be_enabled(self) -> None:
        agg = MultiTimeframeAggregator(
            detector=SMCDetector(), enable_ob_test_trigger=True,
        )
        assert agg._enable_ob_test_trigger is True

    def test_intra_call_zone_dedup_in_source(self) -> None:
        """Verify generate_setups() contains intra-call zone dedup logic."""
        import inspect
        src = inspect.getsource(MultiTimeframeAggregator.generate_setups)
        assert "zones_used_this_call" in src
        # Must appear 3 times: declaration, check, and add
        assert src.count("zones_used_this_call") >= 3


class TestComputeH1ATR:
    """Sprint 4: Tests for H1 ATR(14) computation used by adaptive SL."""

    def test_none_returns_zero(self) -> None:
        agg = MultiTimeframeAggregator(detector=SMCDetector())
        assert agg._compute_h1_atr(None) == 0.0

    def test_insufficient_data_returns_zero(self) -> None:
        agg = MultiTimeframeAggregator(detector=SMCDetector())
        df = pl.DataFrame({"high": [2350.0] * 5, "low": [2340.0] * 5, "close": [2345.0] * 5})
        assert agg._compute_h1_atr(df) == 0.0

    def test_known_atr_value(self) -> None:
        """20 bars with constant range=10 ($10) should give ATR = 1000 points."""
        agg = MultiTimeframeAggregator(detector=SMCDetector())
        rows = []
        price = 2000.0
        for _ in range(20):
            rows.append({"high": price + 5.0, "low": price - 5.0, "close": price})
        df = pl.DataFrame(rows)
        atr = agg._compute_h1_atr(df)
        # bar range = 10.0 ($10) = 1000 points. True Range = max(H-L, |H-prevC|, |L-prevC|)
        # With constant close at midpoint: TR = H-L = 10.0, ATR = 10.0/$0.01 = 1000 pts
        assert atr == pytest.approx(1000.0, rel=0.01)

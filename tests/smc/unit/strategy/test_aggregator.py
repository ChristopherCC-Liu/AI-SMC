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

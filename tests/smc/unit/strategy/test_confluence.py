"""Unit tests for smc.strategy.confluence — multi-factor scoring."""

from __future__ import annotations

import pytest

from smc.data.schemas import Timeframe
from smc.strategy.confluence import TRADEABLE_THRESHOLD, score_confluence
from smc.strategy.types import BiasDirection, EntrySignal, TradeZone


class TestScoreConfluence:
    def test_high_quality_setup_scores_high(
        self,
        bullish_bias: BiasDirection,
        bullish_overlap_zone: TradeZone,
        sample_entry_signal: EntrySignal,
    ) -> None:
        """A+ setup: strong bias + overlap zone + choch trigger + 2.5:1 RR."""
        score = score_confluence(bullish_bias, bullish_overlap_zone, sample_entry_signal)
        assert score >= TRADEABLE_THRESHOLD
        assert score > 0.45

    def test_score_bounded_0_to_1(
        self,
        bullish_bias: BiasDirection,
        bullish_ob_zone: TradeZone,
        sample_entry_signal: EntrySignal,
    ) -> None:
        score = score_confluence(bullish_bias, bullish_ob_zone, sample_entry_signal)
        assert 0.0 <= score <= 1.0

    def test_neutral_bias_scores_low(
        self,
        neutral_bias: BiasDirection,
        bullish_ob_zone: TradeZone,
        sample_entry_signal: EntrySignal,
    ) -> None:
        """Neutral bias should tank the HTF alignment component."""
        score = score_confluence(neutral_bias, bullish_ob_zone, sample_entry_signal)
        # HTF component is 0, so overall score should be lower
        high_score = score_confluence(
            BiasDirection(direction="bullish", confidence=0.9, key_levels=(2340.0, 2360.0, 2380.0, 2390.0, 2400.0, 2410.0), rationale="strong"),
            bullish_ob_zone,
            sample_entry_signal,
        )
        assert score < high_score

    def test_overlap_zone_scores_higher_than_fvg(
        self,
        bullish_bias: BiasDirection,
        sample_entry_signal: EntrySignal,
    ) -> None:
        overlap_zone = TradeZone(
            zone_high=2352.0, zone_low=2348.0, zone_type="ob_fvg_overlap",
            direction="long", timeframe=Timeframe.H1, confidence=0.95,
        )
        fvg_zone = TradeZone(
            zone_high=2352.0, zone_low=2348.0, zone_type="fvg",
            direction="long", timeframe=Timeframe.H1, confidence=0.5,
        )
        score_overlap = score_confluence(bullish_bias, overlap_zone, sample_entry_signal)
        score_fvg = score_confluence(bullish_bias, fvg_zone, sample_entry_signal)
        assert score_overlap > score_fvg

    def test_choch_trigger_scores_higher_than_rejection(
        self,
        bullish_bias: BiasDirection,
        bullish_ob_zone: TradeZone,
    ) -> None:
        choch_entry = EntrySignal(
            entry_price=2350.0, stop_loss=2347.0, take_profit_1=2357.5,
            take_profit_2=2362.0, risk_points=300.0, reward_points=750.0,
            rr_ratio=2.5, trigger_type="choch_in_zone", direction="long", grade="A",
        )
        rejection_entry = EntrySignal(
            entry_price=2350.0, stop_loss=2347.0, take_profit_1=2357.5,
            take_profit_2=2362.0, risk_points=300.0, reward_points=750.0,
            rr_ratio=2.5, trigger_type="ob_test_rejection", direction="long", grade="A",
        )
        score_choch = score_confluence(bullish_bias, bullish_ob_zone, choch_entry)
        score_reject = score_confluence(bullish_bias, bullish_ob_zone, rejection_entry)
        assert score_choch > score_reject

    def test_higher_rr_scores_better(
        self,
        bullish_bias: BiasDirection,
        bullish_ob_zone: TradeZone,
    ) -> None:
        entry_2_5rr = EntrySignal(
            entry_price=2350.0, stop_loss=2347.0, take_profit_1=2357.5,
            take_profit_2=2362.0, risk_points=300.0, reward_points=750.0,
            rr_ratio=2.5, trigger_type="choch_in_zone", direction="long", grade="A",
        )
        entry_3rr = EntrySignal(
            entry_price=2350.0, stop_loss=2347.0, take_profit_1=2359.0,
            take_profit_2=2365.0, risk_points=300.0, reward_points=900.0,
            rr_ratio=3.0, trigger_type="choch_in_zone", direction="long", grade="A",
        )
        score_2_5rr = score_confluence(bullish_bias, bullish_ob_zone, entry_2_5rr)
        score_3rr = score_confluence(bullish_bias, bullish_ob_zone, entry_3rr)
        assert score_3rr > score_2_5rr

    def test_weights_sum_to_1(self) -> None:
        """Verify the weight constants sum to 1.0."""
        from smc.strategy.confluence import (
            _W_ENTRY_TRIGGER,
            _W_HTF_ALIGNMENT,
            _W_LIQUIDITY,
            _W_RR_RATIO,
            _W_ZONE_QUALITY,
        )
        total = _W_HTF_ALIGNMENT + _W_ZONE_QUALITY + _W_ENTRY_TRIGGER + _W_RR_RATIO + _W_LIQUIDITY
        assert total == pytest.approx(1.0)

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


class TestScoreConfluenceMacroBias:
    """Round 4 Alt-B W2: macro_bias parameter integration tests."""

    def test_zero_macro_bias_is_backward_compatible(
        self,
        bullish_bias: BiasDirection,
        bullish_ob_zone: TradeZone,
        sample_entry_signal: EntrySignal,
    ) -> None:
        """Default macro_bias=0.0 must produce the same score as pre-macro call."""
        score_no_macro = score_confluence(bullish_bias, bullish_ob_zone, sample_entry_signal)
        score_zero_macro = score_confluence(
            bullish_bias, bullish_ob_zone, sample_entry_signal, macro_bias=0.0
        )
        assert score_no_macro == score_zero_macro

    def test_aligned_macro_bias_boosts_score(
        self,
        bullish_bias: BiasDirection,
        bullish_ob_zone: TradeZone,
        sample_entry_signal: EntrySignal,
    ) -> None:
        """SMC bullish + positive macro_bias (also bullish) → higher score."""
        score_base = score_confluence(bullish_bias, bullish_ob_zone, sample_entry_signal)
        score_boosted = score_confluence(
            bullish_bias, bullish_ob_zone, sample_entry_signal, macro_bias=0.20
        )
        assert score_boosted > score_base

    def test_opposed_macro_bias_penalises_score(
        self,
        bullish_bias: BiasDirection,
        bullish_ob_zone: TradeZone,
        sample_entry_signal: EntrySignal,
    ) -> None:
        """SMC bullish + negative macro_bias (bearish macro) → lower score."""
        score_base = score_confluence(bullish_bias, bullish_ob_zone, sample_entry_signal)
        score_penalised = score_confluence(
            bullish_bias, bullish_ob_zone, sample_entry_signal, macro_bias=-0.20
        )
        assert score_penalised < score_base

    def test_neutral_smc_bias_ignores_macro(
        self,
        neutral_bias: BiasDirection,
        bullish_ob_zone: TradeZone,
        sample_entry_signal: EntrySignal,
    ) -> None:
        """When SMC bias is neutral, macro_bias has no effect (bias_sign = 0)."""
        score_no_macro = score_confluence(neutral_bias, bullish_ob_zone, sample_entry_signal)
        score_with_macro = score_confluence(
            neutral_bias, bullish_ob_zone, sample_entry_signal, macro_bias=0.20
        )
        assert score_no_macro == score_with_macro

    def test_score_clamped_to_1_with_large_macro_bias(
        self,
        bullish_bias: BiasDirection,
        bullish_overlap_zone: TradeZone,
        sample_entry_signal: EntrySignal,
    ) -> None:
        """Even with maximum macro_bias=0.3 on a high-quality setup, score <= 1.0."""
        score = score_confluence(
            bullish_bias, bullish_overlap_zone, sample_entry_signal, macro_bias=0.30
        )
        assert score <= 1.0

    def test_score_clamped_to_0_with_large_negative_macro_bias(
        self,
        bullish_bias: BiasDirection,
        bullish_ob_zone: TradeZone,
    ) -> None:
        """With macro_bias=-0.3 on a weak setup, score stays >= 0.0."""
        weak_entry = EntrySignal(
            entry_price=2350.0, stop_loss=2349.0, take_profit_1=2351.5,
            take_profit_2=2352.0, risk_points=100.0, reward_points=150.0,
            rr_ratio=1.5, trigger_type="ob_test_rejection", direction="long", grade="C",
        )
        score = score_confluence(
            bullish_bias, bullish_ob_zone, weak_entry, macro_bias=-0.30
        )
        assert score >= 0.0

    def test_bearish_smc_with_negative_macro_aligns(
        self,
    ) -> None:
        """Bearish SMC bias + negative macro_bias (bearish macro) → score boost."""
        bearish_bias = BiasDirection(
            direction="bearish",
            confidence=0.8,
            key_levels=(2400.0, 2390.0, 2380.0),
            rationale="D1 bearish BOS sequence",
        )
        bearish_zone = TradeZone(
            zone_high=2385.0, zone_low=2382.0, zone_type="ob",
            direction="short", timeframe=__import__("smc.data.schemas", fromlist=["Timeframe"]).Timeframe.H1,
            confidence=0.75,
        )
        entry = EntrySignal(
            entry_price=2384.0, stop_loss=2386.5, take_profit_1=2379.0,
            take_profit_2=2374.0, risk_points=250.0, reward_points=500.0,
            rr_ratio=2.0, trigger_type="choch_in_zone", direction="short", grade="B",
        )
        score_base = score_confluence(bearish_bias, bearish_zone, entry)
        score_aligned = score_confluence(bearish_bias, bearish_zone, entry, macro_bias=-0.10)
        # Bearish SMC + bearish macro (negative macro_bias) → boost
        # bias_sign = -1, directional_bonus = (-1) * (-0.10) = +0.10
        assert score_aligned > score_base


class TestEffectiveThreshold:
    """Sprint 6: effective_threshold returns tier-only floor (regime removed)."""

    def test_tier1_uses_base_threshold(self) -> None:
        from smc.strategy.confluence import TRADEABLE_THRESHOLD, effective_threshold
        assert effective_threshold("Tier 1: D1+H4") == TRADEABLE_THRESHOLD

    def test_tier2_uses_tier2_floor(self) -> None:
        from smc.strategy.confluence import TIER2_CONFLUENCE_FLOOR, effective_threshold
        assert effective_threshold("Tier 2: H4-only") == TIER2_CONFLUENCE_FLOOR

    def test_tier3_uses_tier3_floor(self) -> None:
        from smc.strategy.confluence import TIER3_CONFLUENCE_FLOOR, effective_threshold
        assert effective_threshold("Tier 3: D1-only") == TIER3_CONFLUENCE_FLOOR

    def test_unknown_tier_uses_base(self) -> None:
        from smc.strategy.confluence import TRADEABLE_THRESHOLD, effective_threshold
        assert effective_threshold("Some other rationale") == TRADEABLE_THRESHOLD

    def test_no_regime_param(self) -> None:
        """Sprint 6: effective_threshold no longer accepts regime parameter."""
        import inspect
        from smc.strategy.confluence import effective_threshold
        sig = inspect.signature(effective_threshold)
        assert list(sig.parameters.keys()) == ["bias_rationale"]

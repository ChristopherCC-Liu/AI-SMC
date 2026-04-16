"""Unit tests for smc.strategy.aggregator_v1_1 — v1.1 fvg_fill inversion aggregator.

Validates that v1.1 is a same-count variant of v1:
- v1.1 produces the SAME number of setups as v1 (no adds, no removes)
- fvg_fill SHORT setups are flipped to LONG in non-trending regimes
- Flipped longs have SL below entry and TP above entry
- Trending regime produces identical output to v1
- v1 parameters are unchanged (threshold, cooldown, zones)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import polars as pl
import pytest

from smc.data.schemas import Timeframe
from smc.smc_core.detector import SMCDetector
from smc.strategy.aggregator import MultiTimeframeAggregator
from smc.strategy.aggregator_v1_1 import AggregatorV1_1
from smc.strategy.types import (
    BiasDirection,
    EntrySignal,
    TradeSetup,
    TradeZone,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestAggregatorV1_1Construction:
    """v1.1 inherits v1 constructor — no new params."""

    def test_inherits_from_v1(self) -> None:
        assert issubclass(AggregatorV1_1, MultiTimeframeAggregator)

    def test_default_construction(self) -> None:
        detector = SMCDetector(swing_length=10)
        agg = AggregatorV1_1(detector=detector)
        assert agg.detector.swing_length_map == {
            Timeframe.D1: 5,
            Timeframe.H4: 7,
            Timeframe.H1: 10,
            Timeframe.M15: 10,
        }

    def test_v1_parameters_unchanged(self) -> None:
        """v1.1 must preserve all v1 class-level parameters."""
        agg = AggregatorV1_1(detector=SMCDetector())
        assert agg._MAX_ENTRIES_PER_ZONE == 1
        assert agg._ZONE_COOLDOWN_HOURS == 24
        assert agg._enable_ob_test_trigger is False


# ---------------------------------------------------------------------------
# Pipeline: same count property
# ---------------------------------------------------------------------------


class TestAggregatorV1_1Pipeline:
    """v1.1 pipeline produces SAME count as v1 (no adds, no removes)."""

    def test_empty_data_returns_empty(self) -> None:
        agg = AggregatorV1_1(detector=SMCDetector())
        result = agg.generate_setups({}, current_price=2350.0)
        assert result == ()

    def test_missing_timeframes_returns_empty(
        self, sample_ohlcv_df: pl.DataFrame,
    ) -> None:
        agg = AggregatorV1_1(detector=SMCDetector())
        result = agg.generate_setups(
            {Timeframe.H4: sample_ohlcv_df, Timeframe.H1: sample_ohlcv_df},
            current_price=2350.0,
        )
        assert result == ()

    def test_full_pipeline_returns_tuples(
        self, sample_ohlcv_df: pl.DataFrame,
    ) -> None:
        detector = SMCDetector(swing_length=5)
        agg = AggregatorV1_1(detector=detector)
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

    def test_v1_1_same_count_as_v1(
        self, sample_ohlcv_df: pl.DataFrame,
    ) -> None:
        """v1.1 must produce exactly the same number of setups as v1."""
        data = {
            Timeframe.D1: sample_ohlcv_df,
            Timeframe.H4: sample_ohlcv_df,
            Timeframe.H1: sample_ohlcv_df,
            Timeframe.M15: sample_ohlcv_df,
        }
        v1_agg = MultiTimeframeAggregator(detector=SMCDetector(swing_length=5))
        v1_1_agg = AggregatorV1_1(detector=SMCDetector(swing_length=5))

        v1_setups = v1_agg.generate_setups(data, current_price=2350.0)
        v1_1_setups = v1_1_agg.generate_setups(data, current_price=2350.0)

        assert len(v1_1_setups) == len(v1_setups)

    def test_setups_sorted_by_confluence(
        self, sample_ohlcv_df: pl.DataFrame,
    ) -> None:
        detector = SMCDetector(swing_length=5)
        agg = AggregatorV1_1(detector=detector)
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


# ---------------------------------------------------------------------------
# Flip logic: unit tests on _flip_setup
# ---------------------------------------------------------------------------


_NOW = datetime(2024, 6, 10, 12, 0, 0, tzinfo=timezone.utc)


def _make_short_fvg_fill_setup(
    *,
    entry_price: float = 2350.00,
    sl: float = 2355.00,
    tp1: float = 2337.50,
    tp2: float = 2330.00,
) -> TradeSetup:
    """Build a SHORT fvg_fill_in_zone setup for flip testing."""
    risk_points = abs(sl - entry_price) / 0.01
    reward_points = abs(entry_price - tp1) / 0.01
    rr = reward_points / risk_points if risk_points > 0 else 0.0

    entry = EntrySignal(
        entry_price=entry_price,
        stop_loss=sl,
        take_profit_1=tp1,
        take_profit_2=tp2,
        risk_points=risk_points,
        reward_points=reward_points,
        rr_ratio=round(rr, 2),
        trigger_type="fvg_fill_in_zone",
        direction="short",
        grade="B",
    )
    bias = BiasDirection(
        direction="bearish",
        confidence=0.75,
        key_levels=(2340.0, 2360.0),
        rationale="Tier 1: D1 bearish, H4 confirms.",
    )
    zone = TradeZone(
        zone_high=2355.00,
        zone_low=2348.00,
        zone_type="ob",
        direction="short",
        timeframe=Timeframe.H1,
        confidence=0.8,
    )
    return TradeSetup(
        entry_signal=entry,
        bias=bias,
        zone=zone,
        confluence_score=0.65,
        generated_at=_NOW,
    )


def _make_long_choch_setup() -> TradeSetup:
    """Build a LONG choch_in_zone setup (should NOT be flipped)."""
    entry = EntrySignal(
        entry_price=2350.00,
        stop_loss=2347.70,
        take_profit_1=2355.75,
        take_profit_2=2385.00,
        risk_points=230.0,
        reward_points=575.0,
        rr_ratio=2.5,
        trigger_type="choch_in_zone",
        direction="long",
        grade="A",
    )
    bias = BiasDirection(
        direction="bullish",
        confidence=0.8,
        key_levels=(2340.0, 2360.0),
        rationale="Tier 1: D1 bullish, H4 confirms.",
    )
    zone = TradeZone(
        zone_high=2352.00,
        zone_low=2348.00,
        zone_type="ob",
        direction="long",
        timeframe=Timeframe.H1,
        confidence=0.8,
    )
    return TradeSetup(
        entry_signal=entry,
        bias=bias,
        zone=zone,
        confluence_score=0.72,
        generated_at=_NOW,
    )


class TestFlipSetup:
    """Unit tests for _flip_setup static method."""

    def test_flip_direction_short_to_long(self) -> None:
        setup = _make_short_fvg_fill_setup()
        flipped = AggregatorV1_1._flip_setup(setup)
        assert flipped.entry_signal.direction == "long"

    def test_flip_zone_direction(self) -> None:
        setup = _make_short_fvg_fill_setup()
        flipped = AggregatorV1_1._flip_setup(setup)
        assert flipped.zone.direction == "long"

    def test_flip_sl_below_entry(self) -> None:
        """Flipped long must have SL below entry price."""
        setup = _make_short_fvg_fill_setup(entry_price=2350.0, sl=2355.0)
        flipped = AggregatorV1_1._flip_setup(setup)
        assert flipped.entry_signal.stop_loss < flipped.entry_signal.entry_price

    def test_flip_tp_above_entry(self) -> None:
        """Flipped long must have TP1 and TP2 above entry price."""
        setup = _make_short_fvg_fill_setup(
            entry_price=2350.0, tp1=2337.50, tp2=2330.00,
        )
        flipped = AggregatorV1_1._flip_setup(setup)
        assert flipped.entry_signal.take_profit_1 > flipped.entry_signal.entry_price
        assert flipped.entry_signal.take_profit_2 > flipped.entry_signal.entry_price

    def test_flip_preserves_sl_distance(self) -> None:
        """SL distance from entry must be identical before and after flip."""
        setup = _make_short_fvg_fill_setup(entry_price=2350.0, sl=2355.0)
        flipped = AggregatorV1_1._flip_setup(setup)
        original_dist = abs(setup.entry_signal.stop_loss - setup.entry_signal.entry_price)
        flipped_dist = abs(flipped.entry_signal.entry_price - flipped.entry_signal.stop_loss)
        assert flipped_dist == pytest.approx(original_dist, abs=0.01)

    def test_flip_preserves_tp1_distance(self) -> None:
        setup = _make_short_fvg_fill_setup(entry_price=2350.0, tp1=2337.50)
        flipped = AggregatorV1_1._flip_setup(setup)
        original_dist = abs(setup.entry_signal.entry_price - setup.entry_signal.take_profit_1)
        flipped_dist = abs(flipped.entry_signal.take_profit_1 - flipped.entry_signal.entry_price)
        assert flipped_dist == pytest.approx(original_dist, abs=0.01)

    def test_flip_preserves_tp2_distance(self) -> None:
        setup = _make_short_fvg_fill_setup(entry_price=2350.0, tp2=2330.00)
        flipped = AggregatorV1_1._flip_setup(setup)
        original_dist = abs(setup.entry_signal.entry_price - setup.entry_signal.take_profit_2)
        flipped_dist = abs(flipped.entry_signal.take_profit_2 - flipped.entry_signal.entry_price)
        assert flipped_dist == pytest.approx(original_dist, abs=0.01)

    def test_flip_preserves_rr_ratio(self) -> None:
        setup = _make_short_fvg_fill_setup()
        flipped = AggregatorV1_1._flip_setup(setup)
        assert flipped.entry_signal.rr_ratio == setup.entry_signal.rr_ratio

    def test_flip_preserves_risk_points(self) -> None:
        setup = _make_short_fvg_fill_setup()
        flipped = AggregatorV1_1._flip_setup(setup)
        assert flipped.entry_signal.risk_points == setup.entry_signal.risk_points

    def test_flip_preserves_trigger_type(self) -> None:
        setup = _make_short_fvg_fill_setup()
        flipped = AggregatorV1_1._flip_setup(setup)
        assert flipped.entry_signal.trigger_type == "fvg_fill_in_zone"

    def test_flip_preserves_grade(self) -> None:
        setup = _make_short_fvg_fill_setup()
        flipped = AggregatorV1_1._flip_setup(setup)
        assert flipped.entry_signal.grade == setup.entry_signal.grade

    def test_flip_preserves_confluence_score(self) -> None:
        setup = _make_short_fvg_fill_setup()
        flipped = AggregatorV1_1._flip_setup(setup)
        assert flipped.confluence_score == setup.confluence_score

    def test_flip_preserves_bias(self) -> None:
        """Bias is NOT flipped — it reflects the original HTF assessment."""
        setup = _make_short_fvg_fill_setup()
        flipped = AggregatorV1_1._flip_setup(setup)
        assert flipped.bias == setup.bias

    def test_flip_preserves_generated_at(self) -> None:
        setup = _make_short_fvg_fill_setup()
        flipped = AggregatorV1_1._flip_setup(setup)
        assert flipped.generated_at == setup.generated_at

    def test_flip_produces_new_objects(self) -> None:
        """Immutability: flip must return new objects, not mutate originals."""
        setup = _make_short_fvg_fill_setup()
        flipped = AggregatorV1_1._flip_setup(setup)
        assert flipped is not setup
        assert flipped.entry_signal is not setup.entry_signal
        assert flipped.zone is not setup.zone


# ---------------------------------------------------------------------------
# Regime-gated inversion logic
# ---------------------------------------------------------------------------


class TestRegimeGatedInversion:
    """Verify inversion only happens in non-trending regimes."""

    def test_trending_regime_no_flip(self) -> None:
        """In a trending regime, fvg_fill SHORT stays SHORT."""
        short_setup = _make_short_fvg_fill_setup()

        with patch.object(
            MultiTimeframeAggregator,
            "generate_setups",
            return_value=(short_setup,),
        ), patch(
            "smc.strategy.aggregator_v1_1.classify_regime",
            return_value="trending",
        ):
            agg = AggregatorV1_1(detector=SMCDetector())
            result = agg.generate_setups({Timeframe.D1: pl.DataFrame()}, current_price=2350.0)

        assert len(result) == 1
        assert result[0].entry_signal.direction == "short"

    def test_ranging_regime_flips(self) -> None:
        """In a ranging regime, fvg_fill SHORT is flipped to LONG."""
        short_setup = _make_short_fvg_fill_setup()

        with patch.object(
            MultiTimeframeAggregator,
            "generate_setups",
            return_value=(short_setup,),
        ), patch(
            "smc.strategy.aggregator_v1_1.classify_regime",
            return_value="ranging",
        ):
            agg = AggregatorV1_1(detector=SMCDetector())
            result = agg.generate_setups({Timeframe.D1: pl.DataFrame()}, current_price=2350.0)

        assert len(result) == 1
        assert result[0].entry_signal.direction == "long"

    def test_transitional_regime_flips(self) -> None:
        """In a transitional regime, fvg_fill SHORT is flipped to LONG."""
        short_setup = _make_short_fvg_fill_setup()

        with patch.object(
            MultiTimeframeAggregator,
            "generate_setups",
            return_value=(short_setup,),
        ), patch(
            "smc.strategy.aggregator_v1_1.classify_regime",
            return_value="transitional",
        ):
            agg = AggregatorV1_1(detector=SMCDetector())
            result = agg.generate_setups({Timeframe.D1: pl.DataFrame()}, current_price=2350.0)

        assert len(result) == 1
        assert result[0].entry_signal.direction == "long"

    def test_non_fvg_fill_not_flipped(self) -> None:
        """choch_in_zone LONG should never be flipped regardless of regime."""
        long_setup = _make_long_choch_setup()

        with patch.object(
            MultiTimeframeAggregator,
            "generate_setups",
            return_value=(long_setup,),
        ), patch(
            "smc.strategy.aggregator_v1_1.classify_regime",
            return_value="ranging",
        ):
            agg = AggregatorV1_1(detector=SMCDetector())
            result = agg.generate_setups({Timeframe.D1: pl.DataFrame()}, current_price=2350.0)

        assert len(result) == 1
        assert result[0].entry_signal.direction == "long"
        assert result[0].entry_signal.trigger_type == "choch_in_zone"

    def test_mixed_setups_only_qualifying_flipped(self) -> None:
        """With mixed setups, only fvg_fill SHORT is flipped in ranging."""
        short_fvg = _make_short_fvg_fill_setup()
        long_choch = _make_long_choch_setup()

        with patch.object(
            MultiTimeframeAggregator,
            "generate_setups",
            return_value=(long_choch, short_fvg),
        ), patch(
            "smc.strategy.aggregator_v1_1.classify_regime",
            return_value="ranging",
        ):
            agg = AggregatorV1_1(detector=SMCDetector())
            result = agg.generate_setups({Timeframe.D1: pl.DataFrame()}, current_price=2350.0)

        assert len(result) == 2
        # First setup: long_choch — unchanged
        assert result[0].entry_signal.trigger_type == "choch_in_zone"
        assert result[0].entry_signal.direction == "long"
        # Second setup: short_fvg — flipped to long
        assert result[1].entry_signal.trigger_type == "fvg_fill_in_zone"
        assert result[1].entry_signal.direction == "long"

    def test_fvg_fill_long_not_flipped(self) -> None:
        """fvg_fill LONG should NOT be flipped (only SHORT is inverted)."""
        long_fvg_entry = EntrySignal(
            entry_price=2350.00,
            stop_loss=2345.00,
            take_profit_1=2362.50,
            take_profit_2=2370.00,
            risk_points=500.0,
            reward_points=1250.0,
            rr_ratio=2.5,
            trigger_type="fvg_fill_in_zone",
            direction="long",
            grade="B",
        )
        long_fvg_setup = TradeSetup(
            entry_signal=long_fvg_entry,
            bias=BiasDirection(
                direction="bullish",
                confidence=0.8,
                key_levels=(2340.0,),
                rationale="Tier 1: D1 bullish.",
            ),
            zone=TradeZone(
                zone_high=2352.0,
                zone_low=2348.0,
                zone_type="ob",
                direction="long",
                timeframe=Timeframe.H1,
                confidence=0.8,
            ),
            confluence_score=0.65,
            generated_at=_NOW,
        )

        with patch.object(
            MultiTimeframeAggregator,
            "generate_setups",
            return_value=(long_fvg_setup,),
        ), patch(
            "smc.strategy.aggregator_v1_1.classify_regime",
            return_value="ranging",
        ):
            agg = AggregatorV1_1(detector=SMCDetector())
            result = agg.generate_setups({Timeframe.D1: pl.DataFrame()}, current_price=2350.0)

        assert len(result) == 1
        assert result[0].entry_signal.direction == "long"


# ---------------------------------------------------------------------------
# Zone management inheritance
# ---------------------------------------------------------------------------


class TestV1_1ZoneManagement:
    """v1.1 inherits v1 zone management (cooldown, anti-clustering)."""

    def test_zone_cooldown_inherited(self) -> None:
        agg = AggregatorV1_1(detector=SMCDetector())
        loss_time = datetime(2024, 6, 10, 12, 0, 0, tzinfo=timezone.utc)
        agg.record_zone_loss(2352.00, 2348.00, "long", loss_time)

        expected_until = loss_time + timedelta(hours=24)
        key = (2352.00, 2348.00, "long")
        assert key in agg._zone_cooldowns
        assert agg._zone_cooldowns[key] == expected_until

    def test_zone_anti_clustering_inherited(self) -> None:
        agg = AggregatorV1_1(detector=SMCDetector())
        agg.mark_zone_active(2352.00, 2348.00, "long")
        assert (2352.00, 2348.00, "long") in agg._active_zones

    def test_clear_cooldowns_inherited(self) -> None:
        agg = AggregatorV1_1(detector=SMCDetector())
        agg.record_zone_loss(
            2352.00, 2348.00, "long",
            datetime(2024, 6, 10, tzinfo=timezone.utc),
        )
        agg.clear_cooldowns()
        assert len(agg._zone_cooldowns) == 0

    def test_clear_active_zones_inherited(self) -> None:
        agg = AggregatorV1_1(detector=SMCDetector())
        agg.mark_zone_active(2352.00, 2348.00, "long")
        agg.clear_active_zones()
        assert len(agg._active_zones) == 0

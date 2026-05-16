"""Phase B step 3 — regime_classifier_v2 tests."""

from __future__ import annotations

import pytest

from smc.hedgerock.regime_classifier_v2 import (
    DEFAULT_CONFIDENCE_FLOOR,
    classify_regime_v2,
)


@pytest.mark.unit
def test_compressed_volatility_classifies_as_range() -> None:
    a = classify_regime_v2(volatility_rank=0.20, h4_trend_bars=1, hh_count=2, ll_count=2)
    assert a.regime == "range"
    assert a.confidence >= DEFAULT_CONFIDENCE_FLOOR
    assert "compressed" in a.reason or "range" in a.reason


@pytest.mark.unit
def test_clean_uptrend_classifies_as_trend_up() -> None:
    a = classify_regime_v2(volatility_rank=0.55, h4_trend_bars=5, hh_count=8, ll_count=2)
    assert a.regime == "trend_up"
    assert a.confidence >= DEFAULT_CONFIDENCE_FLOOR


@pytest.mark.unit
def test_clean_downtrend_classifies_as_trend_down() -> None:
    a = classify_regime_v2(volatility_rank=0.55, h4_trend_bars=5, hh_count=2, ll_count=8)
    assert a.regime == "trend_down"


@pytest.mark.unit
def test_breakout_fires_on_top_decile_volatility_with_directional_swings() -> None:
    a = classify_regime_v2(volatility_rank=0.90, h4_trend_bars=4, hh_count=7, ll_count=2)
    # breakout is higher priority than trend_up at this vol+trend combo.
    assert a.regime == "breakout"


@pytest.mark.unit
def test_news_high_overrides_price_based_classification() -> None:
    """News=high beats trend_up because of priority. Even a clean uptrend
    becomes 'news' regime — rule_engine should react accordingly."""
    a = classify_regime_v2(
        volatility_rank=0.50, h4_trend_bars=4, hh_count=6, ll_count=2,
        news_intensity="high",
    )
    assert a.regime == "news"


@pytest.mark.unit
def test_crisis_fires_on_extreme_vol_plus_high_news() -> None:
    a = classify_regime_v2(
        volatility_rank=0.96, h4_trend_bars=2, hh_count=3, ll_count=4,
        news_intensity="high",
    )
    assert a.regime == "crisis"


@pytest.mark.unit
def test_crisis_fires_on_extreme_spread_alone() -> None:
    a = classify_regime_v2(
        volatility_rank=0.50, h4_trend_bars=1, hh_count=0, ll_count=0,
        spread_pts=250,
    )
    assert a.regime == "crisis"
    assert "spread" in a.reason


@pytest.mark.unit
def test_low_confidence_falls_through_to_unknown() -> None:
    """Mid-vol with HH/LL imbalance not strong enough for trend, vol not
    compressed enough for range#1, swings not balanced enough for
    range#2 → no rule fires → unknown."""
    # vol_rank=0.35 is not <0.30 (range#1 no), trend_bars=1 < 3 (trend no),
    # |hh-ll|=2 > 1 (range#2 no), vol<0.85 (breakout no), no news/spread.
    a = classify_regime_v2(volatility_rank=0.35, h4_trend_bars=1, hh_count=4, ll_count=2)
    assert a.regime == "unknown"


@pytest.mark.unit
def test_assessment_records_all_voting_rules() -> None:
    """The assessment should expose every rule that fired, not only the winner —
    decision_log inspection depends on this."""
    a = classify_regime_v2(
        volatility_rank=0.55, h4_trend_bars=5, hh_count=8, ll_count=2,
        news_intensity="medium",
    )
    rule_names = {v[0] for v in a.rule_votes}
    # Both 'news' and 'trend_up' should have voted; news (priority 5) wins.
    assert "news" in rule_names
    assert "trend_up" in rule_names
    assert a.regime == "news"


@pytest.mark.unit
def test_unknown_carries_best_below_floor_confidence_for_diagnostics() -> None:
    """An unknown verdict still surfaces the *best* sub-floor confidence
    so the rule engine can see how close we got."""
    # Same fixture as the no-fire case above — confirm confidence is
    # bounded below the floor (no rule contributed).
    a = classify_regime_v2(volatility_rank=0.35, h4_trend_bars=1, hh_count=4, ll_count=2)
    assert a.regime == "unknown"
    assert 0.0 <= a.confidence < DEFAULT_CONFIDENCE_FLOOR

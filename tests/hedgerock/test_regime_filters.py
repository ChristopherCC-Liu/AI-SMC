"""Test Phase 1 regime filters mirror HedgeRock_v2.mq5 inline filter behavior."""
from __future__ import annotations

import pytest

from smc.hedgerock.regime_filters import (
    H1_ATR_RATIO_THRESHOLD,
    H4_TREND_THRESHOLD_PCT,
    RANGE_RATIO_THRESHOLD,
    FilterInputs,
    build_filter_inputs,
    compute_filters,
)


@pytest.mark.unit
def test_no_halt_in_calm_market():
    """Calm market: small H4 trend, normal ATR, contained range — no halt."""
    inputs = FilterInputs(
        h4_close_now=2700.0,
        h4_close_lookback=2710.0,  # 0.37% change → below 2% threshold
        h1_atr_now=10.0,
        h1_atr_lookback_avg=10.0,  # 1.0× → below 1.5× threshold
        h1_recent_high=2705.0,
        h1_recent_low=2695.0,  # range 10 (24 bars)
        h1_reference_high=2730.0,
        h1_reference_low=2670.0,  # range 60 (168 bars)
        # ratio = 10/60 = 0.17 < 0.6 threshold
    )
    result = compute_filters(inputs)
    assert result.halt is False
    assert result.triggered == ()


@pytest.mark.unit
def test_halt_on_h4_trend():
    """H4 trend > 2% → Filter A triggers."""
    inputs = FilterInputs(
        h4_close_now=2900.0,
        h4_close_lookback=2700.0,  # +7.4% → above 2% threshold
        h1_atr_now=10.0,
        h1_atr_lookback_avg=10.0,
        h1_recent_high=2900.0,
        h1_recent_low=2890.0,
        h1_reference_high=2900.0,
        h1_reference_low=2700.0,  # large range but recent small relative
    )
    result = compute_filters(inputs)
    assert result.halt is True
    assert "h4_trend" in result.triggered


@pytest.mark.unit
def test_halt_on_atr_breakout():
    """H1 ATR > 1.5× lookback avg → Filter B triggers."""
    inputs = FilterInputs(
        h4_close_now=2700.0,
        h4_close_lookback=2705.0,  # tiny trend, no Filter A
        h1_atr_now=20.0,           # current ATR
        h1_atr_lookback_avg=10.0,  # 2.0× → above 1.5× threshold
        h1_recent_high=2705.0,
        h1_recent_low=2695.0,
        h1_reference_high=2730.0,
        h1_reference_low=2670.0,
    )
    result = compute_filters(inputs)
    assert result.halt is True
    assert "h1_atr_breakout" in result.triggered


@pytest.mark.unit
def test_halt_on_range_expansion():
    """24-bar range > 60% of weekly → Filter C triggers."""
    inputs = FilterInputs(
        h4_close_now=2700.0,
        h4_close_lookback=2705.0,  # no Filter A
        h1_atr_now=10.0,
        h1_atr_lookback_avg=10.0,  # no Filter B
        h1_recent_high=2730.0,
        h1_recent_low=2680.0,  # range 50 (24 bars)
        h1_reference_high=2730.0,
        h1_reference_low=2680.0,  # range 50 (168 bars) → ratio 1.0
    )
    result = compute_filters(inputs)
    assert result.halt is True
    assert "range_expansion" in result.triggered


@pytest.mark.unit
def test_multiple_filters_trigger():
    """All 3 filters can trigger simultaneously."""
    inputs = FilterInputs(
        h4_close_now=3000.0,
        h4_close_lookback=2700.0,  # +11% Filter A
        h1_atr_now=20.0,
        h1_atr_lookback_avg=10.0,  # 2× Filter B
        h1_recent_high=3000.0,
        h1_recent_low=2700.0,
        h1_reference_high=3000.0,
        h1_reference_low=2700.0,  # ratio 1.0 Filter C
    )
    result = compute_filters(inputs)
    assert result.halt is True
    assert len(result.triggered) == 3
    assert {"h4_trend", "h1_atr_breakout", "range_expansion"} == set(result.triggered)


@pytest.mark.unit
def test_zero_lookback_safe():
    """Zero lookback values don't divide-by-zero (filter returns 0 ratio)."""
    inputs = FilterInputs(
        h4_close_now=2700.0,
        h4_close_lookback=0.0,  # bad data → trend_pct=0
        h1_atr_now=10.0,
        h1_atr_lookback_avg=0.0,  # bad data → atr_ratio=0
        h1_recent_high=2700.0,
        h1_recent_low=2700.0,  # recent_range=0
        h1_reference_high=2700.0,
        h1_reference_low=2700.0,  # reference_range=0 → range_ratio=0
    )
    result = compute_filters(inputs)
    # All ratios computed as 0 → no halt (graceful with bad data)
    assert result.halt is False
    assert result.h4_trend_pct == 0.0
    assert result.h1_atr_ratio == 0.0
    assert result.range_ratio == 0.0


@pytest.mark.unit
def test_build_filter_inputs_validation():
    """build_filter_inputs raises if sequences too short."""
    with pytest.raises(ValueError, match="h4_closes need"):
        build_filter_inputs(
            h4_closes=[2700.0] * 10,  # need 25
            h1_closes=[2700.0] * 800,
            h1_highs=[2700.0] * 800,
            h1_lows=[2700.0] * 800,
            h1_atrs=[10.0] * 800,
        )


@pytest.mark.unit
def test_build_filter_inputs_happy():
    """build_filter_inputs produces correct snapshot from raw series."""
    h4 = [2700.0 + i for i in range(30)]   # ascending
    h1_close = [2700.0 + i * 0.1 for i in range(800)]
    h1_high = [c + 5.0 for c in h1_close]
    h1_low = [c - 5.0 for c in h1_close]
    h1_atr = [10.0] * 800
    inputs = build_filter_inputs(h4, h1_close, h1_high, h1_low, h1_atr)
    assert inputs.h4_close_now == h4[-1]
    assert inputs.h4_close_lookback == h4[-25]  # -1 - 24
    assert inputs.h1_atr_now == h1_atr[-1]
    assert inputs.h1_atr_lookback_avg == 10.0  # all same


@pytest.mark.unit
def test_thresholds_match_ea_constants():
    """Threshold constants match EA inline filter values verbatim."""
    assert H4_TREND_THRESHOLD_PCT == 0.02  # EA: trend_pct > 0.02
    assert H1_ATR_RATIO_THRESHOLD == 1.5   # EA: atr_now > 1.5 * atr_ref
    assert RANGE_RATIO_THRESHOLD == 0.6    # EA: recent_range > 0.60 * weekly_range

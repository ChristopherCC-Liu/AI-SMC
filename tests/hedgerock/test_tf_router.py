"""Tests for hedgerock.tf_router."""

from __future__ import annotations

import pytest

from smc.hedgerock.tf_router import (
    DEAD_VOL_THRESHOLD,
    EXTREME_VOL_THRESHOLD,
    TREND_BAR_FLOOR,
    TREND_SWING_DELTA,
    TimeframeRoute,
    route_timeframe,
)


# ---------------------------------------------------------------------------
# Threshold sanity (so a typo in constants is caught immediately)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_thresholds_make_sense() -> None:
    assert 0.0 < DEAD_VOL_THRESHOLD < EXTREME_VOL_THRESHOLD < 1.0
    assert TREND_SWING_DELTA > 0
    assert TREND_BAR_FLOOR > 0


# ---------------------------------------------------------------------------
# Branch 1: extreme volatility → M5
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("vol", [0.85, 0.90, 0.99, 1.00])
def test_extreme_vol_picks_m5(vol: float) -> None:
    route = route_timeframe(
        volatility_rank=vol, hh_count=0, ll_count=0, h4_trend_bars=0
    )
    assert route.timeframe == "M5"
    assert "extreme volatility" in route.reason
    assert 0.6 <= route.confidence <= 1.0


@pytest.mark.unit
def test_extreme_vol_overrides_strong_trend() -> None:
    """Extreme vol should win even if trend signals are strong — vol kills you faster."""
    route = route_timeframe(
        volatility_rank=0.99, hh_count=10, ll_count=0, h4_trend_bars=20
    )
    assert route.timeframe == "M5"


# ---------------------------------------------------------------------------
# Branch 2: dead volatility → H4
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("vol", [0.00, 0.05, 0.10, 0.15])
def test_dead_vol_picks_h4(vol: float) -> None:
    route = route_timeframe(
        volatility_rank=vol, hh_count=0, ll_count=0, h4_trend_bars=0
    )
    assert route.timeframe == "H4"
    assert "dead volatility" in route.reason
    assert 0.6 <= route.confidence <= 1.0


@pytest.mark.unit
def test_dead_vol_overrides_trend_signals() -> None:
    """Even with HH-LL imbalance, a dead market means H4 to avoid grinding noise."""
    route = route_timeframe(
        volatility_rank=0.05, hh_count=10, ll_count=0, h4_trend_bars=20
    )
    assert route.timeframe == "H4"


# ---------------------------------------------------------------------------
# Branch 3: directional trend → H1
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_strong_uptrend_picks_h1() -> None:
    route = route_timeframe(
        volatility_rank=0.5, hh_count=8, ll_count=2, h4_trend_bars=5
    )
    assert route.timeframe == "H1"
    assert "directional trend" in route.reason


@pytest.mark.unit
def test_strong_downtrend_picks_h1() -> None:
    route = route_timeframe(
        volatility_rank=0.5, hh_count=2, ll_count=8, h4_trend_bars=6
    )
    assert route.timeframe == "H1"


@pytest.mark.unit
@pytest.mark.parametrize(
    "hh, ll, bars",
    [
        (TREND_SWING_DELTA, 0, TREND_BAR_FLOOR),  # exact threshold — included
        (5, 0, 4),  # threshold values
        (10, 0, 10),  # well past threshold
    ],
)
def test_trend_threshold_inclusive(hh: int, ll: int, bars: int) -> None:
    route = route_timeframe(
        volatility_rank=0.5, hh_count=hh, ll_count=ll, h4_trend_bars=bars
    )
    assert route.timeframe == "H1"


@pytest.mark.unit
def test_trend_confidence_scales_with_strength() -> None:
    weak = route_timeframe(
        volatility_rank=0.5, hh_count=5, ll_count=0, h4_trend_bars=4
    )
    strong = route_timeframe(
        volatility_rank=0.5, hh_count=10, ll_count=0, h4_trend_bars=10
    )
    assert strong.timeframe == "H1"
    assert weak.timeframe == "H1"
    assert strong.confidence > weak.confidence


# ---------------------------------------------------------------------------
# Branch 4: default → M15
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mid_vol_no_trend_picks_m15() -> None:
    route = route_timeframe(
        volatility_rank=0.5, hh_count=2, ll_count=2, h4_trend_bars=1
    )
    assert route.timeframe == "M15"
    assert "consolidation" in route.reason


@pytest.mark.unit
def test_swing_imbalance_below_threshold_picks_m15() -> None:
    # |hh-ll| = 4 < TREND_SWING_DELTA = 5
    route = route_timeframe(
        volatility_rank=0.5, hh_count=5, ll_count=1, h4_trend_bars=10
    )
    assert route.timeframe == "M15"


@pytest.mark.unit
def test_trend_bars_below_threshold_picks_m15() -> None:
    # h4_trend_bars=3 < TREND_BAR_FLOOR=4
    route = route_timeframe(
        volatility_rank=0.5, hh_count=8, ll_count=0, h4_trend_bars=3
    )
    assert route.timeframe == "M15"


# ---------------------------------------------------------------------------
# Boundary cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_threshold_just_above_dead_picks_m15_or_h1() -> None:
    """vol just above DEAD_VOL_THRESHOLD with no trend → M15."""
    route = route_timeframe(
        volatility_rank=DEAD_VOL_THRESHOLD + 0.01,
        hh_count=0,
        ll_count=0,
        h4_trend_bars=0,
    )
    assert route.timeframe == "M15"


@pytest.mark.unit
def test_threshold_just_below_extreme_picks_h1_when_trending() -> None:
    """vol just below EXTREME_VOL_THRESHOLD + strong trend → H1."""
    route = route_timeframe(
        volatility_rank=EXTREME_VOL_THRESHOLD - 0.01,
        hh_count=8,
        ll_count=0,
        h4_trend_bars=5,
    )
    assert route.timeframe == "H1"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("bad", [-0.01, -1.0, 1.01, 100.0])
def test_volatility_rank_out_of_range_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="volatility_rank"):
        route_timeframe(
            volatility_rank=bad, hh_count=0, ll_count=0, h4_trend_bars=0
        )


@pytest.mark.unit
def test_negative_hh_count_raises() -> None:
    with pytest.raises(ValueError):
        route_timeframe(
            volatility_rank=0.5, hh_count=-1, ll_count=0, h4_trend_bars=0
        )


@pytest.mark.unit
def test_negative_ll_count_raises() -> None:
    with pytest.raises(ValueError):
        route_timeframe(
            volatility_rank=0.5, hh_count=0, ll_count=-1, h4_trend_bars=0
        )


@pytest.mark.unit
def test_negative_trend_bars_raises() -> None:
    with pytest.raises(ValueError):
        route_timeframe(
            volatility_rank=0.5, hh_count=0, ll_count=0, h4_trend_bars=-1
        )


# ---------------------------------------------------------------------------
# Result properties
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_route_is_frozen() -> None:
    route = route_timeframe(
        volatility_rank=0.5, hh_count=0, ll_count=0, h4_trend_bars=0
    )
    with pytest.raises((AttributeError, TypeError)):
        route.timeframe = "M5"  # type: ignore[misc]


@pytest.mark.unit
def test_all_branches_return_known_timeframe() -> None:
    """Spot-check every branch returns a valid TF — paranoia check."""
    cases = [
        # (vol, hh, ll, bars, expected)
        (0.95, 0, 0, 0, "M5"),
        (0.05, 0, 0, 0, "H4"),
        (0.5, 8, 0, 5, "H1"),
        (0.5, 2, 2, 1, "M15"),
    ]
    for vol, hh, ll, bars, expected in cases:
        route = route_timeframe(
            volatility_rank=vol, hh_count=hh, ll_count=ll, h4_trend_bars=bars
        )
        assert route.timeframe == expected, (
            f"({vol}, {hh}, {ll}, {bars}) expected {expected}, got {route.timeframe}"
        )
        assert isinstance(route, TimeframeRoute)
        assert 0.0 <= route.confidence <= 1.0


@pytest.mark.unit
def test_reason_strings_are_non_empty() -> None:
    cases = [
        (0.95, 0, 0, 0),
        (0.05, 0, 0, 0),
        (0.5, 8, 0, 5),
        (0.5, 2, 2, 1),
    ]
    for vol, hh, ll, bars in cases:
        route = route_timeframe(
            volatility_rank=vol, hh_count=hh, ll_count=ll, h4_trend_bars=bars
        )
        assert route.reason
        assert len(route.reason) > 10

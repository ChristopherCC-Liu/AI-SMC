"""Unit tests for smc.strategy.mode_router — pure-function mode routing."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from smc.strategy.mode_router import route_trading_mode
from smc.strategy.range_types import RangeBounds


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_range_bounds() -> RangeBounds:
    return RangeBounds(
        upper=2380.00,
        lower=2340.00,
        width_points=4000.0,
        midpoint=2360.00,
        detected_at=datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc),
        source="ob_boundaries",
        confidence=0.85,
    )


# ---------------------------------------------------------------------------
# Priority 1: Asian session → hold regardless of AI
# ---------------------------------------------------------------------------

class TestAsianSessionHold:
    def test_asian_bullish_high_conf(self, sample_range_bounds: RangeBounds) -> None:
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.9,
            regime="trending",
            session="ASIAN",
            range_bounds=sample_range_bounds,
        )
        assert result.mode == "trending"
        assert "Asian" in result.reason

    def test_asian_neutral_with_range(self, sample_range_bounds: RangeBounds) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="ranging",
            session="ASIAN",
            range_bounds=sample_range_bounds,
        )
        assert result.mode == "trending"
        assert "Asian" in result.reason

    def test_asian_bearish_no_range(self) -> None:
        result = route_trading_mode(
            ai_direction="bearish",
            ai_confidence=0.7,
            regime="transitional",
            session="ASIAN",
            range_bounds=None,
        )
        assert result.mode == "trending"
        assert "Asian" in result.reason


# ---------------------------------------------------------------------------
# Priority 2: AI bullish/bearish + high confidence → trending
# ---------------------------------------------------------------------------

class TestHighConfTrending:
    def test_bullish_high_conf(self) -> None:
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.8,
            regime="trending",
            session="LONDON",
            range_bounds=None,
        )
        assert result.mode == "trending"
        assert "bullish" in result.reason

    def test_bearish_exactly_threshold(self) -> None:
        result = route_trading_mode(
            ai_direction="bearish",
            ai_confidence=0.5,
            regime="transitional",
            session="NEW_YORK",
            range_bounds=None,
        )
        assert result.mode == "trending"
        assert "bearish" in result.reason

    def test_bullish_high_conf_ignores_range(
        self, sample_range_bounds: RangeBounds,
    ) -> None:
        """Even with range_bounds present, high AI conviction → trending."""
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.7,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
        )
        assert result.mode == "trending"


# ---------------------------------------------------------------------------
# Priority 3a: Neutral + range exists + regime != trending → ranging
# ---------------------------------------------------------------------------

class TestNeutralWithRangeRanging:
    def test_neutral_range_ranging_regime(
        self, sample_range_bounds: RangeBounds,
    ) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
        )
        assert result.mode == "ranging"
        assert result.range_bounds is not None

    def test_neutral_range_transitional_regime(
        self, sample_range_bounds: RangeBounds,
    ) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="transitional",
            session="NEW_YORK",
            range_bounds=sample_range_bounds,
        )
        assert result.mode == "ranging"

    def test_low_conf_bullish_with_range(
        self, sample_range_bounds: RangeBounds,
    ) -> None:
        """Bullish direction but low confidence → falls to priority 3."""
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.3,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
        )
        assert result.mode == "ranging"


# ---------------------------------------------------------------------------
# Regime trending + neutral → still trending (don't range in a trend)
# ---------------------------------------------------------------------------

class TestRegimeTrendingBlocksRanging:
    def test_neutral_range_trending_regime(
        self, sample_range_bounds: RangeBounds,
    ) -> None:
        """Even with range_bounds, regime='trending' prevents ranging mode.
        Falls through to v1_passthrough instead."""
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="trending",
            session="LONDON",
            range_bounds=sample_range_bounds,
        )
        assert result.mode == "v1_passthrough"
        assert "v1 pipeline" in result.reason


# ---------------------------------------------------------------------------
# Fallback: neutral + no range → trending (hold)
# ---------------------------------------------------------------------------

class TestV1Passthrough:
    """When AI is unsure and no range exists, v1 pipeline runs autonomously."""

    def test_neutral_no_range(self) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.1,
            regime="ranging",
            session="LONDON",
            range_bounds=None,
        )
        assert result.mode == "v1_passthrough"
        assert "v1 pipeline" in result.reason

    def test_low_conf_bearish_no_range(self) -> None:
        result = route_trading_mode(
            ai_direction="bearish",
            ai_confidence=0.2,
            regime="transitional",
            session="NEW_YORK",
            range_bounds=None,
        )
        assert result.mode == "v1_passthrough"
        assert "v1 pipeline" in result.reason

    def test_neutral_trending_regime_no_range(self) -> None:
        """The key fix: neutral AI + trending regime + no range used to deadlock."""
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="trending",
            session="LONDON",
            range_bounds=None,
        )
        assert result.mode == "v1_passthrough"
        assert "AI unsure" in result.reason


# ---------------------------------------------------------------------------
# Context passthrough
# ---------------------------------------------------------------------------

class TestContextFields:
    def test_fields_preserved(self, sample_range_bounds: RangeBounds) -> None:
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.9,
            regime="trending",
            session="LONDON",
            range_bounds=sample_range_bounds,
        )
        assert result.ai_direction == "bullish"
        assert result.ai_confidence == 0.9
        assert result.regime == "trending"
        assert result.range_bounds == sample_range_bounds

    def test_frozen(self) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.1,
            regime="ranging",
            session="LONDON",
            range_bounds=None,
        )
        with pytest.raises(Exception):
            result.mode = "trending"  # type: ignore[misc]

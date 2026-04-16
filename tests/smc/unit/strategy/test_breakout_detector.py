"""Unit tests for smc.strategy.breakout_detector — ATR-buffered breakout detection."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from smc.smc_core.constants import XAUUSD_POINT_SIZE
from smc.strategy.breakout_detector import BreakoutDetector
from smc.strategy.range_types import RangeBounds


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def detector() -> BreakoutDetector:
    return BreakoutDetector(atr_buffer_mult=0.25)


@pytest.fixture()
def range_bounds() -> RangeBounds:
    return RangeBounds(
        upper=2380.00,
        lower=2340.00,
        width_points=4000.0,
        midpoint=2360.00,
        detected_at=datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc),
        source="ob_boundaries",
        confidence=0.85,
    )


# H1 ATR = 500 points → buffer = 500 * 0.01 * 0.25 = $1.25
_H1_ATR_POINTS = 500.0
_BUFFER = _H1_ATR_POINTS * XAUUSD_POINT_SIZE * 0.25  # 1.25


# ---------------------------------------------------------------------------
# Price within range → "none"
# ---------------------------------------------------------------------------

class TestWithinRange:
    def test_midpoint(
        self, detector: BreakoutDetector, range_bounds: RangeBounds,
    ) -> None:
        assert detector.check_breakout(2360.00, range_bounds, _H1_ATR_POINTS) == "none"

    def test_near_upper(
        self, detector: BreakoutDetector, range_bounds: RangeBounds,
    ) -> None:
        assert detector.check_breakout(2379.50, range_bounds, _H1_ATR_POINTS) == "none"

    def test_near_lower(
        self, detector: BreakoutDetector, range_bounds: RangeBounds,
    ) -> None:
        assert detector.check_breakout(2340.50, range_bounds, _H1_ATR_POINTS) == "none"


# ---------------------------------------------------------------------------
# Price above upper + buffer → "bullish_breakout"
# ---------------------------------------------------------------------------

class TestBullishBreakout:
    def test_clear_bullish_breakout(
        self, detector: BreakoutDetector, range_bounds: RangeBounds,
    ) -> None:
        # upper=2380, buffer=1.25 → need > 2381.25
        price = 2382.00
        assert detector.check_breakout(price, range_bounds, _H1_ATR_POINTS) == "bullish_breakout"

    def test_just_above_buffer(
        self, detector: BreakoutDetector, range_bounds: RangeBounds,
    ) -> None:
        # Exactly at upper + buffer + epsilon
        price = range_bounds.upper + _BUFFER + 0.01
        assert detector.check_breakout(price, range_bounds, _H1_ATR_POINTS) == "bullish_breakout"


# ---------------------------------------------------------------------------
# Price below lower - buffer → "bearish_breakout"
# ---------------------------------------------------------------------------

class TestBearishBreakout:
    def test_clear_bearish_breakout(
        self, detector: BreakoutDetector, range_bounds: RangeBounds,
    ) -> None:
        # lower=2340, buffer=1.25 → need < 2338.75
        price = 2338.00
        assert detector.check_breakout(price, range_bounds, _H1_ATR_POINTS) == "bearish_breakout"

    def test_just_below_buffer(
        self, detector: BreakoutDetector, range_bounds: RangeBounds,
    ) -> None:
        price = range_bounds.lower - _BUFFER - 0.01
        assert detector.check_breakout(price, range_bounds, _H1_ATR_POINTS) == "bearish_breakout"


# ---------------------------------------------------------------------------
# Price just outside boundary but within buffer → "none" (not a breakout yet)
# ---------------------------------------------------------------------------

class TestWithinBuffer:
    def test_above_upper_within_buffer(
        self, detector: BreakoutDetector, range_bounds: RangeBounds,
    ) -> None:
        # upper=2380, buffer=1.25 → 2380.50 is outside range but within buffer
        price = 2380.50
        assert detector.check_breakout(price, range_bounds, _H1_ATR_POINTS) == "none"

    def test_below_lower_within_buffer(
        self, detector: BreakoutDetector, range_bounds: RangeBounds,
    ) -> None:
        # lower=2340, buffer=1.25 → 2339.50 is outside range but within buffer
        price = 2339.50
        assert detector.check_breakout(price, range_bounds, _H1_ATR_POINTS) == "none"

    def test_exactly_at_upper_plus_buffer(
        self, detector: BreakoutDetector, range_bounds: RangeBounds,
    ) -> None:
        """Boundary case: exactly at upper + buffer is NOT a breakout (need >)."""
        price = range_bounds.upper + _BUFFER  # 2381.25
        assert detector.check_breakout(price, range_bounds, _H1_ATR_POINTS) == "none"

    def test_exactly_at_lower_minus_buffer(
        self, detector: BreakoutDetector, range_bounds: RangeBounds,
    ) -> None:
        """Boundary case: exactly at lower - buffer is NOT a breakout (need <)."""
        price = range_bounds.lower - _BUFFER  # 2338.75
        assert detector.check_breakout(price, range_bounds, _H1_ATR_POINTS) == "none"


# ---------------------------------------------------------------------------
# Custom multiplier
# ---------------------------------------------------------------------------

class TestCustomMultiplier:
    def test_zero_buffer(self, range_bounds: RangeBounds) -> None:
        """With mult=0, any price outside boundary is a breakout."""
        det = BreakoutDetector(atr_buffer_mult=0.0)
        assert det.check_breakout(2380.01, range_bounds, _H1_ATR_POINTS) == "bullish_breakout"
        assert det.check_breakout(2339.99, range_bounds, _H1_ATR_POINTS) == "bearish_breakout"

    def test_large_buffer(self, range_bounds: RangeBounds) -> None:
        """With mult=1.0, buffer is 500*0.01*1.0=$5.00 — need > 2385 or < 2335."""
        det = BreakoutDetector(atr_buffer_mult=1.0)
        assert det.check_breakout(2384.00, range_bounds, _H1_ATR_POINTS) == "none"
        assert det.check_breakout(2386.00, range_bounds, _H1_ATR_POINTS) == "bullish_breakout"

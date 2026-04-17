"""Unit tests for smc.strategy.mode_router — priority-based mode routing."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from smc.strategy.hysteresis import HysteresisState
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
        duration_bars=20,
    )


# ---------------------------------------------------------------------------
# ASIAN_CORE session: Round 4.5 — ranging ALLOWED, trending still blocked
# ---------------------------------------------------------------------------


class TestAsianCoreSession:
    def test_asian_core_neutral_no_range(self) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="ASIAN_CORE",
            range_bounds=None,
        )
        assert result.mode == "v1_passthrough"

    def test_asian_core_with_range_and_guards_enters_ranging(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """Round 4.5: ASIAN_CORE now allows ranging when range + guards pass."""
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="ASIAN_CORE",
            range_bounds=sample_range_bounds,
            guards_passed=True,
        )
        assert result.mode == "ranging"

    def test_asian_core_bullish_high_conf_still_v1_passthrough(self) -> None:
        """ASIAN_CORE continues to block trending even at high confidence."""
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.9,
            regime="trending",
            session="ASIAN_CORE",
            range_bounds=None,
        )
        assert result.mode == "v1_passthrough"


# ---------------------------------------------------------------------------
# Priority 1: AI strong directional + confidence >= 0.5 + NOT ASIAN_CORE
# ---------------------------------------------------------------------------


class TestPriority1Trending:
    def test_london_bullish_0_6(self) -> None:
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.6,
            regime="trending",
            session="LONDON",
            range_bounds=None,
        )
        assert result.mode == "trending"
        assert "bullish" in result.reason

    def test_bullish_exactly_threshold(self) -> None:
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.5,
            regime="transitional",
            session="LONDON",
            range_bounds=None,
        )
        assert result.mode == "trending"

    def test_bearish_high_conf_new_york(self) -> None:
        result = route_trading_mode(
            ai_direction="bearish",
            ai_confidence=0.8,
            regime="trending",
            session="NEW YORK",
            range_bounds=None,
        )
        assert result.mode == "trending"
        assert "bearish" in result.reason

    def test_asian_non_core_bullish_allowed(self) -> None:
        """'ASIAN' session (not 'ASIAN_CORE') is permitted for trending."""
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.7,
            regime="trending",
            session="ASIAN",
            range_bounds=None,
        )
        assert result.mode == "trending"

    def test_trending_priority_over_range_and_guards(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """Even with range_bounds + guards_passed, high AI confidence wins."""
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.7,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=True,
        )
        assert result.mode == "trending"

    def test_below_conf_threshold_does_not_trend(self) -> None:
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.49,
            regime="trending",
            session="LONDON",
            range_bounds=None,
        )
        assert result.mode != "trending"


# ---------------------------------------------------------------------------
# Priority 2: range + 5 guards pass + active session → ranging
# ---------------------------------------------------------------------------


class TestPriority2Ranging:
    def test_asian_london_transition_with_range_and_guards(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="ASIAN_LONDON_TRANSITION",
            range_bounds=sample_range_bounds,
            guards_passed=True,
        )
        assert result.mode == "ranging"
        assert result.range_bounds is not None

    def test_london_with_range_and_guards(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=True,
        )
        assert result.mode == "ranging"

    def test_new_york_with_range_and_guards(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.1,
            regime="transitional",
            session="NEW YORK",
            range_bounds=sample_range_bounds,
            guards_passed=True,
        )
        assert result.mode == "ranging"

    def test_guards_failed_gives_v1_passthrough(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=False,
        )
        assert result.mode == "v1_passthrough"

    def test_no_range_bounds_gives_v1_passthrough(self) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="LONDON",
            range_bounds=None,
            guards_passed=True,
        )
        assert result.mode == "v1_passthrough"

    def test_low_conf_bullish_with_range_and_guards(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """Bullish but below threshold → falls to Priority 2 ranging."""
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.3,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=True,
        )
        assert result.mode == "ranging"


# ---------------------------------------------------------------------------
# Fallback: v1_passthrough
# ---------------------------------------------------------------------------


class TestV1Passthrough:
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
            session="NEW YORK",
            range_bounds=None,
        )
        assert result.mode == "v1_passthrough"
        assert "v1 pipeline" in result.reason

    def test_neutral_trending_regime_no_range(self) -> None:
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


# ---------------------------------------------------------------------------
# Hysteresis integration: route_trading_mode output fed through HysteresisState
# ---------------------------------------------------------------------------


class TestModeRouterWithHysteresis:
    def test_bar1_ranging_bar2_break_no_flip(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """bar1 proposes ranging, bar2 breaks → committed mode stays v1_passthrough."""
        state = HysteresisState()

        mode1 = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=True,
        )
        committed1 = state.update(mode1.mode)
        assert committed1 == "v1_passthrough"  # pending, not committed yet

        mode2 = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=False,  # guards fail → break
        )
        committed2 = state.update(mode2.mode)
        assert committed2 == "v1_passthrough"  # no flip

    def test_bar1_bar2_ranging_flips(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """Two consecutive ranging proposals commit to ranging."""
        state = HysteresisState()
        committed = "v1_passthrough"

        for _ in range(2):
            mode = route_trading_mode(
                ai_direction="neutral",
                ai_confidence=0.2,
                regime="ranging",
                session="LONDON",
                range_bounds=sample_range_bounds,
                guards_passed=True,
            )
            committed = state.update(mode.mode)

        assert committed == "ranging"

    def test_flipped_ranging_bar1_bar2_break_flips_back(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """Flipped to ranging; two consecutive break bars flip back to v1_passthrough."""
        state = HysteresisState(current_mode="ranging")
        committed = "ranging"

        for _ in range(2):
            mode = route_trading_mode(
                ai_direction="neutral",
                ai_confidence=0.2,
                regime="ranging",
                session="LONDON",
                range_bounds=sample_range_bounds,
                guards_passed=False,  # guards fail
            )
            committed = state.update(mode.mode)

        assert committed == "v1_passthrough"

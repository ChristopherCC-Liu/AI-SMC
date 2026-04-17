"""Integration tests for dual-mode routing: mode_router dispatch.

Each test verifies the TradingMode emitted by route_trading_mode() for a
specific session × AI × range_bounds × guards combination.
"""
import pytest
from datetime import datetime, timezone

from smc.strategy.mode_router import route_trading_mode
from smc.strategy.range_types import RangeBounds


def make_bounds(width_pts: float = 1000.0, duration: int = 12) -> RangeBounds:
    return RangeBounds(
        upper=2400.0,
        lower=2390.0,
        width_points=width_pts,
        midpoint=2395.0,
        detected_at=datetime.now(tz=timezone.utc),
        source="ob_boundaries",
        confidence=0.7,
        duration_bars=duration,
    )


class TestDualModeRouting:
    # --- Case 1: ASIAN_CORE + AI neutral → v1_passthrough ---
    def test_asian_core_neutral_v1_passthrough(self):
        mode = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="transitional",
            session="ASIAN_CORE",
            range_bounds=None,
            guards_passed=False,
        )
        assert mode.mode == "v1_passthrough"

    # --- Case 2: ASIAN_CORE + range + guards pass → still v1_passthrough (session blocks ranging) ---
    def test_asian_core_with_range_blocked(self):
        mode = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="ranging",
            session="ASIAN_CORE",
            range_bounds=make_bounds(),
            guards_passed=True,
        )
        assert mode.mode == "v1_passthrough"

    # --- Case 3: ASIAN_LONDON_TRANSITION + range + guards pass → ranging ---
    def test_asian_london_transition_ranging(self):
        mode = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="ranging",
            session="ASIAN_LONDON_TRANSITION",
            range_bounds=make_bounds(),
            guards_passed=True,
        )
        assert mode.mode == "ranging"

    # --- Case 4: LONDON + range + guards pass → ranging ---
    def test_london_ranging_allowed(self):
        mode = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="ranging",
            session="LONDON",
            range_bounds=make_bounds(),
            guards_passed=True,
        )
        assert mode.mode == "ranging"

    # --- Case 5: LONDON + AI bullish 0.6 → trending ---
    def test_london_trending(self):
        mode = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.6,
            regime="trending",
            session="LONDON",
            range_bounds=None,
            guards_passed=False,
        )
        assert mode.mode == "trending"

    # --- Case 6: NEW_YORK + AI bearish 0.7 → trending ---
    def test_ny_trending(self):
        mode = route_trading_mode(
            ai_direction="bearish",
            ai_confidence=0.7,
            regime="trending",
            session="NEW YORK",
            range_bounds=None,
            guards_passed=False,
        )
        assert mode.mode == "trending"

    # --- Case 7: LONDON + range width<800 (guards fail) → v1_passthrough ---
    def test_london_narrow_range_fallback(self):
        mode = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="ranging",
            session="LONDON",
            range_bounds=make_bounds(width_pts=700),
            guards_passed=False,
        )
        assert mode.mode == "v1_passthrough"

    # --- Case 8: LONDON + range but guards_passed=False → v1_passthrough ---
    def test_london_guards_fail_fallback(self):
        mode = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="ranging",
            session="LONDON",
            range_bounds=make_bounds(),
            guards_passed=False,
        )
        assert mode.mode == "v1_passthrough"

    # --- Case 9: LONDON + range + duration=10 + guards_passed=False → v1_passthrough ---
    def test_london_short_duration_fallback(self):
        mode = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="ranging",
            session="LONDON",
            range_bounds=make_bounds(duration=10),
            guards_passed=False,
        )
        assert mode.mode == "v1_passthrough"

    # --- Case 10: ASIAN_LONDON_TRANSITION + AI bullish 0.6 → trending ---
    def test_asian_london_transition_trending(self):
        mode = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.6,
            regime="trending",
            session="ASIAN_LONDON_TRANSITION",
            range_bounds=None,
            guards_passed=False,
        )
        assert mode.mode == "trending"

    # --- Case 11: LONDON + AI neutral + no range → v1_passthrough ---
    def test_london_neutral_no_range(self):
        mode = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.4,
            regime="transitional",
            session="LONDON",
            range_bounds=None,
            guards_passed=False,
        )
        assert mode.mode == "v1_passthrough"

    # --- Case 12: ASIAN_CORE + AI bullish 0.6 → v1_passthrough (trending blocked in ASIAN_CORE) ---
    def test_asian_core_bullish_no_trending(self):
        mode = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.6,
            regime="trending",
            session="ASIAN_CORE",
            range_bounds=None,
            guards_passed=False,
        )
        assert mode.mode == "v1_passthrough"

    # --- Case 13: LONDON + AI bullish 0.4 (below threshold) → v1_passthrough ---
    def test_london_weak_bullish_fallback(self):
        mode = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.4,
            regime="transitional",
            session="LONDON",
            range_bounds=None,
            guards_passed=False,
        )
        assert mode.mode == "v1_passthrough"

    # --- Case 14: LATE NY + range + guards pass → ranging ---
    def test_late_ny_ranging(self):
        mode = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="ranging",
            session="LATE NY",
            range_bounds=make_bounds(),
            guards_passed=True,
        )
        assert mode.mode == "ranging"

    # --- Case 15 (4.6-R USER CATCH): LONDON + trending regime → v1_passthrough ---
    # Reversed from prior "range priority over trending regime" contract.
    # 4/17 14:00+ rally real-world miss: ranging mode 在 sustained trending
    # breakout 里等 M15 CHoCH 反转 = 逆势死等. v1_passthrough 让 HTF bias +
    # confluence 主导可 follow trend. ASIAN_CORE 保留 ranging 优先 (Asian 反转
    # 力强, regime trending 多假信号).
    def test_london_trending_regime_yields_to_v1_passthrough(self):
        mode = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="trending",
            session="LONDON",
            range_bounds=make_bounds(),
            guards_passed=True,
        )
        assert mode.mode == "v1_passthrough"

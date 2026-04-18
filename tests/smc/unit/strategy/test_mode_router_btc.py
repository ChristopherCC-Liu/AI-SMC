"""Unit tests for mode_router with BTC InstrumentConfig injection.

Covers:
- BTC ranging_sessions used correctly (both HIGH_VOL and LOW_VOL allow ranging)
- asian_core_session_name=None means the Asian-core gate is skipped (pass-through)
  for ALL sessions including ones that would otherwise be blocked.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from smc.instruments import get_instrument_config
from smc.strategy.mode_router import route_trading_mode
from smc.strategy.range_types import RangeBounds


@pytest.fixture()
def btc_cfg():
    return get_instrument_config("BTCUSD")


@pytest.fixture()
def sample_range_bounds() -> RangeBounds:
    return RangeBounds(
        upper=70_000.0,
        lower=68_000.0,
        width_points=2_000.0,
        midpoint=69_000.0,
        detected_at=datetime(2024, 6, 10, 15, 0, tzinfo=timezone.utc),
        source="ob_boundaries",
        confidence=0.85,
        duration_bars=20,
    )


# ---------------------------------------------------------------------------
# BTC ranging_sessions injection
# ---------------------------------------------------------------------------


class TestBtcRangingSessions:
    def test_high_vol_ranging_allowed(
        self, btc_cfg, sample_range_bounds: RangeBounds
    ) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="HIGH_VOL",
            range_bounds=sample_range_bounds,
            guards_passed=True,
            cfg=btc_cfg,
        )
        assert result.mode == "ranging"

    def test_low_vol_ranging_allowed(
        self, btc_cfg, sample_range_bounds: RangeBounds
    ) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="LOW_VOL",
            range_bounds=sample_range_bounds,
            guards_passed=True,
            cfg=btc_cfg,
        )
        assert result.mode == "ranging"

    def test_unknown_session_no_ranging(
        self, btc_cfg, sample_range_bounds: RangeBounds
    ) -> None:
        """A session name not in BTC ranging_sessions falls through to v1_passthrough."""
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="LONDON",  # XAU session — not in BTC cfg
            range_bounds=sample_range_bounds,
            guards_passed=True,
            cfg=btc_cfg,
        )
        assert result.mode == "v1_passthrough"


# ---------------------------------------------------------------------------
# asian_core_session_name=None → gate pass-through
# ---------------------------------------------------------------------------


class TestBtcAsianCoreNone:
    def test_high_vol_bullish_trending_allowed(self, btc_cfg) -> None:
        """With asian_core_session_name=None, HIGH_VOL does NOT block trending."""
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.9,
            regime="trending",
            session="HIGH_VOL",
            range_bounds=None,
            cfg=btc_cfg,
        )
        assert result.mode == "trending"

    def test_low_vol_bullish_trending_allowed(self, btc_cfg) -> None:
        """LOW_VOL (would be 'ASIAN_CORE'-equivalent for XAU) also allows trending."""
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.9,
            regime="trending",
            session="LOW_VOL",
            range_bounds=None,
            cfg=btc_cfg,
        )
        assert result.mode == "trending"

    def test_bearish_high_conf_low_vol_trending(self, btc_cfg) -> None:
        result = route_trading_mode(
            ai_direction="bearish",
            ai_confidence=0.75,
            regime="trending",
            session="LOW_VOL",
            range_bounds=None,
            cfg=btc_cfg,
        )
        assert result.mode == "trending"


# ---------------------------------------------------------------------------
# XAU default path unchanged (no cfg kwarg)
# ---------------------------------------------------------------------------


class TestXauDefaultPath:
    def test_xau_asian_core_blocks_trending(self) -> None:
        """Default (no cfg) still blocks trending in ASIAN_CORE for XAU."""
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.9,
            regime="trending",
            session="ASIAN_CORE",
            range_bounds=None,
        )
        assert result.mode == "v1_passthrough"

    def test_xau_london_trending_allowed(self) -> None:
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.7,
            regime="trending",
            session="LONDON",
            range_bounds=None,
        )
        assert result.mode == "trending"

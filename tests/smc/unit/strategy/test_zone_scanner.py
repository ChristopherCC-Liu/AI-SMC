"""Unit tests for smc.strategy.zone_scanner — H1 trade zone identification."""

from __future__ import annotations

import pytest

from smc.data.schemas import Timeframe
from smc.smc_core.types import FairValueGap, OrderBlock, SMCSnapshot
from smc.strategy.types import BiasDirection, TradeZone
from smc.strategy.zone_scanner import scan_zones


class TestScanZones:
    def test_neutral_bias_returns_empty(
        self,
        h1_bullish_snapshot: SMCSnapshot,
        neutral_bias: BiasDirection,
    ) -> None:
        zones = scan_zones(h1_bullish_snapshot, neutral_bias)
        assert zones == ()

    def test_bullish_bias_returns_bullish_zones(
        self,
        h1_bullish_snapshot: SMCSnapshot,
        bullish_bias: BiasDirection,
    ) -> None:
        zones = scan_zones(h1_bullish_snapshot, bullish_bias)
        assert len(zones) > 0
        for zone in zones:
            assert zone.direction == "long"

    def test_max_3_zones(
        self,
        h1_bullish_snapshot: SMCSnapshot,
        bullish_bias: BiasDirection,
    ) -> None:
        zones = scan_zones(h1_bullish_snapshot, bullish_bias)
        assert len(zones) <= 3

    def test_zones_sorted_by_confidence_descending(
        self,
        h1_bullish_snapshot: SMCSnapshot,
        bullish_bias: BiasDirection,
    ) -> None:
        zones = scan_zones(h1_bullish_snapshot, bullish_bias)
        if len(zones) > 1:
            confidences = [z.confidence for z in zones]
            assert confidences == sorted(confidences, reverse=True)

    def test_ob_fvg_overlap_gets_bonus(
        self,
        bullish_bias: BiasDirection,
    ) -> None:
        """When an OB and FVG overlap spatially, the zone should get an overlap bonus."""
        from tests.smc.unit.strategy.conftest import _make_snapshot, _ts

        snapshot = _make_snapshot(
            timeframe=Timeframe.H1,
            trend="bullish",
            order_blocks=(
                OrderBlock(ts_start=_ts(2), ts_end=_ts(3), high=2355.00, low=2350.00, ob_type="bullish", timeframe=Timeframe.H1, mitigated=False),
            ),
            fvgs=(
                FairValueGap(ts=_ts(4), high=2354.00, low=2351.00, fvg_type="bullish", timeframe=Timeframe.H1, filled_pct=0.0, fully_filled=False),
            ),
        )
        zones = scan_zones(snapshot, bullish_bias)
        assert len(zones) > 0
        # At least one zone should be ob_fvg_overlap
        overlap_zones = [z for z in zones if z.zone_type == "ob_fvg_overlap"]
        assert len(overlap_zones) > 0

    def test_mitigated_obs_excluded(
        self,
        bullish_bias: BiasDirection,
    ) -> None:
        """Mitigated order blocks should not produce zones."""
        from tests.smc.unit.strategy.conftest import _make_snapshot, _ts

        snapshot = _make_snapshot(
            timeframe=Timeframe.H1,
            trend="bullish",
            order_blocks=(
                OrderBlock(ts_start=_ts(2), ts_end=_ts(3), high=2355.00, low=2350.00, ob_type="bullish", timeframe=Timeframe.H1, mitigated=True, mitigated_at=_ts(10)),
            ),
        )
        zones = scan_zones(snapshot, bullish_bias)
        assert zones == ()

    def test_filled_fvgs_excluded(
        self,
        bullish_bias: BiasDirection,
    ) -> None:
        """Fully filled FVGs should not produce zones."""
        from tests.smc.unit.strategy.conftest import _make_snapshot, _ts

        snapshot = _make_snapshot(
            timeframe=Timeframe.H1,
            trend="bullish",
            fvgs=(
                FairValueGap(ts=_ts(4), high=2356.00, low=2352.00, fvg_type="bullish", timeframe=Timeframe.H1, filled_pct=1.0, fully_filled=True),
            ),
        )
        zones = scan_zones(snapshot, bullish_bias)
        assert zones == ()

    def test_bearish_obs_excluded_for_bullish_bias(
        self,
        bullish_bias: BiasDirection,
    ) -> None:
        """Bearish OBs should be excluded when bias is bullish."""
        from tests.smc.unit.strategy.conftest import _make_snapshot, _ts

        snapshot = _make_snapshot(
            timeframe=Timeframe.H1,
            trend="bullish",
            order_blocks=(
                OrderBlock(ts_start=_ts(2), ts_end=_ts(3), high=2380.00, low=2376.00, ob_type="bearish", timeframe=Timeframe.H1, mitigated=False),
            ),
        )
        zones = scan_zones(snapshot, bullish_bias)
        assert zones == ()

    def test_empty_snapshot_returns_empty(
        self,
        h1_empty_snapshot: SMCSnapshot,
        bullish_bias: BiasDirection,
    ) -> None:
        zones = scan_zones(h1_empty_snapshot, bullish_bias)
        assert zones == ()

    def test_zone_timeframe_matches_snapshot(
        self,
        h1_bullish_snapshot: SMCSnapshot,
        bullish_bias: BiasDirection,
    ) -> None:
        zones = scan_zones(h1_bullish_snapshot, bullish_bias)
        for zone in zones:
            assert zone.timeframe == Timeframe.H1

    def test_confidence_bounded(
        self,
        h1_bullish_snapshot: SMCSnapshot,
        bullish_bias: BiasDirection,
    ) -> None:
        zones = scan_zones(h1_bullish_snapshot, bullish_bias)
        for zone in zones:
            assert 0.0 <= zone.confidence <= 1.0

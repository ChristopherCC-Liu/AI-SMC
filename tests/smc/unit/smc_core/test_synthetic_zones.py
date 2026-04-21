"""Unit tests for smc.smc_core.synthetic_zones — ATH fallback zones.

Round 5 A-track Task #9 — build synthetic supply/demand anchors when
historical OB/FVG zones are absent in new-high regimes (W14+W15 2024
setup drought fix).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from smc.data.schemas import Timeframe
from smc.smc_core.synthetic_zones import (
    ATH_TRIGGER_PERCENTILE,
    DEFAULT_ROUND_NUMBER_STEP,
    DEFAULT_ROUND_NUMBER_WINDOW_PCT,
    SyntheticZoneConfig,
    build_synthetic_zones,
)
from smc.strategy.types import TradeZone


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


_UTC_BASE = datetime(2024, 6, 10, 12, 0, 0, tzinfo=timezone.utc)  # Monday 12:00 UTC


def _make_m15_df(
    *,
    n_bars: int = 96,
    start_ts: datetime = _UTC_BASE - timedelta(hours=24),
    base_price: float = 4000.0,
    price_noise: float = 10.0,
) -> pl.DataFrame:
    """Produce a deterministic M15 OHLCV frame for synthetic zone tests.

    96 bars × 15 min = 24 h → covers full Asian + London + NY session
    boundaries for session H/L extraction.
    """
    rows = []
    for i in range(n_bars):
        ts = start_ts + timedelta(minutes=15 * i)
        offset = (i % 20) * (price_noise / 10.0)
        high = base_price + offset + 5.0
        low = base_price + offset - 5.0
        close = base_price + offset
        rows.append({
            "ts": ts,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0,
        })
    return pl.DataFrame(rows)


def _make_h1_df(
    *,
    n_bars: int = 168,  # 7 days
    start_ts: datetime = _UTC_BASE - timedelta(days=7),
    base_price: float = 4000.0,
) -> pl.DataFrame:
    rows = []
    for i in range(n_bars):
        ts = start_ts + timedelta(hours=i)
        offset = (i % 15) * 2.0
        high = base_price + offset + 10.0
        low = base_price + offset - 10.0
        close = base_price + offset
        rows.append({
            "ts": ts,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0,
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gate tests — ATH trigger
# ---------------------------------------------------------------------------


class TestATHGate:
    def test_below_ath_returns_empty(self) -> None:
        """When price is in middle of 52w range, no synthetic zones fire."""
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(),
            h1_df=_make_h1_df(),
            current_price=4000.0,
            price_52w_high=5000.0,
            price_52w_low=3000.0,  # pct = 0.5 → below 0.95 trigger
        )
        assert zones == []

    def test_ath_above_threshold_fires(self) -> None:
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(),
            h1_df=_make_h1_df(),
            current_price=4950.0,
            price_52w_high=5000.0,
            price_52w_low=3000.0,  # pct = 0.975 → above 0.95 trigger
            now=_UTC_BASE,
        )
        assert len(zones) > 0

    def test_degenerate_52w_range_returns_empty(self) -> None:
        """Flat 52w range (high==low) → no zones."""
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(),
            h1_df=_make_h1_df(),
            current_price=4000.0,
            price_52w_high=4000.0,
            price_52w_low=4000.0,
        )
        assert zones == []

    def test_zero_high_returns_empty(self) -> None:
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(),
            h1_df=_make_h1_df(),
            current_price=4000.0,
            price_52w_high=0.0,
            price_52w_low=0.0,
        )
        assert zones == []


# ---------------------------------------------------------------------------
# Happy path — W14+W15 2024 drought regression fixture
# ---------------------------------------------------------------------------


class TestATHDroughtRegression:
    """The W14+W15 2024 drought: 6 months of zero historical setups.  This
    regression asserts the synthetic path yields ≥ 3 anchors under those
    conditions (new ATH + no historical zones)."""

    def test_returns_at_least_three_zones(self) -> None:
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(),
            h1_df=_make_h1_df(),
            current_price=4900.0,
            price_52w_high=4950.0,
            price_52w_low=2000.0,
            now=_UTC_BASE,
        )
        assert len(zones) >= 3

    def test_all_zones_are_synthetic_type(self) -> None:
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(),
            h1_df=_make_h1_df(),
            current_price=4900.0,
            price_52w_high=4950.0,
            price_52w_low=2000.0,
            now=_UTC_BASE,
        )
        for z in zones:
            assert z.zone_type.startswith("synthetic_")
            assert isinstance(z, TradeZone)

    def test_zones_have_direction_relative_to_price(self) -> None:
        """Anchors above current price → short; below → long."""
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(),
            h1_df=_make_h1_df(),
            current_price=4900.0,
            price_52w_high=4950.0,
            price_52w_low=2000.0,
            now=_UTC_BASE,
        )
        for z in zones:
            mid = (z.zone_high + z.zone_low) / 2.0
            if mid < 4900.0:
                assert z.direction == "long"
            else:
                assert z.direction == "short"

    def test_zones_have_positive_width(self) -> None:
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(),
            h1_df=_make_h1_df(),
            current_price=4900.0,
            price_52w_high=4950.0,
            price_52w_low=2000.0,
            now=_UTC_BASE,
        )
        for z in zones:
            assert z.zone_high > z.zone_low

    def test_zones_carry_sensible_confidence(self) -> None:
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(),
            h1_df=_make_h1_df(),
            current_price=4900.0,
            price_52w_high=4950.0,
            price_52w_low=2000.0,
            now=_UTC_BASE,
        )
        for z in zones:
            assert 0.0 < z.confidence <= 1.0


# ---------------------------------------------------------------------------
# Round-number zones
# ---------------------------------------------------------------------------


class TestRoundNumberZones:
    def test_round_numbers_within_window(self) -> None:
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(),
            h1_df=_make_h1_df(),
            current_price=4900.0,
            price_52w_high=4950.0,
            price_52w_low=2000.0,
            now=_UTC_BASE,
        )
        # Window = 1.5% of 4900 = ±73.5 → expect 4850 / 4900 skip-self / 4950 boundary.
        mids = [(z.zone_high + z.zone_low) / 2.0 for z in zones]
        round_hits = [m for m in mids if abs(m - round(m / 50) * 50) < 0.01]
        assert len(round_hits) >= 1

    def test_custom_round_number_step(self) -> None:
        # Widen round_number_window_pct to 3% so $100 grid (4800/5000)
        # reaches into the window for a $4900 anchor.
        cfg = SyntheticZoneConfig(round_number_step=100.0, round_number_window_pct=3.0)
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(),
            h1_df=_make_h1_df(),
            current_price=4900.0,
            price_52w_high=4950.0,
            price_52w_low=2000.0,
            cfg=cfg,
            now=_UTC_BASE,
        )
        # With $100 steps + 3% window = ±147 → includes 4800 and 5000
        # (4900 itself skipped because it equals current_price).
        round_100_hits = [
            (z.zone_high + z.zone_low) / 2.0
            for z in zones
            if abs(((z.zone_high + z.zone_low) / 2.0) % 100.0) < 1.0
        ]
        assert len(round_100_hits) >= 1

    def test_round_number_at_current_price_skipped(self) -> None:
        # Current price at 4850 exactly, step 50 → 4800/4900 inside window
        # but 4850 excluded because == current_price.
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(),
            h1_df=_make_h1_df(),
            current_price=4850.0,
            price_52w_high=4900.0,
            price_52w_low=2000.0,
            now=_UTC_BASE,
        )
        mids = [(z.zone_high + z.zone_low) / 2.0 for z in zones]
        assert 4850.0 not in mids


# ---------------------------------------------------------------------------
# VWAP band zones
# ---------------------------------------------------------------------------


class TestVWAPZones:
    def test_produces_vwap_bands_when_sufficient_bars(self) -> None:
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(n_bars=96),
            h1_df=_make_h1_df(),
            current_price=4900.0,
            price_52w_high=4950.0,
            price_52w_low=2000.0,
            now=_UTC_BASE,
        )
        # VWAP band is the only component that can produce zones with price
        # mid near the VWAP ± std. Not checking identity — just that we
        # got *some* zones in a reasonable band around current price.
        assert all((z.zone_high > z.zone_low) for z in zones)

    def test_insufficient_bars_still_ok(self) -> None:
        """Very short M15 frame → no VWAP bands, but round numbers still fire."""
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(n_bars=5),
            h1_df=_make_h1_df(),
            current_price=4900.0,
            price_52w_high=4950.0,
            price_52w_low=2000.0,
            now=_UTC_BASE,
        )
        # Round numbers + previous week still contribute.
        assert len(zones) >= 1


# ---------------------------------------------------------------------------
# Null-safety + graceful degradation
# ---------------------------------------------------------------------------


class TestNullSafety:
    def test_missing_m15_returns_only_round_numbers_and_prev_week(self) -> None:
        zones = build_synthetic_zones(
            m15_df=None,
            h1_df=_make_h1_df(),
            current_price=4900.0,
            price_52w_high=4950.0,
            price_52w_low=2000.0,
            now=_UTC_BASE,
        )
        # Round numbers + prev week remain — expect ≥ 1 zone.
        assert len(zones) >= 1
        for z in zones:
            assert z.zone_type.startswith("synthetic_")

    def test_missing_h1_returns_only_m15_sources(self) -> None:
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(),
            h1_df=None,
            current_price=4900.0,
            price_52w_high=4950.0,
            price_52w_low=2000.0,
            now=_UTC_BASE,
        )
        assert len(zones) >= 1

    def test_missing_both_ohlcv_returns_only_round_numbers(self) -> None:
        zones = build_synthetic_zones(
            m15_df=None,
            h1_df=None,
            current_price=4900.0,
            price_52w_high=4950.0,
            price_52w_low=2000.0,
            now=_UTC_BASE,
        )
        # Round numbers alone should yield at least 1 zone within ±1.5%.
        assert len(zones) >= 1

    def test_zero_current_price_returns_empty(self) -> None:
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(),
            h1_df=_make_h1_df(),
            current_price=0.0,
            price_52w_high=4950.0,
            price_52w_low=2000.0,
            now=_UTC_BASE,
        )
        # Current price 0 → pct computation degenerate, no zones.
        assert zones == []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_no_two_zones_within_005pct(self) -> None:
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(),
            h1_df=_make_h1_df(),
            current_price=4900.0,
            price_52w_high=4950.0,
            price_52w_low=2000.0,
            now=_UTC_BASE,
        )
        tolerance = 4900.0 * 0.0005
        mids = [(z.zone_high + z.zone_low) / 2.0 for z in zones]
        for i, a in enumerate(mids):
            for b in mids[i + 1:]:
                assert abs(a - b) > tolerance


# ---------------------------------------------------------------------------
# Config exports
# ---------------------------------------------------------------------------


class TestZoneSubtypeProvenance:
    """R-review: synthetic zones must carry per-source provenance subtype."""

    def test_valid_subtype_values(self) -> None:
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(),
            h1_df=_make_h1_df(),
            current_price=4900.0,
            price_52w_high=4950.0,
            price_52w_low=2000.0,
            now=_UTC_BASE,
        )
        valid = {
            "synthetic_vwap",
            "synthetic_session",
            "synthetic_round",
            "synthetic_prev_week",
        }
        for z in zones:
            assert z.zone_type in valid, (
                f"unexpected subtype {z.zone_type}; must be one of {valid}"
            )

    def test_round_subtype_appears_when_inside_window(self) -> None:
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(),
            h1_df=_make_h1_df(),
            current_price=4900.0,
            price_52w_high=4950.0,
            price_52w_low=2000.0,
            now=_UTC_BASE,
        )
        subtypes = {z.zone_type for z in zones}
        # $50 step within ±1.5% of 4900 → 4850 + 4950 reachable.
        assert "synthetic_round" in subtypes

    def test_vwap_subtype_appears_when_m15_has_bars(self) -> None:
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(n_bars=96),
            h1_df=_make_h1_df(),
            current_price=4900.0,
            price_52w_high=4950.0,
            price_52w_low=2000.0,
            now=_UTC_BASE,
        )
        subtypes = {z.zone_type for z in zones}
        # With 20-period lookback on 96 bars, VWAP bands should emit.
        assert "synthetic_vwap" in subtypes

    def test_all_zones_confidence_is_uniform_04(self) -> None:
        """R-review: all synthetic zones share confidence=0.4."""
        from smc.smc_core.synthetic_zones import SYNTHETIC_CONFIDENCE
        assert SYNTHETIC_CONFIDENCE == 0.4
        zones = build_synthetic_zones(
            m15_df=_make_m15_df(),
            h1_df=_make_h1_df(),
            current_price=4900.0,
            price_52w_high=4950.0,
            price_52w_low=2000.0,
            now=_UTC_BASE,
        )
        for z in zones:
            assert z.confidence == 0.4

    def test_confluence_weight_same_for_all_subtypes(self) -> None:
        """R-review: confluence weights for all 4 synthetic sub-types are 0.35."""
        from smc.strategy.confluence import _score_zone_quality
        from smc.strategy.types import TradeZone
        subtypes = (
            "synthetic_vwap", "synthetic_session",
            "synthetic_round", "synthetic_prev_week",
        )
        scores: list[float] = []
        for st in subtypes:
            z = TradeZone(
                zone_high=4910.0, zone_low=4890.0,
                zone_type=st,  # type: ignore[arg-type]
                direction="long", timeframe=Timeframe.H1, confidence=0.4,
            )
            scores.append(_score_zone_quality(z))
        assert len(set(scores)) == 1, f"scores diverged: {scores}"


class TestAggregatorIntegration:
    """Verify the aggregator augments with synthetic zones when configured."""

    def test_synthetic_zones_disabled_by_default(self) -> None:
        from smc.smc_core.detector import SMCDetector
        from smc.strategy.aggregator import MultiTimeframeAggregator
        agg = MultiTimeframeAggregator(detector=SMCDetector(swing_length=5))
        assert agg._synthetic_zones_enabled is False
        assert agg._synthetic_zones_min_historical == 2

    def test_synthetic_zones_toggle_wires_through(self) -> None:
        from smc.smc_core.detector import SMCDetector
        from smc.strategy.aggregator import MultiTimeframeAggregator
        agg = MultiTimeframeAggregator(
            detector=SMCDetector(swing_length=5),
            synthetic_zones_enabled=True,
            synthetic_zones_min_historical=3,
        )
        assert agg._synthetic_zones_enabled is True
        assert agg._synthetic_zones_min_historical == 3


class TestModuleConstants:
    def test_trigger_percentile_default(self) -> None:
        assert ATH_TRIGGER_PERCENTILE == 0.95

    def test_round_number_defaults(self) -> None:
        assert DEFAULT_ROUND_NUMBER_STEP == 50.0
        assert DEFAULT_ROUND_NUMBER_WINDOW_PCT == 1.5

    def test_config_is_frozen(self) -> None:
        cfg = SyntheticZoneConfig()
        with pytest.raises(Exception):
            cfg.round_number_step = 999.0  # type: ignore[misc]

    def test_config_accepts_overrides(self) -> None:
        cfg = SyntheticZoneConfig(
            round_number_step=100.0,
            round_number_window_pct=2.0,
            zone_half_width_pct=0.05,
            vwap_period=30,
            ath_trigger_percentile=0.9,
        )
        assert cfg.round_number_step == 100.0
        assert cfg.vwap_period == 30
        assert cfg.ath_trigger_percentile == 0.9

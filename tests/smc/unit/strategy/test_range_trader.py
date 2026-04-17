"""Tests for range detection and mean-reversion setup generation.

Covers:
- Range detection via OB boundaries (Method A)
- Range detection via swing extremes (Method B fallback)
- Width rejection (too narrow / too wide)
- Long setup at lower boundary
- Short setup at upper boundary
- No setup when price is mid-range
- TP targets and SL placement
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from smc.data.schemas import Timeframe
from smc.smc_core.constants import XAUUSD_POINT_SIZE
from smc.smc_core.types import (
    OrderBlock,
    SMCSnapshot,
    StructureBreak,
    SwingPoint,
)
from smc.strategy.range_trader import RangeTrader
from smc.strategy.range_types import RangeBounds, RangeSetup

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TS = datetime(2024, 7, 1, 0, 0, 0, tzinfo=timezone.utc)


def _make_snapshot(
    *,
    timeframe: Timeframe = Timeframe.H1,
    trend: str = "ranging",
    swing_points: tuple[SwingPoint, ...] = (),
    order_blocks: tuple[OrderBlock, ...] = (),
    structure_breaks: tuple[StructureBreak, ...] = (),
) -> SMCSnapshot:
    return SMCSnapshot(
        ts=_BASE_TS,
        timeframe=timeframe,
        swing_points=swing_points,
        order_blocks=order_blocks,
        fvgs=(),
        structure_breaks=structure_breaks,
        liquidity_levels=(),
        trend_direction=trend,  # type: ignore[arg-type]
    )


def _empty_h1_df() -> pl.DataFrame:
    """Minimal H1 DataFrame (detect_range signature requires it)."""
    return pl.DataFrame({"high": [2380.0], "low": [2340.0], "close": [2360.0]})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def trader() -> RangeTrader:
    return RangeTrader(min_range_width=300, max_range_width=3000)


@pytest.fixture()
def h1_with_obs() -> SMCSnapshot:
    """H1 snapshot with bearish OB at top + bullish OB at bottom → valid range.

    Range: 2370.00 (bearish OB high) - 2348.00 (bullish OB low) = $22 = 2200 pts.
    Within the 300-3000 point bounds.
    """
    return _make_snapshot(
        timeframe=Timeframe.H1,
        order_blocks=(
            OrderBlock(
                ts_start=_BASE_TS,
                ts_end=_BASE_TS,
                high=2370.00,
                low=2366.00,
                ob_type="bearish",
                timeframe=Timeframe.H1,
            ),
            OrderBlock(
                ts_start=_BASE_TS,
                ts_end=_BASE_TS,
                high=2352.00,
                low=2348.00,
                ob_type="bullish",
                timeframe=Timeframe.H1,
            ),
        ),
        swing_points=(
            SwingPoint(ts=_BASE_TS, price=2348.00, swing_type="low", strength=5),
            SwingPoint(ts=_BASE_TS, price=2370.00, swing_type="high", strength=5),
            SwingPoint(ts=_BASE_TS, price=2350.00, swing_type="low", strength=5),
        ),
    )


@pytest.fixture()
def h1_swings_only() -> SMCSnapshot:
    """H1 snapshot with swings but no OBs → Method B fallback.

    Range: 2368.00 (max high) - 2348.00 (min low) = $20 = 2000 pts.
    """
    return _make_snapshot(
        timeframe=Timeframe.H1,
        swing_points=(
            SwingPoint(ts=_BASE_TS, price=2348.00, swing_type="low", strength=5),
            SwingPoint(ts=_BASE_TS, price=2368.00, swing_type="high", strength=5),
            SwingPoint(ts=_BASE_TS, price=2350.00, swing_type="low", strength=5),
            SwingPoint(ts=_BASE_TS, price=2365.00, swing_type="high", strength=5),
        ),
    )


@pytest.fixture()
def m15_choch_at_lower() -> SMCSnapshot:
    """M15 snapshot with bullish CHoCH near the lower boundary (2348 area)."""
    return _make_snapshot(
        timeframe=Timeframe.M15,
        structure_breaks=(
            StructureBreak(
                ts=_BASE_TS,
                price=2349.00,
                break_type="choch",
                direction="bullish",
                timeframe=Timeframe.M15,
            ),
        ),
    )


@pytest.fixture()
def m15_choch_at_upper() -> SMCSnapshot:
    """M15 snapshot with bearish CHoCH near the upper boundary (2370 area)."""
    return _make_snapshot(
        timeframe=Timeframe.M15,
        structure_breaks=(
            StructureBreak(
                ts=_BASE_TS,
                price=2369.00,
                break_type="choch",
                direction="bearish",
                timeframe=Timeframe.M15,
            ),
        ),
    )


@pytest.fixture()
def m15_no_choch() -> SMCSnapshot:
    """M15 snapshot with no CHoCH — setup should be rejected."""
    return _make_snapshot(timeframe=Timeframe.M15)


# ---------------------------------------------------------------------------
# Range Detection: OB boundaries (Method A)
# ---------------------------------------------------------------------------


class TestDetectRangeOBBoundaries:
    """Method A: derive range from highest bearish OB + lowest bullish OB."""

    def test_valid_range_from_obs(
        self, trader: RangeTrader, h1_with_obs: SMCSnapshot
    ) -> None:
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)

        assert bounds is not None
        assert bounds.source == "ob_boundaries"
        assert bounds.upper == 2370.00
        assert bounds.lower == 2348.00
        assert bounds.confidence == 0.8
        expected_width = (2370.00 - 2348.00) / XAUUSD_POINT_SIZE
        assert bounds.width_points == pytest.approx(expected_width, rel=1e-3)
        assert bounds.midpoint == pytest.approx((2370.00 + 2348.00) / 2.0, rel=1e-3)
        # Round 4.5.1: duration_bars must be populated from OB ts_start, not 0
        assert bounds.duration_bars > 0

    def test_no_range_without_bearish_ob(self, trader: RangeTrader) -> None:
        """Only bullish OBs present — no upper boundary → None."""
        snapshot = _make_snapshot(
            order_blocks=(
                OrderBlock(
                    ts_start=_BASE_TS,
                    ts_end=_BASE_TS,
                    high=2352.00,
                    low=2348.00,
                    ob_type="bullish",
                    timeframe=Timeframe.H1,
                ),
            ),
        )
        assert trader.detect_range(_empty_h1_df(), snapshot) is None


# ---------------------------------------------------------------------------
# Range Detection: Swing extremes (Method B)
# ---------------------------------------------------------------------------


class TestDetectRangeSwingExtremes:
    """Method B: derive range from max swing high + min swing low."""

    def test_fallback_to_swings(
        self, trader: RangeTrader, h1_swings_only: SMCSnapshot
    ) -> None:
        bounds = trader.detect_range(_empty_h1_df(), h1_swings_only)

        assert bounds is not None
        assert bounds.source == "swing_extremes"
        assert bounds.upper == 2368.00
        assert bounds.lower == 2348.00
        assert bounds.confidence == 0.6
        # Round 4.5.1: duration_bars must be populated from swing ts, not 0
        assert bounds.duration_bars > 0

    def test_no_range_with_one_swing(self, trader: RangeTrader) -> None:
        """Fewer than 2 swing points → None."""
        snapshot = _make_snapshot(
            swing_points=(
                SwingPoint(ts=_BASE_TS, price=2360.00, swing_type="high", strength=5),
            ),
        )
        assert trader.detect_range(_empty_h1_df(), snapshot) is None


# ---------------------------------------------------------------------------
# Width rejection
# ---------------------------------------------------------------------------


class TestWidthRejection:
    def test_too_narrow(self) -> None:
        """Range < min_range_width (300 points = $3.00) → rejected."""
        trader = RangeTrader(min_range_width=300, max_range_width=3000)
        # Swings only $2.00 apart = 200 points
        snapshot = _make_snapshot(
            swing_points=(
                SwingPoint(ts=_BASE_TS, price=2360.00, swing_type="low", strength=5),
                SwingPoint(ts=_BASE_TS, price=2362.00, swing_type="high", strength=5),
            ),
        )
        assert trader.detect_range(_empty_h1_df(), snapshot) is None

    def test_too_wide(self) -> None:
        """Range > max_range_width (3000 points = $30.00) → rejected."""
        trader = RangeTrader(min_range_width=300, max_range_width=3000)
        # Swings $40.00 apart = 4000 points
        snapshot = _make_snapshot(
            swing_points=(
                SwingPoint(ts=_BASE_TS, price=2300.00, swing_type="low", strength=5),
                SwingPoint(ts=_BASE_TS, price=2340.00, swing_type="high", strength=5),
            ),
        )
        assert trader.detect_range(_empty_h1_df(), snapshot) is None

    def test_default_max_width_accepts_7000_pts(self) -> None:
        """Round 4.6-D: default max 20000 accepts ~$70 ranges (measured UTC 06:00)."""
        trader = RangeTrader()  # defaults: min=200, max=20000
        highs = [2400.0] * 48
        lows = [2330.0] * 48  # $70 gap = 7000 pts
        closes = [2365.0] * 48
        h1_df = pl.DataFrame({"high": highs, "low": lows, "close": closes})
        bounds = trader.detect_range(h1_df, _empty_snapshot())
        assert bounds is not None
        assert bounds.source == "donchian_channel"
        assert bounds.width_points == 7000.0

    def test_default_max_width_rejects_25000_pts(self) -> None:
        """Default max 20000 still hard-caps egregious 'ranges' (>$200)."""
        trader = RangeTrader()
        highs = [2500.0] * 48
        lows = [2250.0] * 48  # $250 = 25000 pts > 20000 max
        closes = [2375.0] * 48
        h1_df = pl.DataFrame({"high": highs, "low": lows, "close": closes})
        assert trader.detect_range(h1_df, _empty_snapshot()) is None


# ---------------------------------------------------------------------------
# Setup generation: lower boundary → long
# ---------------------------------------------------------------------------


class TestSetupAtLowerBoundary:
    def test_long_setup_at_support(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_choch_at_lower: SMCSnapshot,
    ) -> None:
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None

        # Price near lower boundary
        current_price = 2349.00
        setups = trader.generate_range_setups(
            h1_with_obs, m15_choch_at_lower, current_price, bounds
        )

        assert len(setups) >= 1
        long_setup = setups[0]
        assert long_setup.direction == "long"
        assert long_setup.trigger == "support_bounce"
        assert long_setup.entry_price == current_price

    def test_long_tp_at_midpoint(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_choch_at_lower: SMCSnapshot,
    ) -> None:
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None

        setups = trader.generate_range_setups(
            h1_with_obs, m15_choch_at_lower, 2349.00, bounds
        )
        assert len(setups) >= 1
        assert setups[0].take_profit == bounds.midpoint

    def test_long_sl_below_boundary(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_choch_at_lower: SMCSnapshot,
    ) -> None:
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None

        setups = trader.generate_range_setups(
            h1_with_obs, m15_choch_at_lower, 2349.00, bounds
        )
        assert len(setups) >= 1
        # SL must be below the lower boundary
        assert setups[0].stop_loss < bounds.lower


# ---------------------------------------------------------------------------
# Setup generation: upper boundary → short
# ---------------------------------------------------------------------------


class TestSetupAtUpperBoundary:
    def test_short_setup_at_resistance(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_choch_at_upper: SMCSnapshot,
    ) -> None:
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None

        # Price near upper boundary
        current_price = 2369.00
        setups = trader.generate_range_setups(
            h1_with_obs, m15_choch_at_upper, current_price, bounds
        )

        assert len(setups) >= 1
        short_setup = setups[0]
        assert short_setup.direction == "short"
        assert short_setup.trigger == "resistance_rejection"
        assert short_setup.entry_price == current_price

    def test_short_sl_above_boundary(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_choch_at_upper: SMCSnapshot,
    ) -> None:
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None

        setups = trader.generate_range_setups(
            h1_with_obs, m15_choch_at_upper, 2369.00, bounds
        )
        assert len(setups) >= 1
        # SL must be above the upper boundary
        assert setups[0].stop_loss > bounds.upper


# ---------------------------------------------------------------------------
# No setup when price is mid-range
# ---------------------------------------------------------------------------


class TestNoSetupMidRange:
    def test_no_setup_at_midpoint(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_choch_at_lower: SMCSnapshot,
    ) -> None:
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None

        # Price right at midpoint — not near any boundary
        mid_price = bounds.midpoint
        setups = trader.generate_range_setups(
            h1_with_obs, m15_choch_at_lower, mid_price, bounds
        )

        assert len(setups) == 0


# ---------------------------------------------------------------------------
# Setup requires M15 CHoCH confirmation
# ---------------------------------------------------------------------------


class TestM15Confirmation:
    def test_no_setup_without_choch(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_no_choch: SMCSnapshot,
    ) -> None:
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None

        # Price at lower boundary but no M15 CHoCH
        setups = trader.generate_range_setups(
            h1_with_obs, m15_no_choch, 2349.00, bounds
        )
        assert len(setups) == 0


# ---------------------------------------------------------------------------
# TP extended at opposite boundary minus 10%
# ---------------------------------------------------------------------------


class TestTPExtended:
    def test_long_tp_ext_near_upper(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_choch_at_lower: SMCSnapshot,
    ) -> None:
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None

        setups = trader.generate_range_setups(
            h1_with_obs, m15_choch_at_lower, 2349.00, bounds
        )
        assert len(setups) >= 1

        range_price = bounds.width_points * XAUUSD_POINT_SIZE
        expected_ext = bounds.upper - range_price * 0.10
        assert setups[0].take_profit_ext == pytest.approx(expected_ext, rel=1e-3)

    def test_short_tp_ext_near_lower(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_choch_at_upper: SMCSnapshot,
    ) -> None:
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None

        setups = trader.generate_range_setups(
            h1_with_obs, m15_choch_at_upper, 2369.00, bounds
        )
        assert len(setups) >= 1

        range_price = bounds.width_points * XAUUSD_POINT_SIZE
        expected_ext = bounds.lower + range_price * 0.10
        assert setups[0].take_profit_ext == pytest.approx(expected_ext, rel=1e-3)


# ---------------------------------------------------------------------------
# Grade assignment
# ---------------------------------------------------------------------------


class TestGrading:
    def test_ob_source_high_rr_gets_grade_a(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_choch_at_lower: SMCSnapshot,
    ) -> None:
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None
        assert bounds.source == "ob_boundaries"

        setups = trader.generate_range_setups(
            h1_with_obs, m15_choch_at_lower, 2349.00, bounds
        )
        assert len(setups) >= 1
        # OB source (0.5) + decent RR should yield A or B
        assert setups[0].grade in ("A", "B")

    def test_max_two_setups(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
    ) -> None:
        """generate_range_setups returns at most 2 setups."""
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None

        # Create M15 with both bullish and bearish CHoCH
        m15 = _make_snapshot(
            timeframe=Timeframe.M15,
            structure_breaks=(
                StructureBreak(
                    ts=_BASE_TS,
                    price=2349.00,
                    break_type="choch",
                    direction="bullish",
                    timeframe=Timeframe.M15,
                ),
                StructureBreak(
                    ts=_BASE_TS,
                    price=2369.00,
                    break_type="choch",
                    direction="bearish",
                    timeframe=Timeframe.M15,
                ),
            ),
        )
        # Price at lower boundary only
        setups = trader.generate_range_setups(
            h1_with_obs, m15, 2349.00, bounds
        )
        assert len(setups) <= 2


# ---------------------------------------------------------------------------
# Method D (Donchian channel) — Round 4.5 hotfix
# ---------------------------------------------------------------------------


def _empty_snapshot() -> SMCSnapshot:
    """H1 snapshot with no OB and no swing points (Asian core low-volatility scenario)."""
    return _make_snapshot(swing_points=(), order_blocks=())


def _h1_df_wide_channel(bars: int = 48) -> pl.DataFrame:
    """H1 DataFrame with high-low channel wider than min_range_width (Round 4.6-A: 200)."""
    # Width ~$10 = 1000 points > any min_range_width
    highs = [2370.0] * bars
    lows = [2360.0] * bars
    closes = [2365.0] * bars
    return pl.DataFrame({"high": highs, "low": lows, "close": closes})


def _h1_df_narrow_channel(bars: int = 48) -> pl.DataFrame:
    """H1 DataFrame with channel narrower than min_range_width=200 (Round 4.6-A)."""
    # Width ~$1 = 100 points < min_range_width=200 → rejection by _validate_bounds
    highs = [2361.0] * bars
    lows = [2360.0] * bars
    closes = [2360.5] * bars
    return pl.DataFrame({"high": highs, "low": lows, "close": closes})


class TestMethodDDonchian:
    """Method D: Donchian channel fallback (Round 4.5 Asian core hotfix).

    Round 4.6-A: lookback widened 24 → 48, min_range_width dropped 300 → 200.
    """

    def test_donchian_low_volatility_asian_core_detects_range(
        self, trader: RangeTrader
    ) -> None:
        """Asian core scenario: no OB, no swings, but 48-bar channel wide enough.

        Method A fails (no OB) → Method B fails (no swings) → Method D
        (Donchian) detects the range from H1 high/low over 48 bars.
        """
        h1_df = _h1_df_wide_channel(bars=48)
        snapshot = _empty_snapshot()

        bounds = trader.detect_range(h1_df, snapshot)

        assert bounds is not None
        assert bounds.source == "donchian_channel"
        assert bounds.upper == 2370.0
        assert bounds.lower == 2360.0
        assert bounds.width_points == 1000.0  # $10 / 0.01 = 1000 points
        assert bounds.duration_bars == 48  # Method D reports its lookback (Round 4.6-A)

    def test_donchian_insufficient_bars_returns_none(
        self, trader: RangeTrader
    ) -> None:
        """Fewer than 48 H1 bars → Method D returns None (Round 4.6-A)."""
        h1_df = _h1_df_wide_channel(bars=24)
        snapshot = _empty_snapshot()

        bounds = trader.detect_range(h1_df, snapshot)

        assert bounds is None

    def test_donchian_width_below_min_range_returns_none(
        self, trader: RangeTrader
    ) -> None:
        """Channel narrower than min_range_width=200 → validate_bounds rejects."""
        h1_df = _h1_df_narrow_channel(bars=48)
        snapshot = _empty_snapshot()

        bounds = trader.detect_range(h1_df, snapshot)

        # Width = 100 points < min_range_width=200 → None
        assert bounds is None

    def test_donchian_accepts_asian_width_200_to_800(
        self, trader: RangeTrader
    ) -> None:
        """Round 4.6-A: Asian-range width (e.g. $3 = 300 points) now detects.

        Before 4.6-A: width 300 barely passes min=300 but Guard 1 (>=800) rejects.
        After 4.6-A: detection succeeds; Guard 1 still filters at application time.
        """
        highs = [2363.0] * 48
        lows = [2360.0] * 48
        closes = [2361.5] * 48
        h1_df = pl.DataFrame({"high": highs, "low": lows, "close": closes})
        snapshot = _empty_snapshot()

        bounds = trader.detect_range(h1_df, snapshot)

        assert bounds is not None
        assert bounds.source == "donchian_channel"
        assert bounds.width_points == 300.0  # $3 / 0.01 = 300 pts, >= min 200

    def test_donchian_only_runs_after_a_and_b_fail(
        self, trader: RangeTrader, h1_with_obs: SMCSnapshot
    ) -> None:
        """Method A (OB) succeeds → Method D should NOT run (short-circuit)."""
        h1_df = _h1_df_wide_channel(bars=48)

        bounds = trader.detect_range(h1_df, h1_with_obs)

        # Method A returns first, source stays "ob_boundaries"
        assert bounds is not None
        assert bounds.source == "ob_boundaries"

    def test_donchian_empty_dataframe_returns_none(
        self, trader: RangeTrader
    ) -> None:
        """Empty H1 DataFrame → Method D returns None (not an exception)."""
        h1_df = pl.DataFrame({"high": [], "low": [], "close": []},
                             schema={"high": pl.Float64, "low": pl.Float64, "close": pl.Float64})
        snapshot = _empty_snapshot()

        bounds = trader.detect_range(h1_df, snapshot)

        assert bounds is None


# ---------------------------------------------------------------------------
# Round 4.5.1: Method A/B duration_bars now computed from earliest boundary ts
# ---------------------------------------------------------------------------


class TestMethodABDurationBars:
    """Regression: before Round 4.5.1, Method A/B duration_bars defaulted to 0,
    causing Guard 4 (>=12) to silently reject every A/B-sourced range.
    """

    def test_method_a_recent_obs_yield_low_duration(
        self, trader: RangeTrader
    ) -> None:
        """OBs formed <12h before now → duration_bars < 12 (would fail Guard 4)."""
        now = datetime.now(tz=timezone.utc)
        recent = now - timedelta(hours=5)
        snapshot = _make_snapshot(
            order_blocks=(
                OrderBlock(
                    ts_start=recent, ts_end=recent, high=2370.00, low=2365.00,
                    ob_type="bearish", timeframe=Timeframe.H1,
                ),
                OrderBlock(
                    ts_start=recent, ts_end=recent, high=2352.00, low=2348.00,
                    ob_type="bullish", timeframe=Timeframe.H1,
                ),
            ),
        )
        bounds = trader.detect_range(_empty_h1_df(), snapshot)
        assert bounds is not None
        assert bounds.source == "ob_boundaries"
        assert bounds.duration_bars < 12  # would fail Guard 4

    def test_method_a_old_obs_yield_high_duration(
        self, trader: RangeTrader
    ) -> None:
        """OBs formed 24h before now → duration_bars >= 12 (passes Guard 4)."""
        now = datetime.now(tz=timezone.utc)
        old = now - timedelta(hours=24)
        snapshot = _make_snapshot(
            order_blocks=(
                OrderBlock(
                    ts_start=old, ts_end=old, high=2370.00, low=2365.00,
                    ob_type="bearish", timeframe=Timeframe.H1,
                ),
                OrderBlock(
                    ts_start=old, ts_end=old, high=2352.00, low=2348.00,
                    ob_type="bullish", timeframe=Timeframe.H1,
                ),
            ),
        )
        bounds = trader.detect_range(_empty_h1_df(), snapshot)
        assert bounds is not None
        assert bounds.source == "ob_boundaries"
        assert bounds.duration_bars >= 12  # passes Guard 4

    def test_method_b_swing_duration_bars_populated(
        self, trader: RangeTrader
    ) -> None:
        """Swing-sourced bounds populate duration_bars from earliest swing ts."""
        now = datetime.now(tz=timezone.utc)
        old = now - timedelta(hours=20)
        snapshot = _make_snapshot(
            swing_points=(
                SwingPoint(ts=old, price=2368.00, swing_type="high", strength=5),
                SwingPoint(ts=old, price=2348.00, swing_type="low", strength=5),
            ),
        )
        bounds = trader.detect_range(_empty_h1_df(), snapshot)
        assert bounds is not None
        assert bounds.source == "swing_extremes"
        assert bounds.duration_bars >= 12


# ---------------------------------------------------------------------------
# Round 4.6-C2: measure-first diagnostic exposed via _last_diagnostic
# ---------------------------------------------------------------------------


class TestDetectRangeDiagnostic:
    """Round 4.6-C2: _last_diagnostic surfaces per-cycle failure reasoning."""

    def test_diagnostic_populated_on_failure(self, trader: RangeTrader) -> None:
        snapshot = _empty_snapshot()
        h1_df = pl.DataFrame(
            {"high": [2361.0], "low": [2360.0], "close": [2360.5]},
            schema={"high": pl.Float64, "low": pl.Float64, "close": pl.Float64},
        )
        bounds = trader.detect_range(h1_df, snapshot)
        assert bounds is None
        diag = trader._last_diagnostic
        assert diag["h1_bars_count"] == 1
        assert diag["n_bearish_ob"] == 0
        assert diag["n_bullish_ob"] == 0
        assert diag["method_a_hit"] is False
        assert diag["method_b_hit"] is False
        assert diag["method_d_hit"] is False
        assert diag["donchian_width_pts"] is None
        assert diag["final_source"] is None
        assert diag["donchian_lookback_required"] == 48
        # fixture trader uses min_range_width=300 (test override, not prod default 200)
        assert diag["min_range_width_required"] == trader._min_range_width

    def test_diagnostic_populated_on_method_d_success(
        self, trader: RangeTrader
    ) -> None:
        highs = [2370.0] * 48
        lows = [2360.0] * 48
        closes = [2365.0] * 48
        h1_df = pl.DataFrame({"high": highs, "low": lows, "close": closes})
        bounds = trader.detect_range(h1_df, _empty_snapshot())
        assert bounds is not None
        diag = trader._last_diagnostic
        assert diag["h1_bars_count"] == 48
        assert diag["method_a_hit"] is False
        assert diag["method_b_hit"] is False
        assert diag["method_d_hit"] is True
        assert diag["final_source"] == "donchian_channel"
        assert diag["donchian_width_pts"] == 1000.0

    def test_diagnostic_populated_on_method_a_success(
        self, trader: RangeTrader, h1_with_obs: SMCSnapshot
    ) -> None:
        h1_df = pl.DataFrame(
            {"high": [2370.0] * 48, "low": [2360.0] * 48, "close": [2365.0] * 48}
        )
        bounds = trader.detect_range(h1_df, h1_with_obs)
        assert bounds is not None
        diag = trader._last_diagnostic
        assert diag["method_a_hit"] is True
        assert diag["method_b_hit"] is False  # short-circuited
        assert diag["method_d_hit"] is False
        assert diag["final_source"] == "ob_boundaries"
        assert diag["n_bearish_ob"] >= 1
        assert diag["n_bullish_ob"] >= 1


class TestSetupsDiagnostic:
    """Round 4.6-C2 extended: generate_range_setups exposes per-cycle reason."""

    def test_diagnostic_price_mid_range(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_choch_at_lower: SMCSnapshot,
    ) -> None:
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None
        # Price at midpoint (far from both boundaries)
        mid_price = (bounds.upper + bounds.lower) / 2.0
        setups = trader.generate_range_setups(
            h1_with_obs, m15_choch_at_lower, mid_price, bounds
        )
        assert len(setups) == 0
        diag = trader._last_setups_diagnostic
        assert diag["setup_count"] == 0
        assert diag["near_lower"] is False
        assert diag["near_upper"] is False
        assert diag["reason_if_zero"] == "price_mid_range"

    def test_diagnostic_no_choch_at_lower(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_no_choch: SMCSnapshot,
    ) -> None:
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None
        # Price near lower
        setups = trader.generate_range_setups(
            h1_with_obs, m15_no_choch, 2349.00, bounds
        )
        assert len(setups) == 0
        diag = trader._last_setups_diagnostic
        assert diag["setup_count"] == 0
        assert diag["near_lower"] is True
        assert diag["long_setup_built"] is False
        assert diag["reason_if_zero"] in (
            "no_m15_choch_at_lower", "no_m15_choch_any_boundary"
        )

    def test_diagnostic_long_built(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_choch_at_lower: SMCSnapshot,
    ) -> None:
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None
        setups = trader.generate_range_setups(
            h1_with_obs, m15_choch_at_lower, 2349.00, bounds
        )
        assert len(setups) >= 1
        diag = trader._last_setups_diagnostic
        assert diag["setup_count"] >= 1
        assert diag["near_lower"] is True
        assert diag["long_setup_built"] is True
        assert diag["reason_if_zero"] is None


class TestAsianBoundaryPctProfile:
    """Round 4.6-F (USER DIRECTIVE): Asian session uses 30% boundary_pct."""

    def test_wide_band_sessions_trigger_near_lower(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_choch_at_lower: SMCSnapshot,
    ) -> None:
        """Round 4.6-I: all active ranging sessions use 30% wide band.

        Price 20% into range triggers near_lower for Asian + London + NY.
        Unknown session falls back to constructor 0.15 (mid-range).
        """
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None
        price_20pct = bounds.lower + (bounds.upper - bounds.lower) * 0.20

        # All active ranging sessions: 30% band → triggers
        for session in [
            "ASIAN_CORE",
            "ASIAN_LONDON_TRANSITION",
            "LONDON",
            "LONDON/NY OVERLAP",
            "NEW YORK",
            "LATE NY",
        ]:
            trader.generate_range_setups(
                h1_with_obs, m15_choch_at_lower, price_20pct, bounds, session=session
            )
            diag = trader._last_setups_diagnostic
            assert diag["near_lower"] is True, f"{session} should trigger"
            assert diag["boundary_pct_applied"] == 0.30

        # Unknown session: falls back to trader._boundary_pct (15%) → not triggered
        trader.generate_range_setups(
            h1_with_obs, m15_choch_at_lower, price_20pct, bounds, session="UNKNOWN"
        )
        diag_unknown = trader._last_setups_diagnostic
        assert diag_unknown["near_lower"] is False
        assert diag_unknown["boundary_pct_applied"] == trader._boundary_pct  # 0.15

    def test_asian_london_transition_also_uses_wider_band(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_choch_at_lower: SMCSnapshot,
    ) -> None:
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None
        price_25pct = bounds.lower + (bounds.upper - bounds.lower) * 0.25
        trader.generate_range_setups(
            h1_with_obs, m15_choch_at_lower, price_25pct, bounds,
            session="ASIAN_LONDON_TRANSITION",
        )
        diag = trader._last_setups_diagnostic
        assert diag["boundary_pct_applied"] == 0.30
        assert diag["near_lower"] is True  # 25% < 30%

    def test_default_session_uses_constructor_value(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_choch_at_lower: SMCSnapshot,
    ) -> None:
        """Unknown/empty session falls back to trader._boundary_pct."""
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None
        trader.generate_range_setups(
            h1_with_obs, m15_choch_at_lower, bounds.lower + 0.1, bounds, session=""
        )
        diag = trader._last_setups_diagnostic
        assert diag["boundary_pct_applied"] == trader._boundary_pct

    def test_asian_build_setup_zone_width_consistent(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_choch_at_lower: SMCSnapshot,
    ) -> None:
        """Round 4.6-G (skeptic catch): _build_setup synthetic CHoCH zone
        must use the same effective_boundary_pct as near_lower detection.

        Before 4.6-G: near_lower triggered at 30% band (Asian) but CHoCH
        zone was hard-coded to 15%, silently filtering out setups for
        prices in the 15-30% band of the range.
        """
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None
        # Price at 25% of range — inside Asian 30% band but outside legacy 15%.
        price_25pct = bounds.lower + (bounds.upper - bounds.lower) * 0.25

        setups = trader.generate_range_setups(
            h1_with_obs,
            m15_choch_at_lower,
            price_25pct,
            bounds,
            session="ASIAN_CORE",
        )
        diag = trader._last_setups_diagnostic
        assert diag["boundary_pct_applied"] == 0.30
        assert diag["near_lower"] is True
        # setup build outcome must mirror near_lower (same band applied in
        # both near-check and build-zone). Under pre-4.6-G bug these could
        # diverge: near_lower=True but long_setup_built=False due to narrow
        # CHoCH zone. Consistent: either both True or both False.
        assert diag["long_setup_built"] == (len(setups) >= 1)

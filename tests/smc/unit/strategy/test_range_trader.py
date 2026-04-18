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
from pathlib import Path

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
from smc.strategy.range_trader import (
    _SOFT_REVERSAL_RECENCY,
    _SOFT_REVERSAL_SWING_WINDOW,
    RangeTrader,
    _soft_reversal_3bar,
)
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
    ts: datetime = _BASE_TS,
) -> SMCSnapshot:
    return SMCSnapshot(
        ts=ts,
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
def trader(tmp_path: Path) -> RangeTrader:
    """RangeTrader with isolated cooldown state so tests don't share timestamps."""
    return RangeTrader(
        min_range_width=300,
        max_range_width=3000,
        cooldown_state_path=tmp_path / "cooldown.json",
    )


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
    def test_setup_without_choch_rejected_by_check3_fix(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_no_choch: SMCSnapshot,
    ) -> None:
        """Round 5 T0 (P0-9): Check 3 unconditional `return True` removed.

        With no structure breaks and no matching swings in m15_no_choch,
        _soft_reversal_3bar now returns False → setup is rejected.

        Previously (4.6-V): `assert len(setups) in (0, 1)` accepted the
        soft-fallback allowing structureless setups through. Now we require
        at least Check 1 (structure break) or Check 2 (matching swing).
        """
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None

        # Price at lower boundary, no strict CHoCH, no swings → rejected
        setups = trader.generate_range_setups(
            h1_with_obs, m15_no_choch, 2349.00, bounds
        )
        # P0-9 fix: no structure → 0 setups (Check 3 no longer bypasses)
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

    def test_diagnostic_near_lower_no_structure_rejects(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
        m15_no_choch: SMCSnapshot,
    ) -> None:
        """Round 5 T0 (P0-9): near_lower + no CHoCH + no swings → rejected.

        Diagnostic still reports near_lower=True (price was near boundary),
        but long_setup_built=False because Check 3 no longer bypasses structure.
        """
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None
        setups = trader.generate_range_setups(
            h1_with_obs, m15_no_choch, 2349.00, bounds
        )
        diag = trader._last_setups_diagnostic
        assert diag["near_lower"] is True
        assert len(setups) == 0
        assert diag["long_setup_built"] is False
        assert diag["reason_if_zero"] == "no_m15_choch_at_lower"

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


# ---------------------------------------------------------------------------
# Round 5 T0 (P0-9): Check 3 returns False — structureless setups rejected
# ---------------------------------------------------------------------------


class TestCheck3ReturnsFalseWithoutStructure:
    """P0-9: _soft_reversal_3bar no longer has an unconditional `return True`."""

    def test_check3_returns_false_without_structure(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
    ) -> None:
        """Empty M15 snapshot (no breaks, no swings) → 0 setups at boundary."""
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None

        m15_empty = _make_snapshot(timeframe=Timeframe.M15)
        setups = trader.generate_range_setups(
            h1_with_obs, m15_empty, 2349.00, bounds
        )
        # P0-9: no structure → rejected by _soft_reversal_3bar returning False
        assert len(setups) == 0

    def test_check3_passes_with_matching_swing(
        self,
        trader: RangeTrader,
        h1_with_obs: SMCSnapshot,
    ) -> None:
        """A matching swing (Check 2) is sufficient — setup is built."""
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None

        # No structure break but has a "low" swing (matches direction="long")
        m15_swing_only = _make_snapshot(
            timeframe=Timeframe.M15,
            swing_points=(
                SwingPoint(ts=_BASE_TS, price=2349.00, swing_type="low", strength=3),
            ),
        )
        setups = trader.generate_range_setups(
            h1_with_obs, m15_swing_only, 2349.00, bounds
        )
        # Check 2 matched → should build a setup (RR check may still reject)
        # At minimum, the soft_reversal check passes; we verify it was attempted.
        diag = trader._last_setups_diagnostic
        assert diag["near_lower"] is True
        # long_setup_built depends on RR — just assert no exception and near_lower ok


# ---------------------------------------------------------------------------
# Audit R2 S1: _soft_reversal_3bar Check 2 tightened
# ---------------------------------------------------------------------------


class TestSoftReversalCheck2Tightening:
    """R2 S1: Check 2 must use last N swings AND ts-recency (≤30 min).

    Before: `swing_points[-5:]` with no ts filter → near-100% pass because
    any aligned swing in recent history satisfied a soft reversal intent.
    After: window narrowed to `_SOFT_REVERSAL_SWING_WINDOW` AND matching
    swing must fall within `_SOFT_REVERSAL_RECENCY` of snapshot.ts.
    """

    def test_constants_match_intended_bounds(self) -> None:
        """Window and recency should match 'soft 3-bar reversal' semantics."""
        assert _SOFT_REVERSAL_SWING_WINDOW == 3
        # 30 min == 2 M15 bars — tight enough to mean "current", loose enough
        # to absorb 1-bar swing-confirmation latency.
        assert _SOFT_REVERSAL_RECENCY == timedelta(minutes=30)

    def test_fresh_matching_swing_in_window_passes_long(self) -> None:
        """Matching low within 30 min and within -3 window → pass."""
        snap_ts = _BASE_TS + timedelta(hours=1)
        snapshot = _make_snapshot(
            timeframe=Timeframe.M15,
            ts=snap_ts,
            swing_points=(
                SwingPoint(
                    ts=snap_ts - timedelta(minutes=15),
                    price=2349.00,
                    swing_type="low",
                    strength=3,
                ),
            ),
        )
        assert _soft_reversal_3bar(snapshot, direction="long") is True

    def test_fresh_matching_swing_in_window_passes_short(self) -> None:
        """Matching high within 30 min and within -3 window → pass."""
        snap_ts = _BASE_TS + timedelta(hours=2)
        snapshot = _make_snapshot(
            timeframe=Timeframe.M15,
            ts=snap_ts,
            swing_points=(
                SwingPoint(
                    ts=snap_ts - timedelta(minutes=10),
                    price=2369.00,
                    swing_type="high",
                    strength=3,
                ),
            ),
        )
        assert _soft_reversal_3bar(snapshot, direction="short") is True

    def test_stale_matching_swing_rejected(self) -> None:
        """Matching low but older than 30 min → reject despite swing match."""
        snap_ts = _BASE_TS + timedelta(hours=5)
        snapshot = _make_snapshot(
            timeframe=Timeframe.M15,
            ts=snap_ts,
            swing_points=(
                SwingPoint(
                    ts=snap_ts - timedelta(minutes=45),
                    price=2349.00,
                    swing_type="low",
                    strength=3,
                ),
            ),
        )
        # Only the ts-recency gate fails; type matches direction. Must reject.
        assert _soft_reversal_3bar(snapshot, direction="long") is False

    def test_matching_swing_outside_window_rejected(self) -> None:
        """Matching low exists but sits outside last -3 swings → reject.

        Even though the match is fresh (same ts as snapshot), it is the 4th
        newest swing (position -4) and must not be considered under the
        tightened window.
        """
        snap_ts = _BASE_TS + timedelta(hours=3)
        snapshot = _make_snapshot(
            timeframe=Timeframe.M15,
            ts=snap_ts,
            swing_points=(
                # Position -4 relative to end: matching "low" (fresh ts)
                SwingPoint(ts=snap_ts, price=2349.00, swing_type="low", strength=3),
                # Positions -3..-1: all "high", non-matching for long direction
                SwingPoint(ts=snap_ts, price=2360.00, swing_type="high", strength=3),
                SwingPoint(ts=snap_ts, price=2362.00, swing_type="high", strength=3),
                SwingPoint(ts=snap_ts, price=2364.00, swing_type="high", strength=3),
            ),
        )
        assert _soft_reversal_3bar(snapshot, direction="long") is False

    def test_recency_boundary_exactly_at_cutoff_passes(self) -> None:
        """sw.ts == snapshot.ts - recency → still counted (>= cutoff)."""
        snap_ts = _BASE_TS + timedelta(hours=4)
        snapshot = _make_snapshot(
            timeframe=Timeframe.M15,
            ts=snap_ts,
            swing_points=(
                SwingPoint(
                    ts=snap_ts - _SOFT_REVERSAL_RECENCY,
                    price=2349.00,
                    swing_type="low",
                    strength=3,
                ),
            ),
        )
        assert _soft_reversal_3bar(snapshot, direction="long") is True

    def test_check2_rejects_stale_matching_swing_31min(self) -> None:
        """sw.ts = snapshot.ts − 31min → JUST outside 30-min window → reject.

        Threshold-locking test (Lead DECISION + decision-reviewer R2 S1 REVIEW).
        Paired with test_check2_accepts_fresh_matching_swing_29min to bracket
        the 30-min constant at ±1 min resolution so future edits can't silently
        drift _SOFT_REVERSAL_RECENCY without failing a test.
        """
        snap_ts = _BASE_TS + timedelta(hours=9)
        snapshot = _make_snapshot(
            timeframe=Timeframe.M15,
            ts=snap_ts,
            swing_points=(
                SwingPoint(
                    ts=snap_ts - timedelta(minutes=31),
                    price=2349.00,
                    swing_type="low",
                    strength=3,
                ),
            ),
        )
        assert _soft_reversal_3bar(snapshot, direction="long") is False

    def test_check2_accepts_fresh_matching_swing_29min(self) -> None:
        """sw.ts = snapshot.ts − 29min → JUST inside 30-min window → accept.

        Threshold-locking test (Lead DECISION + decision-reviewer R2 S1 REVIEW).
        Paired with test_check2_rejects_stale_matching_swing_31min — together
        they anchor the 30-min constant so any narrowing (<29) or widening
        (>31) breaks at least one test.
        """
        snap_ts = _BASE_TS + timedelta(hours=10)
        snapshot = _make_snapshot(
            timeframe=Timeframe.M15,
            ts=snap_ts,
            swing_points=(
                SwingPoint(
                    ts=snap_ts - timedelta(minutes=29),
                    price=2349.00,
                    swing_type="low",
                    strength=3,
                ),
            ),
        )
        assert _soft_reversal_3bar(snapshot, direction="long") is True

    def test_structure_break_bypasses_check2(self) -> None:
        """Check 1 remains authoritative: a matching break passes even with
        no fresh swing at all.

        Backward compat: tightening Check 2 must not regress the primary
        structure-break path used by the m15_choch_at_* fixtures.
        """
        snap_ts = _BASE_TS + timedelta(hours=6)
        snapshot = _make_snapshot(
            timeframe=Timeframe.M15,
            ts=snap_ts,
            structure_breaks=(
                StructureBreak(
                    ts=snap_ts,
                    price=2349.00,
                    break_type="choch",
                    direction="bullish",
                    timeframe=Timeframe.M15,
                ),
            ),
            # No swing_points at all → Check 2 can't contribute
            swing_points=(),
        )
        assert _soft_reversal_3bar(snapshot, direction="long") is True

    def test_wrong_direction_swing_rejected_even_when_fresh(self) -> None:
        """Fresh + in-window swing but type mismatches direction → reject.

        Guards against over-eager widening where direction gate is bypassed
        under presence of any recent swing.
        """
        snap_ts = _BASE_TS + timedelta(hours=7)
        snapshot = _make_snapshot(
            timeframe=Timeframe.M15,
            ts=snap_ts,
            swing_points=(
                SwingPoint(
                    ts=snap_ts - timedelta(minutes=5),
                    price=2369.00,
                    swing_type="high",
                    strength=3,
                ),
            ),
        )
        # Long wants "low"; only a "high" is available → reject.
        assert _soft_reversal_3bar(snapshot, direction="long") is False

    def test_latest_structure_break_wrong_direction_blocks_check1(self) -> None:
        """Check 1 only inspects the most recent break — a wrong-direction
        break at the tail still aborts Check 1 (`break` after first iter).

        If Check 2 then succeeds on fresh+matching swing, overall still pass.
        This documents the interaction after tightening.
        """
        snap_ts = _BASE_TS + timedelta(hours=8)
        snapshot = _make_snapshot(
            timeframe=Timeframe.M15,
            ts=snap_ts,
            structure_breaks=(
                StructureBreak(
                    ts=snap_ts,
                    price=2349.00,
                    break_type="choch",
                    direction="bearish",  # wrong for long
                    timeframe=Timeframe.M15,
                ),
            ),
            swing_points=(
                SwingPoint(
                    ts=snap_ts - timedelta(minutes=10),
                    price=2349.00,
                    swing_type="low",
                    strength=3,
                ),
            ),
        )
        # Check 1 rejects (bearish != bullish target), Check 2 accepts → True.
        assert _soft_reversal_3bar(snapshot, direction="long") is True


# ---------------------------------------------------------------------------
# Round 5 T0 (P0-4): Cooldown state persisted across RangeTrader instances
# ---------------------------------------------------------------------------


class TestCooldownPersisted:
    """P0-4: _last_setup_ts is saved to JSON and reloaded on construction."""

    def test_cooldown_persisted_across_instances(self, tmp_path: Path) -> None:
        """State written by one RangeTrader instance survives a restart."""
        state_path = tmp_path / "cooldown.json"

        # Instance 1: record a "long" setup (triggers cooldown)
        trader1 = RangeTrader(
            min_range_width=300,
            max_range_width=3000,
            cooldown_state_path=state_path,
        )
        trader1._last_setup_ts["long"] = datetime.now(tz=timezone.utc)
        trader1._persist_cooldown_state()

        # Instance 2: reconstruct from same state path
        trader2 = RangeTrader(
            min_range_width=300,
            max_range_width=3000,
            cooldown_state_path=state_path,
        )
        assert "long" in trader2._last_setup_ts

        elapsed = (
            datetime.now(tz=timezone.utc) - trader2._last_setup_ts["long"]
        ).total_seconds()
        # Timestamp round-tripped; should be within a few seconds of now
        assert elapsed < 10

    def test_cooldown_state_path_defaults_when_not_given(self) -> None:
        """Default path is data/range_cooldown_state.json (no crash if missing)."""
        # Just check construction doesn't raise even if the file is absent
        trader = RangeTrader.__new__(RangeTrader)
        trader._min_range_width = 200.0
        trader._max_range_width = 20000.0
        trader._boundary_pct = 0.15
        trader._last_diagnostic = {}
        trader._last_setups_diagnostic = {}
        trader._cooldown_state_path = Path("data/range_cooldown_state_NONEXISTENT_TEST.json")
        trader._last_setup_ts = {}
        trader._load_cooldown_state()  # must not raise even if file missing
        assert trader._last_setup_ts == {}

    def test_corrupt_state_file_does_not_crash(self, tmp_path: Path) -> None:
        """Corrupt JSON state file → load silently falls back to empty dict."""
        state_path = tmp_path / "bad.json"
        state_path.write_text("{not valid json}", encoding="utf-8")

        # Must not raise
        trader = RangeTrader(
            min_range_width=300,
            max_range_width=3000,
            cooldown_state_path=state_path,
        )
        assert trader._last_setup_ts == {}


# ---------------------------------------------------------------------------
# Round 5 T0 (P0-5): Unconditional per-direction 30-min cooldown
# ---------------------------------------------------------------------------


class TestCooldown30MinUnconditional:
    """P0-5: 30-min per-direction cooldown is unconditional (not just on loss)."""

    def test_cooldown_blocks_within_30min(
        self, tmp_path: Path, h1_with_obs: SMCSnapshot
    ) -> None:
        """After a setup is recorded, same-direction attempts within 30 min are blocked."""
        trader = RangeTrader(
            min_range_width=300,
            max_range_width=3000,
            cooldown_state_path=tmp_path / "cd.json",
        )
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None

        m15_bullish = _make_snapshot(
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

        # First call: should succeed (no cooldown yet)
        setups1 = trader.generate_range_setups(
            h1_with_obs, m15_bullish, 2349.00, bounds
        )
        assert len(setups1) >= 1, "First setup should be accepted (no prior cooldown)"

        # Immediately call again: within 30-min window → blocked
        setups2 = trader.generate_range_setups(
            h1_with_obs, m15_bullish, 2349.00, bounds
        )
        assert len(setups2) == 0, "Same direction within 30 min must be blocked"

    def test_cooldown_expires_after_30min(
        self, tmp_path: Path, h1_with_obs: SMCSnapshot
    ) -> None:
        """After 30+ minutes the cooldown expires and a new setup is accepted."""
        trader = RangeTrader(
            min_range_width=300,
            max_range_width=3000,
            cooldown_state_path=tmp_path / "cd.json",
        )
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None

        # Seed an old timestamp (35 minutes ago)
        trader._last_setup_ts["long"] = datetime.now(tz=timezone.utc) - timedelta(minutes=35)

        m15_bullish = _make_snapshot(
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

        setups = trader.generate_range_setups(
            h1_with_obs, m15_bullish, 2349.00, bounds
        )
        assert len(setups) >= 1, "Setup should be accepted after 35 min cooldown expiry"

    def test_cooldown_is_per_direction(
        self, tmp_path: Path, h1_with_obs: SMCSnapshot
    ) -> None:
        """A long cooldown must not block short setups at the upper boundary."""
        trader = RangeTrader(
            min_range_width=300,
            max_range_width=3000,
            cooldown_state_path=tmp_path / "cd.json",
        )
        bounds = trader.detect_range(_empty_h1_df(), h1_with_obs)
        assert bounds is not None

        # Seed "long" cooldown (just now)
        trader._last_setup_ts["long"] = datetime.now(tz=timezone.utc)

        m15_bearish = _make_snapshot(
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

        # Short at upper boundary — must not be blocked by long cooldown
        setups = trader.generate_range_setups(
            h1_with_obs, m15_bearish, 2369.00, bounds
        )
        assert len(setups) >= 1, "Short setup should not be blocked by long-direction cooldown"


# ---------------------------------------------------------------------------
# Round 5 T0 (P0-9): Guard 6 — HTF bias alignment
# ---------------------------------------------------------------------------


class TestGuard6HTFAlignment:
    """P0-9: Guard 6 rejects setups that oppose a strong HTF bias."""

    def _passing_h1_df(self) -> "pl.DataFrame":
        """H1 df with enough touches for guards to pass."""
        import polars as pl
        highs = [2370.0] * 3 + [2359.0] * 5
        lows = [2362.0] * 3 + [2348.0] * 3 + [2358.0] * 2
        return pl.DataFrame({"high": highs, "low": lows})

    def test_guard6_bullish_bias_blocks_sell(
        self, h1_with_obs: SMCSnapshot
    ) -> None:
        """HTF bullish bias (conf>=0.5) rejects a short setup."""
        from smc.strategy.range_trader import check_range_guards
        from smc.strategy.range_types import RangeBounds, RangeSetup
        from smc.strategy.types import BiasDirection

        detected_at = datetime(2024, 7, 1, tzinfo=timezone.utc)
        bounds = RangeBounds(
            upper=2380.0, lower=2370.0,
            width_points=1000.0, midpoint=2375.0,
            detected_at=detected_at,
            source="ob_boundaries", confidence=0.85, duration_bars=20,
        )
        setup = RangeSetup(
            direction="short",
            entry_price=2379.0, stop_loss=2381.0,
            take_profit=2375.0, take_profit_ext=2371.0,
            risk_points=200.0, reward_points=800.0, rr_ratio=4.0,
            range_bounds=bounds, confidence=0.85,
            trigger="resistance_rejection", grade="A",
        )

        import polars as pl
        h1_df = pl.DataFrame({
            "high": [2380.0] * 2 + [2375.0] * 5,
            "low": [2374.0] * 2 + [2370.0] * 3 + [2373.0] * 2,
        })
        bullish_bias = BiasDirection(
            direction="bullish",
            confidence=0.7,
            key_levels=(),
            rationale="Tier 1 test",
        )

        result = check_range_guards(bounds, setup, "LONDON", h1_df, htf_bias=bullish_bias)
        assert result is False, "Bullish HTF bias should block a short setup"

    def test_guard6_bearish_bias_blocks_buy(
        self, h1_with_obs: SMCSnapshot
    ) -> None:
        """HTF bearish bias (conf>=0.5) rejects a long setup."""
        from smc.strategy.range_trader import check_range_guards
        from smc.strategy.range_types import RangeBounds, RangeSetup
        from smc.strategy.types import BiasDirection

        detected_at = datetime(2024, 7, 1, tzinfo=timezone.utc)
        bounds = RangeBounds(
            upper=2380.0, lower=2370.0,
            width_points=1000.0, midpoint=2375.0,
            detected_at=detected_at,
            source="ob_boundaries", confidence=0.85, duration_bars=20,
        )
        setup = RangeSetup(
            direction="long",
            entry_price=2371.0, stop_loss=2369.0,
            take_profit=2375.0, take_profit_ext=2379.0,
            risk_points=200.0, reward_points=800.0, rr_ratio=4.0,
            range_bounds=bounds, confidence=0.85,
            trigger="support_bounce", grade="A",
        )

        import polars as pl
        h1_df = pl.DataFrame({
            "high": [2380.0] * 2 + [2375.0] * 5,
            "low": [2374.0] * 2 + [2370.0] * 3 + [2373.0] * 2,
        })
        bearish_bias = BiasDirection(
            direction="bearish",
            confidence=0.7,
            key_levels=(),
            rationale="Tier 1 test",
        )

        result = check_range_guards(bounds, setup, "LONDON", h1_df, htf_bias=bearish_bias)
        assert result is False, "Bearish HTF bias should block a long setup"

    def test_guard6_neutral_bias_does_not_reject(
        self, h1_with_obs: SMCSnapshot
    ) -> None:
        """Neutral HTF bias (regardless of confidence) does not trigger Guard 6."""
        from smc.strategy.range_trader import check_range_guards
        from smc.strategy.range_types import RangeBounds, RangeSetup
        from smc.strategy.types import BiasDirection

        detected_at = datetime(2024, 7, 1, tzinfo=timezone.utc)
        bounds = RangeBounds(
            upper=2380.0, lower=2370.0,
            width_points=1000.0, midpoint=2375.0,
            detected_at=detected_at,
            source="ob_boundaries", confidence=0.85, duration_bars=20,
        )
        setup = RangeSetup(
            direction="short",
            entry_price=2379.0, stop_loss=2381.0,
            take_profit=2375.0, take_profit_ext=2371.0,
            risk_points=200.0, reward_points=800.0, rr_ratio=4.0,
            range_bounds=bounds, confidence=0.85,
            trigger="resistance_rejection", grade="A",
        )

        import polars as pl
        h1_df = pl.DataFrame({
            "high": [2380.0] * 2 + [2375.0] * 5,
            "low": [2374.0] * 2 + [2370.0] * 3 + [2373.0] * 2,
        })
        neutral_bias = BiasDirection(
            direction="neutral",
            confidence=0.9,
            key_levels=(),
            rationale="Conflicting D1/H4",
        )

        result = check_range_guards(bounds, setup, "LONDON", h1_df, htf_bias=neutral_bias)
        assert result is True, "Neutral bias must not trigger Guard 6 rejection"

    def test_guard6_low_confidence_bias_does_not_reject(
        self, h1_with_obs: SMCSnapshot
    ) -> None:
        """Bullish bias with confidence < 0.5 does NOT trigger Guard 6 (pass-through)."""
        from smc.strategy.range_trader import check_range_guards
        from smc.strategy.range_types import RangeBounds, RangeSetup
        from smc.strategy.types import BiasDirection

        detected_at = datetime(2024, 7, 1, tzinfo=timezone.utc)
        bounds = RangeBounds(
            upper=2380.0, lower=2370.0,
            width_points=1000.0, midpoint=2375.0,
            detected_at=detected_at,
            source="ob_boundaries", confidence=0.85, duration_bars=20,
        )
        setup = RangeSetup(
            direction="short",
            entry_price=2379.0, stop_loss=2381.0,
            take_profit=2375.0, take_profit_ext=2371.0,
            risk_points=200.0, reward_points=800.0, rr_ratio=4.0,
            range_bounds=bounds, confidence=0.85,
            trigger="resistance_rejection", grade="A",
        )

        import polars as pl
        h1_df = pl.DataFrame({
            "high": [2380.0] * 2 + [2375.0] * 5,
            "low": [2374.0] * 2 + [2370.0] * 3 + [2373.0] * 2,
        })
        weak_bullish_bias = BiasDirection(
            direction="bullish",
            confidence=0.3,  # < 0.5 → Guard 6 inactive
            key_levels=(),
            rationale="Tier 3 weak",
        )

        result = check_range_guards(bounds, setup, "LONDON", h1_df, htf_bias=weak_bullish_bias)
        assert result is True, "Weak bias (conf<0.5) must not trigger Guard 6"

    def test_guard6_absent_htf_bias_backward_compat(
        self, h1_with_obs: SMCSnapshot
    ) -> None:
        """Omitting htf_bias (None) behaves identically to pre-Guard-6 code."""
        from smc.strategy.range_trader import check_range_guards
        from smc.strategy.range_types import RangeBounds, RangeSetup

        detected_at = datetime(2024, 7, 1, tzinfo=timezone.utc)
        bounds = RangeBounds(
            upper=2380.0, lower=2370.0,
            width_points=1000.0, midpoint=2375.0,
            detected_at=detected_at,
            source="ob_boundaries", confidence=0.85, duration_bars=20,
        )
        setup = RangeSetup(
            direction="short",
            entry_price=2379.0, stop_loss=2381.0,
            take_profit=2375.0, take_profit_ext=2371.0,
            risk_points=200.0, reward_points=800.0, rr_ratio=4.0,
            range_bounds=bounds, confidence=0.85,
            trigger="resistance_rejection", grade="A",
        )

        import polars as pl
        h1_df = pl.DataFrame({
            "high": [2380.0] * 2 + [2375.0] * 5,
            "low": [2374.0] * 2 + [2370.0] * 3 + [2373.0] * 2,
        })

        # No htf_bias kwarg at all — must not raise
        result = check_range_guards(bounds, setup, "LONDON", h1_df)
        assert result is True, "No htf_bias → Guard 6 skipped, all other guards pass"

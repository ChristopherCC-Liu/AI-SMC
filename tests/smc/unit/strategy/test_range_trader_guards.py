"""Unit tests for the 5 range guards and boundary touch counter.

Covers:
- check_range_guards: each of the 5 guards individually (failure + all-pass)
- _count_boundary_touches: upper/lower/both/none, tolerance ratio effect
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from smc.strategy.range_trader import (
    _count_boundary_touches,
    check_bounds_only_guards,
    check_range_guards,
    get_last_guards_diagnostic,
)
from smc.strategy.range_types import RangeBounds, RangeSetup

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

_DETECTED_AT = datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc)

# width_points=1000 → price_width = 10.0 → upper=2380, lower=2370
_PASSING_BOUNDS = RangeBounds(
    upper=2380.0,
    lower=2370.0,
    width_points=1000.0,
    midpoint=2375.0,
    detected_at=_DETECTED_AT,
    source="ob_boundaries",
    confidence=0.85,
    duration_bars=20,
)

_PASSING_SETUP = RangeSetup(
    direction="long",
    entry_price=2371.0,
    stop_loss=2369.0,
    take_profit=2375.0,
    take_profit_ext=2379.0,
    risk_points=200.0,
    reward_points=400.0,
    rr_ratio=2.0,
    range_bounds=_PASSING_BOUNDS,
    confidence=0.85,
    trigger="support_bounce",
    grade="A",
)


def _make_touching_df(
    bounds: RangeBounds,
    n_upper: int = 2,
    n_lower: int = 0,
    n_mid: int = 5,
) -> pl.DataFrame:
    """Build H1 DataFrame with controlled boundary-touching bars."""
    highs: list[float] = []
    lows: list[float] = []

    # Bars whose high touches upper boundary exactly
    for _ in range(n_upper):
        highs.append(bounds.upper)
        lows.append(bounds.upper - 1.0)

    # Bars whose low touches lower boundary exactly
    for _ in range(n_lower):
        highs.append(bounds.lower + 1.0)
        lows.append(bounds.lower)

    # Non-touching mid-range bars
    for _ in range(n_mid):
        highs.append(bounds.midpoint + 0.5)
        lows.append(bounds.midpoint - 0.5)

    return pl.DataFrame({"high": highs, "low": lows})


# ---------------------------------------------------------------------------
# check_range_guards: all-pass baseline
# ---------------------------------------------------------------------------


class TestCheckRangeGuardsAllPass:
    def test_all_five_guards_pass(self) -> None:
        h1_df = _make_touching_df(_PASSING_BOUNDS, n_upper=2)
        assert check_range_guards(_PASSING_BOUNDS, _PASSING_SETUP, "LONDON", h1_df) is True


# ---------------------------------------------------------------------------
# Guard 1: width_points >= 800
# ---------------------------------------------------------------------------


class TestGuard1Width:
    def test_width_below_800_fails(self) -> None:
        narrow = RangeBounds(
            upper=2380.0,
            lower=2372.2,
            width_points=780.0,  # < 800
            midpoint=2376.1,
            detected_at=_DETECTED_AT,
            source="ob_boundaries",
            confidence=0.85,
            duration_bars=20,
        )
        h1_df = _make_touching_df(narrow, n_upper=2)
        setup = _PASSING_SETUP.model_copy(update={"range_bounds": narrow})
        assert check_range_guards(narrow, setup, "LONDON", h1_df) is False

    def test_width_exactly_800_passes(self) -> None:
        bounds = RangeBounds(
            upper=2380.0,
            lower=2372.0,
            width_points=800.0,
            midpoint=2376.0,
            detected_at=_DETECTED_AT,
            source="ob_boundaries",
            confidence=0.85,
            duration_bars=20,
        )
        h1_df = _make_touching_df(bounds, n_upper=2)
        setup = _PASSING_SETUP.model_copy(update={"range_bounds": bounds})
        assert check_range_guards(bounds, setup, "LONDON", h1_df) is True


# ---------------------------------------------------------------------------
# Guard 2: rr_ratio >= 1.2
# ---------------------------------------------------------------------------


class TestGuard2RR:
    def test_rr_below_1_2_fails(self) -> None:
        low_rr_setup = RangeSetup(
            direction="long",
            entry_price=2371.0,
            stop_loss=2369.0,
            take_profit=2372.0,
            take_profit_ext=2374.0,
            risk_points=200.0,
            reward_points=200.0,
            rr_ratio=1.1,  # < 1.2
            range_bounds=_PASSING_BOUNDS,
            confidence=0.85,
            trigger="support_bounce",
            grade="C",
        )
        h1_df = _make_touching_df(_PASSING_BOUNDS, n_upper=2)
        assert check_range_guards(_PASSING_BOUNDS, low_rr_setup, "LONDON", h1_df) is False

    def test_rr_exactly_1_2_passes(self) -> None:
        passing_rr_setup = RangeSetup(
            direction="long",
            entry_price=2371.0,
            stop_loss=2369.0,
            take_profit=2373.4,
            take_profit_ext=2376.0,
            risk_points=200.0,
            reward_points=240.0,
            rr_ratio=1.2,  # exactly 1.2
            range_bounds=_PASSING_BOUNDS,
            confidence=0.85,
            trigger="support_bounce",
            grade="B",
        )
        h1_df = _make_touching_df(_PASSING_BOUNDS, n_upper=2)
        assert check_range_guards(_PASSING_BOUNDS, passing_rr_setup, "LONDON", h1_df) is True


# ---------------------------------------------------------------------------
# Guard 3: >= 2 boundary touches (tolerance = width * 5%)
# ---------------------------------------------------------------------------


class TestGuard3Touches:
    def test_one_touch_fails(self) -> None:
        h1_df = _make_touching_df(_PASSING_BOUNDS, n_upper=1, n_lower=0, n_mid=5)
        assert check_range_guards(_PASSING_BOUNDS, _PASSING_SETUP, "LONDON", h1_df) is False

    def test_zero_touches_fails(self) -> None:
        h1_df = _make_touching_df(_PASSING_BOUNDS, n_upper=0, n_lower=0, n_mid=5)
        assert check_range_guards(_PASSING_BOUNDS, _PASSING_SETUP, "LONDON", h1_df) is False

    def test_two_touches_passes(self) -> None:
        h1_df = _make_touching_df(_PASSING_BOUNDS, n_upper=2, n_lower=0, n_mid=5)
        assert check_range_guards(_PASSING_BOUNDS, _PASSING_SETUP, "LONDON", h1_df) is True

    def test_mixed_upper_lower_touches_count(self) -> None:
        h1_df = _make_touching_df(_PASSING_BOUNDS, n_upper=1, n_lower=1, n_mid=5)
        assert check_range_guards(_PASSING_BOUNDS, _PASSING_SETUP, "LONDON", h1_df) is True


# ---------------------------------------------------------------------------
# Guard 4: duration_bars >= 12
# ---------------------------------------------------------------------------


class TestGuard4Duration:
    def test_duration_below_12_fails(self) -> None:
        short = RangeBounds(
            upper=2380.0,
            lower=2370.0,
            width_points=1000.0,
            midpoint=2375.0,
            detected_at=_DETECTED_AT,
            source="ob_boundaries",
            confidence=0.85,
            duration_bars=10,  # < 12
        )
        h1_df = _make_touching_df(short, n_upper=2)
        setup = _PASSING_SETUP.model_copy(update={"range_bounds": short})
        assert check_range_guards(short, setup, "LONDON", h1_df) is False

    def test_duration_exactly_12_passes(self) -> None:
        exact = RangeBounds(
            upper=2380.0,
            lower=2370.0,
            width_points=1000.0,
            midpoint=2375.0,
            detected_at=_DETECTED_AT,
            source="ob_boundaries",
            confidence=0.85,
            duration_bars=12,  # exactly 12
        )
        h1_df = _make_touching_df(exact, n_upper=2)
        setup = _PASSING_SETUP.model_copy(update={"range_bounds": exact})
        assert check_range_guards(exact, setup, "LONDON", h1_df) is True

    def test_duration_zero_default_fails(self) -> None:
        """Default duration_bars=0 (unknown) fails guard 4."""
        zero_dur = RangeBounds(
            upper=2380.0,
            lower=2370.0,
            width_points=1000.0,
            midpoint=2375.0,
            detected_at=_DETECTED_AT,
            source="ob_boundaries",
            confidence=0.85,
            # duration_bars defaults to 0
        )
        h1_df = _make_touching_df(zero_dur, n_upper=2)
        setup = _PASSING_SETUP.model_copy(update={"range_bounds": zero_dur})
        assert check_range_guards(zero_dur, setup, "LONDON", h1_df) is False


# ---------------------------------------------------------------------------
# Guard 5: lot_multiplier applied downstream — no direct block
# ---------------------------------------------------------------------------


class TestGuard5LotMultiplier:
    def test_guard5_does_not_block_when_g1_to_g4_pass(self) -> None:
        """Guard 5 is enforced downstream; all-pass guards return True."""
        h1_df = _make_touching_df(_PASSING_BOUNDS, n_upper=3)
        assert check_range_guards(_PASSING_BOUNDS, _PASSING_SETUP, "LONDON", h1_df) is True


# ---------------------------------------------------------------------------
# _count_boundary_touches: tolerance ratio effect
# ---------------------------------------------------------------------------


class TestCountBoundaryTouches:
    def test_counts_upper_touches(self) -> None:
        df = _make_touching_df(_PASSING_BOUNDS, n_upper=3, n_lower=0, n_mid=5)
        assert _count_boundary_touches(df, _PASSING_BOUNDS, tolerance_ratio=0.05) == 3

    def test_counts_lower_touches(self) -> None:
        df = _make_touching_df(_PASSING_BOUNDS, n_upper=0, n_lower=2, n_mid=5)
        assert _count_boundary_touches(df, _PASSING_BOUNDS, tolerance_ratio=0.05) == 2

    def test_counts_both_boundaries(self) -> None:
        df = _make_touching_df(_PASSING_BOUNDS, n_upper=2, n_lower=2, n_mid=5)
        assert _count_boundary_touches(df, _PASSING_BOUNDS, tolerance_ratio=0.05) == 4

    def test_zero_touches_all_mid_range(self) -> None:
        df = _make_touching_df(_PASSING_BOUNDS, n_upper=0, n_lower=0, n_mid=5)
        assert _count_boundary_touches(df, _PASSING_BOUNDS, tolerance_ratio=0.05) == 0

    def test_wider_tolerance_includes_nearby_bars(self) -> None:
        # Bar 0.3 away from upper boundary
        df = pl.DataFrame({
            "high": [_PASSING_BOUNDS.upper - 0.3],
            "low": [_PASSING_BOUNDS.midpoint],
        })
        # tolerance_ratio=0.05: tol_pts=50, tol_price=0.50 → 0.3 <= 0.5 → counts
        assert _count_boundary_touches(df, _PASSING_BOUNDS, tolerance_ratio=0.05) == 1
        # tolerance_ratio=0.01: tol_pts=10, tol_price=0.10 → 0.3 > 0.1 → doesn't count
        assert _count_boundary_touches(df, _PASSING_BOUNDS, tolerance_ratio=0.01) == 0

    def test_exact_boundary_hit_zero_tolerance(self) -> None:
        """A bar landing exactly on boundary counts even at tolerance_ratio=0."""
        df = pl.DataFrame({
            "high": [_PASSING_BOUNDS.upper],
            "low": [_PASSING_BOUNDS.lower - 5.0],  # far from lower
        })
        assert _count_boundary_touches(df, _PASSING_BOUNDS, tolerance_ratio=0.0) == 1

    def test_single_bar_touching_both_boundaries_counted_once(self) -> None:
        """One bar can touch upper (via high) and lower (via low) — still 1 bar."""
        df = pl.DataFrame({
            "high": [_PASSING_BOUNDS.upper],
            "low": [_PASSING_BOUNDS.lower],
        })
        # Counted as 1 bar (OR logic in loop)
        assert _count_boundary_touches(df, _PASSING_BOUNDS, tolerance_ratio=0.05) == 1


# ---------------------------------------------------------------------------
# Round 4.6-B: Session-aware Guard 1 (width) + Guard 4 (duration)
# ---------------------------------------------------------------------------


def _asian_bounds(width_points: float, duration_bars: int) -> RangeBounds:
    price_width = width_points * 0.01  # XAUUSD_POINT_SIZE
    lower = 4800.0
    upper = lower + price_width
    return RangeBounds(
        upper=round(upper, 2),
        lower=round(lower, 2),
        width_points=width_points,
        midpoint=round((upper + lower) / 2.0, 2),
        detected_at=_DETECTED_AT,
        source="donchian_channel",
        confidence=0.5,
        duration_bars=duration_bars,
    )


def _asian_setup(bounds: RangeBounds, rr: float = 2.0) -> RangeSetup:
    return RangeSetup(
        direction="long",
        entry_price=bounds.lower + 0.5,
        stop_loss=bounds.lower - 0.5,
        take_profit=bounds.midpoint,
        take_profit_ext=bounds.upper - 0.5,
        risk_points=100.0,
        reward_points=100.0 * rr,
        rr_ratio=rr,
        range_bounds=bounds,
        confidence=0.5,
        trigger="support_bounce",
        grade="B",
    )


class TestAsianSessionGuardProfile:
    """Round 4.6-B: ASIAN_CORE / ASIAN_LONDON_TRANSITION relax width & duration."""

    def test_asian_width_500_passes_london_fails(self) -> None:
        bounds = _asian_bounds(width_points=500.0, duration_bars=20)
        setup = _asian_setup(bounds)
        df = _make_touching_df(bounds, n_upper=2, n_lower=2, n_mid=5)
        assert check_range_guards(bounds, setup, "ASIAN_CORE", df) is True
        assert check_range_guards(bounds, setup, "LONDON", df) is False

    def test_asian_width_400_edge_passes(self) -> None:
        bounds = _asian_bounds(width_points=400.0, duration_bars=20)
        setup = _asian_setup(bounds)
        df = _make_touching_df(bounds, n_upper=2, n_lower=2, n_mid=5)
        assert check_range_guards(bounds, setup, "ASIAN_CORE", df) is True

    def test_asian_width_399_below_edge_fails(self) -> None:
        bounds = _asian_bounds(width_points=399.0, duration_bars=20)
        setup = _asian_setup(bounds)
        df = _make_touching_df(bounds, n_upper=2, n_lower=2, n_mid=5)
        assert check_range_guards(bounds, setup, "ASIAN_CORE", df) is False

    def test_asian_duration_8_edge_passes(self) -> None:
        bounds = _asian_bounds(width_points=500.0, duration_bars=8)
        setup = _asian_setup(bounds)
        df = _make_touching_df(bounds, n_upper=2, n_lower=2, n_mid=5)
        assert check_range_guards(bounds, setup, "ASIAN_CORE", df) is True

    def test_asian_duration_7_below_edge_fails(self) -> None:
        bounds = _asian_bounds(width_points=500.0, duration_bars=7)
        setup = _asian_setup(bounds)
        df = _make_touching_df(bounds, n_upper=2, n_lower=2, n_mid=5)
        assert check_range_guards(bounds, setup, "ASIAN_CORE", df) is False

    def test_asian_london_transition_also_relaxed(self) -> None:
        bounds = _asian_bounds(width_points=500.0, duration_bars=8)
        setup = _asian_setup(bounds)
        df = _make_touching_df(bounds, n_upper=2, n_lower=2, n_mid=5)
        assert (
            check_range_guards(bounds, setup, "ASIAN_LONDON_TRANSITION", df) is True
        )

    def test_unknown_session_uses_default(self) -> None:
        """Session name not in _ASIAN_SESSIONS falls back to 800/12 defaults."""
        bounds = _asian_bounds(width_points=500.0, duration_bars=20)
        setup = _asian_setup(bounds)
        df = _make_touching_df(bounds, n_upper=2, n_lower=2, n_mid=5)
        assert check_range_guards(bounds, setup, "FOO_SESSION", df) is False

    def test_asian_rr_still_enforced(self) -> None:
        """Asian profile relaxes width/duration but RR>=1.2 still required."""
        bounds = _asian_bounds(width_points=500.0, duration_bars=20)
        setup = _asian_setup(bounds, rr=1.0)
        df = _make_touching_df(bounds, n_upper=2, n_lower=2, n_mid=5)
        assert check_range_guards(bounds, setup, "ASIAN_CORE", df) is False


class TestGuardsDiagnostic:
    """Round 4.6-C2: check_range_guards exposes per-call decision trace."""

    def test_diagnostic_on_pass(self) -> None:
        df = _make_touching_df(_PASSING_BOUNDS, n_upper=2, n_lower=2, n_mid=5)
        assert check_range_guards(_PASSING_BOUNDS, _PASSING_SETUP, "LONDON", df) is True
        diag = get_last_guards_diagnostic()
        assert diag["all_passed"] is True
        assert diag["session"] == "LONDON"
        assert diag["is_asian_profile"] is False
        assert diag["min_width_required"] == 800.0
        assert diag["width_pass"] is True
        assert diag["rr_pass"] is True
        assert diag["touches_pass"] is True
        assert diag["duration_pass"] is True

    def test_diagnostic_on_width_fail(self) -> None:
        narrow = RangeBounds(
            upper=2380.0,
            lower=2378.0,
            width_points=200.0,
            midpoint=2379.0,
            detected_at=_DETECTED_AT,
            source="ob_boundaries",
            confidence=0.85,
            duration_bars=20,
        )
        df = _make_touching_df(narrow, n_upper=2, n_lower=2, n_mid=5)
        assert check_range_guards(narrow, _PASSING_SETUP, "LONDON", df) is False
        diag = get_last_guards_diagnostic()
        assert diag["all_passed"] is False
        assert diag["width_pass"] is False
        assert diag["width_points"] == 200.0

    def test_diagnostic_asian_profile_flag(self) -> None:
        bounds = _asian_bounds(width_points=500.0, duration_bars=20)
        setup = _asian_setup(bounds)
        df = _make_touching_df(bounds, n_upper=2, n_lower=2, n_mid=5)
        check_range_guards(bounds, setup, "ASIAN_CORE", df)
        diag = get_last_guards_diagnostic()
        assert diag["is_asian_profile"] is True
        assert diag["min_width_required"] == 400.0
        assert diag["min_duration_required"] == 8


class TestCheckBoundsOnlyGuards:
    """Round 4.6-E: bounds-level subset used by live_demo pre mode_router."""

    def test_asian_bounds_pass(self) -> None:
        bounds = _asian_bounds(width_points=500.0, duration_bars=20)
        df = _make_touching_df(bounds, n_upper=2, n_lower=2, n_mid=5)
        assert check_bounds_only_guards(bounds, "ASIAN_CORE", df) is True

    def test_london_width_fail(self) -> None:
        bounds = _asian_bounds(width_points=500.0, duration_bars=20)
        df = _make_touching_df(bounds, n_upper=2, n_lower=2, n_mid=5)
        # width 500 < London 800
        assert check_bounds_only_guards(bounds, "LONDON", df) is False

    def test_touches_fail(self) -> None:
        bounds = _asian_bounds(width_points=500.0, duration_bars=20)
        df = _make_touching_df(bounds, n_upper=1, n_lower=0, n_mid=5)  # only 1 touch
        assert check_bounds_only_guards(bounds, "ASIAN_CORE", df) is False

    def test_duration_fail(self) -> None:
        bounds = _asian_bounds(width_points=500.0, duration_bars=5)  # <8 Asian
        df = _make_touching_df(bounds, n_upper=2, n_lower=2, n_mid=5)
        assert check_bounds_only_guards(bounds, "ASIAN_CORE", df) is False

    def test_ignores_rr_setup_level_gate(self) -> None:
        """Unlike check_range_guards, no setup/rr required — fires on bounds alone."""
        bounds = _asian_bounds(width_points=500.0, duration_bars=20)
        df = _make_touching_df(bounds, n_upper=2, n_lower=2, n_mid=5)
        # No setup arg at all, must not crash or demand one
        assert check_bounds_only_guards(bounds, "ASIAN_CORE", df) is True

    def test_diagnostic_stage_flag(self) -> None:
        bounds = _asian_bounds(width_points=500.0, duration_bars=20)
        df = _make_touching_df(bounds, n_upper=2, n_lower=2, n_mid=5)
        check_bounds_only_guards(bounds, "ASIAN_CORE", df)
        diag = get_last_guards_diagnostic()
        assert diag["stage"] == "bounds_only"
        assert diag["rr_pass"] is None  # deferred to setup-level check

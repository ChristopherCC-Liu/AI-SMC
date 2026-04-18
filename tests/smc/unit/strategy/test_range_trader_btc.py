"""BTC-specific tests for range_trader parameterization.

Validates that injecting BTCUSD_CONFIG produces the expected cfg-driven behavior
for guards, boundary_pct, helper, and RangeTrader initialization.

All XAU backward-compat tests remain in test_range_trader.py / test_range_trader_guards.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from smc.instruments.btcusd import BTCUSD_CONFIG
from smc.instruments.xauusd import XAUUSD_CONFIG
from smc.strategy.range_trader import (
    RangeTrader,
    _min_range_width_resolved,
    check_bounds_only_guards,
    check_range_guards,
    get_last_guards_diagnostic,
)
from smc.strategy.range_types import RangeBounds, RangeSetup

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DETECTED_AT = datetime(2024, 7, 1, 12, 0, tzinfo=timezone.utc)

# BTC range: width_points=150000 means ~1500 pts at BTC price_size=0.01
# upper=90000, lower=88500 → width=1500 pts (generous, passes BTC guard_width_high=1500)
_BTC_PASSING_BOUNDS = RangeBounds(
    upper=90000.0,
    lower=88500.0,
    width_points=150000.0,  # (90000 - 88500) / 0.01 = 150000 pts
    midpoint=89250.0,
    detected_at=_DETECTED_AT,
    source="ob_boundaries",
    confidence=0.85,
    duration_bars=9,  # >= BTC guard_duration_high=8
)

_BTC_PASSING_SETUP = RangeSetup(
    direction="long",
    entry_price=88510.0,
    stop_loss=88490.0,
    take_profit=89250.0,
    take_profit_ext=89985.0,
    risk_points=2000.0,
    reward_points=3000.0,
    rr_ratio=1.5,  # >= BTC guard_rr_min=1.5
    range_bounds=_BTC_PASSING_BOUNDS,
    confidence=0.85,
    trigger="support_bounce",
    grade="A",
)


def _make_touching_df_btc(
    bounds: RangeBounds,
    n_upper: int = 2,
    n_lower: int = 0,
    n_mid: int = 5,
) -> pl.DataFrame:
    """Build H1 DataFrame with controlled boundary-touching bars for BTC."""
    highs: list[float] = []
    lows: list[float] = []
    for _ in range(n_upper):
        highs.append(bounds.upper)
        lows.append(bounds.upper - 100.0)
    for _ in range(n_lower):
        highs.append(bounds.lower + 100.0)
        lows.append(bounds.lower)
    for _ in range(n_mid):
        highs.append(bounds.midpoint + 50.0)
        lows.append(bounds.midpoint - 50.0)
    return pl.DataFrame({"high": highs, "low": lows})


# ---------------------------------------------------------------------------
# check_range_guards with BTC cfg: guard_width_low/high, duration_low/high
# ---------------------------------------------------------------------------


class TestCheckRangeGuardsBTC:
    """BTC cfg injected → guard thresholds come from BTCUSD_CONFIG."""

    def test_btc_all_guards_pass(self) -> None:
        df = _make_touching_df_btc(_BTC_PASSING_BOUNDS, n_upper=2)
        assert (
            check_range_guards(
                _BTC_PASSING_BOUNDS, _BTC_PASSING_SETUP, "HIGH_VOL", df, cfg=BTCUSD_CONFIG
            )
            is True
        )

    def test_btc_guard_width_high_threshold(self) -> None:
        """BTC guard_width_high=1500. A range with width=1400 pts must fail."""
        # width=1400 pts → price width = 1400 * 0.01 = 14.0
        narrow_btc = RangeBounds(
            upper=90000.0,
            lower=89986.0,  # 14.0 apart → 1400 pts
            width_points=1400.0,
            midpoint=89993.0,
            detected_at=_DETECTED_AT,
            source="ob_boundaries",
            confidence=0.85,
            duration_bars=9,
        )
        narrow_setup = _BTC_PASSING_SETUP.model_copy(update={"range_bounds": narrow_btc})
        df = _make_touching_df_btc(narrow_btc, n_upper=2)
        result = check_range_guards(
            narrow_btc, narrow_setup, "HIGH_VOL", df, cfg=BTCUSD_CONFIG
        )
        assert result is False, "BTC width 1400 < guard_width_high=1500 must fail"

    def test_btc_guard_width_exactly_1500_passes(self) -> None:
        """Exactly at guard_width_high=1500 pts → passes."""
        edge_btc = RangeBounds(
            upper=90000.0,
            lower=89985.0,  # 15.0 apart → 1500 pts
            width_points=1500.0,
            midpoint=89992.5,
            detected_at=_DETECTED_AT,
            source="ob_boundaries",
            confidence=0.85,
            duration_bars=9,
        )
        edge_setup = _BTC_PASSING_SETUP.model_copy(update={"range_bounds": edge_btc})
        df = _make_touching_df_btc(edge_btc, n_upper=2)
        result = check_range_guards(
            edge_btc, edge_setup, "HIGH_VOL", df, cfg=BTCUSD_CONFIG
        )
        assert result is True, "BTC width exactly 1500 must pass"

    def test_btc_guard_duration_high_8_passes(self) -> None:
        """BTC guard_duration_high=8. duration_bars=8 must pass for non-Asian session."""
        dur8_btc = _BTC_PASSING_BOUNDS.model_copy(update={"duration_bars": 8})
        dur8_setup = _BTC_PASSING_SETUP.model_copy(update={"range_bounds": dur8_btc})
        df = _make_touching_df_btc(dur8_btc, n_upper=2)
        assert (
            check_range_guards(dur8_btc, dur8_setup, "HIGH_VOL", df, cfg=BTCUSD_CONFIG) is True
        )

    def test_btc_guard_duration_7_fails(self) -> None:
        """duration_bars=7 < BTC guard_duration_high=8 must fail."""
        dur7_btc = _BTC_PASSING_BOUNDS.model_copy(update={"duration_bars": 7})
        dur7_setup = _BTC_PASSING_SETUP.model_copy(update={"range_bounds": dur7_btc})
        df = _make_touching_df_btc(dur7_btc, n_upper=2)
        assert (
            check_range_guards(dur7_btc, dur7_setup, "HIGH_VOL", df, cfg=BTCUSD_CONFIG) is False
        )

    def test_btc_guard_rr_min_1_5(self) -> None:
        """BTC guard_rr_min=1.5. rr=1.4 must fail; rr=1.5 passes."""
        low_rr_setup = _BTC_PASSING_SETUP.model_copy(update={"rr_ratio": 1.4})
        df = _make_touching_df_btc(_BTC_PASSING_BOUNDS, n_upper=2)
        assert (
            check_range_guards(
                _BTC_PASSING_BOUNDS, low_rr_setup, "HIGH_VOL", df, cfg=BTCUSD_CONFIG
            )
            is False
        ), "RR 1.4 < BTC guard_rr_min=1.5 must fail"

        exact_rr_setup = _BTC_PASSING_SETUP.model_copy(update={"rr_ratio": 1.5})
        assert (
            check_range_guards(
                _BTC_PASSING_BOUNDS, exact_rr_setup, "HIGH_VOL", df, cfg=BTCUSD_CONFIG
            )
            is True
        ), "RR exactly 1.5 must pass"

    def test_btc_asian_sessions_empty_all_sessions_use_high_thresholds(self) -> None:
        """BTC has no Asian sessions (frozenset()); all sessions use guard_width_high."""
        df = _make_touching_df_btc(_BTC_PASSING_BOUNDS, n_upper=2)
        # BTC's asian_sessions is empty, so any session name uses guard_width_high
        for session in ("HIGH_VOL", "LOW_VOL", "ASIAN_CORE"):
            result = check_range_guards(
                _BTC_PASSING_BOUNDS, _BTC_PASSING_SETUP, session, df, cfg=BTCUSD_CONFIG
            )
            diag = get_last_guards_diagnostic()
            assert diag["is_asian_profile"] is False, f"{session} should not be Asian for BTC"
            assert diag["min_width_required"] == BTCUSD_CONFIG.guard_width_high

    def test_btc_diagnostic_shows_correct_thresholds(self) -> None:
        """Diagnostic records BTC-specific width/duration thresholds."""
        df = _make_touching_df_btc(_BTC_PASSING_BOUNDS, n_upper=2)
        check_range_guards(
            _BTC_PASSING_BOUNDS, _BTC_PASSING_SETUP, "HIGH_VOL", df, cfg=BTCUSD_CONFIG
        )
        diag = get_last_guards_diagnostic()
        assert diag["min_width_required"] == BTCUSD_CONFIG.guard_width_high  # 1500.0
        assert diag["min_duration_required"] == BTCUSD_CONFIG.guard_duration_high  # 8


# ---------------------------------------------------------------------------
# check_bounds_only_guards with BTC cfg
# ---------------------------------------------------------------------------


class TestCheckBoundsOnlyGuardsBTC:
    """check_bounds_only_guards uses BTC thresholds when cfg=BTCUSD_CONFIG."""

    def test_btc_bounds_pass(self) -> None:
        df = _make_touching_df_btc(_BTC_PASSING_BOUNDS, n_upper=2)
        assert (
            check_bounds_only_guards(_BTC_PASSING_BOUNDS, "HIGH_VOL", df, cfg=BTCUSD_CONFIG)
            is True
        )

    def test_btc_width_fail(self) -> None:
        narrow_btc = _BTC_PASSING_BOUNDS.model_copy(update={"width_points": 1000.0})
        df = _make_touching_df_btc(narrow_btc, n_upper=2)
        assert (
            check_bounds_only_guards(narrow_btc, "HIGH_VOL", df, cfg=BTCUSD_CONFIG) is False
        )

    def test_btc_duration_fail(self) -> None:
        dur5_btc = _BTC_PASSING_BOUNDS.model_copy(update={"duration_bars": 5})
        df = _make_touching_df_btc(dur5_btc, n_upper=2)
        assert (
            check_bounds_only_guards(dur5_btc, "HIGH_VOL", df, cfg=BTCUSD_CONFIG) is False
        )


# ---------------------------------------------------------------------------
# boundary_pct: BTC cfg uses boundary_pct_default=0.15 / boundary_pct_wide=0.25
# ---------------------------------------------------------------------------


class TestBTCBoundaryPctDefault:
    """RangeTrader with BTC cfg uses BTCUSD_CONFIG.boundary_pct_default as fallback."""

    def test_btc_boundary_pct_default_is_0_15(self, tmp_path: Path) -> None:
        trader = RangeTrader(cfg=BTCUSD_CONFIG, cooldown_state_path=tmp_path / "cd.json")
        assert trader._boundary_pct == BTCUSD_CONFIG.boundary_pct_default  # 0.15

    def test_btc_wide_band_sessions_uses_boundary_pct_wide(self, tmp_path: Path) -> None:
        """BTC wide_band_sessions contains HIGH_VOL + LOW_VOL → uses boundary_pct_wide=0.25."""
        from smc.data.schemas import Timeframe
        from smc.smc_core.types import SMCSnapshot, StructureBreak

        trader = RangeTrader(cfg=BTCUSD_CONFIG, cooldown_state_path=tmp_path / "cd.json")

        # Build a BTC range
        bounds = _BTC_PASSING_BOUNDS

        # Empty M15 with just a bullish CHoCH
        m15_choch = SMCSnapshot(
            ts=_DETECTED_AT,
            timeframe=Timeframe.M15,
            swing_points=(),
            order_blocks=(),
            fvgs=(),
            structure_breaks=(
                StructureBreak(
                    ts=_DETECTED_AT,
                    price=bounds.lower + 10.0,
                    break_type="choch",
                    direction="bullish",
                    timeframe=Timeframe.M15,
                ),
            ),
            liquidity_levels=(),
            trend_direction="ranging",
        )
        m15_empty = SMCSnapshot(
            ts=_DETECTED_AT,
            timeframe=Timeframe.M15,
            swing_points=(),
            order_blocks=(),
            fvgs=(),
            structure_breaks=(),
            liquidity_levels=(),
            trend_direction="ranging",
        )
        h1_empty = SMCSnapshot(
            ts=_DETECTED_AT,
            timeframe=Timeframe.H1,
            swing_points=(),
            order_blocks=(),
            fvgs=(),
            structure_breaks=(),
            liquidity_levels=(),
            trend_direction="ranging",
        )

        # Price 20% into range (inside BTC 25% wide band, but NOT inside XAU 30% wide band)
        price_20pct = bounds.lower + (bounds.upper - bounds.lower) * 0.20

        # HIGH_VOL is in BTC wide_band_sessions — should use boundary_pct_wide=0.25
        trader.generate_range_setups(h1_empty, m15_choch, price_20pct, bounds, session="HIGH_VOL")
        diag = trader._last_setups_diagnostic
        assert diag["boundary_pct_applied"] == BTCUSD_CONFIG.boundary_pct_wide  # 0.25
        assert diag["near_lower"] is True, "20% < 25% wide band should trigger"

    def test_btc_unknown_session_uses_boundary_pct_default(self, tmp_path: Path) -> None:
        """Unknown session → boundary_pct_default=0.15."""
        from smc.data.schemas import Timeframe
        from smc.smc_core.types import SMCSnapshot

        trader = RangeTrader(cfg=BTCUSD_CONFIG, cooldown_state_path=tmp_path / "cd.json")
        bounds = _BTC_PASSING_BOUNDS

        m15_empty = SMCSnapshot(
            ts=_DETECTED_AT,
            timeframe=Timeframe.M15,
            swing_points=(),
            order_blocks=(),
            fvgs=(),
            structure_breaks=(),
            liquidity_levels=(),
            trend_direction="ranging",
        )
        h1_empty = SMCSnapshot(
            ts=_DETECTED_AT,
            timeframe=Timeframe.H1,
            swing_points=(),
            order_blocks=(),
            fvgs=(),
            structure_breaks=(),
            liquidity_levels=(),
            trend_direction="ranging",
        )

        # Price 20% into range — beyond 15% default, so should NOT trigger near_lower
        price_20pct = bounds.lower + (bounds.upper - bounds.lower) * 0.20
        trader.generate_range_setups(h1_empty, m15_empty, price_20pct, bounds, session="UNKNOWN")
        diag = trader._last_setups_diagnostic
        assert diag["boundary_pct_applied"] == BTCUSD_CONFIG.boundary_pct_default  # 0.15
        assert diag["near_lower"] is False, "20% > 15% default should not trigger"


# ---------------------------------------------------------------------------
# _min_range_width_resolved helper
# ---------------------------------------------------------------------------


class TestMinRangeWidthResolved:
    """Helper resolves XAU (points-based) and BTC (pct-based) paths correctly."""

    def test_xau_path_uses_points_times_point_size(self) -> None:
        """XAU: min_range_width_points=200, point_size=0.01 → returns 2.0."""
        result = _min_range_width_resolved(XAUUSD_CONFIG, current_price=2400.0)
        expected = XAUUSD_CONFIG.min_range_width_points * XAUUSD_CONFIG.point_size  # type: ignore[operator]
        assert result == pytest.approx(expected, rel=1e-9)
        # Concrete: 200 * 0.01 = 2.0
        assert result == pytest.approx(2.0, rel=1e-9)

    def test_btc_path_uses_pct_of_price(self) -> None:
        """BTC: min_range_width_pct=2.0 (meaning 2%), current_price=90000 → returns 1800.0."""
        current_price = 90000.0
        result = _min_range_width_resolved(BTCUSD_CONFIG, current_price=current_price)
        expected = (BTCUSD_CONFIG.min_range_width_pct / 100.0) * current_price  # type: ignore[operator]
        assert result == pytest.approx(expected, rel=1e-9)
        # Concrete: (2.0/100) * 90000 = 1800.0
        assert result == pytest.approx(1800.0, rel=1e-9)

    def test_btc_path_scales_with_price(self) -> None:
        """BTC pct-path scales linearly with price."""
        price_low = 50000.0
        price_high = 100000.0
        result_low = _min_range_width_resolved(BTCUSD_CONFIG, current_price=price_low)
        result_high = _min_range_width_resolved(BTCUSD_CONFIG, current_price=price_high)
        assert result_high == pytest.approx(result_low * 2.0, rel=1e-9)

    def test_xau_path_ignores_current_price(self) -> None:
        """XAU path: result independent of current_price."""
        r1 = _min_range_width_resolved(XAUUSD_CONFIG, current_price=2000.0)
        r2 = _min_range_width_resolved(XAUUSD_CONFIG, current_price=4000.0)
        assert r1 == pytest.approx(r2, rel=1e-9)


# ---------------------------------------------------------------------------
# RangeTrader initialization with BTC cfg
# ---------------------------------------------------------------------------


class TestRangeTraderBTCInit:
    """RangeTrader(cfg=BTCUSD_CONFIG) initializes correctly."""

    def test_btc_init_stores_cfg(self, tmp_path: Path) -> None:
        trader = RangeTrader(cfg=BTCUSD_CONFIG, cooldown_state_path=tmp_path / "cd.json")
        assert trader._cfg is BTCUSD_CONFIG

    def test_btc_init_min_range_width_fallback_zero(self, tmp_path: Path) -> None:
        """BTC has min_range_width_points=None → fallback to 0.0."""
        trader = RangeTrader(cfg=BTCUSD_CONFIG, cooldown_state_path=tmp_path / "cd.json")
        assert trader._min_range_width == 0.0

    def test_btc_init_max_range_width_fallback_inf(self, tmp_path: Path) -> None:
        """BTC has max_range_width_points=None → fallback to inf."""
        trader = RangeTrader(cfg=BTCUSD_CONFIG, cooldown_state_path=tmp_path / "cd.json")
        assert trader._max_range_width == float("inf")

    def test_btc_init_explicit_min_max_override_cfg(self, tmp_path: Path) -> None:
        """Explicit min_range_width / max_range_width kwargs override cfg-derived values."""
        trader = RangeTrader(
            cfg=BTCUSD_CONFIG,
            min_range_width=500.0,
            max_range_width=100000.0,
            cooldown_state_path=tmp_path / "cd.json",
        )
        assert trader._min_range_width == 500.0
        assert trader._max_range_width == 100000.0

    def test_btc_init_boundary_pct_from_cfg(self, tmp_path: Path) -> None:
        """boundary_pct defaults to cfg.boundary_pct_default when not passed."""
        trader = RangeTrader(cfg=BTCUSD_CONFIG, cooldown_state_path=tmp_path / "cd.json")
        assert trader._boundary_pct == BTCUSD_CONFIG.boundary_pct_default  # 0.15

    def test_btc_init_explicit_boundary_pct_overrides_cfg(self, tmp_path: Path) -> None:
        """Explicit boundary_pct kwarg overrides cfg."""
        trader = RangeTrader(
            cfg=BTCUSD_CONFIG,
            boundary_pct=0.20,
            cooldown_state_path=tmp_path / "cd.json",
        )
        assert trader._boundary_pct == 0.20

    def test_xau_default_init_unchanged(self, tmp_path: Path) -> None:
        """RangeTrader() (no args) → XAU config, same values as before parameterization."""
        trader = RangeTrader(cooldown_state_path=tmp_path / "cd.json")
        assert trader._cfg.symbol == "XAUUSD"
        assert trader._min_range_width == XAUUSD_CONFIG.min_range_width_points  # 200.0
        assert trader._max_range_width == XAUUSD_CONFIG.max_range_width_points  # 20000.0
        assert trader._boundary_pct == XAUUSD_CONFIG.boundary_pct_default  # 0.15

    def test_xau_legacy_kwarg_min_max_still_work(self, tmp_path: Path) -> None:
        """XAU explicit min/max kwargs still accepted (backward-compat path)."""
        trader = RangeTrader(
            min_range_width=300.0,
            max_range_width=3000.0,
            cooldown_state_path=tmp_path / "cd.json",
        )
        assert trader._min_range_width == 300.0
        assert trader._max_range_width == 3000.0
        assert trader._cfg.symbol == "XAUUSD"

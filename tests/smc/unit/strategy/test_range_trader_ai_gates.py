"""Round 9 P0-A/B/C tests — RangeTrader AI-aware gates.

Covers three gates that default OFF for production safety:

P0-A — D1 SMA50 trend filter (`range_trend_filter_enabled`):
    Block long support_bounce when D1 trend is materially down
    (slope <= -0.05%/bar AND close <= -1.0% below SMA50). Mirror for short.

P0-B — AI regime gates (`range_ai_regime_gate_enabled`):
    TREND_DOWN blocks range BUY, TREND_UP/ATH_BREAKOUT block range SELL,
    CONSOLIDATION allows both, TRANSITION raises minimum RR floor to 2.0.
    All gates require AIRegimeAssessment.confidence >= 0.6.

P0-C — Donchian validity per AI regime (`range_require_regime_valid`):
    detect_range returns None when AI sees a confident trend regime
    (TREND_UP/TREND_DOWN/ATH_BREAKOUT, conf >= 0.6).

Each gate is independently toggleable; all default behaviors must be
byte-identical to Round 8 production when flags are OFF.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from smc.ai.models import AIRegimeAssessment
from smc.ai.param_router import route
from smc.data.schemas import Timeframe
from smc.smc_core.types import OrderBlock, SMCSnapshot, StructureBreak, SwingPoint
from smc.strategy.range_trader import RangeTrader

_BASE_TS = datetime(2024, 7, 1, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_assessment(
    regime: str,
    confidence: float,
    *,
    direction: str = "neutral",
) -> AIRegimeAssessment:
    """Build a frozen AIRegimeAssessment for testing the gates.

    Tests construct minimal valid assessments — `param_preset`, `reasoning`,
    `assessed_at`, and `cost_usd` are real values (no mocking) so the type
    contract used by the gates stays representative of production.
    """
    return AIRegimeAssessment(
        regime=regime,  # type: ignore[arg-type]
        trend_direction=direction,  # type: ignore[arg-type]
        confidence=confidence,
        param_preset=route(regime),  # type: ignore[arg-type]
        reasoning="test fixture",
        assessed_at=_BASE_TS,
        source="atr_fallback",
        cost_usd=0.0,
    )


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
    return pl.DataFrame({"high": [2380.0], "low": [2340.0], "close": [2360.0]})


def _h1_df_donchian_channel(bars: int = 48) -> pl.DataFrame:
    """48-bar Donchian channel ($10 wide) — guarantees Method D detection."""
    highs = [2370.0] * bars
    lows = [2360.0] * bars
    closes = [2365.0] * bars
    return pl.DataFrame({"high": highs, "low": lows, "close": closes})


def _empty_snapshot() -> SMCSnapshot:
    return _make_snapshot(swing_points=(), order_blocks=())


def _h1_obs_snapshot() -> SMCSnapshot:
    """OB-bounded range $22 wide (2348-2370) — used for setup-builder tests."""
    return _make_snapshot(
        order_blocks=(
            OrderBlock(
                ts_start=_BASE_TS, ts_end=_BASE_TS,
                high=2370.00, low=2366.00,
                ob_type="bearish", timeframe=Timeframe.H1,
            ),
            OrderBlock(
                ts_start=_BASE_TS, ts_end=_BASE_TS,
                high=2352.00, low=2348.00,
                ob_type="bullish", timeframe=Timeframe.H1,
            ),
        ),
        swing_points=(
            SwingPoint(ts=_BASE_TS, price=2348.00, swing_type="low", strength=5),
            SwingPoint(ts=_BASE_TS, price=2370.00, swing_type="high", strength=5),
        ),
    )


def _m15_choch(direction: str, price: float) -> SMCSnapshot:
    bullish = direction == "long"
    return _make_snapshot(
        timeframe=Timeframe.M15,
        structure_breaks=(
            StructureBreak(
                ts=_BASE_TS,
                price=price,
                break_type="choch",
                direction="bullish" if bullish else "bearish",
                timeframe=Timeframe.M15,
            ),
        ),
    )


def _d1_df_with_slope(slope_pct: float, close_offset_pct: float) -> pl.DataFrame:
    """Build a D1 frame whose SMA50 metrics match target slope + close offset.

    Returns 60 D1 closes shaped so that:
      - SMA50_now (mean of last 50 closes) = 2400.0
      - SMA50_5ago (mean of closes[-55:-5]) implies the requested slope:
            slope_pct = (sma_now - sma_5ago) / sma_now * 100 / 5
      - The latest close lands at close_offset_pct above/below SMA50_now.

    Construction is piecewise-constant: closes[5:10] hold the residual
    needed to anchor SMA50_5ago, closes[10:59] hold a single value that
    averages to SMA50_now once the final close at index 59 is fixed,
    and closes[0:5] are kept identical to closes[5:10] for symmetry.
    """
    sma_now = 2400.0
    sma_5ago = sma_now / (1.0 + (slope_pct / 100.0) * 5.0)
    final_close = sma_now * (1.0 + close_offset_pct / 100.0)

    # closes[10:60] (50 bars) must average to sma_now.  Make 49 bars
    # equal to ``other_value`` and pin the final bar at ``final_close``.
    other_value = (sma_now * 50 - final_close) / 49

    # closes[5:55] (50 bars) must average to sma_5ago.  Bars [10:55] are
    # already ``other_value`` (45 of them); compute the head value for
    # bars [5:10] so the 50-bar mean lands exactly on sma_5ago.
    head_value = (sma_5ago * 50 - 45 * other_value) / 5

    closes = [head_value] * 5 + [head_value] * 5 + [other_value] * 49 + [final_close]
    return pl.DataFrame({
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
    })


# ---------------------------------------------------------------------------
# P0-A: D1 SMA50 trend filter
# ---------------------------------------------------------------------------


class TestP0ATrendFilter:
    """Round 9 P0-A: D1 SMA50 trend filter blocks counter-trend setups."""

    def _trader(self, tmp_path: Path, *, enabled: bool = True) -> RangeTrader:
        return RangeTrader(
            min_range_width=300, max_range_width=3000,
            cooldown_state_path=tmp_path / "cd.json",
            trend_filter_enabled=enabled,
        )

    def test_tuesday_2026_04_21_regression_blocks_long(
        self, tmp_path: Path
    ) -> None:
        """Regression fixture: D1 SMA50 slope -0.057%, close -2.1% below.

        Recreates the Tuesday 2026-04-21 incident — a $4779 BUY taken
        during a confirmed D1 downtrend. With the trend filter ON the
        long support_bounce should be rejected and the diagnostic key
        ``long_trend_filter_blocked`` should record slope+close metrics.
        """
        trader = self._trader(tmp_path, enabled=True)
        d1_df = _d1_df_with_slope(slope_pct=-0.057, close_offset_pct=-2.1)
        snap = _h1_obs_snapshot()
        m15 = _m15_choch("long", 2349.0)
        bounds = trader.detect_range(_empty_h1_df(), snap)
        assert bounds is not None

        setups = trader.generate_range_setups(
            snap, m15, 2349.0, bounds, session="LONDON",
            d1_df=d1_df,
        )
        assert len(setups) == 0
        diag = trader._last_setups_diagnostic
        assert "long_trend_filter_blocked" in diag
        assert "slope=-0.057" in str(diag["long_trend_filter_blocked"])

    def test_uptrend_blocks_short(self, tmp_path: Path) -> None:
        """Mirror: D1 SMA50 slope +0.07%, close +1.5% above → block short."""
        trader = self._trader(tmp_path, enabled=True)
        d1_df = _d1_df_with_slope(slope_pct=0.07, close_offset_pct=1.5)
        snap = _h1_obs_snapshot()
        m15 = _m15_choch("short", 2369.0)
        bounds = trader.detect_range(_empty_h1_df(), snap)
        assert bounds is not None

        setups = trader.generate_range_setups(
            snap, m15, 2369.0, bounds, session="LONDON",
            d1_df=d1_df,
        )
        assert len(setups) == 0
        diag = trader._last_setups_diagnostic
        assert "short_trend_filter_blocked" in diag

    def test_default_off_does_not_block(self, tmp_path: Path) -> None:
        """Gate OFF: same downtrend D1 + range BUY → setup builds normally."""
        trader = self._trader(tmp_path, enabled=False)
        d1_df = _d1_df_with_slope(slope_pct=-0.10, close_offset_pct=-3.0)
        snap = _h1_obs_snapshot()
        m15 = _m15_choch("long", 2349.0)
        bounds = trader.detect_range(_empty_h1_df(), snap)
        assert bounds is not None

        setups = trader.generate_range_setups(
            snap, m15, 2349.0, bounds, session="LONDON",
            d1_df=d1_df,
        )
        assert len(setups) >= 1
        diag = trader._last_setups_diagnostic
        assert "long_trend_filter_blocked" not in diag

    def test_d1_df_none_does_not_block(self, tmp_path: Path) -> None:
        """No D1 history → trend filter is skipped (fail-open)."""
        trader = self._trader(tmp_path, enabled=True)
        snap = _h1_obs_snapshot()
        m15 = _m15_choch("long", 2349.0)
        bounds = trader.detect_range(_empty_h1_df(), snap)
        assert bounds is not None

        setups = trader.generate_range_setups(
            snap, m15, 2349.0, bounds, session="LONDON",
            d1_df=None,
        )
        assert len(setups) >= 1

    def test_short_d1_history_does_not_block(self, tmp_path: Path) -> None:
        """< 55 D1 bars → insufficient data → no block."""
        trader = self._trader(tmp_path, enabled=True)
        d1_df = pl.DataFrame({
            "high": [2400.0] * 30,
            "low": [2380.0] * 30,
            "close": [2390.0] * 30,
        })
        snap = _h1_obs_snapshot()
        m15 = _m15_choch("long", 2349.0)
        bounds = trader.detect_range(_empty_h1_df(), snap)
        assert bounds is not None

        setups = trader.generate_range_setups(
            snap, m15, 2349.0, bounds, session="LONDON",
            d1_df=d1_df,
        )
        assert len(setups) >= 1


# ---------------------------------------------------------------------------
# P0-B: AI regime gates
# ---------------------------------------------------------------------------


class TestP0BAIRegimeGates:
    """Round 9 P0-B: per-direction regime gates from AIRegimeAssessment."""

    def _trader(self, tmp_path: Path, *, enabled: bool = True) -> RangeTrader:
        return RangeTrader(
            min_range_width=300, max_range_width=3000,
            cooldown_state_path=tmp_path / "cd.json",
            ai_regime_gate_enabled=enabled,
        )

    def test_trend_down_blocks_long(self, tmp_path: Path) -> None:
        """TREND_DOWN conf 0.78 + near_lower → near_lower forced False."""
        trader = self._trader(tmp_path, enabled=True)
        snap = _h1_obs_snapshot()
        m15 = _m15_choch("long", 2349.0)
        bounds = trader.detect_range(_empty_h1_df(), snap)
        assert bounds is not None
        ai = _make_assessment("TREND_DOWN", 0.78, direction="bearish")

        setups = trader.generate_range_setups(
            snap, m15, 2349.0, bounds, session="LONDON",
            ai_regime_assessment=ai,
        )
        assert all(s.direction != "long" for s in setups)
        diag = trader._last_setups_diagnostic
        assert diag["near_lower"] is False
        assert diag["ai_regime_gate_block"].get("long") == "TREND_DOWN"

    def test_trend_up_blocks_short(self, tmp_path: Path) -> None:
        """TREND_UP conf 0.7 + near_upper → near_upper forced False."""
        trader = self._trader(tmp_path, enabled=True)
        snap = _h1_obs_snapshot()
        m15 = _m15_choch("short", 2369.0)
        bounds = trader.detect_range(_empty_h1_df(), snap)
        assert bounds is not None
        ai = _make_assessment("TREND_UP", 0.7, direction="bullish")

        setups = trader.generate_range_setups(
            snap, m15, 2369.0, bounds, session="LONDON",
            ai_regime_assessment=ai,
        )
        assert all(s.direction != "short" for s in setups)
        diag = trader._last_setups_diagnostic
        assert diag["near_upper"] is False
        assert diag["ai_regime_gate_block"].get("short") == "TREND_UP"

    def test_ath_breakout_blocks_short(self, tmp_path: Path) -> None:
        """ATH_BREAKOUT conf 0.85 + near_upper → near_upper forced False."""
        trader = self._trader(tmp_path, enabled=True)
        snap = _h1_obs_snapshot()
        m15 = _m15_choch("short", 2369.0)
        bounds = trader.detect_range(_empty_h1_df(), snap)
        assert bounds is not None
        ai = _make_assessment("ATH_BREAKOUT", 0.85, direction="bullish")

        setups = trader.generate_range_setups(
            snap, m15, 2369.0, bounds, session="LONDON",
            ai_regime_assessment=ai,
        )
        assert all(s.direction != "short" for s in setups)
        diag = trader._last_setups_diagnostic
        assert diag["ai_regime_gate_block"].get("short") == "ATH_BREAKOUT"

    def test_consolidation_allows_both_directions(self, tmp_path: Path) -> None:
        """CONSOLIDATION conf 0.7 → no gate blocks; both setups attempted."""
        trader = self._trader(tmp_path, enabled=True)
        snap = _h1_obs_snapshot()
        # Use a price exactly at the midpoint with a wide enough boundary
        # band so both lower and upper near-zones overlap; default
        # boundary 0.15 keeps the test simple.
        m15 = _m15_choch("long", 2349.0)
        bounds = trader.detect_range(_empty_h1_df(), snap)
        assert bounds is not None
        ai = _make_assessment("CONSOLIDATION", 0.7, direction="neutral")

        # Price at lower → only long would build, but gate must not block.
        trader.generate_range_setups(
            snap, m15, 2349.0, bounds, session="LONDON",
            ai_regime_assessment=ai,
        )
        diag = trader._last_setups_diagnostic
        assert diag["near_lower"] is True
        assert diag["ai_regime_gate_block"] == {}

    def test_low_confidence_does_not_block(self, tmp_path: Path) -> None:
        """conf 0.55 < 0.6 floor → gate falls through, setup builds."""
        trader = self._trader(tmp_path, enabled=True)
        snap = _h1_obs_snapshot()
        m15 = _m15_choch("long", 2349.0)
        bounds = trader.detect_range(_empty_h1_df(), snap)
        assert bounds is not None
        ai = _make_assessment("TREND_DOWN", 0.55, direction="bearish")

        trader.generate_range_setups(
            snap, m15, 2349.0, bounds, session="LONDON",
            ai_regime_assessment=ai,
        )
        diag = trader._last_setups_diagnostic
        assert diag["near_lower"] is True
        assert diag["ai_regime_gate_block"] == {}

    def test_transition_tightens_rr_floor_blocks_low_rr(
        self, tmp_path: Path
    ) -> None:
        """TRANSITION conf 0.7 + setup with rr 1.6 → blocked.

        Geometry: $5 range (2397.5-2402.5) + entry at 2398.0 produces rr=1.6
        with default ATR-based SL buffer.  Under default RR floor 1.2 the
        setup builds; TRANSITION tightening raises the floor to 2.0 and
        the same setup is rejected.
        """
        snap = _make_snapshot(
            order_blocks=(
                OrderBlock(
                    ts_start=_BASE_TS, ts_end=_BASE_TS,
                    high=2402.50, low=2401.50,
                    ob_type="bearish", timeframe=Timeframe.H1,
                ),
                OrderBlock(
                    ts_start=_BASE_TS, ts_end=_BASE_TS,
                    high=2398.50, low=2397.50,
                    ob_type="bullish", timeframe=Timeframe.H1,
                ),
            ),
            swing_points=(
                SwingPoint(ts=_BASE_TS, price=2397.50, swing_type="low", strength=5),
                SwingPoint(ts=_BASE_TS, price=2402.50, swing_type="high", strength=5),
            ),
        )
        m15 = _m15_choch("long", 2398.0)
        ai = _make_assessment("TRANSITION", 0.7, direction="neutral")

        # Gate OFF — baseline rr=1.6 setup builds.
        trader_off = RangeTrader(
            min_range_width=300, max_range_width=3000,
            cooldown_state_path=tmp_path / "cd_off.json",
            ai_regime_gate_enabled=False,
        )
        bounds_off = trader_off.detect_range(_empty_h1_df(), snap)
        setups_off = trader_off.generate_range_setups(
            snap, m15, 2398.0, bounds_off, session="LONDON",
            ai_regime_assessment=ai,
        )
        assert len(setups_off) == 1
        assert 1.2 <= setups_off[0].rr_ratio < 2.0, (
            f"Test geometry must produce rr in [1.2, 2.0); got "
            f"{setups_off[0].rr_ratio}"
        )

        # Gate ON — TRANSITION raises floor to 2.0, blocks the same setup.
        trader_on = RangeTrader(
            min_range_width=300, max_range_width=3000,
            cooldown_state_path=tmp_path / "cd_on.json",
            ai_regime_gate_enabled=True,
        )
        bounds_on = trader_on.detect_range(_empty_h1_df(), snap)
        setups_on = trader_on.generate_range_setups(
            snap, m15, 2398.0, bounds_on, session="LONDON",
            ai_regime_assessment=ai,
        )
        assert len(setups_on) == 0

    def test_default_off_does_not_block(self, tmp_path: Path) -> None:
        """Gate OFF: TREND_DOWN + range BUY → setup builds (legacy behavior)."""
        trader = self._trader(tmp_path, enabled=False)
        snap = _h1_obs_snapshot()
        m15 = _m15_choch("long", 2349.0)
        bounds = trader.detect_range(_empty_h1_df(), snap)
        assert bounds is not None
        ai = _make_assessment("TREND_DOWN", 0.9, direction="bearish")

        setups = trader.generate_range_setups(
            snap, m15, 2349.0, bounds, session="LONDON",
            ai_regime_assessment=ai,
        )
        assert any(s.direction == "long" for s in setups)
        diag = trader._last_setups_diagnostic
        assert diag["ai_regime_gate_block"] == {}


# ---------------------------------------------------------------------------
# P0-C: Donchian validity per AI regime
# ---------------------------------------------------------------------------


class TestP0CRegimeInvalidatesRange:
    """Round 9 P0-C: AI trend regime invalidates a detected Donchian range."""

    def _trader(self, tmp_path: Path, *, enabled: bool = True) -> RangeTrader:
        return RangeTrader(
            min_range_width=300, max_range_width=3000,
            cooldown_state_path=tmp_path / "cd.json",
            require_regime_valid=enabled,
        )

    def test_trend_up_invalidates_range(self, tmp_path: Path) -> None:
        """TREND_UP conf 0.85 + valid Donchian range → detect_range returns None."""
        trader = self._trader(tmp_path, enabled=True)
        h1_df = _h1_df_donchian_channel(48)
        snap = _empty_snapshot()
        ai = _make_assessment("TREND_UP", 0.85, direction="bullish")

        bounds = trader.detect_range(h1_df, snap, ai_regime_assessment=ai)
        assert bounds is None
        diag = trader._last_diagnostic
        assert diag["range_invalidated_by_regime"] == "TREND_UP"
        assert diag["ai_regime_label"] == "TREND_UP"

    def test_consolidation_does_not_invalidate(self, tmp_path: Path) -> None:
        """CONSOLIDATION conf 0.85 → range is valid (mean-reversion regime)."""
        trader = self._trader(tmp_path, enabled=True)
        h1_df = _h1_df_donchian_channel(48)
        snap = _empty_snapshot()
        ai = _make_assessment("CONSOLIDATION", 0.85, direction="neutral")

        bounds = trader.detect_range(h1_df, snap, ai_regime_assessment=ai)
        assert bounds is not None
        diag = trader._last_diagnostic
        assert diag["range_invalidated_by_regime"] is None

    def test_ai_assessment_none_falls_through(self, tmp_path: Path) -> None:
        """No AI assessment → gate is skipped; range detection runs."""
        trader = self._trader(tmp_path, enabled=True)
        h1_df = _h1_df_donchian_channel(48)
        snap = _empty_snapshot()

        bounds = trader.detect_range(h1_df, snap, ai_regime_assessment=None)
        assert bounds is not None
        diag = trader._last_diagnostic
        assert diag["range_invalidated_by_regime"] is None
        assert diag["ai_regime_label"] is None

    def test_default_off_preserves_legacy_behavior(self, tmp_path: Path) -> None:
        """Gate OFF: TREND_UP conf 0.9 → range still detected (Round 8 behavior).

        Diagnostic still records the regime label so dashboards can show
        "would have invalidated" without changing routing.
        """
        trader = self._trader(tmp_path, enabled=False)
        h1_df = _h1_df_donchian_channel(48)
        snap = _empty_snapshot()
        ai = _make_assessment("TREND_UP", 0.9, direction="bullish")

        bounds = trader.detect_range(h1_df, snap, ai_regime_assessment=ai)
        assert bounds is not None
        diag = trader._last_diagnostic
        assert diag["range_invalidated_by_regime"] is None
        assert diag["ai_regime_label"] == "TREND_UP"
        assert diag["ai_regime_confidence"] == pytest.approx(0.9)

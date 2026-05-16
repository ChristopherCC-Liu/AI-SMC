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

    # --- Case 2: ASIAN_CORE + range + guards pass → ranging (Round 4.5 hotfix)
    # Pre-4.5 behavior: ASIAN_CORE session blocked ranging mode.
    # Round 4.5 hotfix (commit 06868f7) added ASIAN_CORE to ranging_sessions —
    # "Asian 低波反转力强" per user directive. This test was left stale.
    # Audit R2 fixup: aligning assertion with shipped behavior.
    def test_asian_core_with_range_blocked(self):
        mode = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="ranging",
            session="ASIAN_CORE",
            range_bounds=make_bounds(),
            guards_passed=True,
        )
        assert mode.mode == "ranging"

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

    # --- Case 15 (4.6-T-v3): LONDON + trending regime + price IN range → ranging ---
    # 4.6-R over-corrected (suppressed all trending regime ranging). 4.6-T-v3
    # revises: only suppress ranging when price has BROKEN OUT of range.
    # Without current_price, defaults to price_in_range=True → ranging valid
    # (mean-reversion assumption holds within range).
    def test_london_trending_regime_price_in_range_yields_ranging(self):
        mode = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="trending",
            session="LONDON",
            range_bounds=make_bounds(),
            guards_passed=True,
        )
        assert mode.mode == "ranging"


# ---------------------------------------------------------------------------
# R10 P2.1 + P2.2 wiring tests
#
# Verify the live_demo end-to-end pathway: when (a) the caller threads the
# D1 SMA50 slope through to the router and (b) the caller threads ai_direction
# through to range_trader, the new gates fire correctly. These mirror the
# wiring inside scripts/live_demo.determine_action without importing the
# script (which has top-level MetaTrader5 import).
# ---------------------------------------------------------------------------


class TestR10P2WiringThroughLiveDemo:
    """End-to-end wiring: caller-provided kwargs propagate through router + RangeTrader."""

    def test_router_consumes_d1_slope_to_demote_ranging(self):
        """When caller supplies d1_sma50_slope_pct + ai_regime_assessment,
        the router demotes a would-be-ranging cycle to trending under the
        trending_dominance flag — proving the wiring lands.
        """
        from datetime import datetime, timezone

        from smc.ai.models import AIRegimeAssessment
        from smc.ai.param_router import route

        assessment = AIRegimeAssessment(
            regime="TREND_UP",
            trend_direction="bullish",
            confidence=0.65,
            param_preset=route("TREND_UP"),
            reasoning="wiring fixture",
            assessed_at=datetime(2026, 4, 26, 12, 0, tzinfo=timezone.utc),
            source="ai_debate",
            cost_usd=0.0,
        )
        mode = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.30,
            regime="ranging",
            session="LONDON",
            range_bounds=make_bounds(),
            guards_passed=True,
            d1_sma50_slope_pct=0.06,
            trending_dominance_enabled=True,
            ai_regime_assessment=assessment,
            ai_regime_trust_threshold=0.7,
        )
        # P0 trust gate is 0.7 (above 0.65) → P0 cannot fire; the only path
        # to trending here is the dominance demote, proving wiring ⇒ effect.
        assert mode.mode == "trending"
        assert mode.ai_regime_decision == "trending_dominance_over_ranging"

    def test_range_trader_consumes_ai_direction_to_veto_setup(self, tmp_path):
        """When caller supplies ai_direction + confidence, RangeTrader vetoes
        opposing setups under the entry-gate flag.
        """
        from smc.smc_core.types import (
            OrderBlock,
            SMCSnapshot,
            StructureBreak,
            SwingPoint,
        )
        from smc.data.schemas import Timeframe
        from smc.strategy.range_trader import RangeTrader
        import polars as pl

        BASE_TS = datetime(2026, 4, 26, 0, 0, tzinfo=timezone.utc)

        def _snap(structure_breaks=(), order_blocks=(), swing_points=(), tf=Timeframe.H1):
            return SMCSnapshot(
                ts=BASE_TS,
                timeframe=tf,
                swing_points=swing_points,
                order_blocks=order_blocks,
                fvgs=(),
                structure_breaks=structure_breaks,
                liquidity_levels=(),
                trend_direction="ranging",
            )

        h1_snap = _snap(
            order_blocks=(
                OrderBlock(
                    ts_start=BASE_TS, ts_end=BASE_TS,
                    high=2370.00, low=2366.00,
                    ob_type="bearish", timeframe=Timeframe.H1,
                ),
                OrderBlock(
                    ts_start=BASE_TS, ts_end=BASE_TS,
                    high=2352.00, low=2348.00,
                    ob_type="bullish", timeframe=Timeframe.H1,
                ),
            ),
            swing_points=(
                SwingPoint(ts=BASE_TS, price=2348.00, swing_type="low", strength=5),
                SwingPoint(ts=BASE_TS, price=2370.00, swing_type="high", strength=5),
            ),
        )
        m15_short_choch = _snap(
            structure_breaks=(
                StructureBreak(
                    ts=BASE_TS, price=2369.0,
                    break_type="choch", direction="bearish",
                    timeframe=Timeframe.M15,
                ),
            ),
            tf=Timeframe.M15,
        )

        trader = RangeTrader(
            min_range_width=300, max_range_width=3000,
            cooldown_state_path=tmp_path / "cd.json",
            ai_direction_entry_gate_enabled=True,
        )
        h1_df = pl.DataFrame({"high": [2380.0], "low": [2340.0], "close": [2360.0]})
        bounds = trader.detect_range(h1_df, h1_snap)
        assert bounds is not None

        setups = trader.generate_range_setups(
            h1_snap, m15_short_choch,
            current_price=2369.5, bounds=bounds, h1_atr=2.0,
            ai_direction="bullish",
            ai_direction_confidence=0.70,
        )
        # bullish AI must veto the short setup; verify diagnostic key present.
        assert all(s.direction != "short" for s in setups)
        assert "short_ai_direction_blocked" in trader._last_setups_diagnostic

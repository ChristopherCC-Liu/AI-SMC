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
# Audit R3 S3 — session-adaptive ai_confidence threshold
# ---------------------------------------------------------------------------


def _make_cfg_with_thresholds(thresholds):
    """Build a lightweight InstrumentConfig with only the fields route_trading_mode reads.

    Keeps tests independent of the production XAU/BTC configs so changes to
    those don't cause spurious S3 test failures.
    """
    from smc.instruments.types import InstrumentConfig
    return InstrumentConfig(
        symbol="TEST",
        mt5_path="TEST",
        magic=0,
        point_size=0.01,
        contract_size=100.0,
        leverage_ratio=100,
        min_lot=0.01,
        pip_value_per_lot=10.0,
        donchian_lookback=48,
        min_range_width_points=200.0,
        min_range_width_pct=None,
        max_range_width_points=20000.0,
        max_range_width_pct=None,
        boundary_pct_default=0.15,
        boundary_pct_wide=0.30,
        guard_width_low=400.0,
        guard_width_high=800.0,
        guard_duration_low=8,
        guard_duration_high=12,
        guard_rr_min=1.2,
        regime_trending_pct=1.4,
        regime_ranging_pct=1.0,
        sl_atr_multiplier=0.75,
        sl_min_buffer_points=200.0,
        sl_min_buffer_pct=None,
        tp1_rr_ratio=2.5,
        tp2_rr_ratio=4.0,
        sessions={
            "LONDON": (8, 13),
            "ASIAN_LONDON_TRANSITION": (6, 8),
            "LONDON/NY OVERLAP": (13, 16),
            "LATE NY": (21, 24),
            "NEW YORK": (16, 21),
        },
        ranging_sessions=frozenset({"LONDON", "LONDON/NY OVERLAP", "NEW YORK"}),
        asian_sessions=frozenset({"ASIAN_LONDON_TRANSITION"}),
        asian_core_session_name=None,
        wide_band_sessions=frozenset(),
        weekend_flag_active=False,
        use_asian_quota=False,
        consec_loss_limit=3,
        mode_router_thresholds=thresholds,
    )


class TestS3SessionAdaptiveThresholds:
    """Audit R3 S3: ai_confidence cutoff resolved per-session via
    InstrumentConfig.mode_router_thresholds, with 0.45 fallback.
    """

    def test_no_thresholds_uses_default_045(self):
        """cfg.mode_router_thresholds=None → all sessions use 0.45."""
        cfg = _make_cfg_with_thresholds(None)
        mode = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.46,
            regime="trending",
            session="LONDON",
            range_bounds=None,
            guards_passed=False,
            cfg=cfg,
        )
        assert mode.mode == "trending"

    def test_session_not_in_dict_uses_default_045(self):
        """Session key missing → fallback to 0.45."""
        cfg = _make_cfg_with_thresholds({"LONDON": 0.45})
        mode = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.46,
            regime="trending",
            session="UNKNOWN_SESSION",
            range_bounds=None,
            guards_passed=False,
            cfg=cfg,
        )
        assert mode.mode == "trending"

    def test_london_at_045_passes(self):
        """LONDON=0.45 boundary — >= triggers trending."""
        cfg = _make_cfg_with_thresholds({"LONDON": 0.45})
        mode = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.45,
            regime="trending",
            session="LONDON",
            range_bounds=None,
            guards_passed=False,
            cfg=cfg,
        )
        assert mode.mode == "trending"

    def test_lon_ny_overlap_at_041_passes_wider(self):
        """LON/NY OVERLAP=0.40 wider threshold — 0.41 passes
        where previous global 0.45 would have rejected it.
        """
        cfg = _make_cfg_with_thresholds({
            "LONDON/NY OVERLAP": 0.40,
        })
        mode = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.41,
            regime="trending",
            session="LONDON/NY OVERLAP",
            range_bounds=None,
            guards_passed=False,
            cfg=cfg,
        )
        assert mode.mode == "trending"

    def test_alt_at_049_rejects_tighter(self):
        """ASIAN_LONDON_TRANSITION=0.50 tighter — 0.49 rejected
        where previous global 0.45 would have passed.
        """
        cfg = _make_cfg_with_thresholds({
            "ASIAN_LONDON_TRANSITION": 0.50,
        })
        mode = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.49,
            regime="trending",
            session="ASIAN_LONDON_TRANSITION",
            range_bounds=None,
            guards_passed=False,
            cfg=cfg,
        )
        assert mode.mode == "v1_passthrough"

    def test_alt_at_050_passes(self):
        """ASIAN_LONDON_TRANSITION=0.50 boundary — >= triggers trending."""
        cfg = _make_cfg_with_thresholds({
            "ASIAN_LONDON_TRANSITION": 0.50,
        })
        mode = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.50,
            regime="trending",
            session="ASIAN_LONDON_TRANSITION",
            range_bounds=None,
            guards_passed=False,
            cfg=cfg,
        )
        assert mode.mode == "trending"

    def test_late_ny_049_rejects(self):
        """LATE NY=0.50 tighter — mirror ALT rejection path."""
        cfg = _make_cfg_with_thresholds({
            "LATE NY": 0.50,
        })
        mode = route_trading_mode(
            ai_direction="bearish",
            ai_confidence=0.49,
            regime="trending",
            session="LATE NY",
            range_bounds=None,
            guards_passed=False,
            cfg=cfg,
        )
        assert mode.mode == "v1_passthrough"

    def test_xau_prod_cfg_has_expected_thresholds(self):
        """Production XAU config ships with S3 per-session thresholds.

        Prevents regression where config drift silently removes the dict.
        """
        from smc.instruments import get_instrument_config
        cfg = get_instrument_config("XAUUSD")
        assert cfg.mode_router_thresholds is not None
        assert cfg.mode_router_thresholds["LONDON"] == 0.45
        assert cfg.mode_router_thresholds["LONDON/NY OVERLAP"] == 0.40
        assert cfg.mode_router_thresholds["ASIAN_LONDON_TRANSITION"] == 0.50
        assert cfg.mode_router_thresholds["LATE NY"] == 0.50
        # ASIAN_CORE intentionally omitted (hard-blocked upstream)
        assert "ASIAN_CORE" not in cfg.mode_router_thresholds

    def test_btc_prod_cfg_has_expected_thresholds(self):
        """Production BTC config ships with HIGH_VOL/LOW_VOL thresholds."""
        from smc.instruments import get_instrument_config
        cfg = get_instrument_config("BTCUSD")
        assert cfg.mode_router_thresholds is not None
        assert cfg.mode_router_thresholds["HIGH_VOL"] == 0.45
        assert cfg.mode_router_thresholds["LOW_VOL"] == 0.50

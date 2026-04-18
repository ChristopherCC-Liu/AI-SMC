"""Unit tests for InstrumentConfig registry and XAU/BTC backward-compat.

Session hour boundaries used here match scripts/live_demo.py:get_session_info()
(authoritative source).  See xauusd.py module docstring for the discrepancy
with the original prompt specification.
"""
import pytest
from dataclasses import FrozenInstanceError
import smc.instruments  # triggers registry population
from smc.instruments import get_instrument_config


class TestRegistry:
    def test_xauusd_registered(self):
        cfg = get_instrument_config("XAUUSD")
        assert cfg.symbol == "XAUUSD"
        assert cfg.magic == 19760418

    def test_btcusd_registered(self):
        cfg = get_instrument_config("BTCUSD")
        assert cfg.symbol == "BTCUSD"
        assert cfg.magic == 19760419

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown symbol"):
            get_instrument_config("BOGUS")


class TestFrozen:
    def test_cannot_mutate(self):
        cfg = get_instrument_config("XAUUSD")
        with pytest.raises(FrozenInstanceError):
            cfg.magic = 999  # type: ignore


class TestXAUUSDBackwardCompat:
    """XAU values must match the currently-live system byte-for-byte."""

    def test_point_size(self):
        assert get_instrument_config("XAUUSD").point_size == 0.01

    def test_donchian_lookback(self):
        assert get_instrument_config("XAUUSD").donchian_lookback == 48

    def test_guards_width_thresholds(self):
        cfg = get_instrument_config("XAUUSD")
        assert cfg.guard_width_low == 400.0
        assert cfg.guard_width_high == 800.0

    def test_guard_duration_thresholds(self):
        cfg = get_instrument_config("XAUUSD")
        assert cfg.guard_duration_low == 8
        assert cfg.guard_duration_high == 12

    def test_guard_rr_min(self):
        assert get_instrument_config("XAUUSD").guard_rr_min == 1.2

    def test_regime_thresholds(self):
        cfg = get_instrument_config("XAUUSD")
        assert cfg.regime_trending_pct == 1.4
        assert cfg.regime_ranging_pct == 1.0

    def test_sl(self):
        cfg = get_instrument_config("XAUUSD")
        assert cfg.sl_atr_multiplier == 0.75
        assert cfg.sl_min_buffer_points == 200.0

    def test_sessions_complete(self):
        cfg = get_instrument_config("XAUUSD")
        assert "ASIAN_CORE" in cfg.sessions
        assert "LONDON" in cfg.sessions

    def test_asian_sessions(self):
        cfg = get_instrument_config("XAUUSD")
        assert cfg.asian_sessions == frozenset({"ASIAN_CORE", "ASIAN_LONDON_TRANSITION"})


class TestBTCUSDFields:
    def test_mt5_path(self):
        cfg = get_instrument_config("BTCUSD")
        assert cfg.mt5_path == "Bitcoin\\BTCUSD"

    def test_regime_soft(self):
        assert get_instrument_config("BTCUSD").regime_trending_pct == 5.0

    def test_asian_sessions_empty(self):
        assert get_instrument_config("BTCUSD").asian_sessions == frozenset()

    def test_use_asian_quota_false(self):
        assert get_instrument_config("BTCUSD").use_asian_quota is False

    def test_weekend_flag(self):
        assert get_instrument_config("BTCUSD").weekend_flag_active is True

    def test_pct_based_width(self):
        cfg = get_instrument_config("BTCUSD")
        assert cfg.min_range_width_pct == 2.0
        assert cfg.min_range_width_points is None


class TestPipValuePerLot:
    """audit-r2 R2: pip_value_per_lot required field — prevents BTC-via-XAU default."""

    def test_xauusd_pip_value(self):
        cfg = get_instrument_config("XAUUSD")
        assert cfg.pip_value_per_lot == 10.0

    def test_btcusd_pip_value(self):
        cfg = get_instrument_config("BTCUSD")
        assert cfg.pip_value_per_lot == 0.1

    def test_xau_btc_ratio_is_100x(self):
        xau = get_instrument_config("XAUUSD").pip_value_per_lot
        btc = get_instrument_config("BTCUSD").pip_value_per_lot
        assert xau / btc == pytest.approx(100.0)

    def test_pip_value_is_required_no_default(self):
        """Constructing InstrumentConfig without pip_value_per_lot must raise."""
        from smc.instruments.types import InstrumentConfig
        with pytest.raises(TypeError):
            InstrumentConfig(  # type: ignore[call-arg]
                symbol="X",
                mt5_path="X",
                magic=1,
                point_size=0.01,
                contract_size=1.0,
                leverage_ratio=100,
                min_lot=0.01,
                # pip_value_per_lot intentionally omitted
                donchian_lookback=24,
                min_range_width_points=None,
                min_range_width_pct=1.0,
                max_range_width_points=None,
                max_range_width_pct=10.0,
                boundary_pct_default=0.15,
                boundary_pct_wide=0.25,
                guard_width_low=100.0,
                guard_width_high=200.0,
                guard_duration_low=6,
                guard_duration_high=8,
                guard_rr_min=1.5,
                regime_trending_pct=5.0,
                regime_ranging_pct=2.0,
                sl_atr_multiplier=1.0,
                sl_min_buffer_points=None,
                sl_min_buffer_pct=0.3,
                tp1_rr_ratio=2.5,
                tp2_rr_ratio=4.0,
                sessions={},
                ranging_sessions=frozenset(),
                asian_sessions=frozenset(),
                asian_core_session_name=None,
                wide_band_sessions=frozenset(),
                weekend_flag_active=False,
                use_asian_quota=False,
                consec_loss_limit=3,
            )

    def test_pip_value_immutable(self):
        cfg = get_instrument_config("XAUUSD")
        with pytest.raises(FrozenInstanceError):
            cfg.pip_value_per_lot = 999.0  # type: ignore

    def test_btc_risk_budget_verification(self):
        """Sanity: 1% risk on $10k, 300 pt SL, BTC pip_value=0.1 → ~16.67 lots raw."""
        from smc.risk.position_sizer import compute_position_size
        cfg = get_instrument_config("BTCUSD")
        ps = compute_position_size(
            balance_usd=10_000.0,
            risk_pct=1.0,
            sl_distance_points=300.0,
            pip_value_per_lot=cfg.pip_value_per_lot,
            max_lot_size=100.0,
        )
        # risk_usd=100, point_value=0.01, raw_lots = 100 / (300*0.01) = 33.33
        assert ps.lots == pytest.approx(33.33, abs=0.02)

    def test_xau_risk_budget_verification(self):
        """Sanity: 1% risk on $10k, 300 pt SL, XAU pip_value=10.0 → ~0.33 lots raw."""
        from smc.risk.position_sizer import compute_position_size
        cfg = get_instrument_config("XAUUSD")
        ps = compute_position_size(
            balance_usd=10_000.0,
            risk_pct=1.0,
            sl_distance_points=300.0,
            pip_value_per_lot=cfg.pip_value_per_lot,
            max_lot_size=1.0,
        )
        # risk_usd=100, point_value=1.0, raw_lots = 100 / 300 = 0.333
        assert ps.lots == pytest.approx(0.33, abs=0.01)

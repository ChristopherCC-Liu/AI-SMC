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

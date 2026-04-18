"""Unit tests for smc.strategy.session — get_session_info parameterized."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from smc.instruments import get_instrument_config
from smc.strategy.session import get_session_info


def _utc(hour: int) -> datetime:
    return datetime(2024, 6, 10, hour, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def xau_cfg():
    return get_instrument_config("XAUUSD")


@pytest.fixture()
def btc_cfg():
    return get_instrument_config("BTCUSD")


# ---------------------------------------------------------------------------
# XAU sessions
# ---------------------------------------------------------------------------


class TestXauSessions:
    def test_hour3_asian_core(self, xau_cfg) -> None:
        name, penalty = get_session_info(_utc(3), cfg=xau_cfg)
        assert name == "ASIAN_CORE"
        assert penalty == pytest.approx(0.2)

    def test_hour10_london(self, xau_cfg) -> None:
        name, penalty = get_session_info(_utc(10), cfg=xau_cfg)
        assert name == "LONDON"
        assert penalty == pytest.approx(0.0)

    def test_hour17_new_york(self, xau_cfg) -> None:
        name, penalty = get_session_info(_utc(17), cfg=xau_cfg)
        assert name == "NEW YORK"
        assert penalty == pytest.approx(0.0)

    def test_hour0_asian_core_boundary(self, xau_cfg) -> None:
        name, penalty = get_session_info(_utc(0), cfg=xau_cfg)
        assert name == "ASIAN_CORE"
        assert penalty == pytest.approx(0.2)

    def test_hour6_asian_london_transition(self, xau_cfg) -> None:
        name, penalty = get_session_info(_utc(6), cfg=xau_cfg)
        assert name == "ASIAN_LONDON_TRANSITION"
        assert penalty == pytest.approx(0.15)

    def test_hour21_late_ny(self, xau_cfg) -> None:
        name, penalty = get_session_info(_utc(21), cfg=xau_cfg)
        assert name == "LATE NY"
        assert penalty == pytest.approx(0.1)

    def test_hour13_overlap(self, xau_cfg) -> None:
        name, penalty = get_session_info(_utc(13), cfg=xau_cfg)
        assert name == "LONDON/NY OVERLAP"
        assert penalty == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BTC sessions
# ---------------------------------------------------------------------------


class TestBtcSessions:
    def test_hour15_high_vol(self, btc_cfg) -> None:
        name, penalty = get_session_info(_utc(15), cfg=btc_cfg)
        assert name == "HIGH_VOL"
        assert penalty == pytest.approx(0.0)

    def test_hour2_low_vol_midnight_wrap(self, btc_cfg) -> None:
        name, penalty = get_session_info(_utc(2), cfg=btc_cfg)
        assert name == "LOW_VOL"
        assert penalty == pytest.approx(0.0)

    def test_hour22_low_vol_start(self, btc_cfg) -> None:
        name, penalty = get_session_info(_utc(22), cfg=btc_cfg)
        assert name == "LOW_VOL"
        assert penalty == pytest.approx(0.0)

    def test_hour12_high_vol_start(self, btc_cfg) -> None:
        name, penalty = get_session_info(_utc(12), cfg=btc_cfg)
        assert name == "HIGH_VOL"
        assert penalty == pytest.approx(0.0)

    def test_hour11_low_vol_before_high(self, btc_cfg) -> None:
        name, penalty = get_session_info(_utc(11), cfg=btc_cfg)
        assert name == "LOW_VOL"
        assert penalty == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Default cfg (no arg) → XAU behaviour
# ---------------------------------------------------------------------------


class TestDefaultCfg:
    def test_no_cfg_defaults_to_xau(self) -> None:
        """Without cfg kwarg the function defaults to XAUUSD."""
        name, penalty = get_session_info(_utc(3))
        assert name == "ASIAN_CORE"
        assert penalty == pytest.approx(0.2)

    def test_no_args_does_not_raise(self) -> None:
        """Calling with no arguments at all should not raise."""
        name, penalty = get_session_info()
        assert isinstance(name, str)
        assert isinstance(penalty, float)

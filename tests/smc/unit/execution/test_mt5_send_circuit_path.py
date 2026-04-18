"""Tests that circuit_flag_path kwarg overrides the module-level default,
and that legacy callers omitting it still use CIRCUIT_OPEN_PATH as the fallback.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from smc.execution.mt5_send import (
    CIRCUIT_OPEN_PATH,
    TRADE_RETCODE_DONE,
    TRADE_RETCODE_REQUOTE,
    reset_circuit_breaker,
    send_with_retry,
)


def _make_mt5(*, order_send_return=None, order_send_side_effect=None) -> Mock:
    mt5 = Mock()
    mt5.ORDER_TYPE_BUY = 0
    mt5.symbol_info_tick.return_value = None
    if order_send_side_effect is not None:
        mt5.order_send.side_effect = order_send_side_effect
    else:
        r = Mock()
        r.retcode = order_send_return or TRADE_RETCODE_DONE
        r.order = 99999
        r.comment = ""
        mt5.order_send.return_value = r
    return mt5


def _request() -> dict:
    return {"symbol": "XAUUSD", "type": 0, "price": 2300.0, "volume": 0.01}


@pytest.mark.unit
class TestCircuitFlagPathOverride:
    def test_explicit_path_overrides_default(self, tmp_path: Path) -> None:
        """Passing circuit_flag_path uses that path, not CIRCUIT_OPEN_PATH."""
        custom_flag = tmp_path / "custom_circuit.flag"
        mt5 = _make_mt5(
            order_send_side_effect=[
                _make_requote_result(),
                _make_requote_result(),
                _make_requote_result(),
            ]
        )

        send_with_retry(mt5, _request(), backoff_ms=(0, 0, 0), circuit_flag_path=custom_flag)

        assert custom_flag.exists(), "Custom flag path must be written on circuit open"
        # Module-level default must NOT be written
        assert not CIRCUIT_OPEN_PATH.exists(), "Default CIRCUIT_OPEN_PATH must not be written"

    def test_none_falls_back_to_module_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """circuit_flag_path=None must fall back to CIRCUIT_OPEN_PATH (legacy compat)."""
        # Redirect CIRCUIT_OPEN_PATH to a temp location so we don't litter the repo
        fake_default = tmp_path / "default_circuit.flag"
        import smc.execution.mt5_send as mod
        monkeypatch.setattr(mod, "CIRCUIT_OPEN_PATH", fake_default)

        mt5 = _make_mt5(
            order_send_side_effect=[
                _make_requote_result(),
                _make_requote_result(),
                _make_requote_result(),
            ]
        )

        send_with_retry(mt5, _request(), backoff_ms=(0, 0, 0), circuit_flag_path=None)

        assert fake_default.exists(), "None should fall back to (monkeypatched) CIRCUIT_OPEN_PATH"

    def test_open_circuit_via_explicit_path_blocks_order(self, tmp_path: Path) -> None:
        """A pre-existing flag at the explicit path short-circuits order_send."""
        custom_flag = tmp_path / "already_open.flag"
        custom_flag.write_text("pre-opened")
        mt5 = _make_mt5()

        res = send_with_retry(mt5, _request(), circuit_flag_path=custom_flag)

        assert res.success is False
        assert res.attempts == 0
        mt5.order_send.assert_not_called()

    def test_open_circuit_at_default_path_blocked_when_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Legacy call (no circuit_flag_path arg) checks the module-level default."""
        fake_default = tmp_path / "default_open.flag"
        fake_default.write_text("legacy open")
        import smc.execution.mt5_send as mod
        monkeypatch.setattr(mod, "CIRCUIT_OPEN_PATH", fake_default)

        mt5 = _make_mt5()
        res = send_with_retry(mt5, _request())

        assert res.success is False
        assert res.attempts == 0
        mt5.order_send.assert_not_called()

    def test_reset_circuit_breaker_custom_path(self, tmp_path: Path) -> None:
        """reset_circuit_breaker respects a custom path."""
        custom_flag = tmp_path / "cb.flag"
        custom_flag.write_text("opened")

        deleted = reset_circuit_breaker(custom_flag)

        assert deleted is True
        assert not custom_flag.exists()

    def test_successful_order_does_not_write_any_flag(self, tmp_path: Path) -> None:
        """A successful order must not create the circuit flag."""
        custom_flag = tmp_path / "should_not_exist.flag"
        mt5 = _make_mt5()
        result_obj = Mock()
        result_obj.retcode = TRADE_RETCODE_DONE
        result_obj.order = 11111
        result_obj.comment = ""
        mt5.order_send.return_value = result_obj

        res = send_with_retry(mt5, _request(), circuit_flag_path=custom_flag)

        assert res.success is True
        assert not custom_flag.exists()


def _make_requote_result() -> Mock:
    r = Mock()
    r.retcode = TRADE_RETCODE_REQUOTE
    r.order = None
    r.comment = "requote"
    return r

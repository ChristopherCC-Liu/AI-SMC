"""Unit tests for smc.execution.mt5_send — retry, deviation, circuit breaker."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from smc.execution.mt5_send import (
    TRADE_RETCODE_DONE,
    TRADE_RETCODE_MARKET_CLOSED,
    TRADE_RETCODE_REQUOTE,
    compute_dynamic_deviation,
    refresh_price,
    reset_circuit_breaker,
    send_with_retry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mt5(*, order_send_side_effect=None, order_send_return=None) -> Mock:
    """Build a minimal MT5Protocol mock."""
    mt5 = Mock()
    mt5.ORDER_TYPE_BUY = 0
    if order_send_side_effect is not None:
        mt5.order_send.side_effect = order_send_side_effect
    elif order_send_return is not None:
        mt5.order_send.return_value = order_send_return
    return mt5


def _result(retcode: int, ticket: int = 12345, comment: str = "") -> Mock:
    r = Mock()
    r.retcode = retcode
    r.order = ticket
    r.comment = comment
    return r


def _request() -> dict:
    return {"symbol": "XAUUSD", "type": 0, "price": 2300.0, "volume": 0.01}


# ---------------------------------------------------------------------------
# send_with_retry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSendWithRetrySuccess:
    def test_first_attempt_done(self, tmp_path: Path) -> None:
        mt5 = _make_mt5(order_send_return=_result(TRADE_RETCODE_DONE))
        mt5.symbol_info_tick.return_value = None

        res = send_with_retry(mt5, _request(), circuit_flag_path=tmp_path / "cb.flag")

        assert res.success is True
        assert res.retcode == TRADE_RETCODE_DONE
        assert res.ticket == 12345
        assert res.attempts == 1
        assert mt5.order_send.call_count == 1

    def test_second_attempt_done_after_requote(self, tmp_path: Path) -> None:
        mt5 = _make_mt5(
            order_send_side_effect=[
                _result(TRADE_RETCODE_REQUOTE),
                _result(TRADE_RETCODE_DONE),
            ]
        )
        mt5.symbol_info_tick.return_value = None

        res = send_with_retry(
            mt5, _request(), backoff_ms=(0, 0, 0), circuit_flag_path=tmp_path / "cb.flag"
        )

        assert res.success is True
        assert res.attempts == 2
        assert mt5.order_send.call_count == 2


@pytest.mark.unit
class TestSendWithRetryFailures:
    def test_three_requotes_opens_circuit(self, tmp_path: Path) -> None:
        flag = tmp_path / "cb.flag"
        mt5 = _make_mt5(
            order_send_side_effect=[
                _result(TRADE_RETCODE_REQUOTE),
                _result(TRADE_RETCODE_REQUOTE),
                _result(TRADE_RETCODE_REQUOTE),
            ]
        )
        mt5.symbol_info_tick.return_value = None

        res = send_with_retry(mt5, _request(), backoff_ms=(0, 0, 0), circuit_flag_path=flag)

        assert res.success is False
        assert res.attempts == 3
        assert flag.exists()

    def test_market_closed_no_retry(self, tmp_path: Path) -> None:
        flag = tmp_path / "cb.flag"
        mt5 = _make_mt5(order_send_return=_result(TRADE_RETCODE_MARKET_CLOSED))
        mt5.symbol_info_tick.return_value = None

        res = send_with_retry(mt5, _request(), circuit_flag_path=flag)

        assert res.success is False
        assert res.retcode == TRADE_RETCODE_MARKET_CLOSED
        assert res.attempts == 1
        assert mt5.order_send.call_count == 1
        assert not flag.exists()

    def test_three_exceptions_opens_circuit(self, tmp_path: Path) -> None:
        flag = tmp_path / "cb.flag"
        mt5 = _make_mt5(order_send_side_effect=RuntimeError("connection lost"))
        mt5.symbol_info_tick.return_value = None

        res = send_with_retry(mt5, _request(), backoff_ms=(0, 0, 0), circuit_flag_path=flag)

        assert res.success is False
        assert res.attempts == 3
        assert flag.exists()

    def test_circuit_already_open_skips_order_send(self, tmp_path: Path) -> None:
        flag = tmp_path / "cb.flag"
        flag.write_text("pre-existing")
        mt5 = _make_mt5()
        mt5.symbol_info_tick.return_value = None

        res = send_with_retry(mt5, _request(), circuit_flag_path=flag)

        assert res.success is False
        assert res.attempts == 0
        mt5.order_send.assert_not_called()


# ---------------------------------------------------------------------------
# send_with_retry — price refresh behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSendWithRetryPriceRefresh:
    def test_tick_price_injected_into_request(self, tmp_path: Path) -> None:
        """On each attempt the latest tick price should replace the stale request price."""
        mt5 = _make_mt5(order_send_return=_result(TRADE_RETCODE_DONE))
        tick = Mock()
        tick.ask = 2350.50
        mt5.symbol_info_tick.return_value = tick

        res = send_with_retry(mt5, _request(), circuit_flag_path=tmp_path / "cb.flag")

        assert res.success is True
        assert res.final_price == pytest.approx(2350.50)
        sent_request = mt5.order_send.call_args[0][0]
        assert sent_request["price"] == pytest.approx(2350.50)


# ---------------------------------------------------------------------------
# compute_dynamic_deviation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeDynamicDeviation:
    def test_normal_spread_multiplied_by_three(self) -> None:
        mt5 = Mock()
        info = Mock()
        info.spread = 10
        mt5.symbol_info.return_value = info

        dev = compute_dynamic_deviation(mt5, "XAUUSD")

        assert dev == 30

    def test_zero_spread_returns_fallback(self) -> None:
        mt5 = Mock()
        info = Mock()
        info.spread = 0
        mt5.symbol_info.return_value = info

        dev = compute_dynamic_deviation(mt5, "XAUUSD", fallback=75)

        assert dev == 75

    def test_none_info_returns_fallback(self) -> None:
        mt5 = Mock()
        mt5.symbol_info.return_value = None

        dev = compute_dynamic_deviation(mt5, "XAUUSD", fallback=50)

        assert dev == 50

    def test_large_spread_capped_at_500(self) -> None:
        mt5 = Mock()
        info = Mock()
        info.spread = 1000
        mt5.symbol_info.return_value = info

        dev = compute_dynamic_deviation(mt5, "XAUUSD")

        assert dev == 500

    def test_small_spread_floored_at_20(self) -> None:
        mt5 = Mock()
        info = Mock()
        info.spread = 3  # 3 * 3 = 9 < 20
        mt5.symbol_info.return_value = info

        dev = compute_dynamic_deviation(mt5, "XAUUSD")

        assert dev == 20

    def test_exception_returns_fallback(self) -> None:
        mt5 = Mock()
        mt5.symbol_info.side_effect = OSError("rpc error")

        dev = compute_dynamic_deviation(mt5, "XAUUSD", fallback=100)

        assert dev == 100


# ---------------------------------------------------------------------------
# refresh_price
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRefreshPrice:
    def test_tick_none_returns_none(self) -> None:
        mt5 = Mock()
        mt5.symbol_info_tick.return_value = None

        assert refresh_price(mt5, "XAUUSD", is_buy=True) is None

    def test_ask_zero_returns_none(self) -> None:
        mt5 = Mock()
        tick = Mock()
        tick.ask = 0.0
        mt5.symbol_info_tick.return_value = tick

        assert refresh_price(mt5, "XAUUSD", is_buy=True) is None

    def test_bid_zero_returns_none(self) -> None:
        mt5 = Mock()
        tick = Mock()
        tick.bid = 0.0
        mt5.symbol_info_tick.return_value = tick

        assert refresh_price(mt5, "XAUUSD", is_buy=False) is None

    def test_returns_ask_for_buy(self) -> None:
        mt5 = Mock()
        tick = Mock()
        tick.ask = 2300.50
        mt5.symbol_info_tick.return_value = tick

        price = refresh_price(mt5, "XAUUSD", is_buy=True)

        assert price == pytest.approx(2300.50)

    def test_returns_bid_for_sell(self) -> None:
        mt5 = Mock()
        tick = Mock()
        tick.bid = 2299.80
        mt5.symbol_info_tick.return_value = tick

        price = refresh_price(mt5, "XAUUSD", is_buy=False)

        assert price == pytest.approx(2299.80)

    def test_exception_returns_none(self) -> None:
        mt5 = Mock()
        mt5.symbol_info_tick.side_effect = ConnectionError("timeout")

        assert refresh_price(mt5, "XAUUSD", is_buy=True) is None


# ---------------------------------------------------------------------------
# reset_circuit_breaker
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResetCircuitBreaker:
    def test_existing_flag_deleted_returns_true(self, tmp_path: Path) -> None:
        flag = tmp_path / "cb.flag"
        flag.write_text("opened")

        result = reset_circuit_breaker(flag)

        assert result is True
        assert not flag.exists()

    def test_no_flag_returns_false(self, tmp_path: Path) -> None:
        flag = tmp_path / "cb.flag"

        result = reset_circuit_breaker(flag)

        assert result is False

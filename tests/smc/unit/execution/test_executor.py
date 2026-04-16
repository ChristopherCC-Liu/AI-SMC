"""Tests for smc.execution.executor — BrokerPort implementations."""

from __future__ import annotations

import pytest

from smc.execution.executor import SimBrokerPort
from smc.execution.types import OrderRequest


@pytest.mark.unit
class TestSimBrokerPortSendOrder:
    """SimBrokerPort.send_order fills instantly and tracks positions."""

    def test_market_order_long_creates_position(self) -> None:
        broker = SimBrokerPort(initial_balance=10_000.0)
        request = OrderRequest(
            direction="long", lots=0.05,
            entry_price=2350.0, stop_loss=2340.0, take_profit_1=2370.0,
        )
        result = broker.send_order(request)

        assert result.success is True
        assert result.ticket > 0
        assert result.fill_price > 0

        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].ticket == result.ticket
        assert positions[0].direction == "long"
        assert positions[0].lots == 0.05

    def test_market_order_short_creates_position(self) -> None:
        broker = SimBrokerPort(initial_balance=10_000.0)
        request = OrderRequest(
            direction="short", lots=0.03,
            entry_price=2380.0, stop_loss=2390.0, take_profit_1=2360.0,
        )
        result = broker.send_order(request)

        assert result.success is True
        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].direction == "short"

    def test_multiple_orders_tracked(self) -> None:
        broker = SimBrokerPort()
        for _ in range(3):
            broker.send_order(OrderRequest(
                direction="long", lots=0.01,
                entry_price=2350.0, stop_loss=2340.0, take_profit_1=2370.0,
            ))
        positions = broker.get_positions()
        assert len(positions) == 3
        tickets = {p.ticket for p in positions}
        assert len(tickets) == 3  # all unique


@pytest.mark.unit
class TestSimBrokerPortModify:
    """SimBrokerPort.modify_order updates SL/TP immutably."""

    def test_modify_sl(self) -> None:
        broker = SimBrokerPort()
        result = broker.send_order(OrderRequest(
            direction="long", lots=0.01,
            entry_price=2350.0, stop_loss=2340.0, take_profit_1=2370.0,
        ))
        ticket = result.ticket

        mod_result = broker.modify_order(ticket, sl=2345.0)
        assert mod_result.success is True

        positions = broker.get_positions()
        assert positions[0].sl == 2345.0

    def test_modify_tp(self) -> None:
        broker = SimBrokerPort()
        result = broker.send_order(OrderRequest(
            direction="long", lots=0.01,
            entry_price=2350.0, stop_loss=2340.0, take_profit_1=2370.0,
        ))
        mod_result = broker.modify_order(result.ticket, tp=2380.0)
        assert mod_result.success is True
        assert broker.get_positions()[0].tp == 2380.0

    def test_modify_nonexistent_ticket(self) -> None:
        broker = SimBrokerPort()
        result = broker.modify_order(999999, sl=2345.0)
        assert result.success is False
        assert "not found" in result.error_message.lower()


@pytest.mark.unit
class TestSimBrokerPortClosePosition:
    """SimBrokerPort.close_position handles full and partial closes."""

    def test_full_close(self) -> None:
        broker = SimBrokerPort(initial_balance=10_000.0)
        result = broker.send_order(OrderRequest(
            direction="long", lots=0.05,
            entry_price=2350.0, stop_loss=2340.0, take_profit_1=2370.0,
        ))
        close_result = broker.close_position(result.ticket)
        assert close_result.success is True
        assert len(broker.get_positions()) == 0

    def test_partial_close_reduces_lots(self) -> None:
        broker = SimBrokerPort(initial_balance=10_000.0)
        result = broker.send_order(OrderRequest(
            direction="long", lots=0.10,
            entry_price=2350.0, stop_loss=2340.0, take_profit_1=2370.0,
        ))

        close_result = broker.close_position(result.ticket, lots=0.05)
        assert close_result.success is True

        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].lots == 0.05

    def test_close_nonexistent_ticket(self) -> None:
        broker = SimBrokerPort()
        result = broker.close_position(999999)
        assert result.success is False


@pytest.mark.unit
class TestSimBrokerPortAccountInfo:
    """SimBrokerPort.get_account_info reflects balance and positions."""

    def test_initial_balance(self) -> None:
        broker = SimBrokerPort(initial_balance=10_000.0)
        info = broker.get_account_info()
        assert info.balance == 10_000.0
        assert info.equity == 10_000.0
        assert info.margin_used == 0.0
        # No positions: margin_level should be large (not 0), matching MT5 behavior
        assert info.margin_level == 9999.0

    def test_balance_after_position(self) -> None:
        broker = SimBrokerPort(initial_balance=10_000.0)
        broker.send_order(OrderRequest(
            direction="long", lots=0.05,
            entry_price=2350.0, stop_loss=2340.0, take_profit_1=2370.0,
        ))
        info = broker.get_account_info()
        assert info.margin_used > 0
        assert info.margin_free < info.equity


@pytest.mark.unit
class TestSimBrokerPortUpdatePrice:
    """SimBrokerPort.update_price recalculates PnL."""

    def test_long_profit(self) -> None:
        broker = SimBrokerPort(initial_balance=10_000.0, spread_points=0.0)
        result = broker.send_order(OrderRequest(
            direction="long", lots=0.01,
            entry_price=2350.0, stop_loss=2340.0, take_profit_1=2370.0,
        ))
        # Move price up $5 -> 0.01 lot * 100 oz * $5 = $5.00 profit
        broker.update_price(result.ticket, 2355.0)
        positions = broker.get_positions()
        assert positions[0].pnl_usd == pytest.approx(5.0, abs=0.1)

    def test_short_profit(self) -> None:
        broker = SimBrokerPort(initial_balance=10_000.0, spread_points=0.0)
        result = broker.send_order(OrderRequest(
            direction="short", lots=0.01,
            entry_price=2380.0, stop_loss=2390.0, take_profit_1=2360.0,
        ))
        # Move price down $10 -> 0.01 lot * 100 oz * $10 = $10.00 profit
        broker.update_price(result.ticket, 2370.0)
        positions = broker.get_positions()
        assert positions[0].pnl_usd == pytest.approx(10.0, abs=0.1)

    def test_long_loss(self) -> None:
        broker = SimBrokerPort(initial_balance=10_000.0, spread_points=0.0)
        result = broker.send_order(OrderRequest(
            direction="long", lots=0.01,
            entry_price=2350.0, stop_loss=2340.0, take_profit_1=2370.0,
        ))
        broker.update_price(result.ticket, 2340.0)
        positions = broker.get_positions()
        assert positions[0].pnl_usd == pytest.approx(-10.0, abs=0.1)

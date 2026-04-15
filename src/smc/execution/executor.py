"""Broker execution adapters for the AI-SMC trading system.

Defines :class:`BrokerPort` — a Protocol that abstracts the broker API — and
two concrete implementations:

1. :class:`MT5BrokerPort` — real MetaTrader 5 execution (Windows only).
2. :class:`SimBrokerPort` — in-memory simulated execution for testing and
   macOS development.

The rest of the execution layer depends only on :class:`BrokerPort`, so
swapping between real and simulated execution is a one-line change.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Protocol

from smc.execution.types import AccountInfo, OrderRequest, OrderResult, PositionState

__all__ = [
    "BrokerPort",
    "MT5BrokerPort",
    "SimBrokerPort",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BrokerPort Protocol
# ---------------------------------------------------------------------------


class BrokerPort(Protocol):
    """Abstract broker interface — same API for real MT5 and simulation."""

    def send_order(self, request: OrderRequest) -> OrderResult: ...

    def modify_order(
        self,
        ticket: int,
        *,
        sl: float | None = None,
        tp: float | None = None,
    ) -> OrderResult: ...

    def close_position(self, ticket: int, lots: float | None = None) -> OrderResult: ...

    def get_positions(self) -> tuple[PositionState, ...]: ...

    def get_account_info(self) -> AccountInfo: ...


# ---------------------------------------------------------------------------
# MT5BrokerPort (real MT5 — Windows only)
# ---------------------------------------------------------------------------


class MT5BrokerPort:
    """Real MT5 execution via the MetaTrader5 Python package.

    This adapter wraps the ``MetaTrader5`` module functions (``order_send``,
    ``positions_get``, ``account_info``) into the :class:`BrokerPort` interface.

    The MetaTrader5 package is only available on Windows. On macOS/Linux this
    class raises a clear error at construction time.
    """

    # MT5 order type constants
    _ORDER_BUY: int = 0
    _ORDER_SELL: int = 1
    _TRADE_ACTION_DEAL: int = 1  # Market execution
    _TRADE_ACTION_SLTP: int = 6  # Modify SL/TP
    _TRADE_RETCODE_DONE: int = 10009

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        *,
        path: str | None = None,
    ) -> None:
        try:
            import MetaTrader5 as mt5  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "MetaTrader5 package is not available. "
                "MT5BrokerPort requires Windows with MT5 terminal installed. "
                "Use SimBrokerPort for macOS/Linux development."
            ) from exc

        self._mt5 = mt5

        if not mt5.initialize(path=path, login=login, password=password, server=server):
            code, msg = mt5.last_error()
            raise RuntimeError(f"MT5 initialization failed: [{code}] {msg}")

        logger.info("MT5BrokerPort initialized — login=%d server=%s", login, server)

    def send_order(self, request: OrderRequest) -> OrderResult:
        """Send a market or limit order to MT5."""
        mt5 = self._mt5
        order_type = self._ORDER_BUY if request.direction == "long" else self._ORDER_SELL

        mt5_request: dict = {
            "action": self._TRADE_ACTION_DEAL,
            "symbol": request.instrument,
            "volume": request.lots,
            "type": order_type,
            "price": request.entry_price if request.entry_price > 0 else self._get_price(request),
            "sl": request.stop_loss,
            "tp": request.take_profit_1,
            "deviation": 20,
            "magic": 202500,
            "comment": "ai-smc",
            "type_time": 0,  # GTC
            "type_filling": 1,  # IOC
        }

        result = mt5.order_send(mt5_request)
        if result is None:
            code, msg = mt5.last_error()
            return OrderResult(success=False, error_message=f"order_send returned None: [{code}] {msg}")

        if result.retcode != self._TRADE_RETCODE_DONE:
            return OrderResult(
                success=False,
                error_message=f"MT5 retcode {result.retcode}: {result.comment}",
            )

        logger.info(
            "Order filled: ticket=%d price=%.2f lots=%.2f",
            result.order,
            result.price,
            result.volume,
        )
        return OrderResult(
            success=True,
            ticket=result.order,
            fill_price=result.price,
        )

    def modify_order(
        self,
        ticket: int,
        *,
        sl: float | None = None,
        tp: float | None = None,
    ) -> OrderResult:
        """Modify the SL and/or TP of an existing position."""
        mt5 = self._mt5
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return OrderResult(success=False, error_message=f"Position {ticket} not found")

        pos = positions[0]
        mt5_request: dict = {
            "action": self._TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": sl if sl is not None else pos.sl,
            "tp": tp if tp is not None else pos.tp,
        }

        result = mt5.order_send(mt5_request)
        if result is None or result.retcode != self._TRADE_RETCODE_DONE:
            msg = "modify failed" if result is None else f"retcode {result.retcode}"
            return OrderResult(success=False, ticket=ticket, error_message=msg)

        return OrderResult(success=True, ticket=ticket)

    def close_position(self, ticket: int, lots: float | None = None) -> OrderResult:
        """Close a position (fully or partially)."""
        mt5 = self._mt5
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return OrderResult(success=False, error_message=f"Position {ticket} not found")

        pos = positions[0]
        close_lots = lots if lots is not None else pos.volume
        # Close by sending opposite order
        close_type = self._ORDER_SELL if pos.type == self._ORDER_BUY else self._ORDER_BUY

        tick = mt5.symbol_info_tick(pos.symbol)
        price = tick.ask if close_type == self._ORDER_BUY else tick.bid

        mt5_request: dict = {
            "action": self._TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": close_lots,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 202500,
            "comment": "ai-smc-close",
            "type_time": 0,
            "type_filling": 1,
        }

        result = mt5.order_send(mt5_request)
        if result is None or result.retcode != self._TRADE_RETCODE_DONE:
            msg = "close failed" if result is None else f"retcode {result.retcode}"
            return OrderResult(success=False, ticket=ticket, error_message=msg)

        logger.info("Position %d closed: lots=%.2f price=%.2f", ticket, close_lots, result.price)
        return OrderResult(success=True, ticket=ticket, fill_price=result.price)

    def get_positions(self) -> tuple[PositionState, ...]:
        """Fetch all open positions from MT5."""
        mt5 = self._mt5
        positions = mt5.positions_get()
        if positions is None:
            return ()

        result: list[PositionState] = []
        for pos in positions:
            tick = mt5.symbol_info_tick(pos.symbol)
            current_price = tick.bid if tick else 0.0
            result.append(
                PositionState(
                    ticket=pos.ticket,
                    instrument=pos.symbol,
                    direction="long" if pos.type == self._ORDER_BUY else "short",
                    lots=pos.volume,
                    open_price=pos.price_open,
                    current_price=current_price,
                    sl=pos.sl,
                    tp=pos.tp,
                    pnl_usd=pos.profit,
                    open_time=datetime.fromtimestamp(pos.time, tz=timezone.utc),
                )
            )
        return tuple(result)

    def get_account_info(self) -> AccountInfo:
        """Fetch account info from MT5."""
        info = self._mt5.account_info()
        if info is None:
            raise RuntimeError("Failed to get MT5 account info")
        return AccountInfo(
            balance=info.balance,
            equity=info.equity,
            margin_used=info.margin,
            margin_free=info.margin_free,
            margin_level=info.margin_level or 0.0,
        )

    def _get_price(self, request: OrderRequest) -> float:
        """Get current ask/bid for the instrument."""
        tick = self._mt5.symbol_info_tick(request.instrument)
        if tick is None:
            raise RuntimeError(f"Cannot get tick for {request.instrument}")
        return tick.ask if request.direction == "long" else tick.bid


# ---------------------------------------------------------------------------
# SimBrokerPort (simulated execution)
# ---------------------------------------------------------------------------


class SimBrokerPort:
    """Simulated broker for testing and macOS development.

    All fills are instant at the requested price plus a configurable spread.
    Positions are tracked in memory. No external dependencies.

    Args:
        initial_balance: Starting account balance in USD.
        spread_points: Simulated spread in price points (e.g. 0.30 = 30 cents).
        leverage: Simulated leverage ratio for margin calculations.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        spread_points: float = 0.30,
        leverage: float = 50.0,
    ) -> None:
        self._balance = initial_balance
        self._equity = initial_balance
        self._spread = spread_points
        self._leverage = leverage
        self._positions: dict[int, PositionState] = {}
        self._next_ticket: int = 100_001
        self._margin_per_lot: float = 2_000.0  # approximate for XAUUSD at 1:50

    def send_order(self, request: OrderRequest) -> OrderResult:
        """Simulate an instant fill at the requested price (+ half spread)."""
        ticket = self._next_ticket
        self._next_ticket += 1

        # Simulate slippage: fill at entry_price + half spread for longs, - for shorts
        half_spread = self._spread / 2.0
        if request.entry_price > 0:
            base_price = request.entry_price
        else:
            # Market order: use a reasonable default (mid-price = 2350 for testing)
            base_price = 2350.0

        fill_price = (
            base_price + half_spread
            if request.direction == "long"
            else base_price - half_spread
        )

        now = datetime.now(tz=timezone.utc)
        position = PositionState(
            ticket=ticket,
            instrument=request.instrument,
            direction=request.direction,
            lots=request.lots,
            open_price=fill_price,
            current_price=fill_price,
            sl=request.stop_loss,
            tp=request.take_profit_1,
            pnl_usd=0.0,
            open_time=now,
        )

        self._positions = {**self._positions, ticket: position}
        self._update_margin()

        logger.info(
            "SimBroker: filled ticket=%d dir=%s lots=%.2f @ %.2f",
            ticket, request.direction, request.lots, fill_price,
        )

        return OrderResult(success=True, ticket=ticket, fill_price=fill_price)

    def modify_order(
        self,
        ticket: int,
        *,
        sl: float | None = None,
        tp: float | None = None,
    ) -> OrderResult:
        """Modify SL/TP on a simulated position."""
        if ticket not in self._positions:
            return OrderResult(success=False, ticket=ticket, error_message=f"Ticket {ticket} not found")

        old = self._positions[ticket]
        updated = PositionState(
            ticket=old.ticket,
            instrument=old.instrument,
            direction=old.direction,
            lots=old.lots,
            open_price=old.open_price,
            current_price=old.current_price,
            sl=sl if sl is not None else old.sl,
            tp=tp if tp is not None else old.tp,
            pnl_usd=old.pnl_usd,
            open_time=old.open_time,
        )
        self._positions = {**self._positions, ticket: updated}
        return OrderResult(success=True, ticket=ticket)

    def close_position(self, ticket: int, lots: float | None = None) -> OrderResult:
        """Close a simulated position (fully or partially)."""
        if ticket not in self._positions:
            return OrderResult(success=False, ticket=ticket, error_message=f"Ticket {ticket} not found")

        pos = self._positions[ticket]
        close_lots = lots if lots is not None else pos.lots

        if close_lots >= pos.lots:
            # Full close
            self._balance += pos.pnl_usd
            new_positions = {k: v for k, v in self._positions.items() if k != ticket}
            self._positions = new_positions
        else:
            # Partial close — reduce lots, realize proportional PnL
            pnl_portion = pos.pnl_usd * (close_lots / pos.lots)
            self._balance += pnl_portion
            remaining = PositionState(
                ticket=pos.ticket,
                instrument=pos.instrument,
                direction=pos.direction,
                lots=round(pos.lots - close_lots, 2),
                open_price=pos.open_price,
                current_price=pos.current_price,
                sl=pos.sl,
                tp=pos.tp,
                pnl_usd=pos.pnl_usd - pnl_portion,
                open_time=pos.open_time,
            )
            self._positions = {**self._positions, ticket: remaining}

        self._update_margin()

        logger.info("SimBroker: closed ticket=%d lots=%.2f", ticket, close_lots)
        return OrderResult(success=True, ticket=ticket, fill_price=pos.current_price)

    def get_positions(self) -> tuple[PositionState, ...]:
        """Return all open simulated positions."""
        return tuple(self._positions.values())

    def get_account_info(self) -> AccountInfo:
        """Return simulated account snapshot."""
        total_pnl = sum(p.pnl_usd for p in self._positions.values())
        equity = self._balance + total_pnl
        margin_used = sum(p.lots * self._margin_per_lot for p in self._positions.values())
        margin_free = equity - margin_used
        margin_level = (equity / margin_used * 100.0) if margin_used > 0 else 0.0

        return AccountInfo(
            balance=round(self._balance, 2),
            equity=round(equity, 2),
            margin_used=round(margin_used, 2),
            margin_free=round(margin_free, 2),
            margin_level=round(margin_level, 2),
        )

    def update_price(self, ticket: int, current_price: float) -> None:
        """Update the current market price for a position (testing helper).

        Recalculates PnL based on the new price.
        """
        if ticket not in self._positions:
            return

        pos = self._positions[ticket]
        # XAUUSD: 1 standard lot = 100 oz, so $1 move = $100 per lot
        point_value = 100.0  # USD per $1 price move per lot
        price_diff = current_price - pos.open_price
        if pos.direction == "short":
            price_diff = -price_diff

        pnl = price_diff * pos.lots * point_value

        updated = PositionState(
            ticket=pos.ticket,
            instrument=pos.instrument,
            direction=pos.direction,
            lots=pos.lots,
            open_price=pos.open_price,
            current_price=current_price,
            sl=pos.sl,
            tp=pos.tp,
            pnl_usd=round(pnl, 2),
            open_time=pos.open_time,
        )
        self._positions = {**self._positions, ticket: updated}

    def _update_margin(self) -> None:
        """Recalculate equity after position changes."""
        total_pnl = sum(p.pnl_usd for p in self._positions.values())
        self._equity = self._balance + total_pnl

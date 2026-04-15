"""Order lifecycle orchestration for the AI-SMC trading system.

:class:`OrderManager` is the single entry point for all trade execution. It
coordinates risk checking, position sizing, order submission, and position
management (trailing stop-loss, partial take-profit, emergency close).

The manager is deliberately **stateful** — it tracks open positions locally
and provides :meth:`get_local_positions` for reconciliation against the
broker. All position mutations produce new :class:`PositionState` objects
(immutability at the data level).

Usage::

    from smc.execution.executor import SimBrokerPort
    from smc.execution.order_manager import OrderManager
    from smc.risk import DrawdownGuard, compute_position_size

    broker = SimBrokerPort(initial_balance=10_000.0)
    guard = DrawdownGuard(max_daily_loss_pct=3.0, max_drawdown_pct=10.0)
    manager = OrderManager(broker=broker, risk_guard=guard, position_sizer_config={})
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from smc.execution.types import AccountInfo, OrderRequest, OrderResult, PositionState
from smc.risk.position_sizer import compute_position_size

if TYPE_CHECKING:
    from smc.execution.executor import BrokerPort
    from smc.risk.drawdown_guard import DrawdownGuard
    from smc.strategy.types import TradeSetup

__all__ = ["OrderManager"]

logger = logging.getLogger(__name__)


class OrderManager:
    """Orchestrates the full order lifecycle: risk check, size, send, manage.

    Parameters
    ----------
    broker:
        Any :class:`BrokerPort` implementation (real MT5 or simulator).
    risk_guard:
        :class:`DrawdownGuard` for circuit-breaker checks.
    position_sizer_config:
        Keyword arguments forwarded to :func:`compute_position_size`
        (e.g. ``max_lot_size``, ``pip_value_per_lot``).
    trailing_sl_points:
        When price moves this many points in profit, trail the SL.
    partial_close_pct:
        Fraction of position to close at TP1 (e.g. 0.5 = 50%).
    """

    def __init__(
        self,
        broker: BrokerPort,
        risk_guard: DrawdownGuard,
        position_sizer_config: dict | None = None,
        *,
        trailing_sl_points: float = 100.0,
        partial_close_pct: float = 0.5,
        peak_balance: float | None = None,
    ) -> None:
        self._broker = broker
        self._risk_guard = risk_guard
        self._sizer_config = position_sizer_config or {}
        self._trailing_sl_points = trailing_sl_points
        self._partial_close_pct = partial_close_pct
        self._local_positions: dict[int, PositionState] = {}
        self._daily_pnl: float = 0.0
        self._peak_balance: float = peak_balance or 0.0
        self._tp1_closed: set[int] = set()  # tickets where TP1 partial close already done

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_setup(
        self,
        setup: TradeSetup,
        account: AccountInfo,
    ) -> OrderResult | None:
        """Full lifecycle: risk check -> compute size -> send order -> track.

        Returns ``None`` if the risk guard rejects the trade, or an
        :class:`OrderResult` from the broker.
        """
        # Update peak balance
        if account.balance > self._peak_balance:
            self._peak_balance = account.balance

        # Risk check
        budget = self._risk_guard.check_budget(
            balance=account.balance,
            peak_balance=self._peak_balance,
            daily_pnl=self._daily_pnl,
        )
        if not budget.can_trade:
            logger.warning(
                "Trade rejected by risk guard: %s", budget.rejection_reason
            )
            return None

        signal = setup.entry_signal

        # Compute position size — explicit whitelist of sizer params
        sizer_kwargs: dict = {
            "balance_usd": account.balance,
            "risk_pct": self._sizer_config.get("risk_pct", 1.0),
            "sl_distance_points": signal.risk_points,
            "max_lot_size": self._sizer_config.get("max_lot_size", 0.1),
        }
        # Only forward keys that compute_position_size actually accepts
        _ALLOWED_SIZER_KEYS = {"pip_value_per_lot", "min_lot_size", "margin_per_lot"}
        for key in _ALLOWED_SIZER_KEYS:
            if key in self._sizer_config:
                sizer_kwargs[key] = self._sizer_config[key]

        size = compute_position_size(**sizer_kwargs)

        # Build order request
        request = OrderRequest(
            instrument="XAUUSD",
            direction=signal.direction,
            lots=size.lots,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            order_type="market",
        )

        # Send to broker
        result = self._broker.send_order(request)

        if result.success:
            # Fetch latest position state from broker
            positions = self._broker.get_positions()
            for pos in positions:
                if pos.ticket == result.ticket:
                    self._local_positions = {
                        **self._local_positions,
                        result.ticket: pos,
                    }
                    break
            logger.info(
                "Setup executed: ticket=%d dir=%s lots=%.2f entry=%.2f sl=%.2f tp1=%.2f",
                result.ticket,
                signal.direction,
                size.lots,
                result.fill_price,
                signal.stop_loss,
                signal.take_profit_1,
            )
        else:
            logger.error("Order failed: %s", result.error_message)

        return result

    def manage_open_positions(self, current_price: float) -> list[OrderResult]:
        """Check all open positions for trailing SL, partial TP1, and TP2 close.

        Args:
            current_price: Current market price for the instrument.

        Returns:
            List of order results from any modifications or closures.
        """
        results: list[OrderResult] = []

        # Work on a snapshot to avoid mutation during iteration
        for ticket, pos in list(self._local_positions.items()):
            # --- Check SL hit ---
            if self._is_sl_hit(pos, current_price):
                result = self._broker.close_position(ticket)
                if result.success:
                    self._realize_pnl(pos, current_price)
                    self._local_positions = {
                        k: v for k, v in self._local_positions.items() if k != ticket
                    }
                    self._tp1_closed.discard(ticket)
                    logger.info("SL hit: ticket=%d pnl=%.2f", ticket, pos.pnl_usd)
                results.append(result)
                continue

            # --- Check TP2 hit (full close) ---
            if pos.tp > 0 and self._is_tp_hit(pos, current_price, pos.tp):
                result = self._broker.close_position(ticket)
                if result.success:
                    self._realize_pnl(pos, current_price)
                    self._local_positions = {
                        k: v for k, v in self._local_positions.items() if k != ticket
                    }
                    self._tp1_closed.discard(ticket)
                    logger.info("TP2 hit: ticket=%d", ticket)
                results.append(result)
                continue

            # --- Check TP1 partial close ---
            if ticket not in self._tp1_closed:
                tp1 = self._get_tp1(pos)
                if tp1 > 0 and self._is_tp_hit(pos, current_price, tp1):
                    close_lots = round(pos.lots * self._partial_close_pct, 2)
                    if close_lots >= 0.01:
                        result = self._broker.close_position(ticket, lots=close_lots)
                        if result.success:
                            self._tp1_closed.add(ticket)
                            # Trail SL to breakeven after TP1
                            self._broker.modify_order(ticket, sl=pos.open_price)
                            # Refresh local tracking
                            self._refresh_position(ticket)
                            logger.info(
                                "TP1 partial close: ticket=%d lots=%.2f, SL trailed to BE",
                                ticket, close_lots,
                            )
                        results.append(result)
                        continue

            # --- Trailing SL ---
            new_sl = self._compute_trailing_sl(pos, current_price)
            if new_sl is not None and new_sl != pos.sl:
                result = self._broker.modify_order(ticket, sl=new_sl)
                if result.success:
                    self._refresh_position(ticket)
                    logger.info(
                        "Trailing SL: ticket=%d new_sl=%.2f", ticket, new_sl
                    )
                results.append(result)

        return results

    def close_all(self) -> list[OrderResult]:
        """Emergency close all tracked positions.

        Returns a list of :class:`OrderResult` — one per position.
        """
        results: list[OrderResult] = []
        for ticket, pos in list(self._local_positions.items()):
            result = self._broker.close_position(ticket)
            if result.success:
                self._realize_pnl(pos, pos.current_price)
                self._local_positions = {
                    k: v for k, v in self._local_positions.items() if k != ticket
                }
                self._tp1_closed.discard(ticket)
            results.append(result)

        logger.info("close_all: %d positions closed", len(results))
        return results

    def get_local_positions(self) -> dict[int, PositionState]:
        """Return the local position tracking dict (for reconciliation)."""
        return dict(self._local_positions)

    @property
    def daily_pnl(self) -> float:
        """Today's realised P&L in USD."""
        return self._daily_pnl

    def reset_daily_pnl(self) -> None:
        """Reset daily P&L counter (call at start of each trading day)."""
        self._daily_pnl = 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_sl_hit(self, pos: PositionState, current_price: float) -> bool:
        """Check if the stop-loss has been hit."""
        if pos.sl <= 0:
            return False
        if pos.direction == "long":
            return current_price <= pos.sl
        return current_price >= pos.sl

    def _is_tp_hit(
        self, pos: PositionState, current_price: float, tp_level: float
    ) -> bool:
        """Check if a take-profit level has been hit."""
        if tp_level <= 0:
            return False
        if pos.direction == "long":
            return current_price >= tp_level
        return current_price <= tp_level

    def _get_tp1(self, pos: PositionState) -> float:
        """Get TP1 level. For SimBroker, TP on the position is TP1."""
        return pos.tp

    def _compute_trailing_sl(
        self, pos: PositionState, current_price: float
    ) -> float | None:
        """Compute a new trailing SL if price has moved enough in profit.

        Returns the new SL, or None if no trailing should occur.
        """
        if pos.direction == "long":
            profit_points = current_price - pos.open_price
            if profit_points >= self._trailing_sl_points:
                candidate_sl = current_price - self._trailing_sl_points
                if candidate_sl > pos.sl:
                    return round(candidate_sl, 2)
        else:
            profit_points = pos.open_price - current_price
            if profit_points >= self._trailing_sl_points:
                candidate_sl = current_price + self._trailing_sl_points
                if candidate_sl < pos.sl or pos.sl <= 0:
                    return round(candidate_sl, 2)
        return None

    def _realize_pnl(self, pos: PositionState, close_price: float) -> None:
        """Update daily P&L after a position close."""
        # Use the PnL from the position state (broker-calculated)
        self._daily_pnl += pos.pnl_usd

    def _refresh_position(self, ticket: int) -> None:
        """Re-fetch a single position from the broker to update local state."""
        positions = self._broker.get_positions()
        for pos in positions:
            if pos.ticket == ticket:
                self._local_positions = {
                    **self._local_positions,
                    ticket: pos,
                }
                return
        # Position no longer exists at broker — remove locally
        self._local_positions = {
            k: v for k, v in self._local_positions.items() if k != ticket
        }
        self._tp1_closed.discard(ticket)

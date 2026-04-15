"""Execution-layer data types for the AI-SMC trading system.

All models are frozen (immutable) Pydantic BaseModel instances.
These types define the contract between the execution module and the rest
of the system — broker adapters, order management, and reconciliation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

__all__ = [
    "OrderRequest",
    "OrderResult",
    "PositionState",
    "AccountInfo",
    "ReconciliationResult",
]

# ---------------------------------------------------------------------------
# Order Request / Result
# ---------------------------------------------------------------------------


class OrderRequest(BaseModel):
    """Immutable order specification sent to the broker."""

    model_config = ConfigDict(frozen=True)

    instrument: str = "XAUUSD"
    direction: Literal["long", "short"]
    lots: float
    entry_price: float = 0.0  # 0 for market orders
    stop_loss: float
    take_profit_1: float
    take_profit_2: float | None = None
    order_type: Literal["market", "limit"] = "market"


class OrderResult(BaseModel):
    """Immutable result returned by the broker after an order action."""

    model_config = ConfigDict(frozen=True)

    success: bool
    ticket: int = 0
    fill_price: float = 0.0
    error_message: str = ""


# ---------------------------------------------------------------------------
# Position State
# ---------------------------------------------------------------------------


class PositionState(BaseModel):
    """Snapshot of an open position as reported by the broker."""

    model_config = ConfigDict(frozen=True)

    ticket: int
    instrument: str
    direction: Literal["long", "short"]
    lots: float
    open_price: float
    current_price: float
    sl: float
    tp: float
    pnl_usd: float
    open_time: datetime


# ---------------------------------------------------------------------------
# Account Info
# ---------------------------------------------------------------------------


class AccountInfo(BaseModel):
    """Snapshot of the trading account state."""

    model_config = ConfigDict(frozen=True)

    balance: float
    equity: float
    margin_used: float
    margin_free: float
    margin_level: float  # as percentage (e.g. 200.0 = healthy)


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


class ReconciliationResult(BaseModel):
    """Result of comparing local position tracking against broker state."""

    model_config = ConfigDict(frozen=True)

    in_sync: bool
    closed_by_broker: tuple[int, ...] = ()  # tickets we thought open but broker closed
    untracked: tuple[int, ...] = ()  # tickets broker has that we don't track
    pnl_mismatches: tuple[int, ...] = ()  # tickets with PnL divergence
    summary: str = ""

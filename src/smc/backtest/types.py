"""Backtest data types for the AI-SMC trading system.

All models are frozen (immutable) Pydantic BaseModel instances.
Uses POINTS (not pips) as the base unit for all price distances —
matches MT5 internal representation where 1 point = $0.01 for XAUUSD.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class BacktestConfig(BaseModel):
    """Immutable configuration for a single backtest run."""

    model_config = ConfigDict(frozen=True)

    initial_balance: float = 10_000.0
    instrument: str = "XAUUSD"
    spread_points: float = 3.0
    slippage_points: float = 0.5
    commission_per_lot: float = 7.0
    max_concurrent_trades: int = 3
    lot_size: float = 100_000.0


# ---------------------------------------------------------------------------
# Trade Record
# ---------------------------------------------------------------------------


class TradeRecord(BaseModel):
    """Immutable record of a single completed trade."""

    model_config = ConfigDict(frozen=True)

    open_ts: datetime
    open_price: float
    direction: Literal["long", "short"]
    close_ts: datetime
    close_price: float
    lots: float
    pnl_usd: float
    pnl_pct: float
    close_reason: Literal["tp1", "tp2", "sl", "manual", "eod"]
    setup_confluence: float
    trigger_type: str


# ---------------------------------------------------------------------------
# Equity Curve
# ---------------------------------------------------------------------------


class EquityCurve(BaseModel):
    """Immutable equity curve representation over time."""

    model_config = ConfigDict(frozen=True)

    timestamps: tuple[datetime, ...]
    equity: tuple[float, ...]
    drawdown: tuple[float, ...]


# ---------------------------------------------------------------------------
# Backtest Result
# ---------------------------------------------------------------------------


class BacktestResult(BaseModel):
    """Immutable aggregate result of a complete backtest run."""

    model_config = ConfigDict(frozen=True)

    config: BacktestConfig
    trades: tuple[TradeRecord, ...]
    equity_curve: EquityCurve
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float
    profit_factor: float
    win_rate: float
    expectancy: float
    total_trades: int
    start_date: datetime
    end_date: datetime


# ---------------------------------------------------------------------------
# Walk-Forward Summary
# ---------------------------------------------------------------------------


class WalkForwardSummary(BaseModel):
    """Immutable aggregate of out-of-sample walk-forward results."""

    model_config = ConfigDict(frozen=True)

    pooled_sharpe: float
    consistency_ratio: float
    total_oos_trades: int
    windows: int
    results: tuple[BacktestResult, ...]


__all__ = [
    "BacktestConfig",
    "TradeRecord",
    "EquityCurve",
    "BacktestResult",
    "WalkForwardSummary",
]

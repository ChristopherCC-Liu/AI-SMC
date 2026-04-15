"""Monitor data types for the AI-SMC live trading system.

All models are frozen (immutable) Pydantic BaseModel instances.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Health Check Result
# ---------------------------------------------------------------------------


class HealthCheckResult(BaseModel):
    """Result of a single health check."""

    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    detail: str


class HealthStatus(BaseModel):
    """Aggregate health status from all checks."""

    model_config = ConfigDict(frozen=True)

    all_ok: bool
    checks: tuple[HealthCheckResult, ...]
    checked_at: datetime


# ---------------------------------------------------------------------------
# Journal Types
# ---------------------------------------------------------------------------


class JournalEntry(BaseModel):
    """A single row in the trade journal."""

    model_config = ConfigDict(frozen=True)

    ts: datetime
    action: Literal["open", "close", "modify", "sl_hit", "tp_hit", "partial_close", "cycle"]
    ticket: int
    instrument: str
    direction: Literal["long", "short", "none"]
    lots: float
    price: float
    sl: float
    tp: float
    pnl: float
    balance_after: float
    setup_confluence: float
    trigger_type: str
    regime: str


class DailySummary(BaseModel):
    """Summarises one day of trading activity."""

    model_config = ConfigDict(frozen=True)

    date: str  # ISO date string YYYY-MM-DD
    total_trades: int
    winning_trades: int
    losing_trades: int
    gross_pnl: float
    net_pnl: float
    max_drawdown_pct: float
    win_rate: float


# ---------------------------------------------------------------------------
# Cycle Log
# ---------------------------------------------------------------------------


class CycleLog(BaseModel):
    """Record of a single live-loop cycle execution."""

    model_config = ConfigDict(frozen=True)

    cycle_number: int
    bar_close_ts: datetime
    setups_generated: int
    orders_placed: int
    positions_managed: int
    health_ok: bool
    duration_seconds: float

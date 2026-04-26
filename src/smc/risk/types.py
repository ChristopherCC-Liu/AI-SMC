"""Risk management data types for the AI-SMC trading system.

All models are frozen (immutable) Pydantic BaseModel instances.
Uses POINTS (not pips) as the base unit for all price distances —
matches MT5 internal representation where 1 point = $0.01 for XAUUSD.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Position Sizing Result
# ---------------------------------------------------------------------------


class PositionSize(BaseModel):
    """Immutable result of a position size calculation.

    Attributes:
        lots: Computed lot size, clamped to [min_lot_size, max_lot_size].
        risk_usd: Dollar amount at risk (balance * risk_pct).
        risk_points: Stop-loss distance in points.
        margin_required_usd: Estimated margin for the computed lot size.
    """

    model_config = ConfigDict(frozen=True)

    lots: float
    risk_usd: float
    risk_points: float
    margin_required_usd: float


# ---------------------------------------------------------------------------
# Risk Budget
# ---------------------------------------------------------------------------


class RiskBudget(BaseModel):
    """Immutable snapshot of the account's risk budget state.

    Returned by :class:`~smc.risk.drawdown_guard.DrawdownGuard.check_budget`
    and :class:`~smc.risk.persistent_drawdown_guard.PersistentDrawdownGuard.check_budget`
    to indicate whether the account is cleared to trade.

    Attributes:
        can_trade: True if no circuit breaker is active.
        available_risk_pct: Remaining risk capacity as a percentage.
        used_risk_pct: Percentage of equity already at risk.
        daily_loss_pct: Today's realised loss as a percentage of balance.
        total_drawdown_pct: Current drawdown from peak equity.
        rejection_reason: Human-readable reason when can_trade is False.
        block_opens: True when new positions must be blocked. Existing
            positions continue to be managed by their own SL/TP/trail.
            Set by every halt tier (R10 P1.1+).
        force_close: True only at the emergency tier (reserved for a
            future ratio<=0.80 trip). When True the operator's executor
            should market-close all positions immediately. None of the
            tiers shipped in R10 set this flag — it exists so callers can
            structure their handling now and avoid a breaking change later.
    """

    model_config = ConfigDict(frozen=True)

    can_trade: bool
    available_risk_pct: float
    used_risk_pct: float
    daily_loss_pct: float
    total_drawdown_pct: float
    rejection_reason: str | None = None
    block_opens: bool = False
    force_close: bool = False

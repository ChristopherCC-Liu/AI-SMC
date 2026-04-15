"""Performance metrics for backtest evaluation — pure functions.

All functions are stateless and operate on sequences of returns or trade
records.  Annualisation assumes M15 bars: 252 trading days * 24 hours * 4
bars/hour = 24_192 bars per year.
"""

from __future__ import annotations

import math
from typing import Sequence

from smc.backtest.types import TradeRecord

# M15 bars per trading year: 252 days * 24 hours * 4 bars/hour
_PERIODS_PER_YEAR: int = 252 * 24 * 4


# ---------------------------------------------------------------------------
# Risk-adjusted return ratios
# ---------------------------------------------------------------------------


def sharpe_ratio(
    returns: Sequence[float],
    risk_free: float = 0.0,
    periods_per_year: int = _PERIODS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio.

    Args:
        returns: Per-bar (or per-period) returns.
        risk_free: Risk-free rate per period (default 0).
        periods_per_year: Number of periods in one year for annualisation.

    Returns:
        Annualised Sharpe ratio.  Returns 0.0 when there are fewer than 2
        observations or standard deviation is zero.
    """
    n = len(returns)
    if n < 2:
        return 0.0

    excess = [r - risk_free for r in returns]
    mean = sum(excess) / n
    variance = sum((r - mean) ** 2 for r in excess) / (n - 1)
    std = math.sqrt(variance)

    if std == 0.0:
        return 0.0

    return (mean / std) * math.sqrt(periods_per_year)


def sortino_ratio(
    returns: Sequence[float],
    risk_free: float = 0.0,
    periods_per_year: int = _PERIODS_PER_YEAR,
) -> float:
    """Annualised Sortino ratio (downside deviation only).

    Uses the same formula as Sharpe but replaces standard deviation with
    downside deviation (only negative excess returns contribute).

    Returns 0.0 when there are fewer than 2 observations or downside
    deviation is zero.
    """
    n = len(returns)
    if n < 2:
        return 0.0

    excess = [r - risk_free for r in returns]
    mean = sum(excess) / n
    downside_sq = [e ** 2 for e in excess if e < 0.0]

    if not downside_sq:
        return 0.0

    downside_dev = math.sqrt(sum(downside_sq) / (n - 1))

    if downside_dev == 0.0:
        return 0.0

    return (mean / downside_dev) * math.sqrt(periods_per_year)


def calmar_ratio(
    returns: Sequence[float],
    periods_per_year: int = _PERIODS_PER_YEAR,
) -> float:
    """Calmar ratio: annualised return / max drawdown.

    Returns 0.0 when max drawdown is zero (no losses) or insufficient data.
    """
    n = len(returns)
    if n < 2:
        return 0.0

    # Build equity from returns to compute drawdown
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))

    dd = max_drawdown(tuple(equity))
    if dd == 0.0:
        return 0.0

    # Annualise the total return
    total_return = equity[-1] / equity[0] - 1.0
    periods = len(returns)
    annual_return = (1.0 + total_return) ** (periods_per_year / periods) - 1.0

    return annual_return / dd


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


def max_drawdown(equity: Sequence[float]) -> float:
    """Maximum drawdown as a percentage (0.0 to 1.0).

    Args:
        equity: Sequence of equity values (absolute, not returns).

    Returns:
        Maximum peak-to-trough drawdown as a fraction.  Returns 0.0 for
        sequences shorter than 2 or monotonically increasing equity.
    """
    if len(equity) < 2:
        return 0.0

    peak = equity[0]
    worst_dd = 0.0

    for value in equity:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0.0 else 0.0
        if dd > worst_dd:
            worst_dd = dd

    return worst_dd


# ---------------------------------------------------------------------------
# Trade-level metrics
# ---------------------------------------------------------------------------


def profit_factor(trades: Sequence[TradeRecord]) -> float:
    """Gross profit / gross loss.  Returns inf when no losing trades exist."""
    gross_profit = sum(t.pnl_usd for t in trades if t.pnl_usd > 0.0)
    gross_loss = abs(sum(t.pnl_usd for t in trades if t.pnl_usd < 0.0))

    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0.0 else 0.0

    return gross_profit / gross_loss


def win_rate(trades: Sequence[TradeRecord]) -> float:
    """Fraction of winning trades (pnl_usd > 0).  Returns 0.0 for empty input."""
    if not trades:
        return 0.0

    winners = sum(1 for t in trades if t.pnl_usd > 0.0)
    return winners / len(trades)


def expectancy(trades: Sequence[TradeRecord]) -> float:
    """Expected value per trade: avg_win * win_rate - avg_loss * loss_rate.

    Returns 0.0 when there are no trades.
    """
    if not trades:
        return 0.0

    winners = [t.pnl_usd for t in trades if t.pnl_usd > 0.0]
    losers = [t.pnl_usd for t in trades if t.pnl_usd <= 0.0]

    wr = len(winners) / len(trades) if trades else 0.0
    lr = len(losers) / len(trades) if trades else 0.0

    avg_win = sum(winners) / len(winners) if winners else 0.0
    avg_loss = abs(sum(losers) / len(losers)) if losers else 0.0

    return avg_win * wr - avg_loss * lr


__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "profit_factor",
    "win_rate",
    "expectancy",
]

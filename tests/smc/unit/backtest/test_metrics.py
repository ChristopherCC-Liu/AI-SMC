"""Unit tests for backtest performance metrics.

Tests are validated against known analytical results to ensure
correctness of Sharpe, Sortino, Calmar, drawdown, and trade metrics.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from smc.backtest.metrics import (
    calmar_ratio,
    expectancy,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)
from smc.backtest.types import TradeRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade(pnl: float) -> TradeRecord:
    """Create a minimal TradeRecord with the given P&L."""
    ts = datetime(2024, 1, 2, tzinfo=timezone.utc)
    return TradeRecord(
        open_ts=ts,
        open_price=2350.0,
        direction="long",
        close_ts=ts,
        close_price=2350.0 + pnl,
        lots=0.01,
        pnl_usd=pnl,
        pnl_pct=pnl / 10_000.0,
        close_reason="tp1",
        setup_confluence=0.8,
        trigger_type="test",
    )


# ---------------------------------------------------------------------------
# Sharpe Ratio
# ---------------------------------------------------------------------------


class TestSharpeRatio:
    def test_empty_returns(self) -> None:
        assert sharpe_ratio([]) == 0.0

    def test_single_return(self) -> None:
        assert sharpe_ratio([0.01]) == 0.0

    def test_constant_returns(self) -> None:
        """Zero variance -> zero Sharpe."""
        assert sharpe_ratio([0.01, 0.01, 0.01]) == 0.0

    def test_positive_returns(self) -> None:
        """Known case: mean > 0, nonzero std -> positive Sharpe."""
        returns = [0.01, 0.02, 0.015, 0.005, 0.012]
        result = sharpe_ratio(returns, periods_per_year=252)
        assert result > 0.0

    def test_negative_returns(self) -> None:
        """All negative returns -> negative Sharpe."""
        returns = [-0.01, -0.02, -0.015]
        result = sharpe_ratio(returns, periods_per_year=252)
        assert result < 0.0

    def test_known_value(self) -> None:
        """Verify against hand-calculated Sharpe.

        returns = [0.01, -0.01]
        mean = 0.0, std = sqrt((0.01^2 + 0.01^2) / 1) = 0.01414
        Sharpe = 0.0 / 0.01414 * sqrt(252) = 0.0
        """
        result = sharpe_ratio([0.01, -0.01], periods_per_year=252)
        assert abs(result) < 1e-10


# ---------------------------------------------------------------------------
# Sortino Ratio
# ---------------------------------------------------------------------------


class TestSortinoRatio:
    def test_empty(self) -> None:
        assert sortino_ratio([]) == 0.0

    def test_all_positive(self) -> None:
        """No negative returns -> zero downside deviation -> 0.0."""
        assert sortino_ratio([0.01, 0.02, 0.03]) == 0.0

    def test_mixed_returns(self) -> None:
        """With negative returns, Sortino should be computable."""
        returns = [0.02, -0.01, 0.03, -0.005, 0.01]
        result = sortino_ratio(returns, periods_per_year=252)
        # Mean is positive, so Sortino should be positive
        assert result > 0.0


# ---------------------------------------------------------------------------
# Calmar Ratio
# ---------------------------------------------------------------------------


class TestCalmarRatio:
    def test_empty(self) -> None:
        assert calmar_ratio([]) == 0.0

    def test_no_drawdown(self) -> None:
        """Monotonically increasing equity -> zero drawdown -> 0.0."""
        assert calmar_ratio([0.01, 0.01, 0.01]) == 0.0

    def test_with_drawdown(self) -> None:
        """Known drawdown case."""
        returns = [0.10, -0.05, 0.03]
        result = calmar_ratio(returns, periods_per_year=252)
        assert result > 0.0


# ---------------------------------------------------------------------------
# Max Drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_empty(self) -> None:
        assert max_drawdown([]) == 0.0
        assert max_drawdown([100.0]) == 0.0

    def test_monotonically_increasing(self) -> None:
        assert max_drawdown([100.0, 110.0, 120.0, 130.0]) == 0.0

    def test_known_drawdown(self) -> None:
        """Equity: 100 -> 120 -> 90 -> 110
        Peak at 120, trough at 90: drawdown = (120-90)/120 = 0.25
        """
        equity = [100.0, 120.0, 90.0, 110.0]
        result = max_drawdown(equity)
        assert abs(result - 0.25) < 1e-10

    def test_full_drawdown(self) -> None:
        """Equity drops to zero."""
        equity = [100.0, 50.0, 0.0]
        assert max_drawdown(equity) == 1.0

    def test_multiple_drawdowns(self) -> None:
        """Two drawdowns, second is deeper."""
        equity = [100.0, 90.0, 95.0, 80.0, 85.0]
        # First dd: (100-90)/100 = 0.10
        # Second dd: (100-80)/100 = 0.20 (peak is still 100)
        result = max_drawdown(equity)
        assert abs(result - 0.20) < 1e-10


# ---------------------------------------------------------------------------
# Profit Factor
# ---------------------------------------------------------------------------


class TestProfitFactor:
    def test_empty(self) -> None:
        assert profit_factor([]) == 0.0

    def test_all_winners(self) -> None:
        trades = [_make_trade(10.0), _make_trade(20.0)]
        assert profit_factor(trades) == float("inf")

    def test_all_losers(self) -> None:
        trades = [_make_trade(-10.0), _make_trade(-20.0)]
        assert profit_factor(trades) == 0.0

    def test_known_value(self) -> None:
        """Gross profit = 30, gross loss = 10 -> PF = 3.0"""
        trades = [_make_trade(10.0), _make_trade(20.0), _make_trade(-10.0)]
        assert abs(profit_factor(trades) - 3.0) < 1e-10


# ---------------------------------------------------------------------------
# Win Rate
# ---------------------------------------------------------------------------


class TestWinRate:
    def test_empty(self) -> None:
        assert win_rate([]) == 0.0

    def test_all_winners(self) -> None:
        trades = [_make_trade(10.0), _make_trade(5.0)]
        assert win_rate(trades) == 1.0

    def test_mixed(self) -> None:
        trades = [_make_trade(10.0), _make_trade(-5.0), _make_trade(3.0)]
        assert abs(win_rate(trades) - 2 / 3) < 1e-10

    def test_breakeven_counts_as_loss(self) -> None:
        """A trade with pnl=0 is NOT a winner."""
        trades = [_make_trade(0.0)]
        assert win_rate(trades) == 0.0


# ---------------------------------------------------------------------------
# Expectancy
# ---------------------------------------------------------------------------


class TestExpectancy:
    def test_empty(self) -> None:
        assert expectancy([]) == 0.0

    def test_known_value(self) -> None:
        """2 winners (+10, +20), 1 loser (-15)
        avg_win = 15, win_rate = 2/3
        avg_loss = 15, loss_rate = 1/3
        expectancy = 15 * 2/3 - 15 * 1/3 = 10 - 5 = 5.0
        """
        trades = [_make_trade(10.0), _make_trade(20.0), _make_trade(-15.0)]
        result = expectancy(trades)
        assert abs(result - 5.0) < 1e-10

    def test_negative_expectancy(self) -> None:
        """All losers -> negative expectancy."""
        trades = [_make_trade(-10.0), _make_trade(-20.0)]
        result = expectancy(trades)
        assert result < 0.0

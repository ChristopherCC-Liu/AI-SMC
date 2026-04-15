"""Backtest engine for the AI-SMC trading system.

Provides bar-by-bar simulation with realistic fills, performance metrics,
and walk-forward out-of-sample validation.
"""

from smc.backtest.engine import BarBacktestEngine
from smc.backtest.fills import BarOHLC, ExitResult, FillModel
from smc.backtest.metrics import (
    calmar_ratio,
    expectancy,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)
from smc.backtest.types import (
    BacktestConfig,
    BacktestResult,
    EquityCurve,
    TradeRecord,
    WalkForwardSummary,
)
from smc.backtest.adapter import SMCStrategyAdapter
from smc.backtest.walk_forward import aggregate_oos_results, walk_forward_oos

__all__ = [
    # Types
    "BacktestConfig",
    "BacktestResult",
    "EquityCurve",
    "TradeRecord",
    "WalkForwardSummary",
    # Fills
    "FillModel",
    "BarOHLC",
    "ExitResult",
    # Engine
    "BarBacktestEngine",
    # Metrics
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "profit_factor",
    "win_rate",
    "expectancy",
    # Adapter
    "SMCStrategyAdapter",
    # Walk-forward
    "walk_forward_oos",
    "aggregate_oos_results",
]

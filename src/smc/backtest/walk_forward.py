"""Walk-forward out-of-sample (OOS) validation.

Implements a rolling-window approach: train on N months of data, test on
the next M months, then slide forward by S months.  This prevents
overfitting by ensuring the strategy is always evaluated on unseen data.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import polars as pl

from smc.backtest import metrics
from smc.backtest.engine import BarBacktestEngine, TradeSetupLike
from smc.backtest.types import BacktestResult, WalkForwardSummary
from smc.data.lake import ForexDataLake
from smc.data.schemas import Timeframe


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class StrategyLike(Protocol):
    """Minimal protocol for a strategy that produces trade setups.

    The strategy must implement two phases:
        1. ``train()`` — fit/calibrate on historical data
        2. ``generate_setups()`` — produce signals on new data
    """

    def train(self, bars: pl.DataFrame) -> None:
        """Train/calibrate the strategy on historical bars."""
        ...

    def generate_setups(
        self, bars: pl.DataFrame
    ) -> dict[datetime, tuple[TradeSetupLike, ...]]:
        """Generate trade setups keyed by bar timestamp."""
        ...


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------


def _add_months(dt: datetime, months: int) -> datetime:
    """Add *months* calendar months to *dt*, clamping to valid day."""
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    # Clamp day to max days in target month
    import calendar

    max_day = calendar.monthrange(year, month)[1]
    day = min(dt.day, max_day)
    return dt.replace(year=year, month=month, day=day)


def walk_forward_oos(
    engine: BarBacktestEngine,
    strategy: StrategyLike,
    lake: ForexDataLake,
    *,
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
) -> list[BacktestResult]:
    """Run walk-forward OOS validation with rolling windows.

    Window layout::

        |--- train (12mo) ---|--- test (3mo) ---|
                         |--- train (12mo) ---|--- test (3mo) ---|
                                          (slide by step_months)

    Args:
        engine: Configured backtest engine.
        strategy: Strategy implementing train() and generate_setups().
        lake: Data lake for querying OHLCV bars.
        train_months: Length of training window in months.
        test_months: Length of test window in months.
        step_months: Slide step in months.

    Returns:
        List of BacktestResult, one per OOS window.
    """
    instrument = engine.config.instrument
    timeframe = Timeframe.M15

    available = lake.available_range(instrument, timeframe)
    if available is None:
        return []

    data_start, data_end = available
    results: list[BacktestResult] = []

    window_start = data_start
    while True:
        train_end = _add_months(window_start, train_months)
        test_end = _add_months(train_end, test_months)

        # Stop if the test window extends beyond available data
        if test_end > data_end:
            break

        # Query train and test bars
        train_bars = lake.query(instrument, timeframe, window_start, train_end)
        test_bars = lake.query(instrument, timeframe, train_end, test_end)

        if train_bars.is_empty() or test_bars.is_empty():
            window_start = _add_months(window_start, step_months)
            continue

        # Phase 1: Train strategy on in-sample data
        strategy.train(train_bars)

        # Phase 2: Generate setups on OOS data (no look-ahead)
        setups = strategy.generate_setups(test_bars)

        # Phase 3: Run backtest on OOS period
        result = engine.run(setups, test_bars)
        results.append(result)

        # Slide window forward
        window_start = _add_months(window_start, step_months)

    return results


def aggregate_oos_results(results: list[BacktestResult]) -> WalkForwardSummary:
    """Aggregate multiple OOS window results into a summary.

    Pooled Sharpe is computed by concatenating all window returns and
    computing a single Sharpe ratio.

    Consistency ratio is the fraction of OOS windows that produced a
    positive Sharpe ratio (Sharpe > 0).

    Args:
        results: List of BacktestResult from walk_forward_oos().

    Returns:
        WalkForwardSummary with pooled metrics.
    """
    if not results:
        return WalkForwardSummary(
            pooled_sharpe=0.0,
            consistency_ratio=0.0,
            total_oos_trades=0,
            windows=0,
            results=(),
        )

    # Pool all bar returns across windows for a single Sharpe
    pooled_returns: list[float] = []
    total_trades = 0
    positive_windows = 0

    for r in results:
        # Reconstruct bar returns from the equity curve
        eq = r.equity_curve.equity
        for j in range(1, len(eq)):
            prev = eq[j - 1]
            ret = (eq[j] - prev) / prev if prev > 0.0 else 0.0
            pooled_returns.append(ret)

        total_trades += r.total_trades
        if r.sharpe > 0.0:
            positive_windows += 1

    pooled_sharpe = metrics.sharpe_ratio(pooled_returns)
    consistency = positive_windows / len(results) if results else 0.0

    return WalkForwardSummary(
        pooled_sharpe=pooled_sharpe,
        consistency_ratio=consistency,
        total_oos_trades=total_trades,
        windows=len(results),
        results=tuple(results),
    )


__all__ = [
    "walk_forward_oos",
    "aggregate_oos_results",
    "StrategyLike",
]

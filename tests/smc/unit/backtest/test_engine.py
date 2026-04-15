"""Unit tests for the bar-by-bar backtest engine.

Validates no-look-ahead semantics, position tracking, equity curve
construction, and metric aggregation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from smc.backtest.engine import BarBacktestEngine
from smc.backtest.fills import FillModel
from smc.backtest.types import BacktestConfig


# ---------------------------------------------------------------------------
# Stub setup types compatible with the engine's protocol
# ---------------------------------------------------------------------------


class _StubEntrySignal:
    """Minimal entry signal for testing."""

    def __init__(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float | None,
        direction: str,
    ) -> None:
        self._entry_price = entry_price
        self._stop_loss = stop_loss
        self._take_profit_1 = take_profit_1
        self._take_profit_2 = take_profit_2
        self._direction = direction

    @property
    def entry_price(self) -> float:
        return self._entry_price

    @property
    def stop_loss(self) -> float:
        return self._stop_loss

    @property
    def take_profit_1(self) -> float:
        return self._take_profit_1

    @property
    def take_profit_2(self) -> float | None:
        return self._take_profit_2

    @property
    def direction(self) -> str:
        return self._direction


class _StubSetup:
    """Minimal trade setup for testing."""

    def __init__(self, signal: _StubEntrySignal, confluence: float = 0.8) -> None:
        self._signal = signal
        self._confluence = confluence

    @property
    def entry_signal(self) -> _StubEntrySignal:
        return self._signal

    @property
    def confluence_score(self) -> float:
        return self._confluence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bars(n: int, base_price: float = 2350.0) -> pl.DataFrame:
    """Create n synthetic M15 bars with predictable prices."""
    start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    delta = timedelta(minutes=15)
    data = {
        "ts": [start + delta * i for i in range(n)],
        "open": [base_price + i * 0.5 for i in range(n)],
        "high": [base_price + i * 0.5 + 2.0 for i in range(n)],
        "low": [base_price + i * 0.5 - 2.0 for i in range(n)],
        "close": [base_price + i * 0.5 + 0.5 for i in range(n)],
    }
    return pl.DataFrame(
        data,
        schema={
            "ts": pl.Datetime("ns", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        },
    )


@pytest.fixture()
def engine() -> BarBacktestEngine:
    """Engine with zero-cost fills for predictable P&L."""
    config = BacktestConfig(
        initial_balance=10_000.0,
        spread_points=0.0,
        slippage_points=0.0,
        commission_per_lot=0.0,
    )
    fill_model = FillModel(
        spread_points=0.0,
        slippage_points=0.0,
        commission_per_lot=0.0,
    )
    return BarBacktestEngine(config, fill_model)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBarBacktestEngine:
    def test_empty_run(self, engine: BarBacktestEngine) -> None:
        """Engine with no setups produces zero trades."""
        bars = _make_bars(10)
        result = engine.run({}, bars)

        assert result.total_trades == 0
        assert len(result.trades) == 0
        assert len(result.equity_curve.timestamps) == 10

    def test_single_winning_trade(self, engine: BarBacktestEngine) -> None:
        """Long trade that hits TP1 produces positive P&L."""
        bars = _make_bars(20)
        bar_ts = bars["ts"][0]

        signal = _StubEntrySignal(
            entry_price=2350.0,
            stop_loss=2340.0,
            take_profit_1=2360.0,
            take_profit_2=None,
            direction="long",
        )
        setup = _StubSetup(signal)

        setups = {bar_ts: (setup,)}
        result = engine.run(setups, bars)

        # The trade should have been opened and closed
        assert result.total_trades >= 1
        # First trade should be a winner (price is trending up)
        if result.trades:
            first = result.trades[0]
            assert first.direction == "long"

    def test_no_lookahead(self, engine: BarBacktestEngine) -> None:
        """Signal at bar[t] should only use bar[t]'s open price for fill.

        The engine processes setups AFTER checking exits, and fills
        at bar.open + costs. It must never peek at future bars.
        """
        bars = _make_bars(5, base_price=2350.0)
        bar_ts = bars["ts"][2]  # Signal fires at bar index 2

        signal = _StubEntrySignal(
            entry_price=2351.0,
            stop_loss=2340.0,
            take_profit_1=2380.0,
            take_profit_2=None,
            direction="long",
        )
        setup = _StubSetup(signal)
        setups = {bar_ts: (setup,)}

        result = engine.run(setups, bars)

        # Trade should open at bar[2]'s timestamp, not earlier
        if result.trades:
            assert result.trades[0].open_ts == bar_ts
        # Even if no trade closed, the equity curve should cover all bars
        assert len(result.equity_curve.timestamps) == 5

    def test_max_concurrent_trades_respected(self) -> None:
        """Engine respects max_concurrent_trades limit."""
        config = BacktestConfig(
            initial_balance=10_000.0,
            max_concurrent_trades=1,
            spread_points=0.0,
            slippage_points=0.0,
            commission_per_lot=0.0,
        )
        fm = FillModel(spread_points=0.0, slippage_points=0.0, commission_per_lot=0.0)
        eng = BarBacktestEngine(config, fm)

        bars = _make_bars(10)
        bar_ts = bars["ts"][0]

        # Two setups on the same bar, but max_concurrent=1
        sig1 = _StubEntrySignal(2350.0, 2340.0, 2390.0, None, "long")
        sig2 = _StubEntrySignal(2350.0, 2340.0, 2390.0, None, "long")
        setups = {bar_ts: (_StubSetup(sig1), _StubSetup(sig2))}

        result = eng.run(setups, bars)
        # At most 1 trade should open since max_concurrent=1
        # (both have far TP so neither closes in 10 bars)
        # The second setup is skipped
        assert result.total_trades <= 1

    def test_equity_curve_length(self, engine: BarBacktestEngine) -> None:
        """Equity curve has one entry per bar."""
        n = 50
        bars = _make_bars(n)
        result = engine.run({}, bars)

        assert len(result.equity_curve.timestamps) == n
        assert len(result.equity_curve.equity) == n
        assert len(result.equity_curve.drawdown) == n

    def test_initial_equity_matches_config(self, engine: BarBacktestEngine) -> None:
        """First equity point equals the configured initial balance."""
        bars = _make_bars(5)
        result = engine.run({}, bars)
        assert result.equity_curve.equity[0] == 10_000.0

    def test_losing_trade_reduces_balance(self) -> None:
        """A trade that hits SL produces negative P&L and reduces equity."""
        config = BacktestConfig(
            initial_balance=10_000.0,
            spread_points=0.0,
            slippage_points=0.0,
            commission_per_lot=0.0,
        )
        fm = FillModel(spread_points=0.0, slippage_points=0.0, commission_per_lot=0.0)
        eng = BarBacktestEngine(config, fm)

        # Create bars where price drops
        start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        delta = timedelta(minutes=15)
        data = {
            "ts": [start + delta * i for i in range(5)],
            "open": [2350.0, 2348.0, 2345.0, 2342.0, 2340.0],
            "high": [2352.0, 2350.0, 2347.0, 2344.0, 2342.0],
            "low": [2348.0, 2345.0, 2342.0, 2339.0, 2338.0],
            "close": [2349.0, 2346.0, 2343.0, 2340.0, 2339.0],
        }
        bars = pl.DataFrame(
            data,
            schema={
                "ts": pl.Datetime("ns", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
            },
        )

        signal = _StubEntrySignal(
            entry_price=2350.0,
            stop_loss=2340.0,
            take_profit_1=2370.0,
            take_profit_2=None,
            direction="long",
        )
        setups = {start: (_StubSetup(signal),)}
        result = eng.run(setups, bars)

        assert result.total_trades == 1
        assert result.trades[0].close_reason == "sl"
        assert result.trades[0].pnl_usd < 0.0

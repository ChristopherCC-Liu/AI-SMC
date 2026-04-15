"""Unit tests for backtest data types."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from smc.backtest.types import (
    BacktestConfig,
    BacktestResult,
    EquityCurve,
    TradeRecord,
    WalkForwardSummary,
)


# ---------------------------------------------------------------------------
# BacktestConfig
# ---------------------------------------------------------------------------


class TestBacktestConfig:
    def test_defaults(self) -> None:
        cfg = BacktestConfig()
        assert cfg.initial_balance == 10_000.0
        assert cfg.instrument == "XAUUSD"
        assert cfg.spread_points == 3.0
        assert cfg.slippage_points == 0.5
        assert cfg.commission_per_lot == 7.0
        assert cfg.max_concurrent_trades == 3
        assert cfg.lot_size == 100_000.0

    def test_custom_values(self) -> None:
        cfg = BacktestConfig(initial_balance=50_000.0, max_concurrent_trades=5)
        assert cfg.initial_balance == 50_000.0
        assert cfg.max_concurrent_trades == 5

    def test_frozen(self) -> None:
        cfg = BacktestConfig()
        with pytest.raises(Exception):
            cfg.initial_balance = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TradeRecord
# ---------------------------------------------------------------------------


class TestTradeRecord:
    def test_creation(self) -> None:
        ts = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
        trade = TradeRecord(
            open_ts=ts,
            open_price=2350.50,
            direction="long",
            close_ts=ts,
            close_price=2355.00,
            lots=0.01,
            pnl_usd=4.50,
            pnl_pct=0.00045,
            close_reason="tp1",
            setup_confluence=0.85,
            trigger_type="SMCSetup",
        )
        assert trade.direction == "long"
        assert trade.pnl_usd == 4.50

    def test_frozen(self) -> None:
        ts = datetime(2024, 1, 2, tzinfo=timezone.utc)
        trade = TradeRecord(
            open_ts=ts,
            open_price=2350.0,
            direction="short",
            close_ts=ts,
            close_price=2345.0,
            lots=0.01,
            pnl_usd=5.0,
            pnl_pct=0.0005,
            close_reason="tp2",
            setup_confluence=0.9,
            trigger_type="test",
        )
        with pytest.raises(Exception):
            trade.pnl_usd = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EquityCurve
# ---------------------------------------------------------------------------


class TestEquityCurve:
    def test_creation(self) -> None:
        ts1 = datetime(2024, 1, 2, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 3, tzinfo=timezone.utc)
        ec = EquityCurve(
            timestamps=(ts1, ts2),
            equity=(10_000.0, 10_050.0),
            drawdown=(0.0, 0.0),
        )
        assert len(ec.timestamps) == 2
        assert ec.equity[1] == 10_050.0

    def test_frozen(self) -> None:
        ec = EquityCurve(timestamps=(), equity=(), drawdown=())
        with pytest.raises(Exception):
            ec.timestamps = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


class TestBacktestResult:
    def test_creation(self) -> None:
        cfg = BacktestConfig()
        ec = EquityCurve(timestamps=(), equity=(), drawdown=())
        ts = datetime(2024, 1, 2, tzinfo=timezone.utc)
        result = BacktestResult(
            config=cfg,
            trades=(),
            equity_curve=ec,
            sharpe=1.5,
            sortino=2.0,
            calmar=0.8,
            max_drawdown_pct=0.05,
            profit_factor=2.5,
            win_rate=0.6,
            expectancy=15.0,
            total_trades=0,
            start_date=ts,
            end_date=ts,
        )
        assert result.sharpe == 1.5
        assert result.total_trades == 0


# ---------------------------------------------------------------------------
# WalkForwardSummary
# ---------------------------------------------------------------------------


class TestWalkForwardSummary:
    def test_creation(self) -> None:
        summary = WalkForwardSummary(
            pooled_sharpe=1.2,
            consistency_ratio=0.75,
            total_oos_trades=100,
            windows=4,
            results=(),
        )
        assert summary.consistency_ratio == 0.75
        assert summary.windows == 4

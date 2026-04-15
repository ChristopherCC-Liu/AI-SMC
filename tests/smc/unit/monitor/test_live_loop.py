"""Tests for smc.monitor.live_loop.LiveLoop.

Uses a mock broker, mock data fetcher, and mock strategy to verify
one full cycle of the live trading loop without real MT5 or asyncio sleep.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from smc.config import SMCConfig
from smc.data.schemas import SCHEMA_VERSION, Timeframe
from smc.monitor.live_loop import LiveLoop
from smc.monitor.types import CycleLog


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubBroker:
    """Minimal broker stub for testing."""

    def __init__(self, balance: float = 10_000.0) -> None:
        self._balance = balance

    def get_account_info(self) -> dict[str, Any]:
        return {"balance": self._balance, "equity": self._balance, "margin_level": None}

    def get_positions(self, symbol: str) -> list[dict[str, Any]]:
        return []

    def get_current_price(self, symbol: str) -> float | None:
        return 2350.0


class StubDataFetcher:
    """Returns minimal OHLCV DataFrames for testing."""

    def __init__(self) -> None:
        self.call_count = 0

    def fetch(
        self,
        *,
        instrument: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        self.call_count += 1
        n = 50
        base_ts = start
        rows: list[datetime] = []
        bar_minutes = {
            Timeframe.D1: 1440,
            Timeframe.H4: 240,
            Timeframe.H1: 60,
            Timeframe.M15: 15,
            Timeframe.M5: 5,
            Timeframe.M1: 1,
        }.get(timeframe, 15)

        for i in range(n):
            rows.append(base_ts + timedelta(minutes=bar_minutes * i))

        return pl.DataFrame(
            {
                "ts": rows,
                "open": [2350.0] * n,
                "high": [2355.0] * n,
                "low": [2345.0] * n,
                "close": [2352.0] * n,
                "volume": [1000.0] * n,
                "spread": [3.0] * n,
                "timeframe": [timeframe.value] * n,
                "source": ["test"] * n,
                "schema_version": [SCHEMA_VERSION] * n,
            },
            schema={
                "ts": pl.Datetime("ns", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "spread": pl.Float64,
                "timeframe": pl.String,
                "source": pl.String,
                "schema_version": pl.Int32,
            },
        )


class StubStrategy:
    """Strategy stub that returns no setups (safe baseline)."""

    def __init__(self, setups: tuple = ()) -> None:
        self._setups = setups
        self.call_count = 0

    def generate_setups(
        self,
        data: dict[Timeframe, pl.DataFrame],
        current_price: float,
        bar_ts: datetime | None = None,
    ) -> tuple:
        self.call_count += 1
        return self._setups


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stub_broker() -> StubBroker:
    return StubBroker()


@pytest.fixture()
def stub_fetcher() -> StubDataFetcher:
    return StubDataFetcher()


@pytest.fixture()
def stub_strategy() -> StubStrategy:
    return StubStrategy()


@pytest.fixture()
def live_loop(
    tmp_path: Path,
    stub_broker: StubBroker,
    stub_fetcher: StubDataFetcher,
    stub_strategy: StubStrategy,
) -> LiveLoop:
    cfg = SMCConfig(
        env="paper",
        mt5_mock=True,
        data_dir=tmp_path / "data",
        telegram_bot_token="",
        telegram_chat_id="",
    )
    return LiveLoop(
        config=cfg,
        broker=stub_broker,
        data_fetcher=stub_fetcher,
        strategy=stub_strategy,
        instrument="XAUUSD",
        journal_dir=tmp_path / "journal",
    )


# ---------------------------------------------------------------------------
# Single cycle tests
# ---------------------------------------------------------------------------


class TestSingleCycle:
    def test_single_cycle_returns_cycle_log(self, live_loop: LiveLoop) -> None:
        # Use a recent timestamp to pass data freshness check
        bar_ts = datetime.now(tz=timezone.utc)
        result = asyncio.run(live_loop.run_single_cycle(bar_ts=bar_ts))

        assert isinstance(result, CycleLog)
        assert result.cycle_number == 1
        assert result.health_ok is True
        assert result.setups_generated == 0
        assert result.orders_placed == 0

    def test_cycle_increments_counter(self, live_loop: LiveLoop) -> None:
        bar_ts = datetime(2024, 6, 15, 10, 15, 0, tzinfo=timezone.utc)
        asyncio.run(live_loop.run_single_cycle(bar_ts=bar_ts))
        asyncio.run(live_loop.run_single_cycle(bar_ts=bar_ts))
        assert live_loop.cycle_count == 2

    def test_cycle_fetches_data(
        self,
        live_loop: LiveLoop,
        stub_fetcher: StubDataFetcher,
    ) -> None:
        bar_ts = datetime(2024, 6, 15, 10, 15, 0, tzinfo=timezone.utc)
        asyncio.run(live_loop.run_single_cycle(bar_ts=bar_ts))
        # Should fetch D1, H4, H1, M15 = 4 calls
        assert stub_fetcher.call_count == 4

    def test_cycle_calls_strategy(
        self,
        live_loop: LiveLoop,
        stub_strategy: StubStrategy,
    ) -> None:
        bar_ts = datetime(2024, 6, 15, 10, 15, 0, tzinfo=timezone.utc)
        asyncio.run(live_loop.run_single_cycle(bar_ts=bar_ts))
        assert stub_strategy.call_count == 1

    def test_cycle_writes_journal(self, live_loop: LiveLoop) -> None:
        bar_ts = datetime(2024, 6, 15, 10, 15, 0, tzinfo=timezone.utc)
        asyncio.run(live_loop.run_single_cycle(bar_ts=bar_ts))

        from datetime import date

        df = live_loop.journal.read_day(date(2024, 6, 15))
        assert len(df) >= 1  # At least the cycle heartbeat


# ---------------------------------------------------------------------------
# Health check integration
# ---------------------------------------------------------------------------


class TestHealthInCycle:
    def test_healthy_cycle(self, live_loop: LiveLoop) -> None:
        bar_ts = datetime.now(tz=timezone.utc)
        result = asyncio.run(live_loop.run_single_cycle(bar_ts=bar_ts))
        assert result.health_ok is True


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------


class TestErrorResilience:
    def test_data_fetch_failure_does_not_crash(self, tmp_path: Path) -> None:
        class FailingFetcher:
            def fetch(self, **kwargs: Any) -> pl.DataFrame:
                raise RuntimeError("Network error")

        cfg = SMCConfig(
            env="paper",
            mt5_mock=True,
            data_dir=tmp_path / "data",
            telegram_bot_token="",
            telegram_chat_id="",
        )
        loop = LiveLoop(
            config=cfg,
            broker=StubBroker(),
            data_fetcher=FailingFetcher(),
            strategy=StubStrategy(),
            instrument="XAUUSD",
            journal_dir=tmp_path / "journal",
        )
        bar_ts = datetime.now(tz=timezone.utc)
        # Should not raise — errors are caught and logged
        result = asyncio.run(loop.run_single_cycle(bar_ts=bar_ts))
        assert isinstance(result, CycleLog)

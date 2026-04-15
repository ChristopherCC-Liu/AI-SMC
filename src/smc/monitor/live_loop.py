"""Live trading loop for the AI-SMC system.

The loop polls MT5 every M15 bar close, runs the strategy pipeline,
executes trades via the risk/execution layer, and monitors health.

This module provides :class:`LiveLoop` — the main entry point is
:meth:`LiveLoop.run`, an async method that runs indefinitely until
interrupted.

Architecture
------------
The loop is deliberately synchronous in its trading logic (one cycle
per bar close), but uses asyncio for:
- Non-blocking sleep between bars
- Async Telegram alerting
- Graceful Ctrl+C shutdown

The loop does NOT run multiple cycles concurrently — this prevents
race conditions on position state.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from smc.config import SMCConfig
from smc.data.schemas import Timeframe
from smc.monitor.alerter import TelegramAlerter
from smc.monitor.health import HealthMonitor
from smc.monitor.journal import TradeJournal
from smc.monitor.timing import next_bar_close, wait_for_bar_close
from smc.monitor.types import CycleLog, JournalEntry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Broker port protocol — matches execution module's interface
# ---------------------------------------------------------------------------


class BrokerPort(Protocol):
    """Minimal broker interface the live loop needs."""

    def get_account_info(self) -> dict[str, Any]: ...
    def get_positions(self, symbol: str) -> list[dict[str, Any]]: ...
    def get_current_price(self, symbol: str) -> float | None: ...


# ---------------------------------------------------------------------------
# Data fetcher protocol
# ---------------------------------------------------------------------------


class DataFetcher(Protocol):
    """Fetches OHLCV data for the live loop."""

    def fetch(
        self,
        *,
        instrument: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame: ...


# ---------------------------------------------------------------------------
# Strategy runner protocol
# ---------------------------------------------------------------------------


class StrategyRunner(Protocol):
    """Runs the strategy pipeline and returns trade setups."""

    def generate_setups(
        self,
        data: dict[Timeframe, pl.DataFrame],
        current_price: float,
        bar_ts: datetime | None = None,
    ) -> tuple[Any, ...]: ...


# ---------------------------------------------------------------------------
# Live Loop
# ---------------------------------------------------------------------------


class LiveLoop:
    """Main live trading loop.

    Parameters
    ----------
    config:
        Application configuration.
    broker:
        Broker port for account info and position queries.
    data_fetcher:
        Adapter for fetching OHLCV bars.
    strategy:
        Strategy aggregator for generating trade setups.
    instrument:
        Trading instrument symbol.
    journal_dir:
        Directory for the trade journal.  Defaults to ``./logs/journal``.
    """

    # Lookback periods for each timeframe (in bars)
    _LOOKBACK_BARS: dict[Timeframe, int] = {
        Timeframe.D1: 60,
        Timeframe.H4: 120,
        Timeframe.H1: 200,
        Timeframe.M15: 300,
    }

    def __init__(
        self,
        *,
        config: SMCConfig,
        broker: BrokerPort,
        data_fetcher: DataFetcher,
        strategy: StrategyRunner,
        instrument: str = "XAUUSD",
        journal_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._broker = broker
        self._data_fetcher = data_fetcher
        self._strategy = strategy
        self._instrument = instrument

        # Monitor components
        self._health_monitor = HealthMonitor(
            max_daily_loss_pct=config.max_daily_loss_pct,
        )
        self._journal = TradeJournal(
            journal_dir or Path("./logs/journal"),
        )
        self._alerter = TelegramAlerter(
            bot_token=config.telegram_bot_token.get_secret_value(),
            chat_id=config.telegram_chat_id,
        )

        # State
        self._cycle_count: int = 0
        self._running: bool = False
        self._daily_start_balance: float = 0.0
        self._last_bar_ts: datetime | None = None

    @property
    def cycle_count(self) -> int:
        """Number of cycles completed."""
        return self._cycle_count

    @property
    def running(self) -> bool:
        """Whether the loop is currently running."""
        return self._running

    @property
    def journal(self) -> TradeJournal:
        """The trade journal instance."""
        return self._journal

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run the live trading loop until interrupted.

        Handles SIGINT / SIGTERM for graceful shutdown.
        """
        self._running = True
        loop = asyncio.get_event_loop()

        # Register signal handlers for graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._request_stop)

        logger.info(
            "Live loop starting — instrument=%s, mode=%s",
            self._instrument,
            self._config.env,
        )

        try:
            # Get initial balance for daily loss tracking
            account = self._broker.get_account_info()
            self._daily_start_balance = account.get("balance", 0.0)

            while self._running:
                await self._run_cycle()
        except asyncio.CancelledError:
            logger.info("Live loop cancelled")
        finally:
            self._running = False
            logger.info("Live loop stopped after %d cycles", self._cycle_count)

    async def run_single_cycle(self, bar_ts: datetime | None = None) -> CycleLog:
        """Run a single cycle — useful for testing.

        Parameters
        ----------
        bar_ts:
            Override the bar close timestamp (for deterministic tests).

        Returns
        -------
        CycleLog
            Record of what happened in the cycle.
        """
        if self._daily_start_balance <= 0:
            account = self._broker.get_account_info()
            self._daily_start_balance = account.get("balance", 0.0)

        return await self._execute_cycle(bar_ts)

    # ------------------------------------------------------------------
    # Cycle implementation
    # ------------------------------------------------------------------

    async def _run_cycle(self) -> None:
        """Wait for bar close, then execute one cycle."""
        # Wait for next M15 bar close
        bar_ts = await wait_for_bar_close(Timeframe.M15)
        await self._execute_cycle(bar_ts)

    async def _execute_cycle(self, bar_ts: datetime | None = None) -> CycleLog:
        """Execute one full cycle of the trading loop."""
        cycle_start = time.monotonic()
        self._cycle_count += 1

        if bar_ts is None:
            bar_ts = datetime.now(tz=timezone.utc)

        setups_generated = 0
        orders_placed = 0
        positions_managed = 0
        health_ok = True

        try:
            # 1. Fetch latest bars
            data = self._fetch_multi_timeframe(bar_ts)

            # 2. Get current price
            current_price = self._broker.get_current_price(self._instrument)
            if current_price is None:
                logger.warning("Cannot get current price for %s", self._instrument)
                current_price = self._get_last_close(data)

            # 3. Run strategy pipeline
            setups = self._strategy.generate_setups(data, current_price, bar_ts=bar_ts)
            setups_generated = len(setups)

            if setups:
                logger.info(
                    "Cycle %d: %d setup(s) generated at %s",
                    self._cycle_count,
                    setups_generated,
                    bar_ts.isoformat(),
                )

            # 4. For each setup: log intent (execution handled by order_manager)
            # NOTE: actual order execution is delegated to the execution module
            # when it's wired in.  For now, we log the setups.
            for setup in setups:
                self._journal.log_action(JournalEntry(
                    ts=bar_ts,
                    action="open",
                    ticket=0,
                    instrument=self._instrument,
                    direction=setup.entry_signal.direction,
                    lots=0.0,  # Populated by position sizer
                    price=setup.entry_signal.entry_price,
                    sl=setup.entry_signal.stop_loss,
                    tp=setup.entry_signal.take_profit_1,
                    pnl=0.0,
                    balance_after=self._daily_start_balance,
                    setup_confluence=setup.confluence_score,
                    trigger_type=setup.entry_signal.trigger_type,
                    regime="unknown",
                ))
                orders_placed += 1

            # 5. Manage open positions
            positions = self._broker.get_positions(self._instrument)
            positions_managed = len(positions)

            # 6. Health check
            account = self._broker.get_account_info()
            daily_pnl = account.get("balance", 0.0) - self._daily_start_balance
            daily_pnl_pct = (daily_pnl / self._daily_start_balance * 100.0) if self._daily_start_balance > 0 else 0.0

            health = self._health_monitor.check_all(
                broker_connected=True,
                last_bar_ts=bar_ts,
                margin_level_pct=account.get("margin_level"),
                unreconciled_count=0,
                daily_pnl_pct=daily_pnl_pct,
            )
            health_ok = health.all_ok

            if not health.all_ok:
                failed = [c.detail for c in health.checks if not c.passed]
                logger.warning("Health check failed: %s", failed)
                await self._alerter.send_health_alert(failed_checks=failed)

            # 7. Log cycle heartbeat
            self._journal.log_cycle(
                bar_close_ts=bar_ts,
                instrument=self._instrument,
                regime="unknown",
                balance=account.get("balance", 0.0),
            )

            self._last_bar_ts = bar_ts

        except Exception:
            logger.exception("Cycle %d failed", self._cycle_count)
            health_ok = False

        duration = time.monotonic() - cycle_start

        return CycleLog(
            cycle_number=self._cycle_count,
            bar_close_ts=bar_ts,
            setups_generated=setups_generated,
            orders_placed=orders_placed,
            positions_managed=positions_managed,
            health_ok=health_ok,
            duration_seconds=round(duration, 3),
        )

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_multi_timeframe(self, bar_ts: datetime) -> dict[Timeframe, pl.DataFrame]:
        """Fetch OHLCV data for all required timeframes."""
        data: dict[Timeframe, pl.DataFrame] = {}

        for tf, lookback_bars in self._LOOKBACK_BARS.items():
            bar_minutes = {
                Timeframe.D1: 1440,
                Timeframe.H4: 240,
                Timeframe.H1: 60,
                Timeframe.M15: 15,
            }[tf]

            start = bar_ts - timedelta(minutes=bar_minutes * lookback_bars)
            try:
                df = self._data_fetcher.fetch(
                    instrument=self._instrument,
                    timeframe=tf,
                    start=start,
                    end=bar_ts,
                )
                data[tf] = df
            except Exception:
                logger.warning("Failed to fetch %s data", tf.value)

        return data

    @staticmethod
    def _get_last_close(data: dict[Timeframe, pl.DataFrame]) -> float:
        """Extract the last close price from available data."""
        for tf in (Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1):
            df = data.get(tf)
            if df is not None and len(df) > 0:
                return float(df["close"][-1])
        return 0.0

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _request_stop(self) -> None:
        """Signal the loop to stop after the current cycle."""
        logger.info("Shutdown requested — finishing current cycle")
        self._running = False


__all__ = ["LiveLoop", "BrokerPort", "DataFetcher", "StrategyRunner"]

"""Bar-by-bar backtest simulation engine.

Walks M15 bars chronologically and processes trade setups with strict
no-look-ahead semantics: a signal at bar[t] uses only data available
at bar[t]'s open price.

The engine tracks open positions, applies fill simulation, updates
the equity curve, and produces a complete BacktestResult.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

import polars as pl

from smc.backtest import metrics
from smc.backtest.fills import BarOHLC, FillModel
from smc.backtest.types import (
    BacktestConfig,
    BacktestResult,
    EquityCurve,
    TradeRecord,
)
from smc.smc_core.constants import XAUUSD_POINT_SIZE


# ---------------------------------------------------------------------------
# TradeSetup protocol — compatible with strategy-lead's type
# ---------------------------------------------------------------------------


@runtime_checkable
class EntrySignalLike(Protocol):
    """Minimal protocol for an entry signal from the strategy layer."""

    @property
    def entry_price(self) -> float: ...
    @property
    def stop_loss(self) -> float: ...
    @property
    def take_profit_1(self) -> float: ...
    @property
    def take_profit_2(self) -> float | None: ...
    @property
    def direction(self) -> str: ...


@runtime_checkable
class TradeSetupLike(Protocol):
    """Minimal protocol for a trade setup from the strategy layer."""

    @property
    def entry_signal(self) -> EntrySignalLike: ...
    @property
    def confluence_score(self) -> float: ...


# ---------------------------------------------------------------------------
# Internal open-position tracker (mutable during simulation only)
# ---------------------------------------------------------------------------


class _OpenPosition:
    """Mutable tracker for a single open position during simulation."""

    __slots__ = (
        "open_ts",
        "fill_price",
        "direction",
        "lots",
        "sl",
        "tp1",
        "tp2",
        "confluence",
        "trigger_type",
    )

    def __init__(
        self,
        open_ts: datetime,
        fill_price: float,
        direction: str,
        lots: float,
        sl: float,
        tp1: float,
        tp2: float | None,
        confluence: float,
        trigger_type: str,
    ) -> None:
        self.open_ts = open_ts
        self.fill_price = fill_price
        self.direction = direction
        self.lots = lots
        self.sl = sl
        self.tp1 = tp1
        self.tp2 = tp2
        self.confluence = confluence
        self.trigger_type = trigger_type


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BarBacktestEngine:
    """Bar-by-bar backtest engine with no-look-ahead guarantee.

    Args:
        config: Backtest configuration (balance, spread, slippage, etc.).
        fill_model: Fill model for simulating entry/exit prices.
    """

    def __init__(self, config: BacktestConfig, fill_model: FillModel) -> None:
        self._config = config
        self._fill_model = fill_model

    @property
    def config(self) -> BacktestConfig:
        return self._config

    def run(
        self,
        setups_by_bar: dict[datetime, tuple[TradeSetupLike, ...]],
        bars: pl.DataFrame,
    ) -> BacktestResult:
        """Run the backtest over a chronological sequence of M15 bars.

        Processing order per bar:
            1. Check open trade exits against this bar's OHLC
            2. Process new setups that fired at this bar's timestamp
            3. Apply fills for accepted setups
            4. Update equity curve

        Args:
            setups_by_bar: Mapping from bar timestamp to trade setups
                generated *at* that bar (using only prior data).
            bars: Polars DataFrame with columns: ts, open, high, low, close.
                Must be sorted ascending by ts.

        Returns:
            Complete BacktestResult with trades, equity curve, and metrics.
        """
        balance = self._config.initial_balance
        open_positions: list[_OpenPosition] = []
        closed_trades: list[TradeRecord] = []

        ts_list: list[datetime] = []
        equity_list: list[float] = []
        dd_list: list[float] = []
        peak_equity = balance

        # Extract columns as Python lists for fast iteration
        ts_col = bars["ts"].to_list()
        open_col = bars["open"].to_list()
        high_col = bars["high"].to_list()
        low_col = bars["low"].to_list()
        close_col = bars["close"].to_list()

        for i in range(len(ts_col)):
            bar_ts = ts_col[i]
            bar = BarOHLC(
                open=open_col[i],
                high=high_col[i],
                low=low_col[i],
                close=close_col[i],
            )

            # --- Step 1: Check exits on open positions ---
            still_open: list[_OpenPosition] = []
            for pos in open_positions:
                exit_result = self._fill_model.check_exit(
                    direction=pos.direction,  # type: ignore[arg-type]
                    entry_price=pos.fill_price,
                    sl=pos.sl,
                    tp1=pos.tp1,
                    tp2=pos.tp2,
                    bar=bar,
                )
                if exit_result is not None:
                    pnl = self._compute_pnl(
                        pos.direction, pos.fill_price, exit_result.exit_price, pos.lots
                    )
                    pnl_pct = pnl / balance if balance > 0.0 else 0.0
                    balance += pnl
                    closed_trades.append(
                        TradeRecord(
                            open_ts=pos.open_ts,
                            open_price=pos.fill_price,
                            direction=pos.direction,  # type: ignore[arg-type]
                            close_ts=bar_ts,
                            close_price=exit_result.exit_price,
                            lots=pos.lots,
                            pnl_usd=pnl,
                            pnl_pct=pnl_pct,
                            close_reason=exit_result.reason,
                            setup_confluence=pos.confluence,
                            trigger_type=pos.trigger_type,
                        )
                    )
                else:
                    still_open.append(pos)
            open_positions = still_open

            # --- Step 2 & 3: Process new setups and apply fills ---
            bar_setups = setups_by_bar.get(bar_ts, ())
            for setup in bar_setups:
                if len(open_positions) >= self._config.max_concurrent_trades:
                    break

                sig = setup.entry_signal
                fill_price = self._fill_model.simulate_fill(
                    direction=sig.direction,  # type: ignore[arg-type]
                    entry_price=sig.entry_price,
                    bar=bar,
                )
                if fill_price is None:
                    continue

                # Default lot sizing: 0.01 lots (micro lot)
                lots = 0.01

                # Deduct commission
                commission = self._fill_model.commission_per_lot * lots
                balance -= commission

                open_positions.append(
                    _OpenPosition(
                        open_ts=bar_ts,
                        fill_price=fill_price,
                        direction=sig.direction,
                        lots=lots,
                        sl=sig.stop_loss,
                        tp1=sig.take_profit_1,
                        tp2=sig.take_profit_2,
                        confluence=setup.confluence_score,
                        trigger_type=getattr(sig, "trigger_type", type(setup).__name__),
                    )
                )

            # --- Step 4: Update equity curve ---
            # Mark-to-market: include unrealised P&L of open positions
            unrealised = sum(
                self._compute_pnl(p.direction, p.fill_price, bar.close, p.lots)
                for p in open_positions
            )
            current_equity = balance + unrealised
            peak_equity = max(peak_equity, current_equity)
            dd = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0

            ts_list.append(bar_ts)
            equity_list.append(current_equity)
            dd_list.append(dd)

        # --- Compute bar-level returns for ratio metrics ---
        bar_returns: list[float] = []
        for j in range(1, len(equity_list)):
            prev = equity_list[j - 1]
            ret = (equity_list[j] - prev) / prev if prev > 0.0 else 0.0
            bar_returns.append(ret)

        frozen_trades = tuple(closed_trades)
        equity_curve = EquityCurve(
            timestamps=tuple(ts_list),
            equity=tuple(equity_list),
            drawdown=tuple(dd_list),
        )

        start_date = ts_list[0] if ts_list else datetime.min
        end_date = ts_list[-1] if ts_list else datetime.min

        return BacktestResult(
            config=self._config,
            trades=frozen_trades,
            equity_curve=equity_curve,
            sharpe=metrics.sharpe_ratio(bar_returns),
            sortino=metrics.sortino_ratio(bar_returns),
            calmar=metrics.calmar_ratio(bar_returns),
            max_drawdown_pct=metrics.max_drawdown(tuple(equity_list)),
            profit_factor=metrics.profit_factor(frozen_trades),
            win_rate=metrics.win_rate(frozen_trades),
            expectancy=metrics.expectancy(frozen_trades),
            total_trades=len(frozen_trades),
            start_date=start_date,
            end_date=end_date,
        )

    def _compute_pnl(
        self,
        direction: str,
        entry_price: float,
        exit_price: float,
        lots: float,
    ) -> float:
        """Compute P&L in USD for a single trade.

        For XAUUSD: P&L = (exit - entry) * lots * lot_size * point_value
        where lot_size = 100,000 and 1 point = $0.01.

        Simplified: pnl = direction_sign * (exit - entry) * lots * lot_size
        (since XAUUSD is quoted in USD per troy oz, 1 standard lot = 100 oz,
        and the price difference directly gives USD P&L per oz).
        """
        # For XAUUSD: 1 standard lot = 100 troy oz
        # P&L per oz = exit_price - entry_price (for long)
        # Total P&L = (exit - entry) * lots * 100
        contract_size = 100.0  # 1 lot = 100 troy oz for XAUUSD
        sign = 1.0 if direction == "long" else -1.0
        return sign * (exit_price - entry_price) * lots * contract_size


__all__ = [
    "BarBacktestEngine",
    "TradeSetupLike",
    "EntrySignalLike",
]

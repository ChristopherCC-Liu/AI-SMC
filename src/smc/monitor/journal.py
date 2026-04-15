"""Append-only Parquet trade journal for the AI-SMC live trading system.

The journal records every action (open, close, modify, cycle) as a single
row in a date-partitioned Parquet file.  This provides a durable audit trail
for post-trade analysis and compliance.

File layout::

    {journal_dir}/{YYYY-MM-DD}.parquet

Each file is an append-only Parquet table — new rows are appended by reading
the existing file, concatenating, and writing back.  This is efficient enough
for M15-frequency trading (max ~96 writes/day).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from smc.monitor.types import DailySummary, JournalEntry

# ---------------------------------------------------------------------------
# Schema for the journal Parquet files
# ---------------------------------------------------------------------------

_JOURNAL_SCHEMA = {
    "ts": pl.Datetime("ns", "UTC"),
    "action": pl.String,
    "ticket": pl.Int64,
    "instrument": pl.String,
    "direction": pl.String,
    "lots": pl.Float64,
    "price": pl.Float64,
    "sl": pl.Float64,
    "tp": pl.Float64,
    "pnl": pl.Float64,
    "balance_after": pl.Float64,
    "setup_confluence": pl.Float64,
    "trigger_type": pl.String,
    "regime": pl.String,
}


def _empty_journal_df() -> pl.DataFrame:
    """Return an empty DataFrame matching the journal schema."""
    return pl.DataFrame(
        {col: pl.Series([], dtype=dtype) for col, dtype in _JOURNAL_SCHEMA.items()}
    )


class TradeJournal:
    """Append-only Parquet trade log.

    Parameters
    ----------
    journal_dir:
        Directory where date-partitioned Parquet files are stored.
        Created automatically on first write.
    """

    def __init__(self, journal_dir: Path) -> None:
        self._journal_dir = journal_dir

    @property
    def journal_dir(self) -> Path:
        """The directory where journal files are stored."""
        return self._journal_dir

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log_action(self, entry: JournalEntry) -> None:
        """Append one row to the journal.

        Parameters
        ----------
        entry:
            A :class:`JournalEntry` describing the trade action.
        """
        self._journal_dir.mkdir(parents=True, exist_ok=True)

        entry_date = entry.ts.date()
        file_path = self._parquet_path(entry_date)

        new_row = pl.DataFrame(
            {
                "ts": [entry.ts],
                "action": [entry.action],
                "ticket": [entry.ticket],
                "instrument": [entry.instrument],
                "direction": [entry.direction],
                "lots": [entry.lots],
                "price": [entry.price],
                "sl": [entry.sl],
                "tp": [entry.tp],
                "pnl": [entry.pnl],
                "balance_after": [entry.balance_after],
                "setup_confluence": [entry.setup_confluence],
                "trigger_type": [entry.trigger_type],
                "regime": [entry.regime],
            },
            schema=_JOURNAL_SCHEMA,
        )

        if file_path.exists():
            existing = pl.read_parquet(file_path)
            combined = pl.concat([existing, new_row])
        else:
            combined = new_row

        combined.write_parquet(file_path)

    def log_cycle(
        self,
        *,
        bar_close_ts: datetime,
        instrument: str,
        regime: str,
        balance: float,
    ) -> None:
        """Log a cycle heartbeat entry.

        Used to record that the live loop executed a cycle, even if no
        trades were taken.
        """
        entry = JournalEntry(
            ts=bar_close_ts,
            action="cycle",
            ticket=0,
            instrument=instrument,
            direction="none",
            lots=0.0,
            price=0.0,
            sl=0.0,
            tp=0.0,
            pnl=0.0,
            balance_after=balance,
            setup_confluence=0.0,
            trigger_type="cycle",
            regime=regime,
        )
        self.log_action(entry)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_day(self, day: date) -> pl.DataFrame:
        """Read all journal entries for a specific date.

        Returns an empty DataFrame if no journal file exists for the date.
        """
        file_path = self._parquet_path(day)
        if not file_path.exists():
            return _empty_journal_df()
        return pl.read_parquet(file_path)

    def daily_summary(self, day: date) -> DailySummary:
        """Summarise the day's trading activity.

        Parameters
        ----------
        day:
            The date to summarise.

        Returns
        -------
        DailySummary
            Aggregate statistics for the day.
        """
        df = self.read_day(day)

        # Filter to trade actions only (exclude cycle heartbeats)
        trades = df.filter(pl.col("action").is_in(["close", "sl_hit", "tp_hit", "partial_close"]))

        total_trades = len(trades)
        if total_trades == 0:
            return DailySummary(
                date=day.isoformat(),
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                gross_pnl=0.0,
                net_pnl=0.0,
                max_drawdown_pct=0.0,
                win_rate=0.0,
            )

        pnl_series = trades["pnl"]
        winning = int(pnl_series.filter(pnl_series > 0).len())
        losing = int(pnl_series.filter(pnl_series < 0).len())
        gross_pnl = float(pnl_series.filter(pnl_series > 0).sum())
        net_pnl = float(pnl_series.sum())

        # Compute max drawdown from cumulative PnL
        cum_pnl = pnl_series.cum_sum()
        running_peak = cum_pnl.cum_max()
        drawdowns = cum_pnl - running_peak
        max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0

        # Express drawdown as percentage of first balance_after entry
        balance_col = trades["balance_after"]
        first_balance = float(balance_col[0]) if len(balance_col) > 0 else 1.0
        ref_balance = first_balance - float(pnl_series[0]) if first_balance > 0 else 1.0
        max_dd_pct = abs(max_dd / ref_balance * 100.0) if ref_balance > 0 else 0.0

        win_rate = winning / total_trades if total_trades > 0 else 0.0

        return DailySummary(
            date=day.isoformat(),
            total_trades=total_trades,
            winning_trades=winning,
            losing_trades=losing,
            gross_pnl=round(gross_pnl, 2),
            net_pnl=round(net_pnl, 2),
            max_drawdown_pct=round(max_dd_pct, 4),
            win_rate=round(win_rate, 4),
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _parquet_path(self, day: date) -> Path:
        """Return the Parquet file path for the given date."""
        return self._journal_dir / f"{day.isoformat()}.parquet"


__all__ = ["TradeJournal"]

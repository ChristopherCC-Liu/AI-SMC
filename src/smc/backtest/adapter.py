"""Strategy adapter bridging MultiTimeframeAggregator to StrategyLike protocol.

The walk-forward engine expects a ``StrategyLike`` with ``train()`` and
``generate_setups(bars)`` methods.  The ``MultiTimeframeAggregator``
has a different interface: it takes all 4 timeframe DataFrames and a
current price.

``SMCStrategyAdapter`` wraps the Aggregator, fetches multi-TF data
from the data lake internally, and groups the flat setup output by
bar timestamp to satisfy the engine's contract.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from smc.backtest.engine import TradeSetupLike
from smc.data.lake import ForexDataLake
from smc.data.schemas import Timeframe
from smc.strategy.aggregator import MultiTimeframeAggregator

# Duration of each HTF candle — a bar is only "closed" after ts + duration.
_HTF_DURATION: dict[Timeframe, timedelta] = {
    Timeframe.D1: timedelta(hours=24),
    Timeframe.H4: timedelta(hours=4),
    Timeframe.H1: timedelta(hours=1),
}


class SMCStrategyAdapter:
    """Adapts ``MultiTimeframeAggregator`` to the ``StrategyLike`` protocol.

    The adapter satisfies the walk-forward engine's interface by:
    1. ``train()`` — no-op (SMC strategy is stateless)
    2. ``generate_setups(bars)`` — uses the M15 bars' time range to query
       D1/H4/H1 data from the lake, runs the Aggregator per bar, and
       groups results by timestamp.

    Args:
        aggregator: A configured ``MultiTimeframeAggregator``.
        lake: Data lake for querying HTF OHLCV bars.
        instrument: Instrument symbol (default "XAUUSD").
    """

    def __init__(
        self,
        aggregator: MultiTimeframeAggregator,
        lake: ForexDataLake,
        instrument: str = "XAUUSD",
    ) -> None:
        self._aggregator = aggregator
        self._lake = lake
        self._instrument = instrument

    def train(self, bars: pl.DataFrame) -> None:
        """No-op: SMC strategy is stateless and requires no training."""

    def generate_setups(
        self, bars: pl.DataFrame,
    ) -> dict[datetime, tuple[TradeSetupLike, ...]]:
        """Generate trade setups keyed by M15 bar timestamp.

        For each M15 bar, runs the full Aggregator pipeline using the
        bar's close as the current price and all available HTF data up
        to that point.

        Args:
            bars: M15 OHLCV bars from the walk-forward test window.

        Returns:
            Mapping from bar timestamp to trade setups generated at that bar.
        """
        if bars.is_empty():
            return {}

        # Query HTF data covering the full M15 range
        ts_col = bars["ts"]
        start = ts_col.min()
        end = ts_col.max()

        if start is None or end is None:
            return {}

        # HTF lookback: D1/H4 need ~100+ bars for swing detection.
        # 6 months before test window start provides sufficient context.
        from smc.backtest.walk_forward import _add_months

        htf_lookback_start = _add_months(start, -6)
        htf_data = self._query_htf_data(htf_lookback_start, end)
        m15_bars = bars

        # Run Aggregator once per M15 bar with close as current_price
        result: dict[datetime, tuple[TradeSetupLike, ...]] = {}
        close_col = bars["close"].to_list()
        ts_list = ts_col.to_list()

        for i, bar_ts in enumerate(ts_list):
            current_price = close_col[i]

            data: dict[Timeframe, pl.DataFrame] = {}
            # Filter HTF data: only include bars whose close time <= bar_ts
            for tf, htf_df in htf_data.items():
                duration = _HTF_DURATION.get(tf, timedelta(0))
                data[tf] = htf_df.filter(pl.col("ts") + duration <= bar_ts)
            # Filter M15 data to only bars up to and including current bar
            # (no look-ahead)
            m15_up_to = m15_bars.filter(pl.col("ts") <= bar_ts)
            data[Timeframe.M15] = m15_up_to

            setups = self._aggregator.generate_setups(data, current_price)
            if setups:
                result[bar_ts] = tuple(setups)

        return result

    def _query_htf_data(
        self,
        start: datetime,
        end: datetime,
    ) -> dict[Timeframe, pl.DataFrame]:
        """Query D1, H4, H1 data from the lake for the given range."""
        htf_data: dict[Timeframe, pl.DataFrame] = {}
        for tf in (Timeframe.D1, Timeframe.H4, Timeframe.H1):
            df = self._lake.query(self._instrument, tf, start, end)
            if not df.is_empty():
                htf_data[tf] = df
        return htf_data


__all__ = ["SMCStrategyAdapter"]

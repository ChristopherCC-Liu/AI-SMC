"""Fast strategy adapter for walk-forward OOS validation.

Optimized version of ``SMCStrategyAdapter`` that avoids O(n^2) detection
cost by processing M15 bars in chunks rather than per-bar.

The strategy pipeline (detect -> bias -> zones -> entries) is run once per
chunk on a trailing window of M15 data, then the resulting setups are
mapped to individual bar timestamps. This preserves no-look-ahead
semantics while reducing detection calls from ~6000 to ~375 per OOS window.

Chunk size of 4 bars = 1 hour of M15 data, which aligns with H1
candle boundaries for finer-grained entry evaluation.
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

# Number of M15 bars per chunk (4 bars = 1 hour)
_CHUNK_SIZE = 4

# Trailing window size for M15 detection (keep last N bars for context)
_M15_TRAILING_WINDOW = 500


class FastSMCStrategyAdapter:
    """Optimized adapter for walk-forward backtesting.

    Instead of running the full aggregator pipeline per M15 bar (O(n^2)),
    this adapter:
    1. Pre-queries all HTF data once for the test window
    2. Processes M15 bars in chunks of 4 (every 1 hour)
    3. Uses a trailing window of 500 M15 bars for detection context
    4. Maps setups to the chunk's final bar timestamp

    This reduces detection calls from ~6000 to ~1500 per 3-month OOS window
    while maintaining no-look-ahead correctness.
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
        """No-op: SMC strategy is stateless."""

    def generate_setups(
        self,
        bars: pl.DataFrame,
    ) -> dict[datetime, tuple[TradeSetupLike, ...]]:
        """Generate trade setups using chunked processing.

        For every chunk of 16 M15 bars, runs the aggregator once using
        a trailing window of M15 data plus full HTF data up to that point.
        Setups are keyed to the last bar in each chunk.

        Args:
            bars: M15 OHLCV bars from the walk-forward test window.

        Returns:
            Mapping from bar timestamp to trade setups.
        """
        if bars.is_empty():
            return {}

        ts_col = bars["ts"]
        start = ts_col.min()
        end = ts_col.max()

        if start is None or end is None:
            return {}

        # Pre-query HTF data with lookback for sufficient context.
        # D1 swing detection needs ~100+ bars, H4 needs ~200+ bars.
        # 6 months of lookback provides enough history for all HTF timeframes.
        from smc.backtest.walk_forward import _add_months

        htf_lookback_start = _add_months(start, -6)
        htf_data = self._query_htf_data(htf_lookback_start, end)

        # Extract columns
        close_col = bars["close"].to_list()
        ts_list = ts_col.to_list()
        n_bars = len(ts_list)

        result: dict[datetime, tuple[TradeSetupLike, ...]] = {}

        # Process in chunks
        for chunk_end_idx in range(_CHUNK_SIZE - 1, n_bars, _CHUNK_SIZE):
            bar_ts = ts_list[chunk_end_idx]
            current_price = close_col[chunk_end_idx]

            # Build M15 trailing window (no look-ahead)
            window_start_idx = max(0, chunk_end_idx - _M15_TRAILING_WINDOW + 1)
            m15_window = bars[window_start_idx : chunk_end_idx + 1]

            # Filter HTF data: only include bars whose close time <= bar_ts
            data: dict[Timeframe, pl.DataFrame] = {}
            for tf, df in htf_data.items():
                duration = _HTF_DURATION.get(tf, timedelta(0))
                filtered = df.filter(pl.col("ts") + duration <= bar_ts)
                if not filtered.is_empty():
                    data[tf] = filtered
            data[Timeframe.M15] = m15_window

            setups = self._aggregator.generate_setups(data, current_price)
            if setups:
                result[bar_ts] = tuple(setups)

        # Also process the last partial chunk if it exists
        remainder = n_bars % _CHUNK_SIZE
        if remainder > 0 and n_bars > _CHUNK_SIZE:
            last_idx = n_bars - 1
            if last_idx not in [
                i for i in range(_CHUNK_SIZE - 1, n_bars, _CHUNK_SIZE)
            ]:
                bar_ts = ts_list[last_idx]
                current_price = close_col[last_idx]

                window_start_idx = max(0, last_idx - _M15_TRAILING_WINDOW + 1)
                m15_window = bars[window_start_idx : last_idx + 1]

                data = {}
                for tf, df in htf_data.items():
                    duration = _HTF_DURATION.get(tf, timedelta(0))
                    filtered = df.filter(pl.col("ts") + duration <= bar_ts)
                    if not filtered.is_empty():
                        data[tf] = filtered
                data[Timeframe.M15] = m15_window

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


__all__ = ["FastSMCStrategyAdapter"]

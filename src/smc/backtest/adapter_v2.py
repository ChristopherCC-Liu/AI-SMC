"""V2 fast strategy adapter for walk-forward OOS validation.

Optimized adapter for the v2 strategy pipeline (Sprint 7+8), using:
- ``AggregatorV2`` instead of ``MultiTimeframeAggregator``
- ``DirectionEngine`` with file cache (zero-LLM backtest)
- H4 + H1 + M15 data only (no D1)
- Returns ``TradeSetupV2`` with entry_mode and trigger_type v2

Same chunked M15 approach as ``FastSMCStrategyAdapter`` (v1): processes
bars in chunks of 4 (1 hour) with a trailing window of 500 bars.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from smc.backtest.engine import TradeSetupLike
from smc.data.lake import ForexDataLake
from smc.data.schemas import Timeframe
from smc.strategy.aggregator_v2 import AggregatorV2

__all__ = ["FastSMCStrategyAdapterV2"]

# Duration of each HTF candle — a bar is only "closed" after ts + duration.
_HTF_DURATION: dict[Timeframe, timedelta] = {
    Timeframe.H4: timedelta(hours=4),
    Timeframe.H1: timedelta(hours=1),
}

# Number of M15 bars per chunk (4 bars = 1 hour)
_CHUNK_SIZE = 4

# Trailing window size for M15 detection (keep last N bars for context)
_M15_TRAILING_WINDOW = 500


class FastSMCStrategyAdapterV2:
    """Backtest adapter for v2 strategy pipeline.

    Wraps ``AggregatorV2`` for the walk-forward engine, using the same
    chunked M15 processing approach as the v1 ``FastSMCStrategyAdapter``
    but with v2 modules:
    - AI direction (from ``DirectionEngine`` with cache)
    - Dual entry (normal + inverted signals)
    - Simplified cascade (H4 + H1 + M15, no D1)

    The adapter satisfies ``StrategyLike`` protocol with:
    - ``train()`` — no-op (stateless)
    - ``generate_setups()`` — chunked M15 processing returning setups
    """

    def __init__(
        self,
        aggregator_v2: AggregatorV2,
        lake: ForexDataLake,
        instrument: str = "XAUUSD",
    ) -> None:
        self._aggregator = aggregator_v2
        self._lake = lake
        self._instrument = instrument

    def train(self, bars: pl.DataFrame) -> None:
        """No-op: v2 strategy is stateless."""

    def generate_setups(
        self,
        bars: pl.DataFrame,
    ) -> dict[datetime, tuple[TradeSetupLike, ...]]:
        """Generate trade setups using chunked M15 processing with v2 pipeline.

        For every chunk of 4 M15 bars, runs ``AggregatorV2.generate_setups``
        once using a trailing window of M15 data plus full HTF data up to
        that point. Setups are keyed to the last bar in each chunk.

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
        # H4 swing detection needs ~200+ bars, H1 needs ~100+ bars.
        # 6 months of lookback provides enough history.
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

            setups = self._aggregator.generate_setups(
                data, current_price, bar_ts=bar_ts,
            )
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

                setups = self._aggregator.generate_setups(
                    data, current_price, bar_ts=bar_ts,
                )
                if setups:
                    result[bar_ts] = tuple(setups)

        return result

    def _query_htf_data(
        self,
        start: datetime,
        end: datetime,
    ) -> dict[Timeframe, pl.DataFrame]:
        """Query H4, H1 data from the lake (no D1 in v2)."""
        htf_data: dict[Timeframe, pl.DataFrame] = {}
        for tf in (Timeframe.H4, Timeframe.H1):
            df = self._lake.query(self._instrument, tf, start, end)
            if not df.is_empty():
                htf_data[tf] = df
        return htf_data

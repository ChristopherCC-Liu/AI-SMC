"""Pre-computed direction cache for backtest performance.

During backtesting, calling the LLM-based direction engine per H4 candle
would add hours of latency and risk look-ahead contamination (LLM may
embed future knowledge). This module solves both problems:

1. **Performance**: Pre-computes all directions offline, O(log n) lookup.
2. **No look-ahead**: Uses SMA50 + DXY fallback only — deterministic,
   computed from data available at each historical timestamp.

Usage::

    # 1. Build cache (offline, one-time)
    from smc.ai.direction_cache import build_direction_cache
    build_direction_cache(lake, output_path="data/direction_cache.parquet")

    # 2. Load in backtest runner
    from smc.ai.direction_cache import DirectionCacheLookup
    cache = DirectionCacheLookup("data/direction_cache.parquet")
    direction = cache.lookup(bar_timestamp)

    # 3. Or pass to DirectionEngine via cache_path
    engine = DirectionEngine(cache_path="data/direction_cache.parquet")
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from smc.ai.models import AIDirection

logger = logging.getLogger(__name__)

__all__ = ["DirectionCacheLookup", "build_direction_cache"]


# ---------------------------------------------------------------------------
# Cache lookup
# ---------------------------------------------------------------------------


class DirectionCacheLookup:
    """Fast timestamp-indexed lookup of pre-computed direction assessments.

    Loads a parquet file with columns: ts, direction, confidence,
    key_drivers, reasoning, source.  On ``lookup(timestamp)``, returns
    the most recent assessment at or before the given timestamp.

    The lookup is O(log n) via binary search on the sorted ts column.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        if not path.exists():
            raise FileNotFoundError(f"Direction cache not found: {path}")
        self._df = pl.read_parquet(path).sort("ts")
        self._ts_list: list[datetime] = self._df["ts"].to_list()
        self._n = len(self._ts_list)

        if self._n == 0:
            raise ValueError(f"Direction cache is empty: {path}")

        logger.info(
            "Loaded direction cache: %d entries, %s to %s",
            self._n,
            self._ts_list[0],
            self._ts_list[-1],
        )

    @property
    def size(self) -> int:
        return self._n

    @property
    def date_range(self) -> tuple[datetime, datetime]:
        return self._ts_list[0], self._ts_list[-1]

    def lookup(self, ts: datetime) -> AIDirection | None:
        """Find the most recent direction assessment at or before ``ts``.

        Returns None if ``ts`` is before the first cached entry.
        Uses binary search for O(log n) performance.
        """
        if self._n == 0 or ts < self._ts_list[0]:
            return None

        # Binary search: find rightmost entry where ts_list[i] <= ts
        lo, hi = 0, self._n - 1
        result_idx = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._ts_list[mid] <= ts:
                result_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1

        row = self._df.row(result_idx, named=True)
        return self._row_to_direction(row)

    @staticmethod
    def _row_to_direction(row: dict) -> AIDirection:
        """Convert a parquet row dict back to a frozen AIDirection."""
        # key_drivers stored as JSON string in parquet
        key_drivers_raw = row.get("key_drivers", "[]")
        if isinstance(key_drivers_raw, str):
            import json
            try:
                key_drivers = tuple(json.loads(key_drivers_raw))
            except (json.JSONDecodeError, TypeError):
                key_drivers = ()
        elif isinstance(key_drivers_raw, (list, tuple)):
            key_drivers = tuple(key_drivers_raw)
        else:
            key_drivers = ()

        return AIDirection(
            direction=row["direction"],
            confidence=row["confidence"],
            key_drivers=key_drivers,
            reasoning=row.get("reasoning", ""),
            assessed_at=row["ts"],
            source="cache",
            cost_usd=0.0,
        )


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------


def build_direction_cache(
    lake: "ForexDataLake",
    output_path: str | Path,
    *,
    instrument: str = "XAUUSD",
    frequency_hours: int = 4,
) -> pl.DataFrame:
    """Build direction cache from H4 bars using SMA50 + DXY fallback.

    Walks through the full date range at ``frequency_hours`` intervals,
    computing the deterministic SMA fallback direction at each point.
    No LLM is used — this is safe for backtest (no look-ahead).

    Parameters
    ----------
    lake:
        ForexDataLake instance with H4 data.
    output_path:
        Path to write the parquet file.
    instrument:
        Trading instrument (default XAUUSD).
    frequency_hours:
        Direction computation frequency in hours (default 4 = every H4 candle).

    Returns
    -------
    pl.DataFrame
        The cache DataFrame (also saved to output_path).
    """
    import json as _json

    from smc.ai.direction_engine import DirectionEngine, extract_h4_context
    from smc.ai.external_context import ExternalContextFetcher
    from smc.data.schemas import Timeframe

    h4_range = lake.available_range(instrument, Timeframe.H4)
    if h4_range is None:
        raise RuntimeError(f"No H4 data for {instrument}")

    data_start = h4_range[0]
    data_end = h4_range[1]

    # Need enough bars for SMA50
    min_lookback = timedelta(hours=_SMA50_PERIOD * frequency_hours + 24)
    classify_start = data_start + min_lookback

    step = timedelta(hours=frequency_hours)
    current = classify_start

    rows: list[dict] = []
    total = 0

    logger.info(
        "Building direction cache: %s to %s, step=%dh",
        classify_start.date(),
        data_end.date(),
        frequency_hours,
    )

    while current <= data_end:
        h4_df = lake.query(instrument, Timeframe.H4, data_start, current)

        if h4_df.is_empty():
            current += step
            continue

        h4_ctx = extract_h4_context(h4_df)
        # Use SMA fallback only (no LLM, no external data for backtest purity)
        direction = DirectionEngine._sma_dxy_fallback(h4_ctx, None)

        rows.append({
            "ts": current,
            "direction": direction.direction,
            "confidence": direction.confidence,
            "key_drivers": _json.dumps(list(direction.key_drivers)),
            "reasoning": direction.reasoning,
            "source": direction.source,
        })
        total += 1

        if total % 500 == 0:
            logger.info("  ... %d directions computed", total)

        current += step

    logger.info("Cache complete: %d entries, saving to %s", total, output_path)

    df = pl.DataFrame(rows, schema={
        "ts": pl.Datetime("ns", "UTC"),
        "direction": pl.String,
        "confidence": pl.Float64,
        "key_drivers": pl.String,
        "reasoning": pl.String,
        "source": pl.String,
    })

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output)

    logger.info("Saved %d rows to %s", len(df), output)
    return df


# Module-level constant used by build_direction_cache
_SMA50_PERIOD = 50

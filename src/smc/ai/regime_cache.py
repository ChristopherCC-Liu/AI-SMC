"""Pre-computed regime classification cache for backtest performance.

During walk-forward backtesting, calling the LLM-based regime classifier
per H4 candle would add hours of latency (6,750 calls for 15 windows).
This module solves the problem by pre-computing all classifications offline
and storing them as a time-indexed parquet file.

Usage:

    # 1. Build cache (offline, one-time per prompt/model change)
    from smc.ai.regime_cache import build_regime_cache
    build_regime_cache(lake, output_path="data/regime_cache.parquet")

    # 2. Load cache in backtest runner
    from smc.ai.regime_cache import RegimeCacheLookup
    cache = RegimeCacheLookup("data/regime_cache.parquet")
    assessment = cache.lookup(bar_timestamp)

    # 3. Or pass to classify_regime_ai() as fallback
    from smc.ai.regime_classifier import classify_regime_ai
    assessment = classify_regime_ai(d1, h4, cache=cache)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from smc.ai.models import AIRegimeAssessment, MarketRegimeAI, RegimeParams
from smc.ai.param_router import route
from smc.ai.regime_classifier import classify_regime_ai, extract_regime_context

logger = logging.getLogger(__name__)

__all__ = ["RegimeCacheLookup", "build_regime_cache"]


# ---------------------------------------------------------------------------
# Cache lookup
# ---------------------------------------------------------------------------


class RegimeCacheLookup:
    """Fast timestamp-indexed lookup of pre-computed regime classifications.

    Loads a parquet file with columns: ts, regime, trend_direction,
    confidence, source, reasoning. On ``lookup(timestamp)``, returns
    the most recent assessment at or before the given timestamp.

    The lookup is O(log n) via binary search on the sorted ts column.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._df = pl.read_parquet(self._path).sort("ts")
        self._ts_list: list[datetime] = self._df["ts"].to_list()
        self._n = len(self._ts_list)

        if self._n == 0:
            raise ValueError(f"Regime cache is empty: {path}")

        logger.info(
            "Loaded regime cache: %d entries, %s to %s",
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

    def lookup(self, ts: datetime) -> AIRegimeAssessment | None:
        """Find the most recent regime assessment at or before ``ts``.

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
        return self._row_to_assessment(row)

    @staticmethod
    def _row_to_assessment(row: dict) -> AIRegimeAssessment:
        """Convert a parquet row dict back to a frozen AIRegimeAssessment."""
        regime: MarketRegimeAI = row["regime"]
        return AIRegimeAssessment(
            regime=regime,
            trend_direction=row["trend_direction"],
            confidence=row["confidence"],
            param_preset=route(regime),
            reasoning=row["reasoning"],
            assessed_at=row["ts"],
            source=row["source"],
            cost_usd=0.0,  # cache lookups are free
        )


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------


def build_regime_cache(
    lake: "ForexDataLake",
    output_path: str | Path,
    *,
    instrument: str = "XAUUSD",
    frequency_hours: int = 4,
    ai_enabled: bool = False,
) -> pl.DataFrame:
    """Pre-compute regime classifications and save to parquet.

    Walks through the full date range at ``frequency_hours`` intervals,
    extracting D1/H4 data up to each timestamp and running
    ``classify_regime_ai()``.

    Parameters
    ----------
    lake:
        ForexDataLake instance with D1 and H4 data.
    output_path:
        Path to write the parquet file.
    instrument:
        Trading instrument (default XAUUSD).
    frequency_hours:
        Classification frequency in hours (default 4 = every H4 candle).
    ai_enabled:
        If True, runs the full AI debate pipeline (requires LLM).
        If False (default), uses ATR fallback only — fast, deterministic.

    Returns
    -------
    pl.DataFrame
        The cache DataFrame (also saved to output_path).
    """
    from smc.data.lake import ForexDataLake
    from smc.data.schemas import Timeframe

    d1_range = lake.available_range(instrument, Timeframe.D1)
    h4_range = lake.available_range(instrument, Timeframe.H4)

    if d1_range is None:
        raise RuntimeError(f"No D1 data for {instrument}")

    # Start from when we have enough D1 data for SMA50 + ATR
    data_start = d1_range[0]
    data_end = d1_range[1]

    # Need ~65 D1 bars minimum (50 for SMA50 + 14 for ATR + 1)
    min_lookback = timedelta(days=70)
    classify_start = data_start + min_lookback

    step = timedelta(hours=frequency_hours)
    current = classify_start

    rows: list[dict] = []
    total = 0
    skipped = 0

    logger.info(
        "Building regime cache: %s to %s, step=%dh",
        classify_start.date(),
        data_end.date(),
        frequency_hours,
    )

    while current <= data_end:
        # Query D1 and H4 data up to current timestamp (no look-ahead)
        d1_df = lake.query(instrument, Timeframe.D1, data_start, current)
        h4_df = lake.query(instrument, Timeframe.H4, h4_range[0], current) if h4_range else None

        if d1_df.is_empty():
            current += step
            skipped += 1
            continue

        assessment = classify_regime_ai(
            d1_df, h4_df,
            ai_enabled=ai_enabled,
        )

        rows.append({
            "ts": current,
            "regime": assessment.regime,
            "trend_direction": assessment.trend_direction,
            "confidence": assessment.confidence,
            "source": assessment.source,
            "reasoning": assessment.reasoning,
        })
        total += 1

        if total % 500 == 0:
            logger.info("  ... %d classifications computed", total)

        current += step

    logger.info(
        "Cache complete: %d entries (%d skipped), saving to %s",
        total, skipped, output_path,
    )

    df = pl.DataFrame(rows, schema={
        "ts": pl.Datetime("ns", "UTC"),
        "regime": pl.String,
        "trend_direction": pl.String,
        "confidence": pl.Float64,
        "source": pl.String,
        "reasoning": pl.String,
    })

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output)

    logger.info("Saved %d rows to %s", len(df), output)
    return df

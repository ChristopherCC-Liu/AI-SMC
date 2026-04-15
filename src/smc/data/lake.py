"""Data lake query interface for the AI-SMC OHLCV parquet store.

:class:`ForexDataLake` provides read-only access to the partitioned parquet
hierarchy written by :func:`~smc.data.writers.write_forex_partitioned`::

    {root}/{instrument}/{timeframe}/{yyyy}/{mm}.parquet

All returned :class:`polars.DataFrame` instances are sorted ascending by ``ts``
and filtered to the requested ``[start, end)`` interval.

Example::

    from pathlib import Path
    from datetime import datetime, timezone
    from smc.data.lake import ForexDataLake
    from smc.data.schemas import Timeframe

    lake = ForexDataLake(Path("data/lake"))
    df = lake.query(
        "XAUUSD",
        Timeframe.H1,
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from smc.data.schemas import Timeframe


class ForexDataLake:
    """Read-only query interface over a partitioned Parquet data lake.

    The lake layout expected::

        {root}/{instrument}/{timeframe}/{yyyy}/{mm}.parquet

    The class does not cache file handles; each call to :meth:`query` opens
    and closes parquet files independently, making it safe for concurrent use.

    Args:
        root: Base directory of the data lake.  Must already exist when
            performing reads (it is created lazily by writers).
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _instrument_timeframe_root(self, instrument: str, timeframe: Timeframe) -> Path:
        return self._root / instrument / str(timeframe)

    def _collect_parquet_files(
        self,
        instrument: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Path]:
        """Return sorted list of parquet files that may overlap [start, end).

        Uses directory names (year/month) for a coarse pre-filter to avoid
        opening every partition when the date range is narrow.
        """
        base = self._instrument_timeframe_root(instrument, timeframe)
        if not base.exists():
            return []

        files: list[Path] = []
        for year_dir in sorted(base.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue

            # Coarse year filter
            if start is not None and year < start.year:
                continue
            if end is not None and year > end.year:
                continue

            for month_file in sorted(year_dir.iterdir()):
                if month_file.suffix != ".parquet":
                    continue
                try:
                    month = int(month_file.stem)
                except ValueError:
                    continue

                # Coarse month filter
                if start is not None and (year < start.year or (year == start.year and month < start.month)):
                    continue
                if end is not None and (year > end.year or (year == end.year and month > end.month)):
                    continue

                files.append(month_file)

        return sorted(files)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        instrument: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Read OHLCV bars for *instrument*/*timeframe* in ``[start, end)``.

        Args:
            instrument: Symbol name, e.g. ``"XAUUSD"``.
            timeframe: One of the :class:`~smc.data.schemas.Timeframe` values.
            start: Inclusive lower bound (tz-aware, UTC).
            end: Exclusive upper bound (tz-aware, UTC).

        Returns:
            :class:`polars.DataFrame` sorted ascending by ``ts``.  Returns an
            empty DataFrame (with the correct schema) if no data is found.

        Raises:
            ValueError: if *start* or *end* is tz-naive.
        """
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("'start' and 'end' must be tz-aware datetimes.")

        start_utc = start.astimezone(timezone.utc)
        end_utc = end.astimezone(timezone.utc)

        files = self._collect_parquet_files(instrument, timeframe, start_utc, end_utc)
        if not files:
            return _empty_frame()

        frames = [pl.read_parquet(f) for f in files]
        combined = pl.concat(frames)

        # Ensure ts column is Datetime with UTC tz for filtering
        if combined.schema["ts"] != pl.Datetime("ns", "UTC"):
            combined = combined.with_columns(
                pl.col("ts").dt.replace_time_zone("UTC").alias("ts")
            )

        start_pl = pl.lit(start_utc).cast(pl.Datetime("ns", "UTC"))
        end_pl = pl.lit(end_utc).cast(pl.Datetime("ns", "UTC"))

        filtered = combined.filter(
            (pl.col("ts") >= start_pl) & (pl.col("ts") < end_pl)
        )
        return filtered.sort("ts")

    def available_range(
        self,
        instrument: str,
        timeframe: Timeframe,
    ) -> tuple[datetime, datetime] | None:
        """Return the ``(min_ts, max_ts)`` available for *instrument*/*timeframe*.

        Returns:
            A ``(min, max)`` tuple of tz-aware UTC datetimes, or ``None`` if no
            data exists for the given combination.
        """
        files = self._collect_parquet_files(instrument, timeframe)
        if not files:
            return None

        ts_min: datetime | None = None
        ts_max: datetime | None = None

        for f in files:
            # Read only the ts column to keep memory low
            df = pl.read_parquet(f, columns=["ts"])
            if df.is_empty():
                continue
            local_min = df["ts"].min()
            local_max = df["ts"].max()
            if local_min is not None:
                local_min_dt = _to_utc_datetime(local_min)
                ts_min = local_min_dt if ts_min is None else min(ts_min, local_min_dt)
            if local_max is not None:
                local_max_dt = _to_utc_datetime(local_max)
                ts_max = local_max_dt if ts_max is None else max(ts_max, local_max_dt)

        if ts_min is None or ts_max is None:
            return None
        return (ts_min, ts_max)

    def row_count(self, instrument: str, timeframe: Timeframe) -> int:
        """Return the total number of rows for *instrument*/*timeframe*.

        Returns:
            Integer row count, or ``0`` if no data exists.
        """
        files = self._collect_parquet_files(instrument, timeframe)
        total = 0
        for f in files:
            # Use metadata to avoid loading data
            import pyarrow.parquet as pq_meta
            meta = pq_meta.read_metadata(f)
            total += meta.num_rows
        return total

    def list_instruments(self) -> list[str]:
        """Return a sorted list of instrument names present in the lake.

        Returns:
            List of instrument directory names, e.g. ``["EURUSD", "XAUUSD"]``.
        """
        if not self._root.exists():
            return []
        instruments = [
            d.name
            for d in sorted(self._root.iterdir())
            if d.is_dir()
        ]
        return instruments


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _empty_frame() -> pl.DataFrame:
    """Return an empty Polars DataFrame matching the OHLCV schema."""
    return pl.DataFrame(
        {
            "ts": pl.Series([], dtype=pl.Datetime("ns", "UTC")),
            "open": pl.Series([], dtype=pl.Float64),
            "high": pl.Series([], dtype=pl.Float64),
            "low": pl.Series([], dtype=pl.Float64),
            "close": pl.Series([], dtype=pl.Float64),
            "volume": pl.Series([], dtype=pl.Float64),
            "spread": pl.Series([], dtype=pl.Float64),
            "timeframe": pl.Series([], dtype=pl.String),
            "source": pl.Series([], dtype=pl.String),
            "schema_version": pl.Series([], dtype=pl.Int32),
        }
    )


def _to_utc_datetime(value: object) -> datetime:
    """Convert a Polars datetime value to a Python tz-aware UTC datetime."""
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    # Polars may return an int (epoch ns) when the series has no tz info
    if isinstance(value, int):
        return datetime.fromtimestamp(value / 1e9, tz=timezone.utc)
    raise TypeError(f"Cannot convert {type(value)!r} to datetime.")


__all__ = ["ForexDataLake"]

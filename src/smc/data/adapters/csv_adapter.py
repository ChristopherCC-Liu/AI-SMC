"""CSV bootstrap adapter for MT5-exported CSV files.

MetaTrader 5 exports history as tab- or comma-separated files with a fixed
header.  This adapter ingests those files into a :class:`polars.DataFrame`
that conforms to the FOREX OHLCV schema.

Expected MT5 CSV format
-----------------------
::

    <DATE>	<TIME>	<OPEN>	<HIGH>	<LOW>	<CLOSE>	<TICKVOL>	<VOL>	<SPREAD>

Column aliases handled:
- ``Date`` / ``<DATE>`` / ``date``
- ``Time`` / ``<TIME>`` / ``time``
- ``Open``  / ``<OPEN>``  / ``open``
- ``High``  / ``<HIGH>``  / ``high``
- ``Low``   / ``<LOW>``   / ``low``
- ``Close`` / ``<CLOSE>`` / ``close``
- ``Tickvol`` / ``<TICKVOL>`` / ``tickvol`` → mapped to ``volume``
- ``Vol``     / ``<VOL>``     / ``vol``     (ignored; tickvol is used)
- ``Spread``  / ``<SPREAD>``  / ``spread``

Directory layout expected by :meth:`CSVAdapter.fetch`::

    {csv_dir}/{timeframe}/{instrument}.csv

For example::

    data/csv/H1/XAUUSD.csv
    data/csv/M15/XAUUSD.csv

Usage::

    from pathlib import Path
    from datetime import datetime, timezone
    from smc.data.adapters.csv_adapter import CSVAdapter
    from smc.data.schemas import Timeframe

    adapter = CSVAdapter(csv_dir=Path("data/csv"), instrument="XAUUSD")
    df = adapter.fetch(
        instrument="XAUUSD",
        timeframe=Timeframe.H1,
        start=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from smc.data.adapters.base import ForexAdapter, ForexAdapterError, ForexAdapterSpec
from smc.data.schemas import SCHEMA_VERSION, Timeframe

# Canonical column name mapping: lower-case stripped alias -> canonical name
_ALIAS_MAP: dict[str, str] = {
    # date
    "date": "date",
    "<date>": "date",
    # time
    "time": "time",
    "<time>": "time",
    # ohlc
    "open": "open",
    "<open>": "open",
    "high": "high",
    "<high>": "high",
    "low": "low",
    "<low>": "low",
    "close": "close",
    "<close>": "close",
    # volume  (tickvol takes priority over vol)
    "tickvol": "volume",
    "<tickvol>": "volume",
    "vol": "_vol_raw",
    "<vol>": "_vol_raw",
    # spread
    "spread": "spread",
    "<spread>": "spread",
}

_REQUIRED_CANONICAL: frozenset[str] = frozenset(
    {"date", "time", "open", "high", "low", "close", "volume", "spread"}
)


class CSVAdapter:
    """Adapter that reads MT5-exported CSV files from a local directory.

    The adapter is *stateless* — it reads from disk on every :meth:`fetch`
    call and does not cache parsed DataFrames.

    Args:
        csv_dir: Root directory organised as ``{csv_dir}/{timeframe}/{instrument}.csv``.
        instrument: Default instrument; used to populate the ``source`` column.
        source_name: Source label written into the ``source`` column
            (default: ``"csv"``).
    """

    def __init__(
        self,
        csv_dir: Path,
        instrument: str,
        source_name: str = "csv",
    ) -> None:
        self._csv_dir = csv_dir
        self._instrument = instrument
        self._source_name = source_name
        self._spec = ForexAdapterSpec(
            source=source_name,
            instrument=instrument,
            timeframes=tuple(Timeframe),
            description=(
                f"CSV bootstrap adapter reading MT5-exported files "
                f"from {csv_dir}"
            ),
        )

    # ------------------------------------------------------------------
    # ForexAdapter protocol
    # ------------------------------------------------------------------

    @property
    def spec(self) -> ForexAdapterSpec:
        """Immutable adapter specification."""
        return self._spec

    def fetch(
        self,
        *,
        instrument: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Read bars from a CSV file and return an OHLCV DataFrame.

        Args:
            instrument: Symbol to load, e.g. ``"XAUUSD"``.
            timeframe: Timeframe to load.
            start: Inclusive lower bound (tz-aware).
            end: Exclusive upper bound (tz-aware).

        Returns:
            :class:`polars.DataFrame` filtered to ``[start, end)`` and sorted
            ascending by ``ts``.

        Raises:
            ForexAdapterError: if the CSV file is missing, malformed, or
                contains no data for the requested range.
        """
        if start.tzinfo is None or end.tzinfo is None:
            raise ForexAdapterError("'start' and 'end' must be tz-aware datetimes.")

        csv_path = self._resolve_csv_path(instrument, timeframe)
        raw = self._load_csv(csv_path)
        normalised = self._normalise_columns(raw, csv_path)
        df = self._parse_to_ohlcv(normalised, instrument, timeframe)

        start_utc = start.astimezone(timezone.utc)
        end_utc = end.astimezone(timezone.utc)

        start_pl = pl.lit(start_utc).cast(pl.Datetime("ns", "UTC"))
        end_pl = pl.lit(end_utc).cast(pl.Datetime("ns", "UTC"))

        filtered = df.filter(
            (pl.col("ts") >= start_pl) & (pl.col("ts") < end_pl)
        ).sort("ts")

        return filtered

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_csv_path(self, instrument: str, timeframe: Timeframe) -> Path:
        """Return the expected CSV path; raise ForexAdapterError if absent."""
        path = self._csv_dir / str(timeframe) / f"{instrument}.csv"
        if not path.is_file():
            raise ForexAdapterError(
                f"CSV file not found: {path}.  "
                f"Expected layout: {{csv_dir}}/{{timeframe}}/{{instrument}}.csv"
            )
        return path

    def _load_csv(self, path: Path) -> pl.DataFrame:
        """Load a raw CSV, trying tab then comma separators."""
        for sep in ("\t", ",", ";"):
            try:
                df = pl.read_csv(
                    path,
                    separator=sep,
                    infer_schema_length=200,
                    null_values=["", "NULL", "null", "N/A"],
                    ignore_errors=False,
                )
                if df.width >= 7:
                    return df
            except Exception:
                continue
        raise ForexAdapterError(f"Failed to parse CSV file: {path}")

    def _normalise_columns(self, df: pl.DataFrame, path: Path) -> pl.DataFrame:
        """Rename columns to canonical names using ``_ALIAS_MAP``.

        Returns a new DataFrame with renamed columns.
        """
        rename_map: dict[str, str] = {}
        for col in df.columns:
            canonical = _ALIAS_MAP.get(col.strip().lower())
            if canonical is not None:
                rename_map[col] = canonical

        renamed = df.rename(rename_map)

        # Check all required canonical columns are present
        missing = _REQUIRED_CANONICAL - set(renamed.columns)
        if missing:
            raise ForexAdapterError(
                f"CSV file {path} is missing required columns after normalisation: "
                f"{sorted(missing)}.  Found: {renamed.columns}"
            )
        return renamed

    def _parse_to_ohlcv(
        self,
        df: pl.DataFrame,
        instrument: str,
        timeframe: Timeframe,
    ) -> pl.DataFrame:
        """Parse date+time strings into a UTC ``ts`` column and cast types."""
        # Combine Date and Time into a single datetime string
        # MT5 format: "2024.01.02" + "00:00" or "2024-01-02" + "00:00:00"
        combined = df.with_columns(
            (pl.col("date").cast(pl.String) + " " + pl.col("time").cast(pl.String)).alias("_datetime_str")
        )

        # Try to parse the combined datetime string (multiple MT5 formats)
        parsed: pl.DataFrame | None = None
        for fmt in (
            "%Y.%m.%d %H:%M",
            "%Y.%m.%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y %H:%M:%S",
        ):
            try:
                attempt = combined.with_columns(
                    pl.col("_datetime_str")
                    .str.strptime(pl.Datetime("ns"), format=fmt, strict=True)
                    .dt.replace_time_zone("UTC")
                    .alias("ts")
                )
                parsed = attempt
                break
            except Exception:
                continue

        if parsed is None:
            raise ForexAdapterError(
                f"Could not parse datetime column from CSV.  "
                f"Sample value: {combined['_datetime_str'][0]!r}"
            )

        result = (
            parsed
            .select(
                [
                    pl.col("ts"),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Float64),
                    pl.col("spread").cast(pl.Float64),
                ]
            )
            .with_columns(
                [
                    pl.lit(str(timeframe)).alias("timeframe"),
                    pl.lit(f"{self._source_name}:{instrument}").alias("source"),
                    pl.lit(SCHEMA_VERSION).cast(pl.Int32).alias("schema_version"),
                ]
            )
        )
        return result


__all__ = ["CSVAdapter"]

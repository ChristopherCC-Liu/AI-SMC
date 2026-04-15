"""CSV ingest pipeline for the AI-SMC data lake.

Chains CSVAdapter → write_forex_partitioned → build_manifest → write_manifest
into a single ``ingest_csv_files()`` call that returns a :class:`Manifest`.

Usage::

    from pathlib import Path
    from smc.data.ingest import ingest_csv_files
    from smc.data.schemas import Timeframe

    manifest = ingest_csv_files(
        csv_dir=Path("data/csv"),
        instrument="XAUUSD",
        timeframe=Timeframe.H1,
        data_dir=Path("data/lake"),
    )
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from smc.data.adapters.csv_adapter import CSVAdapter
from smc.data.manifest import Manifest, build_manifest, write_manifest
from smc.data.schemas import Timeframe
from smc.data.writers import write_forex_partitioned


def ingest_csv_files(
    *,
    csv_dir: Path,
    instrument: str,
    timeframe: Timeframe,
    data_dir: Path,
) -> Manifest:
    """Ingest CSV files into the partitioned data lake and write a manifest.

    Pipeline steps:
    1. Create a :class:`CSVAdapter` pointed at *csv_dir*.
    2. Fetch all bars for the given *instrument* and *timeframe*.
    3. Write partitioned Parquet files via :func:`write_forex_partitioned`.
    4. Build and persist a :class:`Manifest` recording the ingest.

    Args:
        csv_dir: Directory organised as ``{csv_dir}/{timeframe}/{instrument}.csv``.
        instrument: Symbol name, e.g. ``"XAUUSD"``.
        timeframe: One of the :class:`Timeframe` enum values.
        data_dir: Root of the data lake (parquet + manifests).

    Returns:
        The :class:`Manifest` for the completed ingest.

    Raises:
        ForexAdapterError: if the CSV file is missing or malformed.
        ValueError: if the parsed DataFrame is empty or fails schema validation.
    """
    adapter = CSVAdapter(csv_dir=csv_dir, instrument=instrument)

    # Fetch the full date range — use epoch boundaries to capture everything
    epoch_start = datetime(1970, 1, 2, tzinfo=timezone.utc)
    epoch_end = datetime(2099, 12, 31, tzinfo=timezone.utc)

    df = adapter.fetch(
        instrument=instrument,
        timeframe=timeframe,
        start=epoch_start,
        end=epoch_end,
    )

    if df.is_empty():
        raise ValueError(
            f"No data returned from CSV adapter for {instrument}/{timeframe}."
        )

    # Write partitioned parquet files
    written_files = write_forex_partitioned(
        df,
        instrument=instrument,
        timeframe=timeframe,
        root=data_dir,
    )

    # Extract date range from the ts column
    ts_min_raw = df["ts"].min()
    ts_max_raw = df["ts"].max()

    date_min = _to_utc_datetime(ts_min_raw)
    date_max = _to_utc_datetime(ts_max_raw)

    # Build and write manifest
    source = f"csv:{instrument}"
    manifest = build_manifest(
        source=source,
        source_url=str(csv_dir.resolve()),
        files=sorted(written_files),
        row_count=len(df),
        date_min=date_min,
        date_max=date_max,
    )

    manifests_dir = data_dir / "manifests"
    write_manifest(manifest, manifests_dir)

    return manifest


def _to_utc_datetime(value: object) -> datetime:
    """Convert a Polars datetime value to a Python tz-aware UTC datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    raise TypeError(f"Cannot convert {type(value)!r} to datetime.")


__all__ = ["ingest_csv_files"]

"""Forex-specific partitioned Parquet writer for the AI-SMC data lake.

Partition layout::

    {root}/{instrument}/{timeframe}/{yyyy}/{mm}.parquet

Example::

    data/lake/XAUUSD/H1/2024/01.parquet
    data/lake/XAUUSD/H1/2024/02.parquet

Each partition file contains all bars for that instrument/timeframe/year/month
combination, sorted ascending by ``ts``.

Schema conformance is validated via :func:`~smc.data.schemas.validate_forex_frame`
before any file is written — an invalid frame raises :class:`ValueError` before
touching the filesystem.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from smc.data.manifest import source_to_dir_slug
from smc.data.schemas import FOREX_OHLCV_SCHEMA, Timeframe, validate_forex_frame


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_forex_partitioned(
    df: pl.DataFrame,
    *,
    instrument: str,
    timeframe: Timeframe,
    root: Path,
) -> list[Path]:
    """Write *df* to partitioned Parquet under *root*.

    The frame is partitioned by ``(year, month)`` of the ``ts`` column and
    written to::

        {root}/{instrument}/{timeframe}/{yyyy}/{mm}.parquet

    Existing files for a given partition are **overwritten** — this function
    is idempotent when called with the same data.

    Args:
        df: Polars DataFrame with OHLCV columns.  Must contain a ``ts`` column
            with Polars ``Datetime("ns", "UTC")`` dtype.  Additional columns
            beyond the FOREX schema are accepted but stripped before writing.
        instrument: Symbol name used for the directory hierarchy, e.g.
            ``"XAUUSD"``.
        timeframe: Timeframe enum value, e.g. ``Timeframe.H1``.
        root: Base directory of the data lake.

    Returns:
        List of :class:`Path` objects for every parquet file written, in
        ascending chronological order.

    Raises:
        ValueError: if *df* fails schema validation.
        ValueError: if *df* is empty.
    """
    if df.is_empty():
        raise ValueError("Cannot write an empty DataFrame to the data lake.")

    # Validate schema using PyArrow (convert for validation, then revert)
    arrow_table = df.to_arrow()
    validate_forex_frame(arrow_table)

    # Sort by ts ascending before partitioning
    sorted_df = df.sort("ts")

    # Extract year and month for partitioning (do not mutate the original)
    partitioned = sorted_df.with_columns(
        [
            pl.col("ts").dt.year().alias("_year"),
            pl.col("ts").dt.month().alias("_month"),
        ]
    )

    written_paths: list[Path] = []

    for (year, month), group in partitioned.group_by(["_year", "_month"], maintain_order=True):
        # Drop the helper columns before writing
        group_clean = group.drop(["_year", "_month"])

        # Build partition path
        yyyy = str(int(year)).zfill(4)
        mm = str(int(month)).zfill(2)
        partition_dir = root / instrument / str(timeframe) / yyyy
        partition_dir.mkdir(parents=True, exist_ok=True)
        out_path = partition_dir / f"{mm}.parquet"

        # Write via PyArrow to guarantee schema metadata is preserved
        table = group_clean.to_arrow().cast(FOREX_OHLCV_SCHEMA)
        pq.write_table(
            table,
            out_path,
            compression="snappy",
            write_statistics=True,
        )
        written_paths.append(out_path)

    # Return paths sorted by (year, month)
    written_paths.sort()
    return written_paths


__all__ = [
    "source_to_dir_slug",
    "write_forex_partitioned",
]

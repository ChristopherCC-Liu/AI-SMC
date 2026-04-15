"""Shared internal utilities for smc_core modules.

These helpers are used across multiple sub-detectors.  They are kept in a
single module to avoid copy-paste drift between files.
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

__all__ = [
    "to_aware_utc",
    "ts_from_polars",
]


def to_aware_utc(dt: datetime) -> datetime:
    """Ensure *dt* is timezone-aware (UTC).  If naïve, attach UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def ts_from_polars(ts_series: pl.Series, idx: int) -> datetime:
    """Extract a timezone-aware datetime from a Polars Series at *idx*."""
    ts_raw = ts_series[idx]
    if hasattr(ts_raw, "to_pydatetime"):
        dt = ts_raw.to_pydatetime()
    elif isinstance(ts_raw, datetime):
        dt = ts_raw
    else:
        dt = datetime(ts_raw.year, ts_raw.month, ts_raw.day, tzinfo=timezone.utc)
    return to_aware_utc(dt)

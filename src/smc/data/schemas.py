"""Forex OHLCV PyArrow schema and validation for the AI-SMC data lake.

Every parquet file written to the lake must conform to ``FOREX_OHLCV_SCHEMA``.
The schema is pinned to ``SCHEMA_VERSION = 1`` so that future migrations are
detectable from a manifest hash alone.

All timestamps are stored as ``timestamp[ns, UTC]`` — forex market data is
natively UTC and no timezone conversion is applied on read.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

import pyarrow as pa
import pyarrow.compute as pc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 1
FOREX_TZ: str = "UTC"

# ---------------------------------------------------------------------------
# Timeframe enum
# ---------------------------------------------------------------------------


class Timeframe(StrEnum):
    """Canonical set of MT5 timeframes used by the SMC system."""

    D1 = "D1"
    H4 = "H4"
    H1 = "H1"
    M15 = "M15"
    M5 = "M5"
    M1 = "M1"


# ---------------------------------------------------------------------------
# PyArrow schema
# ---------------------------------------------------------------------------

FOREX_OHLCV_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume", pa.float64(), nullable=False),
        pa.field("spread", pa.float64(), nullable=False),   # in points
        pa.field("timeframe", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("schema_version", pa.int32(), nullable=False),
    ]
)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS: frozenset[str] = frozenset(FOREX_OHLCV_SCHEMA.names)

# PyArrow string (utf8) and large_string (large_utf8) are semantically
# equivalent for our purposes — Polars emits large_string, but the schema
# declares string.  We treat them as compatible in validation so that frames
# produced by Polars pass without a cast round-trip.
_STRING_LIKE: frozenset[pa.DataType] = frozenset({pa.string(), pa.large_string()})


def _types_compatible(actual: pa.DataType, expected: pa.DataType) -> bool:
    """Return True if *actual* is compatible with *expected*.

    Two types are compatible when they are equal or when both belong to the
    string-like group (``pa.string()`` / ``pa.large_string()``).
    """
    if actual == expected:
        return True
    if actual in _STRING_LIKE and expected in _STRING_LIKE:
        return True
    return False


def validate_forex_frame(table: pa.Table) -> None:
    """Validate that *table* conforms to ``FOREX_OHLCV_SCHEMA``.

    Raises:
        ValueError: if any schema or data constraint is violated.

    Checks performed:
    - All required columns are present with the correct types.
    - No future timestamps (UTC now as reference).
    - OHLC price relationships: ``high >= low``, ``high >= open``,
      ``high >= close``, ``low <= open``, ``low <= close``.
    - ``spread >= 0``.
    """
    # --- Column presence & type conformance --------------------------------
    actual_names = set(table.schema.names)
    missing = _REQUIRED_COLUMNS - actual_names
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for field in FOREX_OHLCV_SCHEMA:
        actual_field = table.schema.field(field.name)
        if not _types_compatible(actual_field.type, field.type):
            raise ValueError(
                f"Column '{field.name}' has type {actual_field.type!r}, "
                f"expected {field.type!r}"
            )

    if table.num_rows == 0:
        return

    # --- No future timestamps -----------------------------------------------
    now_ns = int(datetime.now(tz=timezone.utc).timestamp() * 1e9)
    ts_col = table.column("ts").cast(pa.int64())
    max_val = pc.max(ts_col).as_py()
    if max_val is not None and max_val > now_ns:
        raise ValueError(
            "Table contains future timestamps (ts > UTC now)."
        )

    # --- OHLC constraints ---------------------------------------------------
    open_col = table.column("open")
    high_col = table.column("high")
    low_col = table.column("low")
    close_col = table.column("close")
    spread_col = table.column("spread")

    def _any_false(mask: pa.ChunkedArray) -> bool:
        """Return True if any element in a boolean chunked array is False."""
        combined = mask.combine_chunks()
        false_count = pc.sum(pc.equal(combined, False)).as_py()
        return bool(false_count and false_count > 0)

    if _any_false(pc.greater_equal(high_col, low_col)):
        raise ValueError("Constraint violated: high >= low")
    if _any_false(pc.greater_equal(high_col, open_col)):
        raise ValueError("Constraint violated: high >= open")
    if _any_false(pc.greater_equal(high_col, close_col)):
        raise ValueError("Constraint violated: high >= close")
    if _any_false(pc.less_equal(low_col, open_col)):
        raise ValueError("Constraint violated: low <= open")
    if _any_false(pc.less_equal(low_col, close_col)):
        raise ValueError("Constraint violated: low <= close")
    if _any_false(pc.greater_equal(spread_col, pa.scalar(0.0))):
        raise ValueError("Constraint violated: spread >= 0")

"""Unit tests for smc.data.schemas — FOREX_OHLCV_SCHEMA and validate_forex_frame.

Tests cover:
- Valid frame passes without exception
- Missing required column raises ValueError
- Future timestamp raises ValueError
- high < low constraint raises ValueError
- Negative spread raises ValueError
- All canonical Timeframe enum values are valid strings
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from smc.data.schemas import (
    FOREX_OHLCV_SCHEMA,
    SCHEMA_VERSION,
    Timeframe,
    validate_forex_frame,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_table(n: int = 5) -> pa.Table:
    """Build a minimal valid FOREX_OHLCV PyArrow table with `n` rows."""
    rng = np.random.default_rng(0)
    now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    timestamps = [
        (now.timestamp() - (n - i) * 900) * 1e9  # 15-min bars, nanoseconds
        for i in range(n)
    ]
    close_prices = 2350.0 + rng.standard_normal(n) * 2.0

    arrays: dict[str, pa.Array] = {
        "ts": pa.array(
            [int(t) for t in timestamps],
            type=pa.timestamp("ns", tz="UTC"),
        ),
        "open": pa.array(close_prices - rng.uniform(0.1, 0.5, n), type=pa.float64()),
        "high": pa.array(close_prices + rng.uniform(0.5, 2.0, n), type=pa.float64()),
        "low": pa.array(close_prices - rng.uniform(0.5, 2.0, n), type=pa.float64()),
        "close": pa.array(close_prices, type=pa.float64()),
        "volume": pa.array(rng.uniform(100.0, 1000.0, n), type=pa.float64()),
        "spread": pa.array(rng.uniform(2.0, 5.0, n), type=pa.float64()),
        "timeframe": pa.array(["M15"] * n, type=pa.string()),
        "source": pa.array(["test"] * n, type=pa.string()),
        "schema_version": pa.array([SCHEMA_VERSION] * n, type=pa.int32()),
    }
    return pa.table(arrays, schema=FOREX_OHLCV_SCHEMA)


# ---------------------------------------------------------------------------
# Schema & validation tests
# ---------------------------------------------------------------------------


class TestValidFramePasses:
    """A correctly constructed table passes validate_forex_frame without errors."""

    def test_valid_table_no_exception(self) -> None:
        table = _make_valid_table(10)
        validate_forex_frame(table)  # must not raise

    def test_empty_table_no_exception(self) -> None:
        """Empty tables skip data constraints but still need correct columns."""
        empty = _make_valid_table(0)
        validate_forex_frame(empty)  # must not raise

    def test_single_row_table(self) -> None:
        table = _make_valid_table(1)
        validate_forex_frame(table)


class TestMissingColumn:
    """Dropping any required column must raise ValueError."""

    @pytest.mark.parametrize("column", list(FOREX_OHLCV_SCHEMA.names))
    def test_missing_column_raises(self, column: str) -> None:
        table = _make_valid_table(5)
        reduced = table.drop([column])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_forex_frame(reduced)


class TestFutureTimestamp:
    """Rows with ts > UTC now must raise ValueError."""

    def test_future_timestamp_raises(self) -> None:
        table = _make_valid_table(3)

        # Overwrite ts column with a future timestamp (year 2099)
        future_ns = int(datetime(2099, 1, 1, tzinfo=timezone.utc).timestamp() * 1e9)
        future_ts = pa.array([future_ns] * 3, type=pa.timestamp("ns", tz="UTC"))
        table = table.set_column(
            table.schema.get_field_index("ts"),
            "ts",
            future_ts,
        )
        with pytest.raises(ValueError, match="future timestamps"):
            validate_forex_frame(table)

    def test_past_timestamp_ok(self) -> None:
        """A timestamp safely in the past must not raise."""
        table = _make_valid_table(3)
        past_ns = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1e9)
        past_ts = pa.array([past_ns] * 3, type=pa.timestamp("ns", tz="UTC"))
        table = table.set_column(
            table.schema.get_field_index("ts"),
            "ts",
            past_ts,
        )
        validate_forex_frame(table)  # must not raise


class TestHighLowConstraint:
    """high < low in any row must raise ValueError."""

    def test_high_less_than_low_raises(self) -> None:
        table = _make_valid_table(5)
        # Force high < low for the first row by swapping columns
        high_vals = table.column("high").to_pylist()
        low_vals = table.column("low").to_pylist()
        # Make first bar have high < low
        high_vals[0], low_vals[0] = low_vals[0] - 5.0, high_vals[0] + 5.0

        table = table.set_column(
            table.schema.get_field_index("high"),
            "high",
            pa.array(high_vals, type=pa.float64()),
        )
        table = table.set_column(
            table.schema.get_field_index("low"),
            "low",
            pa.array(low_vals, type=pa.float64()),
        )
        with pytest.raises(ValueError, match="high >= low"):
            validate_forex_frame(table)

    def test_high_equals_low_is_ok(self) -> None:
        """A doji bar (high == low == open == close) must not raise."""
        table = _make_valid_table(1)
        price = 2355.0
        for col in ("open", "high", "low", "close"):
            idx = table.schema.get_field_index(col)
            table = table.set_column(
                idx, col, pa.array([price], type=pa.float64())
            )
        validate_forex_frame(table)  # must not raise


class TestNegativeSpread:
    """A negative spread value must raise ValueError."""

    def test_negative_spread_raises(self) -> None:
        table = _make_valid_table(5)
        spreads = [-1.0, 2.5, 3.0, 2.0, 4.0]
        table = table.set_column(
            table.schema.get_field_index("spread"),
            "spread",
            pa.array(spreads, type=pa.float64()),
        )
        with pytest.raises(ValueError, match="spread >= 0"):
            validate_forex_frame(table)

    def test_zero_spread_is_ok(self) -> None:
        """Zero spread (e.g. raw tick data) must not raise."""
        table = _make_valid_table(3)
        spreads = [0.0] * 3
        table = table.set_column(
            table.schema.get_field_index("spread"),
            "spread",
            pa.array(spreads, type=pa.float64()),
        )
        validate_forex_frame(table)  # must not raise


class TestTimeframeEnum:
    """All canonical Timeframe enum members must round-trip via StrEnum."""

    @pytest.mark.parametrize(
        "tf_str",
        ["D1", "H4", "H1", "M15", "M5", "M1"],
    )
    def test_valid_timeframe_string(self, tf_str: str) -> None:
        tf = Timeframe(tf_str)
        assert tf.value == tf_str
        assert str(tf) == tf_str

    def test_invalid_timeframe_raises(self) -> None:
        with pytest.raises(ValueError):
            _ = Timeframe("W1")  # not in the canonical set

    def test_timeframe_in_table(self) -> None:
        """Timeframe value written to a table can be round-tripped via StrEnum."""
        table = _make_valid_table(2)
        tf_col = table.column("timeframe").to_pylist()
        for val in tf_col:
            # Must be constructable from the string in the column
            assert Timeframe(val) == Timeframe.M15


class TestSchemaFieldTypes:
    """FOREX_OHLCV_SCHEMA must declare the exact types the rest of the system depends on."""

    def test_ts_is_timestamp_ns_utc(self) -> None:
        field = FOREX_OHLCV_SCHEMA.field("ts")
        assert field.type == pa.timestamp("ns", tz="UTC")

    def test_ohlcv_are_float64(self) -> None:
        for col in ("open", "high", "low", "close", "volume", "spread"):
            assert FOREX_OHLCV_SCHEMA.field(col).type == pa.float64(), col

    def test_schema_version_is_int32(self) -> None:
        assert FOREX_OHLCV_SCHEMA.field("schema_version").type == pa.int32()

    def test_schema_version_constant(self) -> None:
        assert SCHEMA_VERSION == 1

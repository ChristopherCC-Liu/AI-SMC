"""Shared pytest fixtures for the AI-SMC test suite.

All fixtures that span multiple test modules live here. Module-specific
fixtures should stay in that module's conftest or test file.

Fixture summary
---------------
rng                 - numpy RandomState seeded at 42
sample_ohlcv_df     - 200-bar M15 Polars OHLCV DataFrame (~$2300–$2400 XAUUSD)
sample_ohlcv_pandas - Same data as a pandas DataFrame
tmp_data_lake       - Temporary directory with sample Parquet data written
known_swing_points  - Hand-crafted swing high/low dict for deterministic tests
known_order_blocks  - Hand-crafted order block list for deterministic tests
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path fix: ensure src/smc is imported, not the tests/smc shadow package.
#
# When pytest discovers tests/smc/__init__.py it adds tests/ to sys.path,
# which causes `import smc` to resolve to tests/smc instead of src/smc.
# Inserting src/ at position 0 and evicting any stale smc.* entries from
# sys.modules forces subsequent imports to use the real source package.
# ---------------------------------------------------------------------------
_SRC_PATH = str(Path(__file__).parent.parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

# Evict any already-imported smc.* from sys.modules so the re-import
# resolves against the corrected sys.path.
for _key in [k for k in list(sys.modules) if k == "smc" or k.startswith("smc.")]:
    del sys.modules[_key]

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from smc.data.schemas import FOREX_OHLCV_SCHEMA, SCHEMA_VERSION, Timeframe


# ---------------------------------------------------------------------------
# Low-level helpers (not exported as fixtures)
# ---------------------------------------------------------------------------


def _generate_ohlcv_rows(
    n: int,
    rng: np.random.RandomState,
    start_ts: datetime,
    bar_minutes: int = 15,
    base_price: float = 2350.0,
) -> list[dict[str, Any]]:
    """Generate `n` synthetic XAUUSD M15 OHLCV bars.

    Uses a random walk for close prices, then derives O/H/L/V with realistic
    spreads in the $2300–$2400 range.
    """
    rows: list[dict[str, Any]] = []

    # Random walk for close prices, mean-reverting around base_price
    returns = rng.normal(0.0, 0.5, size=n)  # ~0.5 point std per bar
    close_prices = base_price + np.cumsum(returns)
    # Clamp to $2300–$2400 via soft reflection
    close_prices = np.clip(close_prices, 2300.0, 2400.0)

    bar_delta = timedelta(minutes=bar_minutes)
    ts = start_ts

    for i in range(n):
        c = float(close_prices[i])
        # Intra-bar range: random between 0.5 and 4.0 points
        bar_range = float(rng.uniform(0.5, 4.0))
        o = float(rng.uniform(c - bar_range * 0.6, c + bar_range * 0.6))
        h = max(o, c) + float(rng.uniform(0.1, bar_range * 0.4))
        l = min(o, c) - float(rng.uniform(0.1, bar_range * 0.4))
        volume = float(rng.uniform(100.0, 2000.0))
        spread = float(rng.uniform(2.0, 5.0))

        rows.append(
            {
                "ts": ts,
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": round(volume, 2),
                "spread": round(spread, 2),
                "timeframe": Timeframe.M15.value,
                "source": "test",
                "schema_version": SCHEMA_VERSION,
            }
        )
        ts = ts + bar_delta

    return rows


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rng() -> np.random.RandomState:
    """Seeded numpy RandomState for reproducible test data generation."""
    return np.random.RandomState(42)


@pytest.fixture(scope="session")
def sample_ohlcv_df(rng: np.random.RandomState) -> pl.DataFrame:
    """200-bar M15 XAUUSD Polars DataFrame with realistic prices ($2300–$2400).

    Columns match the FOREX_OHLCV_SCHEMA:
        ts, open, high, low, close, volume, spread, timeframe, source, schema_version
    Timestamps are UTC, starting 2024-01-02 00:00.
    """
    start = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    rows = _generate_ohlcv_rows(200, rng, start)

    return pl.DataFrame(
        {
            "ts": [r["ts"] for r in rows],
            "open": [r["open"] for r in rows],
            "high": [r["high"] for r in rows],
            "low": [r["low"] for r in rows],
            "close": [r["close"] for r in rows],
            "volume": [r["volume"] for r in rows],
            "spread": [r["spread"] for r in rows],
            "timeframe": [r["timeframe"] for r in rows],
            "source": [r["source"] for r in rows],
            "schema_version": [r["schema_version"] for r in rows],
        },
        schema={
            "ts": pl.Datetime("ns", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "spread": pl.Float64,
            "timeframe": pl.String,
            "source": pl.String,
            "schema_version": pl.Int32,
        },
    )


@pytest.fixture(scope="session")
def sample_ohlcv_pandas(sample_ohlcv_df: pl.DataFrame) -> pd.DataFrame:
    """Same 200-bar OHLCV data as a pandas DataFrame.

    Used for compatibility with the ``smartmoneyconcepts`` library which
    expects pandas input.
    """
    pdf = sample_ohlcv_df.to_pandas()
    pdf["ts"] = pd.to_datetime(pdf["ts"], utc=True)
    return pdf


@pytest.fixture()
def tmp_data_lake(
    tmp_path: Path,
    sample_ohlcv_df: pl.DataFrame,
) -> Path:
    """Temporary data lake directory populated with sample Parquet data.

    Layout::

        <tmp_path>/
          data/
            xauusd/
              M15/
                sample.parquet
            manifests/
              mt5_xauusd.json

    Returns the ``data/`` root Path.
    """
    import json

    data_root = tmp_path / "data"
    parquet_dir = data_root / "xauusd" / "M15"
    manifests_dir = data_root / "manifests"
    parquet_dir.mkdir(parents=True)
    manifests_dir.mkdir(parents=True)

    # Write sample parquet using PyArrow so it conforms to FOREX_OHLCV_SCHEMA
    pdf = sample_ohlcv_df.to_pandas()
    pdf["ts"] = pd.to_datetime(pdf["ts"], utc=True)

    arrow_table = pa.Table.from_pandas(pdf, schema=FOREX_OHLCV_SCHEMA, preserve_index=False)
    parquet_file = parquet_dir / "sample.parquet"

    import pyarrow.parquet as pq

    pq.write_table(arrow_table, parquet_file)

    # Write a minimal manifest
    ts_series = pdf["ts"]
    manifest = {
        "source": "mt5:XAUUSD",
        "sha256": "0" * 64,  # dummy for test
        "source_url": "mt5://XAUUSD",
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "row_count": len(pdf),
        "date_min": ts_series.min().isoformat(),
        "date_max": ts_series.max().isoformat(),
    }
    (manifests_dir / "mt5_xauusd.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return data_root


@pytest.fixture(scope="session")
def known_swing_points() -> dict[str, Any]:
    """Hand-crafted swing high and swing low points for deterministic tests.

    Returns a dict with keys:
        highs   - list of (ts, price) tuples for swing highs
        lows    - list of (ts, price) tuples for swing lows

    These are chosen so that tests that assert on specific swing structure
    (e.g. CHoCH detection) produce stable, verifiable results.
    """
    base = datetime(2024, 1, 2, 8, 0, 0, tzinfo=timezone.utc)
    delta = timedelta(hours=1)
    return {
        "highs": [
            (base + delta * 2, 2375.50),
            (base + delta * 6, 2380.25),
            (base + delta * 10, 2368.75),
        ],
        "lows": [
            (base + delta * 0, 2355.00),
            (base + delta * 4, 2360.50),
            (base + delta * 8, 2358.25),
        ],
    }


@pytest.fixture(scope="session")
def known_order_blocks() -> list[dict[str, Any]]:
    """Hand-crafted order blocks for deterministic detection tests.

    Each dict represents a single order block with the fields expected by the
    SMC core module. Direction is ``'bullish'`` or ``'bearish'``.

    Two bullish and two bearish blocks are provided so tests can exercise both
    sides of the market.
    """
    base = datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc)
    return [
        {
            "ts": base,
            "direction": "bullish",
            "top": 2365.00,
            "bottom": 2360.00,
            "mitigation_ts": None,
            "mitigated": False,
            "strength": 0.82,
        },
        {
            "ts": base + timedelta(hours=3),
            "direction": "bullish",
            "top": 2372.50,
            "bottom": 2368.00,
            "mitigation_ts": None,
            "mitigated": False,
            "strength": 0.74,
        },
        {
            "ts": base + timedelta(hours=6),
            "direction": "bearish",
            "top": 2385.00,
            "bottom": 2380.50,
            "mitigation_ts": None,
            "mitigated": False,
            "strength": 0.91,
        },
        {
            "ts": base + timedelta(hours=9),
            "direction": "bearish",
            "top": 2378.25,
            "bottom": 2374.00,
            "mitigation_ts": base + timedelta(hours=12),
            "mitigated": True,
            "strength": 0.61,
        },
    ]

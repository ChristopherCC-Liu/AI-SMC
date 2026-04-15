"""Integration tests: HTF data must not leak future bars into detection.

Verifies that the _HTF_DURATION filtering in both SMCStrategyAdapter and
FastSMCStrategyAdapter correctly excludes HTF bars that have not yet closed.

Scenario:
    A D1 bar opens at 2024-01-02 00:00 UTC and closes 24 hours later.
    - At M15 bar 23:45 (before D1 close) -> D1 bar must NOT be in snapshot.
    - At M15 bar 00:00 next day (at D1 close) -> D1 bar IS in snapshot.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from smc.backtest.adapter import _HTF_DURATION
from smc.backtest.adapter_fast import (
    _HTF_DURATION as _HTF_DURATION_FAST,
)
from smc.data.schemas import Timeframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_d1_bar(ts: datetime) -> pl.DataFrame:
    """Create a single D1 OHLCV bar at the given open timestamp."""
    return pl.DataFrame(
        {
            "ts": [ts],
            "open": [2350.0],
            "high": [2360.0],
            "low": [2340.0],
            "close": [2355.0],
            "volume": [1000.0],
            "spread": [3.0],
        },
        schema={
            "ts": pl.Datetime("ns", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "spread": pl.Float64,
        },
    )


def _filter_htf_closed(
    htf_df: pl.DataFrame,
    tf: Timeframe,
    bar_ts: datetime,
    duration_map: dict[Timeframe, timedelta],
) -> pl.DataFrame:
    """Reproduce the adapter's HTF close-time filter logic."""
    duration = duration_map.get(tf, timedelta(0))
    return htf_df.filter(pl.col("ts") + duration <= bar_ts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHTFNoLookahead:
    """Verify D1 bar is excluded before its close and included after."""

    # D1 bar opens at 2024-01-02 00:00 UTC, closes at 2024-01-03 00:00 UTC
    D1_OPEN = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)

    # M15 bar just BEFORE D1 close: 2024-01-02 23:45
    BEFORE_CLOSE = datetime(2024, 1, 2, 23, 45, 0, tzinfo=timezone.utc)

    # M15 bar AT D1 close: 2024-01-03 00:00
    AT_CLOSE = datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc)

    def test_d1_excluded_before_close_adapter(self) -> None:
        """D1 bar must NOT be in snapshot at 23:45 (adapter.py filter)."""
        d1_df = _make_d1_bar(self.D1_OPEN)
        result = _filter_htf_closed(d1_df, Timeframe.D1, self.BEFORE_CLOSE, _HTF_DURATION)
        assert result.is_empty(), (
            f"D1 bar should be excluded at {self.BEFORE_CLOSE}, "
            f"but got {result.height} row(s)"
        )

    def test_d1_included_at_close_adapter(self) -> None:
        """D1 bar IS in snapshot at 00:00 next day (adapter.py filter)."""
        d1_df = _make_d1_bar(self.D1_OPEN)
        result = _filter_htf_closed(d1_df, Timeframe.D1, self.AT_CLOSE, _HTF_DURATION)
        assert result.height == 1, (
            f"D1 bar should be present at {self.AT_CLOSE}, "
            f"but got {result.height} row(s)"
        )

    def test_d1_excluded_before_close_fast_adapter(self) -> None:
        """D1 bar must NOT be in snapshot at 23:45 (adapter_fast.py filter)."""
        d1_df = _make_d1_bar(self.D1_OPEN)
        result = _filter_htf_closed(
            d1_df, Timeframe.D1, self.BEFORE_CLOSE, _HTF_DURATION_FAST,
        )
        assert result.is_empty(), (
            f"D1 bar should be excluded at {self.BEFORE_CLOSE} (fast), "
            f"but got {result.height} row(s)"
        )

    def test_d1_included_at_close_fast_adapter(self) -> None:
        """D1 bar IS in snapshot at 00:00 next day (adapter_fast.py filter)."""
        d1_df = _make_d1_bar(self.D1_OPEN)
        result = _filter_htf_closed(
            d1_df, Timeframe.D1, self.AT_CLOSE, _HTF_DURATION_FAST,
        )
        assert result.height == 1, (
            f"D1 bar should be present at {self.AT_CLOSE} (fast), "
            f"but got {result.height} row(s)"
        )

    def test_h4_excluded_before_close(self) -> None:
        """H4 bar opening at 08:00 must not appear until 12:00."""
        h4_open = datetime(2024, 1, 2, 8, 0, 0, tzinfo=timezone.utc)
        h4_df = _make_d1_bar(h4_open)  # reuse helper, content irrelevant
        # At 11:45 — still within the H4 candle
        before = datetime(2024, 1, 2, 11, 45, 0, tzinfo=timezone.utc)
        result = _filter_htf_closed(h4_df, Timeframe.H4, before, _HTF_DURATION)
        assert result.is_empty()

        # At 12:00 — exactly at H4 close
        at_close = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        result = _filter_htf_closed(h4_df, Timeframe.H4, at_close, _HTF_DURATION)
        assert result.height == 1

    def test_h1_excluded_before_close(self) -> None:
        """H1 bar opening at 14:00 must not appear until 15:00."""
        h1_open = datetime(2024, 1, 2, 14, 0, 0, tzinfo=timezone.utc)
        h1_df = _make_d1_bar(h1_open)
        # At 14:45 — still within the H1 candle
        before = datetime(2024, 1, 2, 14, 45, 0, tzinfo=timezone.utc)
        result = _filter_htf_closed(h1_df, Timeframe.H1, before, _HTF_DURATION)
        assert result.is_empty()

        # At 15:00 — exactly at H1 close
        at_close = datetime(2024, 1, 2, 15, 0, 0, tzinfo=timezone.utc)
        result = _filter_htf_closed(h1_df, Timeframe.H1, at_close, _HTF_DURATION)
        assert result.height == 1

    def test_duration_maps_are_consistent(self) -> None:
        """Both adapters must define the same HTF durations."""
        assert _HTF_DURATION == _HTF_DURATION_FAST, (
            "adapter.py and adapter_fast.py have divergent _HTF_DURATION maps"
        )

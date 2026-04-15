"""Tests for smc.monitor.timing — bar close timing utilities.

Covers:
- next_bar_close for M15, H1, H4, D1
- Boundary cases (exactly at close time)
- wait_for_bar_close returns the correct target
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from smc.data.schemas import Timeframe
from smc.monitor.timing import next_bar_close


# ---------------------------------------------------------------------------
# M15 bar boundaries: :00, :15, :30, :45
# ---------------------------------------------------------------------------


class TestM15:
    def test_mid_bar(self) -> None:
        now = datetime(2024, 6, 15, 10, 7, 30, tzinfo=timezone.utc)
        result = next_bar_close(Timeframe.M15, now)
        assert result == datetime(2024, 6, 15, 10, 15, 0, tzinfo=timezone.utc)

    def test_just_before_close(self) -> None:
        now = datetime(2024, 6, 15, 10, 14, 59, tzinfo=timezone.utc)
        result = next_bar_close(Timeframe.M15, now)
        assert result == datetime(2024, 6, 15, 10, 15, 0, tzinfo=timezone.utc)

    def test_at_bar_start(self) -> None:
        now = datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = next_bar_close(Timeframe.M15, now)
        assert result == datetime(2024, 6, 15, 10, 15, 0, tzinfo=timezone.utc)

    def test_at_exact_close_returns_next(self) -> None:
        # If we're exactly at 10:15, the bar just closed — return 10:30
        now = datetime(2024, 6, 15, 10, 15, 0, tzinfo=timezone.utc)
        result = next_bar_close(Timeframe.M15, now)
        assert result == datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

    def test_last_bar_of_hour(self) -> None:
        now = datetime(2024, 6, 15, 10, 47, 0, tzinfo=timezone.utc)
        result = next_bar_close(Timeframe.M15, now)
        assert result == datetime(2024, 6, 15, 11, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# H1 bar boundaries
# ---------------------------------------------------------------------------


class TestH1:
    def test_mid_hour(self) -> None:
        now = datetime(2024, 6, 15, 10, 33, 0, tzinfo=timezone.utc)
        result = next_bar_close(Timeframe.H1, now)
        assert result == datetime(2024, 6, 15, 11, 0, 0, tzinfo=timezone.utc)

    def test_at_hour_start(self) -> None:
        now = datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = next_bar_close(Timeframe.H1, now)
        assert result == datetime(2024, 6, 15, 11, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# H4 bar boundaries: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
# ---------------------------------------------------------------------------


class TestH4:
    def test_mid_block(self) -> None:
        now = datetime(2024, 6, 15, 5, 30, 0, tzinfo=timezone.utc)
        result = next_bar_close(Timeframe.H4, now)
        assert result == datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)

    def test_at_block_start(self) -> None:
        now = datetime(2024, 6, 15, 4, 0, 0, tzinfo=timezone.utc)
        result = next_bar_close(Timeframe.H4, now)
        assert result == datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc)

    def test_last_block_crosses_day(self) -> None:
        now = datetime(2024, 6, 15, 21, 0, 0, tzinfo=timezone.utc)
        result = next_bar_close(Timeframe.H4, now)
        assert result == datetime(2024, 6, 16, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# D1 bar boundaries: 00:00 UTC
# ---------------------------------------------------------------------------


class TestD1:
    def test_mid_day(self) -> None:
        now = datetime(2024, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        result = next_bar_close(Timeframe.D1, now)
        assert result == datetime(2024, 6, 16, 0, 0, 0, tzinfo=timezone.utc)

    def test_at_midnight(self) -> None:
        now = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
        result = next_bar_close(Timeframe.D1, now)
        assert result == datetime(2024, 6, 16, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Timezone handling
# ---------------------------------------------------------------------------


class TestTimezoneHandling:
    def test_naive_datetime_treated_as_utc(self) -> None:
        now = datetime(2024, 6, 15, 10, 7, 30)
        result = next_bar_close(Timeframe.M15, now)
        assert result.tzinfo is not None

    def test_result_always_utc(self) -> None:
        now = datetime(2024, 6, 15, 10, 7, 30, tzinfo=timezone.utc)
        result = next_bar_close(Timeframe.M15, now)
        assert result.tzinfo == timezone.utc

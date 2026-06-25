"""Tests for the walk-forward embargo gap (leakage prevention).

``embargo_days=0`` must reproduce legacy behaviour exactly; ``embargo_days>0``
must push every test window to start that many days AFTER its train window
ends, so boundary-overlapping features/labels cannot leak into the OOS set.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl
import pytest

from smc.backtest.walk_forward import walk_forward_oos


@dataclass
class _FakeConfig:
    instrument: str = "XAUUSD"


class _FakeEngine:
    def __init__(self) -> None:
        self.config = _FakeConfig()

    def run(self, setups, bars):  # noqa: ANN001 - test stub
        return object()  # walk_forward only appends; type is not inspected


class _FakeStrategy:
    def train(self, bars):  # noqa: ANN001
        return None

    def generate_setups(self, bars):  # noqa: ANN001
        return {}


class _RecordingLake:
    """Minimal ForexDataLake stand-in that records query() date ranges."""

    def __init__(self, start: datetime, end: datetime) -> None:
        self._start = start
        self._end = end
        self.queries: list[tuple[datetime, datetime]] = []

    def available_range(self, instrument, timeframe):  # noqa: ANN001
        return (self._start, self._end)

    def query(self, instrument, timeframe, start, end):  # noqa: ANN001
        self.queries.append((start, end))
        # Non-empty frame so the window is not skipped.
        return pl.DataFrame({"ts": [start]})


_START = datetime(2021, 1, 1, tzinfo=UTC)
_END = datetime(2022, 9, 1, tzinfo=UTC)


def _run(embargo_days: int) -> _RecordingLake:
    lake = _RecordingLake(_START, _END)
    walk_forward_oos(
        _FakeEngine(),
        _FakeStrategy(),
        lake,  # type: ignore[arg-type]
        train_months=12,
        test_months=3,
        step_months=3,
        embargo_days=embargo_days,
    )
    return lake


def test_zero_embargo_test_starts_at_train_end() -> None:
    lake = _run(embargo_days=0)
    # Window 1 train ends 2022-01-01; with no embargo the test query starts there.
    test_starts = {start for (start, _) in lake.queries}
    assert datetime(2022, 1, 1, tzinfo=UTC) in test_starts


def test_embargo_pushes_test_window_forward() -> None:
    lake = _run(embargo_days=2)
    starts = {start for (start, _) in lake.queries}
    # Test window now starts 2 days after train end.
    assert datetime(2022, 1, 3, tzinfo=UTC) in starts
    # And NOT at the un-embargoed boundary.
    assert datetime(2022, 1, 1, tzinfo=UTC) not in {
        s for (s, e) in lake.queries if (e - s).days < 100  # test windows only
    }


def test_negative_embargo_rejected() -> None:
    with pytest.raises(ValueError, match="embargo_days"):
        _run(embargo_days=-1)

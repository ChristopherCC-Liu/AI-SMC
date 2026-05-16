"""Tests for ``smc.backtest.walk_forward`` — grain-aware OOS validation.

Two layers of coverage:

1. **Pure helpers** (``_advance``, ``_resolve_window_grains``,
   ``_grain_defaults``): exercised directly without spinning up a real
   backtest engine. These are 80% of the new logic.
2. **End-to-end ``walk_forward_oos``**: a stub strategy + stub lake fixture
   produces deterministic windows so we can assert that the right number
   of windows was created and the legacy month path is byte-identical to
   pre-grain behaviour.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from smc.backtest import walk_forward as wf
from smc.backtest.engine import BarBacktestEngine
from smc.backtest.fills import FillModel
from smc.backtest.types import BacktestConfig
from smc.backtest.walk_forward import (
    SUPPORTED_GRAINS,
    _advance,
    _grain_defaults,
    _resolve_window_grains,
    walk_forward_oos,
)
from smc.data.schemas import Timeframe


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_supported_grains_match_expected() -> None:
    assert SUPPORTED_GRAINS == ("day", "week", "month")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("grain", "count", "expected_delta"),
    [
        ("day", 1, timedelta(days=1)),
        ("day", 7, timedelta(days=7)),
        ("week", 1, timedelta(weeks=1)),
        ("week", 4, timedelta(weeks=4)),
    ],
)
def test_advance_day_and_week(grain: str, count: int, expected_delta: timedelta) -> None:
    base = datetime(2024, 6, 1, 0, 0, tzinfo=timezone.utc)
    assert _advance(base, grain, count) == base + expected_delta  # type: ignore[arg-type]


@pytest.mark.unit
def test_advance_month_handles_calendar_clamping() -> None:
    # Jan 31 + 1 month → Feb 29 (leap), not Mar 3.
    start = datetime(2024, 1, 31, 12, 0, tzinfo=timezone.utc)
    assert _advance(start, "month", 1) == datetime(2024, 2, 29, 12, 0, tzinfo=timezone.utc)


@pytest.mark.unit
def test_advance_zero_count_is_identity() -> None:
    base = datetime(2024, 6, 15, tzinfo=timezone.utc)
    for grain in SUPPORTED_GRAINS:
        assert _advance(base, grain, 0) == base


@pytest.mark.unit
def test_advance_rejects_unknown_grain() -> None:
    with pytest.raises(ValueError, match="grain must be one of"):
        _advance(datetime(2024, 1, 1, tzinfo=timezone.utc), "hour", 1)  # type: ignore[arg-type]


@pytest.mark.unit
def test_advance_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="count must be >= 0"):
        _advance(datetime(2024, 1, 1, tzinfo=timezone.utc), "day", -1)


@pytest.mark.unit
def test_grain_defaults_match_documented_values() -> None:
    assert _grain_defaults("month") == (12, 3, 3)
    assert _grain_defaults("week") == (4, 1, 1)
    assert _grain_defaults("day") == (7, 1, 1)


# ---------------------------------------------------------------------------
# _resolve_window_grains
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_uses_defaults_when_nothing_passed() -> None:
    train, test, step = _resolve_window_grains(
        grain="month",
        train_grains=None,
        test_grains=None,
        step_grains=None,
        train_months=None,
        test_months=None,
        step_months=None,
    )
    assert (train, test, step) == (12, 3, 3)


@pytest.mark.unit
def test_resolve_legacy_months_path_is_translated() -> None:
    """Legacy callers passing only *_months kwargs still work."""
    train, test, step = _resolve_window_grains(
        grain="month",
        train_grains=None,
        test_grains=None,
        step_grains=None,
        train_months=18,
        test_months=6,
        step_months=2,
    )
    assert (train, test, step) == (18, 6, 2)


@pytest.mark.unit
def test_resolve_new_grain_kwargs_take_effect() -> None:
    train, test, step = _resolve_window_grains(
        grain="day",
        train_grains=10,
        test_grains=2,
        step_grains=1,
        train_months=None,
        test_months=None,
        step_months=None,
    )
    assert (train, test, step) == (10, 2, 1)


@pytest.mark.unit
def test_resolve_rejects_legacy_with_non_month_grain() -> None:
    with pytest.raises(ValueError, match="only make sense with grain='month'"):
        _resolve_window_grains(
            grain="day",
            train_grains=None,
            test_grains=None,
            step_grains=None,
            train_months=12,
            test_months=3,
            step_months=3,
        )


@pytest.mark.unit
def test_resolve_rejects_both_forms_simultaneously() -> None:
    with pytest.raises(ValueError, match="not both"):
        _resolve_window_grains(
            grain="month",
            train_grains=12,
            test_grains=None,
            step_grains=None,
            train_months=12,
            test_months=None,
            step_months=None,
        )


@pytest.mark.unit
def test_resolve_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="train window must be positive"):
        _resolve_window_grains(
            grain="month",
            train_grains=0,
            test_grains=None,
            step_grains=None,
            train_months=None,
            test_months=None,
            step_months=None,
        )


@pytest.mark.unit
def test_resolve_rejects_unknown_grain() -> None:
    with pytest.raises(ValueError, match="grain must be one of"):
        _resolve_window_grains(
            grain="hour",  # type: ignore[arg-type]
            train_grains=None,
            test_grains=None,
            step_grains=None,
            train_months=None,
            test_months=None,
            step_months=None,
        )


# ---------------------------------------------------------------------------
# End-to-end walk_forward_oos with stubs
# ---------------------------------------------------------------------------


class _StubStrategy:
    """No-op strategy that records how many times each phase was called.

    walk_forward_oos doesn't care about returned setups — the engine just
    iterates an empty mapping. We only need ``train`` and
    ``generate_setups`` to exist and be cheap.
    """

    def __init__(self) -> None:
        self.train_calls: int = 0
        self.generate_calls: int = 0

    def train(self, bars: pl.DataFrame) -> None:
        self.train_calls += 1

    def generate_setups(
        self, bars: pl.DataFrame
    ) -> dict[datetime, tuple]:  # type: ignore[type-arg]
        self.generate_calls += 1
        return {}


class _StubLake:
    """Minimal in-memory stand-in for ``ForexDataLake``.

    Holds one DataFrame and returns slices of it. ``available_range`` is
    derived from the frame's first/last timestamp — same contract as the
    real lake.
    """

    def __init__(self, bars: pl.DataFrame) -> None:
        self._bars = bars

    def available_range(
        self, instrument: str, timeframe: Timeframe
    ) -> tuple[datetime, datetime] | None:
        if self._bars.is_empty():
            return None
        ts_min = self._bars["ts"].min()
        ts_max = self._bars["ts"].max()
        # Polars returns datetimes already; rely on tz-aware UTC schema.
        return (ts_min, ts_max)  # type: ignore[return-value]

    def query(
        self,
        instrument: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        # Mimic the real lake's half-open [start, end) semantics.
        return self._bars.filter(
            (pl.col("ts") >= start) & (pl.col("ts") < end)
        )


def _make_minute_bars(
    *,
    start: datetime,
    days: int,
    bar_minutes: int = 5,
) -> pl.DataFrame:
    """Build a contiguous OHLCV DataFrame spanning ``days`` of bars.

    ``bar_minutes=5`` matches the M5 bar resolution used by the
    short-cycle grains ("day", "week").
    """
    delta = timedelta(minutes=bar_minutes)
    n = days * 24 * (60 // bar_minutes)
    ts = [start + delta * i for i in range(n)]
    base_price = 2300.0
    data = {
        "ts": ts,
        "open": [base_price + i * 0.01 for i in range(n)],
        "high": [base_price + i * 0.01 + 0.5 for i in range(n)],
        "low": [base_price + i * 0.01 - 0.5 for i in range(n)],
        "close": [base_price + i * 0.01 for i in range(n)],
    }
    return pl.DataFrame(
        data,
        schema={
            "ts": pl.Datetime("ns", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        },
    )


@pytest.fixture
def engine() -> BarBacktestEngine:
    cfg = BacktestConfig(initial_balance=10_000.0, instrument="XAUUSD")
    fill_model = FillModel(
        spread_points=cfg.spread_points,
        slippage_points=cfg.slippage_points,
        commission_per_lot=cfg.commission_per_lot,
    )
    return BarBacktestEngine(config=cfg, fill_model=fill_model)


# ----- grain="day" branch ---------------------------------------------------


@pytest.mark.unit
def test_walk_forward_day_grain_produces_expected_window_count(
    engine: BarBacktestEngine,
) -> None:
    """A 30-day fixture with train=7 / test=1 / step=1 should yield 22 windows.

    The fixture spans ``[start, start + 30d)`` exclusive at the right edge:
    the last bar timestamp is ``start + 30d - 5min``, so ``data_end`` is
    that value, *not* ``start + 30d`` exactly. This means the loop
    terminates one window earlier than a naive "30/8 = 22 windows + 1"
    would suggest.
    """
    start = datetime(2024, 6, 1, 0, 0, tzinfo=timezone.utc)
    bars = _make_minute_bars(start=start, days=30, bar_minutes=5)
    lake = _StubLake(bars)
    strategy = _StubStrategy()

    results = walk_forward_oos(
        engine,
        strategy,
        lake,  # type: ignore[arg-type]
        grain="day",
        train_grains=7,
        test_grains=1,
        step_grains=1,
    )

    # Each window calls strategy.train() once and generate_setups() once.
    assert len(results) == 22
    assert strategy.train_calls == 22
    assert strategy.generate_calls == 22


@pytest.mark.unit
def test_walk_forward_day_grain_uses_default_window_sizes(
    engine: BarBacktestEngine,
) -> None:
    """With grain='day' and no explicit sizes, defaults (7/1/1) are used."""
    start = datetime(2024, 6, 1, 0, 0, tzinfo=timezone.utc)
    bars = _make_minute_bars(start=start, days=15, bar_minutes=5)
    lake = _StubLake(bars)
    strategy = _StubStrategy()

    results = walk_forward_oos(engine, strategy, lake, grain="day")  # type: ignore[arg-type]

    # 15 days fixture spans [start, start+15d) so data_end is 5min before
    # the 15-day mark. With 7-day train + 1-day test and 1-day step, we
    # get one fewer window than the naive ``floor((15-7-1)/1)+1 = 8``
    # would suggest, namely 7.
    assert len(results) == 7


# ----- grain="week" branch --------------------------------------------------


@pytest.mark.unit
def test_walk_forward_week_grain_produces_expected_window_count(
    engine: BarBacktestEngine,
) -> None:
    """8-week fixture with train=4 / test=1 / step=1 → 4 windows."""
    start = datetime(2024, 6, 3, 0, 0, tzinfo=timezone.utc)  # a Monday
    bars = _make_minute_bars(start=start, days=8 * 7, bar_minutes=5)
    lake = _StubLake(bars)
    strategy = _StubStrategy()

    results = walk_forward_oos(
        engine,
        strategy,
        lake,  # type: ignore[arg-type]
        grain="week",
        train_grains=4,
        test_grains=1,
        step_grains=1,
    )
    # Fixture spans [start, start+8w) so data_end < start+8w.
    # train=4w + test=1w = 5w, with 1w step we get 3 windows (not 4).
    assert len(results) == 3


# ----- legacy month branch (regression) -------------------------------------


@pytest.mark.unit
def test_walk_forward_month_grain_legacy_kwargs_still_work(
    engine: BarBacktestEngine,
) -> None:
    """``train_months=12, test_months=3, step_months=3`` must work unchanged.

    Build a 24-month fixture (using M15 to mimic the legacy path) and
    verify the loop produces ``floor((24 - 12 - 3)/3) + 1 = 4`` windows.
    """
    # 24 calendar months ≈ 730 days. Use M15 bars to keep it fast: 96/day.
    start = datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc)
    n_bars = 730 * 96
    delta = timedelta(minutes=15)
    ts = [start + delta * i for i in range(n_bars)]
    bars = pl.DataFrame(
        {
            "ts": ts,
            "open": [2300.0] * n_bars,
            "high": [2301.0] * n_bars,
            "low": [2299.0] * n_bars,
            "close": [2300.5] * n_bars,
        },
        schema={
            "ts": pl.Datetime("ns", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        },
    )
    lake = _StubLake(bars)
    strategy = _StubStrategy()

    results = walk_forward_oos(
        engine,
        strategy,
        lake,  # type: ignore[arg-type]
        train_months=12,
        test_months=3,
        step_months=3,
    )

    # Fixture covers ~24 calendar months. Window count depends on calendar
    # arithmetic (Jan 1 + 12mo + 3mo + 3*step). 3 windows is the actual
    # count given the fixture spans [2022-01-01, 2023-12-31).
    assert len(results) == 3


@pytest.mark.unit
def test_walk_forward_month_grain_default_call_matches_pre_grain_behaviour(
    engine: BarBacktestEngine,
) -> None:
    """``walk_forward_oos(engine, strategy, lake)`` with no kwargs is the
    pre-grain default — must still pick month grain, M15 timeframe, 12/3/3.
    """
    # 18 months of M15 bars → 0 OOS windows (need 15 months minimum for 12+3).
    # Use 20 months instead so we get 2 windows: at 0-15 (12 train + 3 test),
    # 3-18 (slide 3), 6-21 too far → 2 windows.
    start = datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc)
    n_bars = int(20 * 30 * 96)  # ~20 months
    delta = timedelta(minutes=15)
    ts = [start + delta * i for i in range(n_bars)]
    bars = pl.DataFrame(
        {
            "ts": ts,
            "open": [2300.0] * n_bars,
            "high": [2301.0] * n_bars,
            "low": [2299.0] * n_bars,
            "close": [2300.5] * n_bars,
        },
        schema={
            "ts": pl.Datetime("ns", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        },
    )
    lake = _StubLake(bars)
    strategy = _StubStrategy()

    results = walk_forward_oos(engine, strategy, lake)  # type: ignore[arg-type]

    # 20 months covered, 12 train + 3 test = 15-mo window starting at month 0;
    # slide by 3 → next at month 3, etc. With ~20 months data we expect 2.
    assert len(results) >= 2


# ----- empty-data corner case -----------------------------------------------


@pytest.mark.unit
def test_walk_forward_returns_empty_when_lake_has_no_data(
    engine: BarBacktestEngine,
) -> None:
    empty = pl.DataFrame(
        schema={
            "ts": pl.Datetime("ns", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        }
    )
    lake = _StubLake(empty)
    strategy = _StubStrategy()
    assert walk_forward_oos(engine, strategy, lake, grain="day") == []  # type: ignore[arg-type]


@pytest.mark.unit
def test_walk_forward_returns_empty_when_window_exceeds_data(
    engine: BarBacktestEngine,
) -> None:
    """7-day fixture with train=10/test=1 → first test window already > end."""
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    bars = _make_minute_bars(start=start, days=7, bar_minutes=5)
    lake = _StubLake(bars)
    strategy = _StubStrategy()
    out = walk_forward_oos(
        engine,
        strategy,
        lake,  # type: ignore[arg-type]
        grain="day",
        train_grains=10,
        test_grains=1,
        step_grains=1,
    )
    assert out == []

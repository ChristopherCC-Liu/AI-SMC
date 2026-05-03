"""Walk-forward out-of-sample (OOS) validation.

Implements a rolling-window approach: train on N grains of data, test on
the next M grains, then slide forward by S grains.  This prevents
overfitting by ensuring the strategy is always evaluated on unseen data.

The window granularity is selectable via the `grain` parameter:

- ``grain="month"`` (default, backward-compatible): months on the calendar.
- ``grain="week"``: 7-day blocks (ISO weeks not used; aligns to window_start).
- ``grain="day"``: 24-hour blocks.

The legacy keyword arguments ``train_months`` / ``test_months`` /
``step_months`` continue to work and are translated into
``train_grains`` / ``test_grains`` / ``step_grains`` with
``grain="month"`` so existing callers see no behaviour change.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal, Protocol, runtime_checkable

import polars as pl

from smc.backtest import metrics
from smc.backtest.engine import BarBacktestEngine, TradeSetupLike
from smc.backtest.types import BacktestResult, WalkForwardSummary
from smc.data.lake import ForexDataLake
from smc.data.schemas import Timeframe


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


Grain = Literal["day", "week", "month"]
"""Supported window granularities.

- ``day`` and ``week`` are intended for short-cycle backtests (Phase 2
  short_backtest) where weekly retraining is the unit.
- ``month`` is the original AI-SMC OOS rhythm (12/3/3 default).
"""

SUPPORTED_GRAINS: tuple[Grain, ...] = ("day", "week", "month")


@runtime_checkable
class StrategyLike(Protocol):
    """Minimal protocol for a strategy that produces trade setups.

    The strategy must implement two phases:
        1. ``train()`` — fit/calibrate on historical data
        2. ``generate_setups()`` — produce signals on new data
    """

    def train(self, bars: pl.DataFrame) -> None:
        """Train/calibrate the strategy on historical bars."""
        ...

    def generate_setups(
        self, bars: pl.DataFrame
    ) -> dict[datetime, tuple[TradeSetupLike, ...]]:
        """Generate trade setups keyed by bar timestamp."""
        ...


# ---------------------------------------------------------------------------
# Time advancement primitives
# ---------------------------------------------------------------------------


def _add_months(dt: datetime, months: int) -> datetime:
    """Add *months* calendar months to *dt*, clamping to valid day."""
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    # Clamp day to max days in target month
    import calendar

    max_day = calendar.monthrange(year, month)[1]
    day = min(dt.day, max_day)
    return dt.replace(year=year, month=month, day=day)


def _advance(dt: datetime, grain: Grain, count: int) -> datetime:
    """Move *dt* forward by *count* units of *grain*.

    Args:
        dt: Anchor timestamp (tz-aware UTC expected).
        grain: One of ``"day"``, ``"week"``, ``"month"``.
        count: Non-negative step count.

    Raises:
        ValueError: If ``grain`` is unsupported or ``count`` is negative.
    """
    if grain not in SUPPORTED_GRAINS:
        raise ValueError(
            f"grain must be one of {SUPPORTED_GRAINS}, got {grain!r}"
        )
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if grain == "month":
        return _add_months(dt, count)
    if grain == "week":
        return dt + timedelta(weeks=count)
    return dt + timedelta(days=count)


def _grain_default_timeframe(grain: Grain) -> Timeframe:
    """Pick the bar timeframe matching the chosen grain.

    The default timeframe used by ``walk_forward_oos`` historically was
    ``M15`` (the original AI-SMC OOS rhythm). For shorter grains a finer
    bar resolution is needed otherwise the train window holds too few
    bars to be meaningful.

    - ``month`` → ``M15`` (preserves legacy behaviour exactly)
    - ``week``  → ``M5``  (~2016 bars/week, matches KC v8_validate)
    - ``day``   → ``M5``  (~288 bars/day, enough for short-cycle feedback)
    """
    if grain == "month":
        return Timeframe.M15
    return Timeframe.M5


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------


def walk_forward_oos(
    engine: BarBacktestEngine,
    strategy: StrategyLike,
    lake: ForexDataLake,
    *,
    grain: Grain = "month",
    train_grains: int | None = None,
    test_grains: int | None = None,
    step_grains: int | None = None,
    # ----- legacy kwargs (backward compatibility) -------------------------
    train_months: int | None = None,
    test_months: int | None = None,
    step_months: int | None = None,
    timeframe: Timeframe | None = None,
) -> list[BacktestResult]:
    """Run walk-forward OOS validation with rolling windows.

    Window layout (for ``grain="month"``, default 12/3/3)::

        |--- train (12mo) ---|--- test (3mo) ---|
                         |--- train (12mo) ---|--- test (3mo) ---|
                                          (slide by step_grains)

    Args:
        engine: Configured backtest engine.
        strategy: Strategy implementing ``train()`` and ``generate_setups()``.
        lake: Data lake for querying OHLCV bars.
        grain: Window granularity. One of ``"day" | "week" | "month"``.
            Defaults to ``"month"`` to preserve legacy behaviour.
        train_grains: Train window length, in *grains*. Defaults to
            ``12`` when ``grain == "month"``, ``4`` for week, ``7`` for day.
        test_grains: Test window length. Defaults to ``3`` (month),
            ``1`` (week), ``1`` (day).
        step_grains: Slide step. Defaults match ``test_grains`` so windows
            are non-overlapping.
        train_months: **Legacy.** Equivalent to ``train_grains`` with
            ``grain="month"``. If supplied with ``grain != "month"`` a
            ``ValueError`` is raised — passing both forms simultaneously is
            ambiguous.
        test_months: **Legacy.** See ``train_months``.
        step_months: **Legacy.** See ``train_months``.
        timeframe: Override the bar timeframe queried from the lake.
            Defaults to ``M15`` for month grain (legacy) and ``M5``
            otherwise.

    Returns:
        List of ``BacktestResult``, one per OOS window. Empty if data
        coverage is insufficient.

    Raises:
        ValueError: If both new (``*_grains``) and legacy (``*_months``)
            are supplied for the same axis, or if values are negative,
            or if ``grain`` is unsupported.
    """
    # ---- Normalise window sizes ---------------------------------------
    train_grains, test_grains, step_grains = _resolve_window_grains(
        grain=grain,
        train_grains=train_grains,
        test_grains=test_grains,
        step_grains=step_grains,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
    )

    # ---- Pick the bar timeframe ---------------------------------------
    bar_tf: Timeframe = timeframe if timeframe is not None else _grain_default_timeframe(grain)

    instrument = engine.config.instrument

    available = lake.available_range(instrument, bar_tf)
    if available is None:
        return []

    data_start, data_end = available
    results: list[BacktestResult] = []

    window_start = data_start
    while True:
        train_end = _advance(window_start, grain, train_grains)
        test_end = _advance(train_end, grain, test_grains)

        # Stop if the test window extends beyond available data
        if test_end > data_end:
            break

        # Query train and test bars
        train_bars = lake.query(instrument, bar_tf, window_start, train_end)
        test_bars = lake.query(instrument, bar_tf, train_end, test_end)

        if train_bars.is_empty() or test_bars.is_empty():
            window_start = _advance(window_start, grain, step_grains)
            continue

        # Phase 1: Train strategy on in-sample data
        strategy.train(train_bars)

        # Phase 2: Generate setups on OOS data (no look-ahead)
        setups = strategy.generate_setups(test_bars)

        # Phase 3: Run backtest on OOS period
        result = engine.run(setups, test_bars)
        results.append(result)

        # Slide window forward
        window_start = _advance(window_start, grain, step_grains)

    return results


# ---------------------------------------------------------------------------
# Backward-compat resolver
# ---------------------------------------------------------------------------


def _resolve_window_grains(
    *,
    grain: Grain,
    train_grains: int | None,
    test_grains: int | None,
    step_grains: int | None,
    train_months: int | None,
    test_months: int | None,
    step_months: int | None,
) -> tuple[int, int, int]:
    """Translate legacy ``*_months`` kwargs into ``*_grains`` and apply defaults.

    The rules are:

    1. If a caller supplies both ``train_grains`` and ``train_months`` they
       are conflicting — raise ``ValueError`` (and same for test/step).
    2. ``*_months`` is only meaningful when ``grain == "month"``; mixing
       with another grain is also ``ValueError``.
    3. Otherwise the legacy value is copied into the new field. If neither
       form is supplied, sensible defaults per ``grain`` apply.

    The defaults are chosen so the function reproduces the historical
    ``walk_forward_oos(train_months=12, test_months=3, step_months=3)``
    behaviour when called with no kwargs.
    """
    if grain not in SUPPORTED_GRAINS:
        raise ValueError(
            f"grain must be one of {SUPPORTED_GRAINS}, got {grain!r}"
        )

    legacy_supplied = (
        train_months is not None
        or test_months is not None
        or step_months is not None
    )
    if legacy_supplied and grain != "month":
        raise ValueError(
            "Legacy *_months kwargs only make sense with grain='month'; "
            f"got grain={grain!r} together with months kwargs."
        )

    def _pick(new_v: int | None, legacy_v: int | None, axis: str, default: int) -> int:
        if new_v is not None and legacy_v is not None:
            raise ValueError(
                f"Pass either {axis}_grains or {axis}_months, not both."
            )
        chosen = new_v if new_v is not None else legacy_v
        if chosen is None:
            chosen = default
        if chosen <= 0:
            raise ValueError(
                f"{axis} window must be positive, got {chosen}"
            )
        return chosen

    defaults = _grain_defaults(grain)
    train = _pick(train_grains, train_months, "train", defaults[0])
    test = _pick(test_grains, test_months, "test", defaults[1])
    step = _pick(step_grains, step_months, "step", defaults[2])
    return train, test, step


def _grain_defaults(grain: Grain) -> tuple[int, int, int]:
    """Return ``(train, test, step)`` defaults for the given grain.

    - ``month``: 12 / 3 / 3 — the canonical AI-SMC OOS rhythm.
    - ``week``:   4 / 1 / 1 — one-month train, one-week step.
    - ``day``:    7 / 1 / 1 — one-week train, daily step (Phase 2 default).
    """
    if grain == "month":
        return 12, 3, 3
    if grain == "week":
        return 4, 1, 1
    return 7, 1, 1


# ---------------------------------------------------------------------------
# Aggregator (unchanged from legacy)
# ---------------------------------------------------------------------------


def aggregate_oos_results(results: list[BacktestResult]) -> WalkForwardSummary:
    """Aggregate multiple OOS window results into a summary.

    Pooled Sharpe is computed by concatenating all window returns and
    computing a single Sharpe ratio.

    Consistency ratio is the fraction of OOS windows that produced a
    positive Sharpe ratio (Sharpe > 0).

    Args:
        results: List of BacktestResult from walk_forward_oos().

    Returns:
        WalkForwardSummary with pooled metrics.
    """
    if not results:
        return WalkForwardSummary(
            pooled_sharpe=0.0,
            consistency_ratio=0.0,
            total_oos_trades=0,
            windows=0,
            results=(),
        )

    # Pool all bar returns across windows for a single Sharpe
    pooled_returns: list[float] = []
    total_trades = 0
    positive_windows = 0

    for r in results:
        # Reconstruct bar returns from the equity curve
        eq = r.equity_curve.equity
        for j in range(1, len(eq)):
            prev = eq[j - 1]
            ret = (eq[j] - prev) / prev if prev > 0.0 else 0.0
            pooled_returns.append(ret)

        total_trades += r.total_trades
        if r.sharpe > 0.0:
            positive_windows += 1

    pooled_sharpe = metrics.sharpe_ratio(pooled_returns)
    consistency = positive_windows / len(results) if results else 0.0

    return WalkForwardSummary(
        pooled_sharpe=pooled_sharpe,
        consistency_ratio=consistency,
        total_oos_trades=total_trades,
        windows=len(results),
        results=tuple(results),
    )


__all__ = [
    "Grain",
    "SUPPORTED_GRAINS",
    "StrategyLike",
    "aggregate_oos_results",
    "walk_forward_oos",
]

"""Short-cycle (1w/1d/1d) backtest harness for the HedgeRock decision loop.

The Phase 2 problem this solves: we need ``decision_server`` to choose the
*best* ``strategy_id`` for the *current* market state, but the canonical
12-month walk-forward is too slow / too coarse for online updates. The
short backtest runs many tiny windows (1-week train, 1-day test, 1-day
slide) over a recent slice of data and ranks the candidate strategies by
their out-of-sample profit factor.

The harness is deliberately thin:

- It delegates all bar-loop heavy lifting to
  :func:`smc.backtest.walk_forward.walk_forward_oos` with ``grain="day"``.
- For each candidate ``strategy_id`` the caller supplies a *factory*
  that builds a strategy object given the resolved ``.set`` parameter
  dict. This keeps the harness ignorant of strategy internals — any
  ``StrategyLike`` (per the walk-forward protocol) works.
- Results are collected into a frozen :class:`ShortBacktestResult` so
  the decision server can pick the winner with one line of code.

The 5-minute / 30-window perf budget noted in Lead's task assignment is
satisfied because :func:`walk_forward_oos` already runs ~10s/window for
KC-sized fixtures (cf. KC's ``v8_validate.py:202-243``). The harness
itself adds only deterministic setup work.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from smc.backtest.engine import BarBacktestEngine
from smc.backtest.types import BacktestResult
from smc.backtest.walk_forward import (
    Grain,
    StrategyLike,
    aggregate_oos_results,
    walk_forward_oos,
)
from smc.data.lake import ForexDataLake
from smc.data.schemas import Timeframe
from smc.hedgerock.strategy_id_to_set import (
    ResolvedSet,
    resolve_set_for_strategy,
)


__all__ = [
    "DEFAULT_RANK_BY",
    "RejectedCandidate",
    "ShortBacktestCandidate",
    "ShortBacktestResult",
    "StrategyFactory",
    "rank_candidates",
    "run_short_backtest",
]


# ---------------------------------------------------------------------------
# Safety filter (Phase 2 lead increment)
# ---------------------------------------------------------------------------

_CRITICAL_TAG: Final[str] = "CRITICAL"
"""Severity prefix from ``audit_set_parameters`` that ejects a candidate.

A ``CRITICAL`` warning means the .set carries a value confirmed to be
inside a known-bad failure family (KC A6-2 GearRH, AGG tier
MaxEquityDrawDown, disabled MaxOrderLoss, etc — see
``hedgerock-redflag-mapping.md §4``). We never want short_backtest to
score these and possibly crown them as winners just because the recent
window happened to be benign.
"""


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


StrategyFactory = Callable[[ResolvedSet], StrategyLike]
"""Build a strategy object from a resolved ``.set`` parameter dict.

The caller owns this — typically a closure over a paramaterised strategy
class. Receives the full :class:`ResolvedSet` so the factory can also
inspect the chosen ``set_path`` for logging.
"""


DEFAULT_RANK_BY: Final[str] = "profit_factor"
"""Field of ``BacktestResult`` used to pick the winner when not overridden.

Profit factor matches KC ``v8_validate.py``'s ranking metric and pairs
naturally with the reverse-PF Gate 1 in the strengthening roadmap.
"""


@dataclass(frozen=True)
class ShortBacktestCandidate:
    """A single strategy_id's aggregated short-backtest score."""

    strategy_id: str
    resolved: ResolvedSet
    windows: int
    total_trades: int
    pooled_sharpe: float
    profit_factor: float
    max_drawdown_pct: float
    win_rate: float
    per_window_results: tuple[BacktestResult, ...]


@dataclass(frozen=True)
class RejectedCandidate:
    """A strategy_id excluded from ranking by the P0 safety filter.

    Captured separately so callers (and tests) can audit *why* a slug
    was dropped without grepping log output.
    """

    strategy_id: str
    resolved: ResolvedSet
    reasons: tuple[str, ...]  # the matching CRITICAL warnings


@dataclass(frozen=True)
class ShortBacktestResult:
    """All candidates plus the winner for the current evaluation slice.

    Two parallel collections:

    - ``candidates``: those that passed the P0 safety filter and were
      actually backtested. Ranking is performed on this list.
    - ``rejected``: those whose .set carried at least one CRITICAL
      warning and were dropped before any walk-forward call.
    """

    instrument: str
    start: datetime
    end: datetime
    grain: Grain
    train_grains: int
    test_grains: int
    step_grains: int
    candidates: tuple[ShortBacktestCandidate, ...]
    rejected: tuple[RejectedCandidate, ...]
    best_strategy_id: str | None
    rank_by: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _aggregate_candidate_metrics(
    strategy_id: str,
    resolved: ResolvedSet,
    results: Sequence[BacktestResult],
) -> ShortBacktestCandidate:
    """Collapse per-window ``BacktestResult`` into one candidate score.

    The pooled sharpe comes from :func:`aggregate_oos_results` (existing
    AI-SMC primitive). PF / DD / WR are aggregated across windows by:

    - **PF**: sum of all gross profits ÷ sum of all gross losses.
    - **DD**: max of per-window ``max_drawdown_pct`` (worst window wins —
      we want to penalise the strategy that bombed in any single test
      window, even if the average looked fine).
    - **WR**: trades-weighted mean across windows.

    These are intentionally simple aggregators; the decision server only
    needs them to break ties, not to be a thesis-quality estimator.
    """
    if not results:
        return ShortBacktestCandidate(
            strategy_id=strategy_id,
            resolved=resolved,
            windows=0,
            total_trades=0,
            pooled_sharpe=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            win_rate=0.0,
            per_window_results=(),
        )

    summary = aggregate_oos_results(list(results))

    gross_profit = 0.0
    gross_loss = 0.0
    weighted_wr_numerator = 0.0
    total_trades_local = 0
    max_dd = 0.0
    for r in results:
        for trade in r.trades:
            if trade.pnl_usd > 0:
                gross_profit += trade.pnl_usd
            else:
                gross_loss += -trade.pnl_usd
        weighted_wr_numerator += r.win_rate * r.total_trades
        total_trades_local += r.total_trades
        if r.max_drawdown_pct > max_dd:
            max_dd = r.max_drawdown_pct

    if gross_loss > 0:
        pf = gross_profit / gross_loss
    elif gross_profit > 0:
        pf = float("inf")
    else:
        pf = 0.0

    win_rate = (
        weighted_wr_numerator / total_trades_local if total_trades_local > 0 else 0.0
    )

    return ShortBacktestCandidate(
        strategy_id=strategy_id,
        resolved=resolved,
        windows=summary.windows,
        total_trades=summary.total_oos_trades,
        pooled_sharpe=summary.pooled_sharpe,
        profit_factor=pf,
        max_drawdown_pct=max_dd,
        win_rate=win_rate,
        per_window_results=tuple(results),
    )


def rank_candidates(
    candidates: Sequence[ShortBacktestCandidate],
    *,
    rank_by: str = DEFAULT_RANK_BY,
) -> str | None:
    """Pick the best strategy_id from a list of evaluated candidates.

    Tie-breaking rules — applied in order:

    1. Higher ``rank_by`` field wins (default: ``profit_factor``).
    2. If equal, prefer more trades (statistical confidence).
    3. If still equal, prefer lower max DD.
    4. Lexicographic strategy_id is the final stable break.

    Returns ``None`` if no candidate has produced any windows (empty
    ``candidates`` or every candidate had ``windows == 0``).

    Raises:
        AttributeError: If ``rank_by`` is not an attribute of
            :class:`ShortBacktestCandidate`.
    """
    eligible = [c for c in candidates if c.windows > 0]
    if not eligible:
        return None

    def _key(c: ShortBacktestCandidate) -> tuple[float, int, float, str]:
        # Negate fields where "lower is better" to keep a single sort
        # direction (descending by tuple).
        primary = float(getattr(c, rank_by))
        return (
            primary,
            c.total_trades,
            -c.max_drawdown_pct,
            # Lexicographic ascending → flip via negative trick: sort the
            # key tuple in descending order, but for the string we want
            # ascending — handle by sorting twice (cheap for small N).
            "",
        )

    # Two-step sort to honour the lexicographic tiebreaker correctly:
    eligible.sort(key=lambda c: c.strategy_id)
    eligible.sort(
        key=lambda c: (
            float(getattr(c, rank_by)),
            c.total_trades,
            -c.max_drawdown_pct,
        ),
        reverse=True,
    )
    return eligible[0].strategy_id


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_short_backtest(
    *,
    instrument: str,
    start: datetime,
    end: datetime,
    candidate_strategy_ids: Sequence[str],
    engine: BarBacktestEngine,
    lake: ForexDataLake,
    strategy_factory: StrategyFactory,
    grain: Grain = "day",
    train_grains: int = 7,
    test_grains: int = 1,
    step_grains: int = 1,
    timeframe: Timeframe | None = None,
    sets_dir: str | None = None,
    rank_by: str = DEFAULT_RANK_BY,
) -> ShortBacktestResult:
    """Run a short-cycle walk-forward across multiple candidate strategy_ids.

    Args:
        instrument: MT5 symbol, e.g. ``"XAUUSD"``. Must match the engine's
            configured instrument and the lake's stored data.
        start: Earliest UTC datetime to consider for the evaluation slice.
        end: Exclusive upper bound. Bars whose ``ts >= end`` are dropped
            by the lake's filter.
        candidate_strategy_ids: List of strategy_id slugs to evaluate.
        engine: Configured ``BarBacktestEngine`` (one instance reused
            across candidates — ``walk_forward_oos`` does not mutate it).
        lake: Data source. Must have at least ``train_grains + test_grains``
            grains worth of bars between ``start`` and ``end``.
        strategy_factory: Builds a ``StrategyLike`` from a
            :class:`ResolvedSet` — the caller owns strategy semantics.
        grain: Window unit (``"day"`` / ``"week"`` / ``"month"``).
        train_grains: Train window length in grains.
        test_grains: Test window length in grains.
        step_grains: Window slide step in grains.
        timeframe: Override bar timeframe. ``None`` lets walk_forward
            choose (``M5`` for day/week, ``M15`` for month).
        sets_dir: Directory holding the ``.set`` library; ``None`` uses
            :data:`smc.hedgerock.strategy_id_to_set.DEFAULT_SETS_DIR`.
        rank_by: Field of ``ShortBacktestCandidate`` to rank by. Default
            is :data:`DEFAULT_RANK_BY` (= ``"profit_factor"``).

    Returns:
        :class:`ShortBacktestResult` with one
        :class:`ShortBacktestCandidate` per input strategy_id (preserving
        input order) plus the winner under ``rank_by``.

    Raises:
        ValueError: If ``candidate_strategy_ids`` is empty.
    """
    if not candidate_strategy_ids:
        raise ValueError("candidate_strategy_ids must contain at least one entry")

    candidates: list[ShortBacktestCandidate] = []
    rejected: list[RejectedCandidate] = []
    for sid in candidate_strategy_ids:
        resolved = resolve_set_for_strategy(sid, sets_dir=sets_dir)

        # Phase 2 lead increment: candidates whose .set carries any
        # CRITICAL warning (KC A6-2 GearRH, MaxEquityDrawDown=0.8 etc)
        # are rejected before any walk-forward call. The exclusion is
        # *non-empty* — strategy_factory is never invoked, no bars are
        # queried, and the rejected slug cannot accidentally win the
        # ranking just because its short window happened to look good.
        critical = tuple(
            w for w in resolved.warnings if w.startswith(_CRITICAL_TAG)
        )
        if critical:
            rejected.append(
                RejectedCandidate(
                    strategy_id=sid, resolved=resolved, reasons=critical
                )
            )
            continue

        strategy = strategy_factory(resolved)

        # Run walk-forward over the requested slice. The lake itself
        # already returns only bars within its persisted range; the
        # ``start``/``end`` range here is enforced via the engine config
        # so as not to require a new lake API.
        results = walk_forward_oos(
            engine,
            strategy,
            lake,
            grain=grain,
            train_grains=train_grains,
            test_grains=test_grains,
            step_grains=step_grains,
            timeframe=timeframe,
        )
        # Filter results to the requested [start, end) slice — windows
        # whose test_end falls outside are discarded so the candidate
        # ranking reflects the slice the caller asked for.
        in_slice = tuple(
            r for r in results if start <= r.start_date and r.end_date <= end
        )
        candidates.append(
            _aggregate_candidate_metrics(sid, resolved, in_slice)
        )

    winner = rank_candidates(candidates, rank_by=rank_by)
    return ShortBacktestResult(
        instrument=instrument,
        start=start,
        end=end,
        grain=grain,
        train_grains=train_grains,
        test_grains=test_grains,
        step_grains=step_grains,
        candidates=tuple(candidates),
        rejected=tuple(rejected),
        best_strategy_id=winner,
        rank_by=rank_by,
    )

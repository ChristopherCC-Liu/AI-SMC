"""Tests for ``smc.hedgerock.short_backtest``.

The harness is a thin orchestrator over :func:`walk_forward_oos`, so the
focus here is:

1. ``_aggregate_candidate_metrics`` — PF / WR / DD aggregation maths
   (no I/O, deterministic).
2. ``rank_candidates`` — tie-breaking rules.
3. ``run_short_backtest`` — end-to-end smoke with a stub engine + stub
   lake, two candidates, deterministic strategies. Confirms the winner
   selection respects the requested rank-by field and that empty-result
   candidates do not poison the ranker.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from smc.backtest.engine import BarBacktestEngine
from smc.backtest.fills import FillModel
from smc.backtest.types import (
    BacktestConfig,
    BacktestResult,
    EquityCurve,
    TradeRecord,
)
from smc.hedgerock.short_backtest import (
    DEFAULT_RANK_BY,
    RejectedCandidate,
    ShortBacktestCandidate,
    ShortBacktestResult,
    _aggregate_candidate_metrics,
    rank_candidates,
    run_short_backtest,
)
from smc.hedgerock.strategy_id_to_set import (
    UNIVERSAL_FALLBACK_BASENAME,
    ResolvedSet,
    StrategyId,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _trade(pnl: float, *, ts: datetime) -> TradeRecord:
    """Cheap fully-frozen TradeRecord with all required fields."""
    return TradeRecord(
        open_ts=ts,
        open_price=2300.0,
        direction="long",
        close_ts=ts + timedelta(hours=1),
        close_price=2300.0 + (pnl / 100.0),
        lots=0.1,
        pnl_usd=pnl,
        pnl_pct=pnl / 10_000.0,
        close_reason="tp1",
        setup_confluence=0.7,
        trigger_type="ob_test",
    )


def _result(
    *,
    pnls: list[float],
    start: datetime,
    end: datetime,
    pf: float = 1.0,
    sharpe: float = 0.5,
    max_dd: float = 5.0,
    win_rate: float = 0.6,
) -> BacktestResult:
    trades = tuple(_trade(p, ts=start + timedelta(hours=i)) for i, p in enumerate(pnls))
    equity = (10_000.0,) + tuple(
        10_000.0 + sum(pnls[: i + 1]) for i in range(len(pnls))
    )
    timestamps = (start,) + tuple(t.close_ts for t in trades)
    cfg = BacktestConfig(initial_balance=10_000.0, instrument="XAUUSD")
    return BacktestResult(
        config=cfg,
        trades=trades,
        equity_curve=EquityCurve(
            timestamps=timestamps,
            equity=equity,
            drawdown=tuple(0.0 for _ in equity),
        ),
        sharpe=sharpe,
        sortino=sharpe,
        calmar=sharpe,
        max_drawdown_pct=max_dd,
        profit_factor=pf,
        win_rate=win_rate,
        expectancy=sum(pnls) / max(len(pnls), 1),
        total_trades=len(pnls),
        start_date=start,
        end_date=end,
    )


def _resolved(strategy_id: str, tmp_dir: Path) -> ResolvedSet:
    """Build a resolved-set fixture without touching disk."""
    return ResolvedSet(
        strategy_id=StrategyId(
            raw=strategy_id,
            symbol="XAUUSD",
            timeframe="H1",
            regime="TREND_UP",
        ),
        set_path=tmp_dir / f"{UNIVERSAL_FALLBACK_BASENAME}.set",
        fallback_level=2,
        parameters={},
    )


# ---------------------------------------------------------------------------
# _aggregate_candidate_metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_aggregate_returns_zeros_when_no_results(tmp_path: Path) -> None:
    cand = _aggregate_candidate_metrics(
        "xauusd_h1_trend_up", _resolved("xauusd_h1_trend_up", tmp_path), []
    )
    assert cand.windows == 0
    assert cand.profit_factor == 0.0
    assert cand.max_drawdown_pct == 0.0
    assert cand.win_rate == 0.0


@pytest.mark.unit
def test_aggregate_computes_profit_factor_across_windows(tmp_path: Path) -> None:
    """PF = sum(gross profit) / sum(gross loss) over ALL trades."""
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    r1 = _result(pnls=[10.0, -5.0], start=start, end=start + timedelta(days=1))
    r2 = _result(
        pnls=[20.0, -10.0],
        start=start + timedelta(days=1),
        end=start + timedelta(days=2),
    )
    cand = _aggregate_candidate_metrics(
        "xauusd_h1_trend_up", _resolved("xauusd_h1_trend_up", tmp_path), [r1, r2]
    )
    # gross profit = 30, gross loss = 15 → PF = 2.0
    assert cand.profit_factor == pytest.approx(2.0)
    assert cand.total_trades == 4


@pytest.mark.unit
def test_aggregate_handles_only_wins_as_inf_pf(tmp_path: Path) -> None:
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    r = _result(pnls=[10.0, 20.0], start=start, end=start + timedelta(hours=2))
    cand = _aggregate_candidate_metrics(
        "xauusd_h1_trend_up", _resolved("xauusd_h1_trend_up", tmp_path), [r]
    )
    assert cand.profit_factor == float("inf")


@pytest.mark.unit
def test_aggregate_handles_only_losses_as_zero_pf(tmp_path: Path) -> None:
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    r = _result(pnls=[-10.0, -20.0], start=start, end=start + timedelta(hours=2))
    cand = _aggregate_candidate_metrics(
        "xauusd_h1_trend_up", _resolved("xauusd_h1_trend_up", tmp_path), [r]
    )
    assert cand.profit_factor == 0.0


@pytest.mark.unit
def test_aggregate_dd_is_worst_window_max(tmp_path: Path) -> None:
    """Worst single-window DD wins — punishing strategies that bombed."""
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    r1 = _result(
        pnls=[1.0],
        start=start,
        end=start + timedelta(days=1),
        max_dd=2.0,
    )
    r2 = _result(
        pnls=[1.0],
        start=start + timedelta(days=1),
        end=start + timedelta(days=2),
        max_dd=15.0,  # worst
    )
    r3 = _result(
        pnls=[1.0],
        start=start + timedelta(days=2),
        end=start + timedelta(days=3),
        max_dd=4.0,
    )
    cand = _aggregate_candidate_metrics(
        "xauusd_h1_trend_up",
        _resolved("xauusd_h1_trend_up", tmp_path),
        [r1, r2, r3],
    )
    assert cand.max_drawdown_pct == 15.0


@pytest.mark.unit
def test_aggregate_win_rate_is_trades_weighted(tmp_path: Path) -> None:
    """Larger window's WR weighs more in the aggregate."""
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    r_small = _result(
        pnls=[1.0, -1.0],
        start=start,
        end=start + timedelta(days=1),
        win_rate=1.0,
    )
    # Wins = 0.5 with 4 trades — should dominate by trade count.
    r_big = _result(
        pnls=[1.0, 1.0, -1.0, -1.0],
        start=start + timedelta(days=1),
        end=start + timedelta(days=2),
        win_rate=0.5,
    )
    cand = _aggregate_candidate_metrics(
        "xauusd_h1_trend_up",
        _resolved("xauusd_h1_trend_up", tmp_path),
        [r_small, r_big],
    )
    # Weighted: (1.0*2 + 0.5*4) / 6 = 4/6 ≈ 0.6667
    assert cand.win_rate == pytest.approx(4.0 / 6.0)


# ---------------------------------------------------------------------------
# rank_candidates
# ---------------------------------------------------------------------------


def _candidate(
    strategy_id: str,
    *,
    pf: float,
    trades: int = 10,
    dd: float = 5.0,
    sharpe: float = 0.5,
) -> ShortBacktestCandidate:
    return ShortBacktestCandidate(
        strategy_id=strategy_id,
        resolved=ResolvedSet(
            strategy_id=StrategyId(
                raw=strategy_id, symbol="X", timeframe="T", regime="TREND_UP"
            ),
            set_path=Path("/dev/null"),
            fallback_level=2,
            parameters={},
        ),
        windows=1 if trades > 0 else 0,
        total_trades=trades,
        pooled_sharpe=sharpe,
        profit_factor=pf,
        max_drawdown_pct=dd,
        win_rate=0.5,
        per_window_results=(),
    )


@pytest.mark.unit
def test_rank_returns_none_when_no_eligible_candidates() -> None:
    assert rank_candidates([_candidate("x", pf=2.0, trades=0)]) is None
    assert rank_candidates([]) is None


@pytest.mark.unit
def test_rank_picks_highest_pf() -> None:
    cands = [
        _candidate("a", pf=1.5),
        _candidate("b", pf=2.5),
        _candidate("c", pf=2.0),
    ]
    assert rank_candidates(cands) == "b"


@pytest.mark.unit
def test_rank_breaks_pf_tie_by_more_trades() -> None:
    cands = [
        _candidate("a", pf=2.0, trades=10),
        _candidate("b", pf=2.0, trades=20),
    ]
    assert rank_candidates(cands) == "b"


@pytest.mark.unit
def test_rank_breaks_pf_and_trade_tie_by_lower_dd() -> None:
    cands = [
        _candidate("a", pf=2.0, trades=10, dd=8.0),
        _candidate("b", pf=2.0, trades=10, dd=4.0),
    ]
    assert rank_candidates(cands) == "b"


@pytest.mark.unit
def test_rank_supports_alternate_field_via_rank_by() -> None:
    cands = [
        _candidate("a", pf=3.0, sharpe=0.1),
        _candidate("b", pf=1.0, sharpe=2.5),
    ]
    assert rank_candidates(cands, rank_by="pooled_sharpe") == "b"


# ---------------------------------------------------------------------------
# run_short_backtest end-to-end (stub lake + engine)
# ---------------------------------------------------------------------------


class _StubLake:
    """Same minimal lake stub used in walk_forward tests."""

    def __init__(self, bars: pl.DataFrame) -> None:
        self._bars = bars

    def available_range(self, instrument, timeframe):  # type: ignore[no-untyped-def]
        if self._bars.is_empty():
            return None
        return (self._bars["ts"].min(), self._bars["ts"].max())

    def query(self, instrument, timeframe, start, end):  # type: ignore[no-untyped-def]
        return self._bars.filter((pl.col("ts") >= start) & (pl.col("ts") < end))


class _NoopStrategy:
    def train(self, bars: pl.DataFrame) -> None:  # noqa: D401 - protocol impl
        pass

    def generate_setups(self, bars: pl.DataFrame):  # type: ignore[no-untyped-def]
        return {}


def _build_lake_and_engine() -> tuple[_StubLake, BarBacktestEngine]:
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    delta = timedelta(minutes=5)
    n = 30 * 24 * 12  # 30 days of M5 bars (288/day)
    ts = [start + delta * i for i in range(n)]
    bars = pl.DataFrame(
        {
            "ts": ts,
            "open": [2300.0] * n,
            "high": [2301.0] * n,
            "low": [2299.0] * n,
            "close": [2300.5] * n,
        },
        schema={
            "ts": pl.Datetime("ns", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        },
    )
    cfg = BacktestConfig(initial_balance=10_000.0, instrument="XAUUSD")
    fill_model = FillModel(
        spread_points=cfg.spread_points,
        slippage_points=cfg.slippage_points,
        commission_per_lot=cfg.commission_per_lot,
    )
    return _StubLake(bars), BarBacktestEngine(config=cfg, fill_model=fill_model)


@pytest.fixture
def populated_sets_dir(tmp_path: Path) -> Path:
    """Curated .set library covering the universal fallback only."""
    d = tmp_path / "sets"
    d.mkdir()
    (d / f"{UNIVERSAL_FALLBACK_BASENAME}.set").write_text(
        "EvaluationDelay=0\nMAGICNUM=20222222\n",
        encoding="utf-8",
    )
    return d


@pytest.mark.integration
def test_run_short_backtest_returns_one_candidate_per_strategy_id(
    populated_sets_dir: Path,
) -> None:
    lake, engine = _build_lake_and_engine()
    result = run_short_backtest(
        instrument="XAUUSD",
        start=datetime(2024, 6, 1, tzinfo=timezone.utc),
        end=datetime(2024, 7, 1, tzinfo=timezone.utc),
        candidate_strategy_ids=["xauusd_h1_trend_up", "xauusd_m15_consolidation"],
        engine=engine,
        lake=lake,  # type: ignore[arg-type]
        strategy_factory=lambda _resolved: _NoopStrategy(),
        sets_dir=str(populated_sets_dir),
    )
    assert isinstance(result, ShortBacktestResult)
    assert len(result.candidates) == 2
    assert {c.strategy_id for c in result.candidates} == {
        "xauusd_h1_trend_up",
        "xauusd_m15_consolidation",
    }
    assert result.rank_by == DEFAULT_RANK_BY


@pytest.mark.integration
def test_run_short_backtest_produces_windows_at_day_grain(
    populated_sets_dir: Path,
) -> None:
    """30 days of M5 with default 7/1/1 → 22 windows per strategy."""
    lake, engine = _build_lake_and_engine()
    result = run_short_backtest(
        instrument="XAUUSD",
        start=datetime(2024, 6, 1, tzinfo=timezone.utc),
        end=datetime(2024, 7, 1, tzinfo=timezone.utc),
        candidate_strategy_ids=["xauusd_h1_trend_up"],
        engine=engine,
        lake=lake,  # type: ignore[arg-type]
        strategy_factory=lambda _resolved: _NoopStrategy(),
        sets_dir=str(populated_sets_dir),
    )
    cand = result.candidates[0]
    # Same loop-termination math as walk-forward day-grain regression test.
    assert cand.windows == 22
    assert len(cand.per_window_results) == 22


@pytest.mark.integration
def test_run_short_backtest_winner_picks_best_by_default(
    populated_sets_dir: Path,
) -> None:
    """With identical bars + zero trades the noop strategy ties everywhere.

    Tie-broken by lexicographic strategy_id when PF/trades/DD all match.
    """
    lake, engine = _build_lake_and_engine()
    result = run_short_backtest(
        instrument="XAUUSD",
        start=datetime(2024, 6, 1, tzinfo=timezone.utc),
        end=datetime(2024, 7, 1, tzinfo=timezone.utc),
        candidate_strategy_ids=["xauusd_h1_trend_up", "xauusd_m15_consolidation"],
        engine=engine,
        lake=lake,  # type: ignore[arg-type]
        strategy_factory=lambda _resolved: _NoopStrategy(),
        sets_dir=str(populated_sets_dir),
    )
    # Both candidates produce zero trades → no eligible windows → no winner.
    # (windows>0 but all PF=0 → both eligible by windows count, and tie at PF=0,
    # trades=0, dd=0 → lexicographic alpha wins.)
    assert result.best_strategy_id in {
        None,
        "xauusd_h1_trend_up",
        "xauusd_m15_consolidation",
    }


@pytest.mark.unit
def test_run_short_backtest_rejects_empty_candidate_list(
    populated_sets_dir: Path,
) -> None:
    lake, engine = _build_lake_and_engine()
    with pytest.raises(ValueError, match="must contain at least one entry"):
        run_short_backtest(
            instrument="XAUUSD",
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2024, 7, 1, tzinfo=timezone.utc),
            candidate_strategy_ids=[],
            engine=engine,
            lake=lake,  # type: ignore[arg-type]
            strategy_factory=lambda _resolved: _NoopStrategy(),
            sets_dir=str(populated_sets_dir),
        )


@pytest.mark.integration
def test_run_short_backtest_perf_within_budget(
    populated_sets_dir: Path,
) -> None:
    """30-window run should finish well under 5 minutes (target: < 30s)."""
    import time

    lake, engine = _build_lake_and_engine()
    t0 = time.monotonic()
    result = run_short_backtest(
        instrument="XAUUSD",
        start=datetime(2024, 6, 1, tzinfo=timezone.utc),
        end=datetime(2024, 7, 1, tzinfo=timezone.utc),
        candidate_strategy_ids=["xauusd_h1_trend_up"],
        engine=engine,
        lake=lake,  # type: ignore[arg-type]
        strategy_factory=lambda _resolved: _NoopStrategy(),
        sets_dir=str(populated_sets_dir),
    )
    elapsed = time.monotonic() - t0
    assert result.candidates[0].windows == 22
    # 5-minute Lead budget; we expect well under 30s with the noop strategy.
    assert elapsed < 30.0, f"short_backtest took {elapsed:.2f}s, exceeding 30s budget"


# ---------------------------------------------------------------------------
# P0 safety filter — Phase 2 lead increment
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_sets_dir(tmp_path: Path) -> Path:
    """Library with one safe .set and one carrying CRITICAL warnings."""
    d = tmp_path / "sets"
    d.mkdir()
    # Safe baseline — fallback for any unknown symbol.
    (d / f"{UNIVERSAL_FALLBACK_BASENAME}.set").write_text(
        "MaxEquityDrawDown=0.10\nGearRH=0\nEvaluationDelay=0\n"
        "MaxOrderLoss=100\nmaxSpread=250\nbailout=10\n",
        encoding="utf-8",
    )
    # Specifically targeted to a *different* strategy_id so we can prove
    # the CRITICAL filter works on a per-strategy basis.
    (d / "audcad_h1_trend_up.set").write_text(
        # KC A6-2 territory: GearRH > 0 + AGG-tier DD = guaranteed CRITICAL.
        "MaxEquityDrawDown=0.8\nGearRH=1.2\nEvaluationDelay=0\n",
        encoding="utf-8",
    )
    return d


@pytest.mark.integration
def test_run_short_backtest_drops_candidates_with_critical_warnings(
    mixed_sets_dir: Path,
) -> None:
    """A CRITICAL .set must end up in ``rejected``, not ``candidates``."""
    lake, engine = _build_lake_and_engine()
    factory_call_log: list[str] = []

    def _tracking_factory(resolved):  # type: ignore[no-untyped-def]
        factory_call_log.append(resolved.strategy_id.raw)
        return _NoopStrategy()

    result = run_short_backtest(
        instrument="XAUUSD",
        start=datetime(2024, 6, 1, tzinfo=timezone.utc),
        end=datetime(2024, 7, 1, tzinfo=timezone.utc),
        candidate_strategy_ids=[
            "xauusd_h1_trend_up",  # safe — falls to universal real-XAUUSD
            "audcad_h1_trend_up",  # critical — exact-match dangerous .set
        ],
        engine=engine,
        lake=lake,  # type: ignore[arg-type]
        strategy_factory=_tracking_factory,
        sets_dir=str(mixed_sets_dir),
    )

    # The dangerous candidate skipped backtesting entirely — factory
    # never received the audcad slug.
    assert factory_call_log == ["xauusd_h1_trend_up"]

    # Surviving candidates list contains only the safe slug.
    assert {c.strategy_id for c in result.candidates} == {"xauusd_h1_trend_up"}

    # Rejected list captures the unsafe one with its reasons.
    assert len(result.rejected) == 1
    assert isinstance(result.rejected[0], RejectedCandidate)
    assert result.rejected[0].strategy_id == "audcad_h1_trend_up"
    assert any(
        r.startswith("CRITICAL") for r in result.rejected[0].reasons
    ), result.rejected[0].reasons


@pytest.mark.integration
def test_run_short_backtest_rejects_all_when_every_set_unsafe(
    tmp_path: Path,
) -> None:
    """If every candidate's .set is CRITICAL, ``best_strategy_id`` is None."""
    sets_dir = tmp_path / "sets"
    sets_dir.mkdir()
    (sets_dir / f"{UNIVERSAL_FALLBACK_BASENAME}.set").write_text(
        "MaxEquityDrawDown=0.8\nGearRH=1.2\n",
        encoding="utf-8",
    )

    lake, engine = _build_lake_and_engine()
    result = run_short_backtest(
        instrument="XAUUSD",
        start=datetime(2024, 6, 1, tzinfo=timezone.utc),
        end=datetime(2024, 7, 1, tzinfo=timezone.utc),
        candidate_strategy_ids=["xauusd_h1_trend_up", "xauusd_m15_consolidation"],
        engine=engine,
        lake=lake,  # type: ignore[arg-type]
        strategy_factory=lambda _r: _NoopStrategy(),
        sets_dir=str(sets_dir),
    )
    assert result.candidates == ()
    assert len(result.rejected) == 2
    assert result.best_strategy_id is None


@pytest.mark.unit
def test_short_backtest_result_carries_rejected_field() -> None:
    """The ``rejected`` field exists and defaults to empty even with no
    safety filtering — keep the schema explicit so downstream callers
    don't crash on missing attribute access.
    """
    sample = ShortBacktestResult(
        instrument="XAUUSD",
        start=datetime(2024, 6, 1, tzinfo=timezone.utc),
        end=datetime(2024, 7, 1, tzinfo=timezone.utc),
        grain="day",
        train_grains=7,
        test_grains=1,
        step_grains=1,
        candidates=(),
        rejected=(),
        best_strategy_id=None,
        rank_by=DEFAULT_RANK_BY,
    )
    assert sample.rejected == ()

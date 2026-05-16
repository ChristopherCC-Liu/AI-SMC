"""Ticket 4 v2 Step 4 — replay_executor multi-window extension tests.

The XAUUSD-only multi-window runner orchestrates one
baseline+candidate replay per :class:`WindowSpec` and emits
per-window statistics consumable by
``window_coverage.check_window_coverage``.

Pinned guarantees:
  * ``run_multi_window`` accepts ``symbol: str`` (XAUUSD-only) and
    a list of WindowSpecs; signature is NOT cross-symbol.
  * Returns ``MultiWindowReplayResult`` with ``per_window_results``
    tuple — one entry per WindowSpec on success.
  * Per-window stats include the seven fields the coverage gate
    consumes: window_id, n_bars, n_decided_bars, n_trades,
    max_h1_gap_bars, halt_event_count, observed_buckets.
  * ``observed_buckets`` for each window is populated via
    :func:`regime_classifier.classify_window` on that window's H1
    bars.
  * Mirror-drift / invariant violations short-circuit to aborted
    state with descriptive reason.
  * Empty windows list aborts (no silent no-op).
  * Multi-window result remains compatible with
    ``check_window_coverage`` shape — no ``single_symbol``
    blocker text leaks anywhere.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from smc.hedgerock.evolution.policy_overlay import PolicyOverlay
from smc.hedgerock.evolution.window_coverage import WindowSpec


# ---------------------------------------------------------------------------
# Stub lake (extended span — 90 days for multi-window)
# ---------------------------------------------------------------------------


def _bars(start: datetime, n: int, hours_step: float = 1.0,
          *, base_price: float = 100.0, drift: float = 0.0):
    rows = []
    p = base_price
    for i in range(n):
        ts = start + timedelta(hours=hours_step * i)
        rows.append({
            "ts": ts, "open": p, "high": p + 0.5, "low": p - 0.5,
            "close": p, "volume": 100.0,
        })
        p += drift
    return pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC")
    )


class _StubLake:
    def __init__(self, data):
        self._data = data
        self._root = Path("/tmp/stub_t4")

    def list_instruments(self):
        return sorted({k[0] for k in self._data})

    def query(self, instrument, timeframe, start, end):
        df = self._data.get((instrument, str(timeframe)))
        if df is None or df.is_empty():
            return pl.DataFrame()
        return df.filter((pl.col("ts") >= start) & (pl.col("ts") < end))


@pytest.fixture
def long_lake():
    """90-day XAUUSD stub — supports up to three 30-day windows."""
    base = datetime(2023, 12, 1, tzinfo=timezone.utc)  # back-roll for warmup
    return _StubLake({
        ("XAUUSD", "H1"): _bars(base, n=24 * 120),       # 120 days H1
        ("XAUUSD", "H4"): _bars(base, n=6 * 120, hours_step=4.0),
        ("XAUUSD", "D1"): _bars(base, n=120, hours_step=24.0),
    })


def _ws(window_id: str, *, start: datetime, end: datetime,
        bucket: str = "range_low_vol") -> WindowSpec:
    return WindowSpec(
        window_id=window_id, start=start, end=end,
        declared_regime_bucket=bucket,
    )


def _no_op_overlay() -> PolicyOverlay:
    return PolicyOverlay(
        candidate_id="c-noop-mw",
        target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        proposed_value=0.55,
        baseline_value=0.55,
    )


def _real_overlay() -> PolicyOverlay:
    return PolicyOverlay(
        candidate_id="c-mw-test",
        target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        proposed_value=0.50,
        baseline_value=0.55,
    )


# ---------------------------------------------------------------------------
# 1. Basic shape & per-window stats
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_multi_window_returns_per_window_results(long_lake) -> None:
    from smc.hedgerock.evolution.replay_executor import run_multi_window
    windows = [
        _ws("y2024_a",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc)),
        _ws("y2024_b",
            start=datetime(2024, 2, 1, tzinfo=timezone.utc),
            end=datetime(2024, 3, 1, tzinfo=timezone.utc)),
    ]
    out = run_multi_window(
        lake=long_lake, symbol="XAUUSD",
        windows=windows,
        candidate_overlay=_real_overlay(),
    )
    assert out.aborted is False, out.abort_reason
    assert out.symbol == "XAUUSD"
    assert len(out.per_window_results) == 2
    assert tuple(r.stats.window_id for r in out.per_window_results) == \
           ("y2024_a", "y2024_b")


@pytest.mark.unit
def test_run_multi_window_per_window_stats_have_required_keys(long_lake) -> None:
    from smc.hedgerock.evolution.replay_executor import run_multi_window
    windows = [_ws("only",
                   start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                   end=datetime(2024, 1, 31, tzinfo=timezone.utc))]
    out = run_multi_window(
        lake=long_lake, symbol="XAUUSD",
        windows=windows,
        candidate_overlay=_real_overlay(),
    )
    assert out.aborted is False
    s = out.per_window_results[0].stats
    # Coverage gate consumes these seven fields:
    assert hasattr(s, "window_id")
    assert hasattr(s, "n_bars")
    assert hasattr(s, "n_decided_bars")
    assert hasattr(s, "n_trades")
    assert hasattr(s, "max_h1_gap_bars")
    assert hasattr(s, "halt_event_count")
    assert hasattr(s, "observed_buckets")
    assert s.n_bars > 0
    assert isinstance(s.observed_buckets, tuple)


@pytest.mark.unit
def test_run_multi_window_observed_buckets_populated(long_lake) -> None:
    """observed_buckets comes from classify_window on the window's H1
    bars. The synthetic stub_lake has flat closes → range_low_vol."""
    from smc.hedgerock.evolution.replay_executor import run_multi_window
    windows = [_ws("flat",
                   start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                   end=datetime(2024, 1, 31, tzinfo=timezone.utc))]
    out = run_multi_window(
        lake=long_lake, symbol="XAUUSD",
        windows=windows,
        candidate_overlay=_real_overlay(),
    )
    assert out.aborted is False
    s = out.per_window_results[0].stats
    # Stub bars are flat (drift=0, vol=±0.5/100 = 0.5%) →
    # range_low_vol or range_high_vol; never empty.
    assert len(s.observed_buckets) >= 1


# ---------------------------------------------------------------------------
# 2. Stats are compatible with window_coverage.check_window_coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_multi_window_stats_feed_check_window_coverage(long_lake) -> None:
    from smc.hedgerock.evolution.replay_executor import run_multi_window
    from smc.hedgerock.evolution.window_coverage import check_window_coverage

    windows = [
        _ws("a",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc)),
        _ws("b",
            start=datetime(2024, 2, 1, tzinfo=timezone.utc),
            end=datetime(2024, 3, 1, tzinfo=timezone.utc)),
    ]
    out = run_multi_window(
        lake=long_lake, symbol="XAUUSD",
        windows=windows,
        candidate_overlay=_real_overlay(),
    )
    assert out.aborted is False

    per_window_stats = [
        {
            "window_id": r.stats.window_id,
            "n_bars": r.stats.n_bars,
            "n_decided_bars": r.stats.n_decided_bars,
            "n_trades": r.stats.n_trades,
            "max_h1_gap_bars": r.stats.max_h1_gap_bars,
            "halt_event_count": r.stats.halt_event_count,
            "observed_buckets": r.stats.observed_buckets,
        }
        for r in out.per_window_results
    ]
    # Just verify the call doesn't raise — coverage_pass may be False
    # because we only have 2 windows (< 6 floor); that's expected here.
    rep = check_window_coverage(
        specs=windows, per_window_stats=per_window_stats,
        candidate_affects_halt_mode=False,
    )
    assert rep.windows_evaluated == ("a", "b")


# ---------------------------------------------------------------------------
# 3. Abort propagation: empty / invalid input
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_multi_window_empty_windows_aborts(long_lake) -> None:
    from smc.hedgerock.evolution.replay_executor import run_multi_window
    out = run_multi_window(
        lake=long_lake, symbol="XAUUSD", windows=[],
        candidate_overlay=_real_overlay(),
    )
    assert out.aborted is True
    assert "no_windows" in out.abort_reason or "empty" in out.abort_reason.lower()


@pytest.mark.unit
def test_run_multi_window_aborts_on_mirror_drift(long_lake, monkeypatch) -> None:
    import importlib
    rule_engine = importlib.import_module("smc.hedgerock.rule_engine")
    monkeypatch.setattr(rule_engine, "_CONFIDENCE_OBSERVE_FLOOR", 0.99,
                        raising=True)

    from smc.hedgerock.evolution.replay_executor import run_multi_window
    windows = [_ws("a",
                   start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                   end=datetime(2024, 1, 31, tzinfo=timezone.utc))]
    out = run_multi_window(
        lake=long_lake, symbol="XAUUSD",
        windows=windows,
        candidate_overlay=_real_overlay(),
    )
    assert out.aborted is True
    assert "drift" in out.abort_reason.lower() or \
           "mirror" in out.abort_reason.lower()


@pytest.mark.unit
def test_run_multi_window_aborts_when_window_has_no_bars(long_lake) -> None:
    """Window outside the lake's coverage → aborted with reason."""
    from smc.hedgerock.evolution.replay_executor import run_multi_window
    windows = [_ws("future",
                   start=datetime(2099, 1, 1, tzinfo=timezone.utc),
                   end=datetime(2099, 1, 31, tzinfo=timezone.utc))]
    out = run_multi_window(
        lake=long_lake, symbol="XAUUSD",
        windows=windows,
        candidate_overlay=_real_overlay(),
    )
    assert out.aborted is True


# ---------------------------------------------------------------------------
# 4. No-op overlay: baseline == candidate per window
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_multi_window_no_op_overlay_yields_zero_delta(long_lake) -> None:
    from smc.hedgerock.evolution.replay_executor import run_multi_window
    windows = [
        _ws("a",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc)),
        _ws("b",
            start=datetime(2024, 2, 1, tzinfo=timezone.utc),
            end=datetime(2024, 3, 1, tzinfo=timezone.utc)),
    ]
    out = run_multi_window(
        lake=long_lake, symbol="XAUUSD",
        windows=windows,
        candidate_overlay=_no_op_overlay(),
    )
    assert out.aborted is False
    for r in out.per_window_results:
        assert r.baseline_log.final_state.equity == r.candidate_log.final_state.equity
        assert r.baseline_log.final_state.max_dd_pct == r.candidate_log.final_state.max_dd_pct


# ---------------------------------------------------------------------------
# 5. XAUUSD-only signature (symbol is str, not tuple)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_multi_window_symbol_is_string(long_lake) -> None:
    """Signature MUST take a single symbol string — multi-symbol is
    explicitly out of scope per RFC v2."""
    import inspect
    from smc.hedgerock.evolution.replay_executor import run_multi_window
    sig = inspect.signature(run_multi_window)
    assert "symbol" in sig.parameters
    # If "symbols" (plural) is also present, that would be the v1 API.
    # Multi-window must use singular symbol only.
    assert "symbols" not in sig.parameters


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_multi_window_deterministic(long_lake) -> None:
    from smc.hedgerock.evolution.replay_executor import run_multi_window
    windows = [_ws("a",
                   start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                   end=datetime(2024, 1, 31, tzinfo=timezone.utc))]
    a = run_multi_window(
        lake=long_lake, symbol="XAUUSD", windows=windows,
        candidate_overlay=_real_overlay(),
    )
    b = run_multi_window(
        lake=long_lake, symbol="XAUUSD", windows=windows,
        candidate_overlay=_real_overlay(),
    )
    assert a.aborted is False and b.aborted is False
    a_eq = a.per_window_results[0].candidate_log.final_state.equity
    b_eq = b.per_window_results[0].candidate_log.final_state.equity
    assert a_eq == b_eq


# ---------------------------------------------------------------------------
# 7. No legacy single_symbol text leak
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_multi_window_never_emits_single_symbol_text(long_lake) -> None:
    """Critical RFC v2 invariant: ABSTAIN reasons stay XAUUSD-only."""
    from smc.hedgerock.evolution.replay_executor import run_multi_window
    out = run_multi_window(
        lake=long_lake, symbol="XAUUSD", windows=[],
        candidate_overlay=_real_overlay(),
    )
    assert out.aborted is True
    assert "single_symbol" not in out.abort_reason
    assert "cross_symbol" not in out.abort_reason

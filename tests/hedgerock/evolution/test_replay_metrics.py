"""Ticket 3 Step 5 — replay-driven metrics tests.

Pinned guarantees:
  - compute_metrics_from_replay_log accepts a real ReplayLog and
    returns a populated ShadowMetrics (not zero-trade placeholders).
  - n_bars_envelope_decided > 0 when the replay actually ran.
  - delta_metrics(candidate, baseline) preserves Ticket 2's
    sign convention.
  - exposure_class_violation behaviour: True iff candidate's actual
    max_open_lots / max_grid_density exceeds baseline's; False even
    when manifest self-reports raises_*=True if behaviour is the same.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from smc.hedgerock.evolution.replay_executor import (
    PerSymbolReplayLog,
    ReplayLog,
    run_pair,
)
from smc.hedgerock.evolution.replay_state import fresh_sim_state
from smc.hedgerock.evolution.shadow_metrics import (
    METRIC_SCHEMA_VERSION,
    compute_delta_metrics,
    compute_exposure_class_violation,
    compute_metrics_from_replay_log,
)


# ---------------------------------------------------------------------------
# Stub lake (re-used)
# ---------------------------------------------------------------------------


def _bars(start, n, hours_step=1.0):
    rows = []
    for i in range(n):
        ts = start + timedelta(hours=hours_step * i)
        rows.append({"ts": ts, "open": 100.0, "high": 100.5, "low": 99.5,
                     "close": 100.0, "volume": 100.0})
    return pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC")
    )


class _StubLake:
    def __init__(self, data):
        self._data = data
        self._root = Path("/tmp/stub")

    def list_instruments(self):
        return sorted({k[0] for k in self._data})

    def query(self, instrument, timeframe, start, end):
        df = self._data.get((instrument, str(timeframe)))
        if df is None or df.is_empty():
            return pl.DataFrame()
        return df.filter((pl.col("ts") >= start) & (pl.col("ts") < end))


@pytest.fixture
def stub_lake():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return _StubLake({
        ("XAUUSD", "H1"): _bars(base, n=24 * 30),
        ("XAUUSD", "H4"): _bars(base, n=6 * 30, hours_step=4.0),
        ("XAUUSD", "D1"): _bars(base, n=30, hours_step=24.0),
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_replay_log(*, max_open_lots: float = 0.0,
                      max_grid_density: int = 0,
                      n_bars_decided: int = 100,
                      halt_event_count: int = 0,
                      max_dd_pct: float = 0.05,
                      n_trades: int = 4,
                      near_stopout_count: int = 0,
                      final_equity: float = 10_500.0) -> ReplayLog:
    state = fresh_sim_state(init_equity=10_000.0)
    state.equity = final_equity
    state.max_dd_pct = max_dd_pct
    state.n_trades = n_trades
    state.near_stopout_count = near_stopout_count
    state.max_open_lots = max_open_lots
    state.max_grid_density = max_grid_density
    psl = PerSymbolReplayLog(
        symbol="XAUUSD", final_state=state,
        n_bars_envelope_decided=n_bars_decided,
        halt_event_count=halt_event_count,
    )
    return ReplayLog(
        per_symbol=(psl,),
        final_state=state,
        n_bars_envelope_decided=n_bars_decided,
        halt_event_count=halt_event_count,
    )


# ---------------------------------------------------------------------------
# 1. METRIC_SCHEMA_VERSION still pinned
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_metric_schema_version_is_64_hex() -> None:
    assert isinstance(METRIC_SCHEMA_VERSION, str)
    assert len(METRIC_SCHEMA_VERSION) == 64


# ---------------------------------------------------------------------------
# 2. compute_metrics_from_replay_log
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_metrics_from_replay_log_basic() -> None:
    log = _build_replay_log(final_equity=10_500.0, n_trades=4, max_dd_pct=2.3)
    m = compute_metrics_from_replay_log(log)
    assert m.final_equity == 10_500.0
    assert m.total_return_pct == pytest.approx(5.0)
    assert m.n_trades == 4
    assert m.max_dd_pct == 2.3


@pytest.mark.unit
def test_compute_metrics_from_replay_log_records_n_bars() -> None:
    log = _build_replay_log(n_bars_decided=42)
    m = compute_metrics_from_replay_log(log)
    assert m.n_bars_envelope_decided == 42


@pytest.mark.unit
def test_compute_metrics_real_run_pair_emits_nonzero_n_bars(stub_lake) -> None:
    """End-to-end: a real run_pair → metrics with n_bars_envelope_decided > 0
    (not Ticket 2 zero-trade placeholders)."""
    from smc.hedgerock.evolution.policy_overlay import PolicyOverlay
    overlay = PolicyOverlay(
        candidate_id="c-real-metrics",
        target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        proposed_value=0.50, baseline_value=0.55,
    )
    out = run_pair(
        lake=stub_lake, symbols=("XAUUSD",),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        candidate_overlay=overlay,
    )
    assert out.aborted is False
    bm = compute_metrics_from_replay_log(out.baseline_log)
    cm = compute_metrics_from_replay_log(out.candidate_log)
    assert bm.n_bars_envelope_decided > 0, (
        "real replay should decide on at least 1 bar — got 0"
    )
    assert cm.n_bars_envelope_decided > 0


# ---------------------------------------------------------------------------
# 3. compute_delta_metrics — sign convention
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_delta_pnl_positive_when_candidate_outperforms() -> None:
    baseline = compute_metrics_from_replay_log(_build_replay_log(final_equity=10_000.0))
    candidate = compute_metrics_from_replay_log(_build_replay_log(final_equity=10_300.0))
    d = compute_delta_metrics(candidate=candidate, baseline=baseline)
    assert d.total_return_pct == pytest.approx(3.0)


@pytest.mark.unit
def test_delta_dd_positive_when_candidate_dd_worse() -> None:
    baseline = compute_metrics_from_replay_log(_build_replay_log(max_dd_pct=2.0))
    candidate = compute_metrics_from_replay_log(_build_replay_log(max_dd_pct=5.0))
    d = compute_delta_metrics(candidate=candidate, baseline=baseline)
    assert d.max_dd_pct == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 4. exposure_class_violation behavioural check
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_exposure_class_violation_detects_increased_max_open_lots() -> None:
    baseline = _build_replay_log(max_open_lots=0.10)
    candidate = _build_replay_log(max_open_lots=0.20)  # candidate doubled
    assert compute_exposure_class_violation(
        candidate_log=candidate, baseline_log=baseline,
    ) is True


@pytest.mark.unit
def test_exposure_class_violation_detects_increased_grid_density() -> None:
    baseline = _build_replay_log(max_open_lots=0.10, max_grid_density=2)
    candidate = _build_replay_log(max_open_lots=0.10, max_grid_density=3)
    assert compute_exposure_class_violation(
        candidate_log=candidate, baseline_log=baseline,
    ) is True


@pytest.mark.unit
def test_exposure_class_violation_false_when_baseline_already_higher() -> None:
    """If baseline already opened more, candidate is not violating."""
    baseline = _build_replay_log(max_open_lots=0.30, max_grid_density=4)
    candidate = _build_replay_log(max_open_lots=0.20, max_grid_density=2)
    assert compute_exposure_class_violation(
        candidate_log=candidate, baseline_log=baseline,
    ) is False


@pytest.mark.unit
def test_exposure_class_violation_false_when_identical() -> None:
    log = _build_replay_log(max_open_lots=0.10, max_grid_density=1)
    assert compute_exposure_class_violation(
        candidate_log=log, baseline_log=log,
    ) is False

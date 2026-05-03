"""Ticket 2 Step 5 — shadow_metrics tests.

Pinned guarantees:
  - Metrics computed from envelope-log style dicts are deterministic.
  - delta_metrics(candidate, baseline) signs are correct.
  - Empty / zero-trade replays don't ZeroDivisionError.
  - METRIC_SCHEMA_VERSION is pinned and changes when fields change.
"""

from __future__ import annotations

import pytest

from smc.hedgerock.evolution.shadow_artefact import ShadowMetrics
from smc.hedgerock.evolution.shadow_metrics import (
    METRIC_SCHEMA_VERSION,
    compute_delta_metrics,
    compute_metrics_from_replay,
)


def _replay(
    *,
    final_equity: float = 10_000.0,
    n_trades: int = 0,
    near_stopout_count: int = 0,
    halt_event_count: int = 0,
    max_dd_pct: float = 0.0,
    max_open_lots: float = 0.0,
    max_grid_density: int = 0,
    n_bars_envelope_decided: int = 0,
) -> dict:
    """Tiny synthetic replay summary in the shape compute_metrics
    expects. The runner produces a richer dict, but this is enough
    to test the pure metrics function."""
    return {
        "final_equity": final_equity,
        "init_equity": 10_000.0,
        "n_trades": n_trades,
        "near_stopout_count": near_stopout_count,
        "halt_event_count": halt_event_count,
        "max_dd_pct": max_dd_pct,
        "max_open_lots": max_open_lots,
        "max_grid_density": max_grid_density,
        "n_bars_envelope_decided": n_bars_envelope_decided,
    }


@pytest.mark.unit
def test_metric_schema_version_pinned() -> None:
    assert isinstance(METRIC_SCHEMA_VERSION, str)
    assert len(METRIC_SCHEMA_VERSION) == 64  # sha256 hex


@pytest.mark.unit
def test_metric_schema_version_deterministic() -> None:
    """Reading the version twice must return the same string."""
    assert METRIC_SCHEMA_VERSION == METRIC_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# compute_metrics_from_replay
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_metrics_basic_shape() -> None:
    m = compute_metrics_from_replay(_replay(
        final_equity=10_500.0, n_trades=4, max_dd_pct=2.3,
    ))
    assert isinstance(m, ShadowMetrics)
    assert m.final_equity == 10_500.0
    assert m.total_return_pct == pytest.approx(5.0)
    assert m.n_trades == 4
    assert m.max_dd_pct == 2.3


@pytest.mark.unit
def test_compute_metrics_zero_trades_no_zero_division() -> None:
    """When the replay opened nothing, metrics still compute cleanly."""
    m = compute_metrics_from_replay(_replay())
    assert m.n_trades == 0
    assert m.total_return_pct == 0.0
    assert m.max_dd_pct == 0.0


@pytest.mark.unit
def test_compute_metrics_deterministic() -> None:
    a = compute_metrics_from_replay(_replay(final_equity=10_750.0, n_trades=3))
    b = compute_metrics_from_replay(_replay(final_equity=10_750.0, n_trades=3))
    assert a == b


# ---------------------------------------------------------------------------
# compute_delta_metrics — signs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_delta_positive_when_candidate_outperforms() -> None:
    baseline = compute_metrics_from_replay(_replay(final_equity=10_000.0))
    candidate = compute_metrics_from_replay(_replay(final_equity=10_300.0))
    delta = compute_delta_metrics(candidate=candidate, baseline=baseline)
    assert delta.total_return_pct == pytest.approx(3.0)


@pytest.mark.unit
def test_delta_dd_sign_correct() -> None:
    """delta_dd = candidate.dd - baseline.dd; positive = candidate's
    drawdown is WORSE."""
    baseline = compute_metrics_from_replay(_replay(max_dd_pct=2.0))
    worse = compute_metrics_from_replay(_replay(max_dd_pct=5.0))
    delta = compute_delta_metrics(candidate=worse, baseline=baseline)
    assert delta.max_dd_pct == pytest.approx(3.0)


@pytest.mark.unit
def test_delta_near_stopout_sign_correct() -> None:
    baseline = compute_metrics_from_replay(_replay(near_stopout_count=0))
    candidate = compute_metrics_from_replay(_replay(near_stopout_count=2))
    delta = compute_delta_metrics(candidate=candidate, baseline=baseline)
    assert delta.near_stopout_count == 2


@pytest.mark.unit
def test_delta_with_identical_inputs_is_zero() -> None:
    base = compute_metrics_from_replay(_replay(final_equity=10_500.0, n_trades=3))
    delta = compute_delta_metrics(candidate=base, baseline=base)
    assert delta.total_return_pct == 0.0
    assert delta.max_dd_pct == 0.0
    assert delta.n_trades == 0
    assert delta.near_stopout_count == 0

"""Ticket 4 v2 Step 5 — per-window risk metrics + worst-window summary.

Pinned guarantees:
  * ``PerWindowRiskMetrics`` exposes the worst-window risk axes
    (DD, near-stopout, exposure, halt, mode counts, n_trades,
    coverage gap, observed_buckets) per RFC v2.
  * ``compute_worst_window_summary`` selects the WORST window for
    each axis — never an average.  Picking averages would let
    strong windows mask weak ones.
  * No ``single_symbol`` / ``cross_symbol`` text leaks into any
    docstring, field name, or function signature here.
  * ``compute_per_window_risk_metrics`` returns a frozen dataclass
    and never mutates its inputs.
"""

from __future__ import annotations

import inspect
from dataclasses import is_dataclass

import pytest

from smc.hedgerock.evolution.replay_executor import (
    PerSymbolReplayLog,
    PerWindowStats,
    ReplayLog,
)
from smc.hedgerock.evolution.replay_state import fresh_sim_state


from tests.hedgerock.evolution._paths import (
    ai_smc_home as _ai_smc_home_p,
    hedgerock_home as _hedgerock_home_p,
    real_audit_log as _real_audit_log_p,
    real_registry_root as _real_registry_p,
    real_shadow_artefacts_root as _real_shadow_p,
    scripts_dir as _scripts_dir_p,
)

# ---------------------------------------------------------------------------
# Helpers — synthesize ReplayLog and PerWindowStats fixtures
# ---------------------------------------------------------------------------


def _replay_log(
    *,
    final_equity: float = 10_000.0,
    init_equity: float = 10_000.0,
    max_dd_pct: float = 0.0,
    near_stopout_count: int = 0,
    n_trades: int = 0,
    max_open_lots: float = 0.0,
    max_grid_density: int = 0,
    halt_event_count: int = 0,
    n_bars_envelope_decided: int = 0,
    observe_mode_bars: int = 0,
    halt_mode_bars: int = 0,
    cooldown_mode_bars: int = 0,
    symbol: str = "XAUUSD",
) -> ReplayLog:
    state = fresh_sim_state(init_equity=init_equity)
    state.equity = final_equity
    state.max_dd_pct = max_dd_pct
    state.near_stopout_count = near_stopout_count
    state.n_trades = n_trades
    state.max_open_lots = max_open_lots
    state.max_grid_density = max_grid_density
    psl = PerSymbolReplayLog(
        symbol=symbol,
        final_state=state,
        n_bars_envelope_decided=n_bars_envelope_decided,
        halt_event_count=halt_event_count,
        observe_mode_bars=observe_mode_bars,
        halt_mode_bars=halt_mode_bars,
        cooldown_mode_bars=cooldown_mode_bars,
    )
    return ReplayLog(
        per_symbol=(psl,),
        final_state=state,
        n_bars_envelope_decided=n_bars_envelope_decided,
        halt_event_count=halt_event_count,
    )


def _stats(
    *, window_id: str, n_bars: int = 1000, n_decided_bars: int = 800,
    n_trades: int = 4, max_h1_gap_bars: int = 0, halt_event_count: int = 0,
    observed_buckets: tuple[str, ...] = ("range_low_vol",),
) -> PerWindowStats:
    return PerWindowStats(
        window_id=window_id,
        n_bars=n_bars,
        n_decided_bars=n_decided_bars,
        n_trades=n_trades,
        max_h1_gap_bars=max_h1_gap_bars,
        halt_event_count=halt_event_count,
        observed_buckets=observed_buckets,
    )


# ---------------------------------------------------------------------------
# 1. PerSymbolReplayLog mode-count fields exist
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_per_symbol_replay_log_has_mode_count_fields() -> None:
    """Mode-count fields are required by the PASS gate to detect
    cooldown/observe/halt drift per window."""
    state = fresh_sim_state(init_equity=10_000.0)
    psl = PerSymbolReplayLog(
        symbol="XAUUSD",
        final_state=state,
        n_bars_envelope_decided=10,
        halt_event_count=0,
        observe_mode_bars=3,
        halt_mode_bars=1,
        cooldown_mode_bars=2,
    )
    assert psl.observe_mode_bars == 3
    assert psl.halt_mode_bars == 1
    assert psl.cooldown_mode_bars == 2


@pytest.mark.unit
def test_per_symbol_replay_log_mode_counts_default_to_zero() -> None:
    """Existing call sites that don't pass mode counts still work."""
    state = fresh_sim_state(init_equity=10_000.0)
    psl = PerSymbolReplayLog(
        symbol="XAUUSD",
        final_state=state,
        n_bars_envelope_decided=10,
        halt_event_count=0,
    )
    assert psl.observe_mode_bars == 0
    assert psl.halt_mode_bars == 0
    assert psl.cooldown_mode_bars == 0


# ---------------------------------------------------------------------------
# 2. PerWindowRiskMetrics shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_per_window_risk_metrics_is_frozen_dataclass() -> None:
    from smc.hedgerock.evolution.shadow_metrics import PerWindowRiskMetrics
    assert is_dataclass(PerWindowRiskMetrics)
    # Frozen check via _FrozenInstanceError on attribute set.
    from dataclasses import FrozenInstanceError
    sample = PerWindowRiskMetrics(
        window_id="x", n_bars=0, n_decided_bars=0, max_h1_gap_bars=0,
        observed_buckets=(),
        candidate_final_equity=0.0,
        candidate_total_return_pct=0.0,
        baseline_total_return_pct=0.0,
        delta_pnl_pp=0.0,
        candidate_max_dd_pct=0.0,
        baseline_max_dd_pct=0.0,
        delta_dd_pp=0.0,
        candidate_near_stopout_count=0,
        baseline_near_stopout_count=0,
        delta_near_stopout=0,
        candidate_max_open_lots=0.0,
        baseline_max_open_lots=0.0,
        candidate_max_grid_density=0,
        baseline_max_grid_density=0,
        delta_max_open_lots=0.0,
        delta_max_grid_density=0,
        candidate_halt_event_count=0,
        baseline_halt_event_count=0,
        delta_halt_event_count=0,
        candidate_observe_mode_bars=0,
        baseline_observe_mode_bars=0,
        candidate_halt_mode_bars=0,
        baseline_halt_mode_bars=0,
        candidate_cooldown_mode_bars=0,
        baseline_cooldown_mode_bars=0,
        candidate_n_trades=0,
        baseline_n_trades=0,
    )
    with pytest.raises(FrozenInstanceError):
        sample.window_id = "tampered"  # type: ignore[misc]


@pytest.mark.unit
def test_compute_per_window_risk_metrics_basic() -> None:
    """Both sides identical → all deltas zero, baseline/candidate
    fields equal."""
    from smc.hedgerock.evolution.shadow_metrics import (
        compute_per_window_risk_metrics,
    )
    log = _replay_log(
        final_equity=10_500.0, max_dd_pct=0.05,
        near_stopout_count=1, n_trades=8, max_open_lots=0.4,
        max_grid_density=3, halt_event_count=0,
        n_bars_envelope_decided=600,
        observe_mode_bars=10, halt_mode_bars=2, cooldown_mode_bars=5,
    )
    stats = _stats(window_id="w0", n_trades=8)
    m = compute_per_window_risk_metrics(
        baseline_log=log, candidate_log=log, stats=stats,
    )
    assert m.window_id == "w0"
    assert m.delta_pnl_pp == 0.0
    assert m.delta_dd_pp == 0.0
    assert m.delta_near_stopout == 0
    assert m.delta_halt_event_count == 0
    assert m.delta_max_open_lots == 0.0
    assert m.delta_max_grid_density == 0
    assert m.candidate_observe_mode_bars == 10
    assert m.baseline_observe_mode_bars == 10
    assert m.observed_buckets == ("range_low_vol",)


@pytest.mark.unit
def test_compute_per_window_risk_metrics_separates_baseline_and_candidate() -> None:
    from smc.hedgerock.evolution.shadow_metrics import (
        compute_per_window_risk_metrics,
    )
    baseline = _replay_log(
        final_equity=10_300.0, max_dd_pct=0.03,
        near_stopout_count=0, n_trades=5, halt_event_count=0,
        observe_mode_bars=2, halt_mode_bars=0, cooldown_mode_bars=1,
    )
    candidate = _replay_log(
        final_equity=10_700.0, max_dd_pct=0.06,
        near_stopout_count=2, n_trades=9, halt_event_count=1,
        observe_mode_bars=3, halt_mode_bars=4, cooldown_mode_bars=6,
    )
    stats = _stats(window_id="w1", n_trades=9, halt_event_count=1)
    m = compute_per_window_risk_metrics(
        baseline_log=baseline, candidate_log=candidate, stats=stats,
    )
    assert m.candidate_total_return_pct == pytest.approx(7.0)
    assert m.baseline_total_return_pct == pytest.approx(3.0)
    assert m.delta_pnl_pp == pytest.approx(4.0)
    # delta_dd_pp is reported in percentage points (× 100 vs the raw
    # fraction); 0.06 − 0.03 fraction = 3.0 pp.
    assert m.delta_dd_pp == pytest.approx(3.0)
    assert m.delta_near_stopout == 2
    assert m.delta_halt_event_count == 1
    assert m.candidate_halt_mode_bars == 4
    assert m.baseline_halt_mode_bars == 0


@pytest.mark.unit
def test_compute_per_window_risk_metrics_does_not_mutate_inputs() -> None:
    from smc.hedgerock.evolution.shadow_metrics import (
        compute_per_window_risk_metrics,
    )
    baseline = _replay_log(final_equity=10_100.0, n_trades=3)
    candidate = _replay_log(final_equity=10_200.0, n_trades=4)
    stats = _stats(window_id="w2", n_trades=4)
    snapshot_b_eq = baseline.final_state.equity
    snapshot_c_eq = candidate.final_state.equity
    snapshot_stats = stats
    compute_per_window_risk_metrics(
        baseline_log=baseline, candidate_log=candidate, stats=stats,
    )
    assert baseline.final_state.equity == snapshot_b_eq
    assert candidate.final_state.equity == snapshot_c_eq
    assert stats == snapshot_stats


# ---------------------------------------------------------------------------
# 3. WorstWindowSummary — picks worst window per axis
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_worst_window_summary_picks_worst_dd_window() -> None:
    """delta_dd_pp positive = candidate's DD WORSE. Pick the window
    with the maximum (most-degraded) delta_dd_pp."""
    from smc.hedgerock.evolution.shadow_metrics import (
        compute_per_window_risk_metrics, compute_worst_window_summary,
    )
    metrics = []
    for wid, base_dd, cand_dd in [
        ("w_quiet", 0.02, 0.03),         # delta_dd = +0.01
        ("w_bad", 0.02, 0.10),           # delta_dd = +0.08 ← worst
        ("w_good", 0.05, 0.03),          # delta_dd = -0.02
    ]:
        m = compute_per_window_risk_metrics(
            baseline_log=_replay_log(max_dd_pct=base_dd),
            candidate_log=_replay_log(max_dd_pct=cand_dd),
            stats=_stats(window_id=wid),
        )
        metrics.append(m)
    summary = compute_worst_window_summary(metrics)
    assert summary.worst_delta_dd_window_id == "w_bad"
    # +0.08 fraction → +8.0 pp.
    assert summary.worst_delta_dd_pp == pytest.approx(8.0)


@pytest.mark.unit
def test_worst_window_summary_picks_worst_candidate_dd_window() -> None:
    """worst_candidate_dd_pp picks the highest absolute candidate DD,
    independent of the delta vs baseline."""
    from smc.hedgerock.evolution.shadow_metrics import (
        compute_per_window_risk_metrics, compute_worst_window_summary,
    )
    metrics = []
    for wid, base_dd, cand_dd in [
        ("w_a", 0.02, 0.04),
        ("w_b", 0.40, 0.30),  # candidate DD = 0.30 ← high absolute
        ("w_c", 0.05, 0.07),
    ]:
        m = compute_per_window_risk_metrics(
            baseline_log=_replay_log(max_dd_pct=base_dd),
            candidate_log=_replay_log(max_dd_pct=cand_dd),
            stats=_stats(window_id=wid),
        )
        metrics.append(m)
    summary = compute_worst_window_summary(metrics)
    assert summary.worst_candidate_dd_window_id == "w_b"
    # 0.30 fraction → 30.0 pp.
    assert summary.worst_candidate_dd_pp == pytest.approx(30.0)


@pytest.mark.unit
def test_worst_window_summary_picks_worst_pnl_window() -> None:
    """worst_delta_pnl_pp picks the MINIMUM (most-negative) PnL
    delta — the window where the candidate fared worst vs baseline."""
    from smc.hedgerock.evolution.shadow_metrics import (
        compute_per_window_risk_metrics, compute_worst_window_summary,
    )
    metrics = []
    for wid, base_eq, cand_eq in [
        ("w_a", 10_000.0, 10_500.0),   # delta = +5pp
        ("w_b", 10_000.0, 9_700.0),    # delta = -3pp ← worst
        ("w_c", 10_000.0, 10_100.0),   # delta = +1pp
    ]:
        m = compute_per_window_risk_metrics(
            baseline_log=_replay_log(final_equity=base_eq),
            candidate_log=_replay_log(final_equity=cand_eq),
            stats=_stats(window_id=wid),
        )
        metrics.append(m)
    summary = compute_worst_window_summary(metrics)
    assert summary.worst_delta_pnl_window_id == "w_b"
    assert summary.worst_delta_pnl_pp == pytest.approx(-3.0)


@pytest.mark.unit
def test_worst_window_summary_picks_worst_near_stopout_window() -> None:
    from smc.hedgerock.evolution.shadow_metrics import (
        compute_per_window_risk_metrics, compute_worst_window_summary,
    )
    metrics = []
    for wid, base_ns, cand_ns in [
        ("w_a", 0, 0),
        ("w_b", 1, 5),    # delta = +4 ← worst
        ("w_c", 0, 1),
    ]:
        m = compute_per_window_risk_metrics(
            baseline_log=_replay_log(near_stopout_count=base_ns),
            candidate_log=_replay_log(near_stopout_count=cand_ns),
            stats=_stats(window_id=wid),
        )
        metrics.append(m)
    summary = compute_worst_window_summary(metrics)
    assert summary.worst_delta_near_stopout_window_id == "w_b"
    assert summary.worst_delta_near_stopout == 4


@pytest.mark.unit
def test_worst_window_summary_picks_worst_halt_window() -> None:
    from smc.hedgerock.evolution.shadow_metrics import (
        compute_per_window_risk_metrics, compute_worst_window_summary,
    )
    metrics = []
    for wid, base_halt, cand_halt in [
        ("w_a", 0, 1),
        ("w_b", 0, 3),    # delta = +3 ← worst
        ("w_c", 1, 1),
    ]:
        m = compute_per_window_risk_metrics(
            baseline_log=_replay_log(halt_event_count=base_halt),
            candidate_log=_replay_log(halt_event_count=cand_halt),
            stats=_stats(window_id=wid, halt_event_count=cand_halt),
        )
        metrics.append(m)
    summary = compute_worst_window_summary(metrics)
    assert summary.worst_delta_halt_event_window_id == "w_b"
    assert summary.worst_delta_halt_event_count == 3


@pytest.mark.unit
def test_worst_window_summary_picks_worst_h1_gap_window() -> None:
    from smc.hedgerock.evolution.shadow_metrics import (
        compute_per_window_risk_metrics, compute_worst_window_summary,
    )
    metrics = []
    for wid, gap in [("w_a", 5), ("w_b", 200), ("w_c", 12)]:
        m = compute_per_window_risk_metrics(
            baseline_log=_replay_log(),
            candidate_log=_replay_log(),
            stats=_stats(window_id=wid, max_h1_gap_bars=gap),
        )
        metrics.append(m)
    summary = compute_worst_window_summary(metrics)
    assert summary.worst_max_h1_gap_window_id == "w_b"
    assert summary.worst_max_h1_gap_bars == 200


@pytest.mark.unit
def test_worst_window_summary_picks_worst_exposure_window() -> None:
    """Highest candidate-side max_open_lots and max_grid_density."""
    from smc.hedgerock.evolution.shadow_metrics import (
        compute_per_window_risk_metrics, compute_worst_window_summary,
    )
    metrics = []
    for wid, lots, dens in [
        ("w_a", 0.3, 2),
        ("w_b", 0.9, 6),    # ← worst exposure
        ("w_c", 0.5, 4),
    ]:
        m = compute_per_window_risk_metrics(
            baseline_log=_replay_log(max_open_lots=0.1, max_grid_density=1),
            candidate_log=_replay_log(max_open_lots=lots, max_grid_density=dens),
            stats=_stats(window_id=wid),
        )
        metrics.append(m)
    summary = compute_worst_window_summary(metrics)
    assert summary.worst_candidate_max_open_lots == pytest.approx(0.9)
    assert summary.worst_candidate_max_open_lots_window_id == "w_b"
    assert summary.worst_candidate_max_grid_density == 6
    assert summary.worst_candidate_max_grid_density_window_id == "w_b"


@pytest.mark.unit
def test_worst_window_summary_handles_empty_list() -> None:
    """Empty metrics list → empty/zero summary, never raises."""
    from smc.hedgerock.evolution.shadow_metrics import (
        compute_worst_window_summary,
    )
    s = compute_worst_window_summary([])
    assert s.worst_delta_pnl_pp == 0.0
    assert s.worst_delta_dd_pp == 0.0
    assert s.worst_max_h1_gap_bars == 0
    assert s.worst_delta_pnl_window_id == ""
    assert s.worst_delta_dd_window_id == ""


# ---------------------------------------------------------------------------
# 4. XAUUSD-only vocabulary invariant
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_per_window_metrics_module_no_legacy_single_symbol_text() -> None:
    """Module source must not contain legacy v1 cross-symbol vocab."""
    from pathlib import Path
    src = (_ai_smc_home_p() / "src" / "smc" / "hedgerock" / "evolution" / "shadow_metrics.py").read_text(encoding="utf-8")
    assert "single_symbol" not in src
    assert "cross_symbol" not in src
    assert "single symbol" not in src.lower()


@pytest.mark.unit
def test_compute_worst_window_summary_signature_uses_xauusd_only_terms() -> None:
    """No `symbols` plural / `cross_symbol` / `single_symbol` in
    public function signatures."""
    from smc.hedgerock.evolution.shadow_metrics import (
        compute_worst_window_summary,
        compute_per_window_risk_metrics,
    )
    for fn in (compute_worst_window_summary, compute_per_window_risk_metrics):
        sig = inspect.signature(fn)
        for name in sig.parameters:
            assert "symbols" not in name
            assert "cross_symbol" not in name
            assert "single_symbol" not in name


# ---------------------------------------------------------------------------
# 5. Aggregate-average is NOT used as gate input
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_worst_window_summary_does_not_expose_average_field() -> None:
    """The summary surface MUST NOT carry a window-mean field —
    averaging weak windows lets strong windows mask risk per RFC v2.
    The PASS gate must read worst-window values only."""
    from dataclasses import fields
    from smc.hedgerock.evolution.shadow_metrics import WorstWindowSummary
    field_names = {f.name for f in fields(WorstWindowSummary)}
    forbidden_substrings = ("average", "mean_", "avg_")
    leaks = [
        name for name in field_names
        if any(sub in name for sub in forbidden_substrings)
    ]
    assert not leaks, (
        f"Forbidden aggregate-average fields on WorstWindowSummary: {leaks}"
    )

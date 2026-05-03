"""Ticket 4 v2 Step 6 — XAUUSD-only multi-window PASS evaluator tests.

The active evaluator consumes:
  * a :class:`CoverageReport` (Step 2)
  * a :class:`WorstWindowSummary` (Step 5)
  * a list of :class:`PerWindowRiskMetrics` (Step 5)
  * mirror_consistency, exposure_class_violation, affects_halt_mode

It returns a :class:`MultiWindowPassEvaluation`. Pinned guarantees:

  * eligible_for_pass=True only when EVERY gate clears (worst-window
    floor on PnL, ceiling on DD/near-stopout/halt, full window-pass
    fraction).
  * blocking_window_id surfaces the binding window — the PASS report
    must point operators at the weak window, not at an aggregate.
  * abstain_reason vocabulary stays XAUUSD-only — never emits
    `single_symbol shadow window` or `cross_symbol` blocker.
  * Worst window dominates aggregate: a single weak window ABSTAINs
    even when the average across windows would clear thresholds.
  * mirror_consistency != "PASS" → unconditional ABSTAIN.
  * coverage_pass=False → propagates first; the evaluator does NOT
    silently approve when coverage is incomplete.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from smc.hedgerock.evolution.shadow_metrics import (
    PerWindowRiskMetrics,
    WorstWindowSummary,
    compute_worst_window_summary,
)
from smc.hedgerock.evolution.window_coverage import CoverageReport


from tests.hedgerock.evolution._paths import (
    ai_smc_home as _ai_smc_home_p,
    hedgerock_home as _hedgerock_home_p,
    real_audit_log as _real_audit_log_p,
    real_registry_root as _real_registry_p,
    real_shadow_artefacts_root as _real_shadow_p,
    scripts_dir as _scripts_dir_p,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _pwm(window_id: str, *,
         delta_pnl_pp: float = 1.0,
         delta_dd_pp: float = 0.0,
         delta_near_stopout: int = 0,
         delta_halt_event_count: int = 0,
         candidate_max_dd_pct: float = 0.005,   # fraction
         candidate_max_open_lots: float = 0.2,
         candidate_max_grid_density: int = 2,
         max_h1_gap_bars: int = 0,
         observed_buckets: tuple[str, ...] = ("range_low_vol",),
         ) -> PerWindowRiskMetrics:
    return PerWindowRiskMetrics(
        window_id=window_id, n_bars=1500, n_decided_bars=1000,
        max_h1_gap_bars=max_h1_gap_bars,
        observed_buckets=observed_buckets,
        candidate_final_equity=10_500.0,
        candidate_total_return_pct=2.0,
        baseline_total_return_pct=2.0 - delta_pnl_pp,
        delta_pnl_pp=delta_pnl_pp,
        candidate_max_dd_pct=candidate_max_dd_pct,
        baseline_max_dd_pct=candidate_max_dd_pct - (delta_dd_pp / 100.0),
        delta_dd_pp=delta_dd_pp,
        candidate_near_stopout_count=delta_near_stopout,
        baseline_near_stopout_count=0,
        delta_near_stopout=delta_near_stopout,
        candidate_max_open_lots=candidate_max_open_lots,
        baseline_max_open_lots=0.1,
        candidate_max_grid_density=candidate_max_grid_density,
        baseline_max_grid_density=1,
        delta_max_open_lots=candidate_max_open_lots - 0.1,
        delta_max_grid_density=candidate_max_grid_density - 1,
        candidate_halt_event_count=delta_halt_event_count,
        baseline_halt_event_count=0,
        delta_halt_event_count=delta_halt_event_count,
        candidate_observe_mode_bars=5,
        baseline_observe_mode_bars=4,
        candidate_halt_mode_bars=0,
        baseline_halt_mode_bars=0,
        candidate_cooldown_mode_bars=0,
        baseline_cooldown_mode_bars=0,
        candidate_n_trades=8,
        baseline_n_trades=8,
    )


def _coverage_pass_report(window_ids: tuple[str, ...]) -> CoverageReport:
    return CoverageReport(
        coverage_pass=True,
        windows_evaluated=window_ids,
        regime_buckets_covered=(
            "range_low_vol", "range_high_vol", "trend_up", "trend_down",
        ),
        halt_event_windows=2,
        no_trade_windows=(),
        shortfall_reasons=(),
        declared_vs_observed_mismatches=(),
    )


def _coverage_fail_report() -> CoverageReport:
    return CoverageReport(
        coverage_pass=False,
        windows_evaluated=("a", "b"),
        regime_buckets_covered=("range_low_vol",),
        halt_event_windows=0,
        no_trade_windows=("a", "b"),
        shortfall_reasons=(
            "insufficient_xauusd_window_coverage: have 2, need >= 6",
            "insufficient_regime_bucket_coverage: covered=['range_low_vol'], need >= 4",
        ),
        declared_vs_observed_mismatches=(),
    )


# ---------------------------------------------------------------------------
# 1. Clean PASS path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clean_path_eligible_for_pass() -> None:
    from smc.hedgerock.evolution.pass_evaluator import (
        evaluate_pass_xauusd_multi_window,
    )
    metrics = [_pwm(f"w{i}", delta_pnl_pp=1.5, delta_dd_pp=0.1)
               for i in range(6)]
    summary = compute_worst_window_summary(metrics)
    coverage = _coverage_pass_report(tuple(m.window_id for m in metrics))
    out = evaluate_pass_xauusd_multi_window(
        coverage_report=coverage,
        worst_summary=summary,
        per_window_metrics=metrics,
        mirror_consistency="PASS",
        exposure_class_violation=False,
        affects_halt_mode=False,
    )
    assert out.eligible_for_pass is True, out.abstain_reason
    assert out.abstain_reason == ""
    assert out.n_windows_total == 6
    assert out.n_windows_passing == 6
    assert out.failing_window_ids == ()


# ---------------------------------------------------------------------------
# 2. Mirror drift blocks immediately
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mirror_drift_blocks_pass() -> None:
    from smc.hedgerock.evolution.pass_evaluator import (
        evaluate_pass_xauusd_multi_window,
    )
    metrics = [_pwm(f"w{i}") for i in range(6)]
    out = evaluate_pass_xauusd_multi_window(
        coverage_report=_coverage_pass_report(tuple(m.window_id for m in metrics)),
        worst_summary=compute_worst_window_summary(metrics),
        per_window_metrics=metrics,
        mirror_consistency="FAIL",
        exposure_class_violation=False,
        affects_halt_mode=False,
    )
    assert out.eligible_for_pass is False
    assert "mirror_consistency" in out.abstain_reason


# ---------------------------------------------------------------------------
# 3. Coverage failure propagates with XAUUSD-only vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_coverage_failure_propagates() -> None:
    from smc.hedgerock.evolution.pass_evaluator import (
        evaluate_pass_xauusd_multi_window,
    )
    metrics = [_pwm("w0"), _pwm("w1")]
    out = evaluate_pass_xauusd_multi_window(
        coverage_report=_coverage_fail_report(),
        worst_summary=compute_worst_window_summary(metrics),
        per_window_metrics=metrics,
        mirror_consistency="PASS",
        exposure_class_violation=False,
        affects_halt_mode=False,
    )
    assert out.eligible_for_pass is False
    assert "xauusd_window_coverage" in out.abstain_reason or \
           "insufficient_xauusd_window_coverage" in out.abstain_reason
    assert out.coverage_pass is False
    # Critical RFC v2 invariant.
    assert "single_symbol" not in out.abstain_reason
    assert "cross_symbol" not in out.abstain_reason


# ---------------------------------------------------------------------------
# 4. Worst-window dominates aggregate
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_single_weak_window_blocks_pass_even_when_average_clears() -> None:
    """5 strong windows + 1 weak window — strong-window average
    would clear the floor but the weak window MUST trigger ABSTAIN."""
    from smc.hedgerock.evolution.pass_evaluator import (
        evaluate_pass_xauusd_multi_window,
    )
    strong = [_pwm(f"w_strong_{i}", delta_pnl_pp=2.0) for i in range(5)]
    weak = _pwm("w_weak", delta_pnl_pp=-1.5)   # below 0.5pp floor
    metrics = strong + [weak]
    out = evaluate_pass_xauusd_multi_window(
        coverage_report=_coverage_pass_report(tuple(m.window_id for m in metrics)),
        worst_summary=compute_worst_window_summary(metrics),
        per_window_metrics=metrics,
        mirror_consistency="PASS",
        exposure_class_violation=False,
        affects_halt_mode=False,
    )
    assert out.eligible_for_pass is False
    assert "delta_pnl" in out.abstain_reason or "pnl" in out.abstain_reason
    # Blocker must point at the weak window, not at an aggregate.
    assert out.blocking_window_id == "w_weak"


@pytest.mark.unit
def test_single_weak_dd_window_blocks_pass() -> None:
    from smc.hedgerock.evolution.pass_evaluator import (
        evaluate_pass_xauusd_multi_window,
    )
    strong = [_pwm(f"w_strong_{i}", delta_dd_pp=0.0) for i in range(5)]
    bad = _pwm("w_dd_bad", delta_dd_pp=2.0)   # > 0.5pp ceiling
    metrics = strong + [bad]
    out = evaluate_pass_xauusd_multi_window(
        coverage_report=_coverage_pass_report(tuple(m.window_id for m in metrics)),
        worst_summary=compute_worst_window_summary(metrics),
        per_window_metrics=metrics,
        mirror_consistency="PASS",
        exposure_class_violation=False,
        affects_halt_mode=False,
    )
    assert out.eligible_for_pass is False
    assert "dd" in out.abstain_reason.lower()
    assert out.blocking_window_id == "w_dd_bad"


@pytest.mark.unit
def test_single_weak_near_stopout_window_blocks_pass() -> None:
    from smc.hedgerock.evolution.pass_evaluator import (
        evaluate_pass_xauusd_multi_window,
    )
    strong = [_pwm(f"w_strong_{i}") for i in range(5)]
    bad = _pwm("w_ns_bad", delta_near_stopout=2)
    metrics = strong + [bad]
    out = evaluate_pass_xauusd_multi_window(
        coverage_report=_coverage_pass_report(tuple(m.window_id for m in metrics)),
        worst_summary=compute_worst_window_summary(metrics),
        per_window_metrics=metrics,
        mirror_consistency="PASS",
        exposure_class_violation=False,
        affects_halt_mode=False,
    )
    assert out.eligible_for_pass is False
    assert "near_stopout" in out.abstain_reason
    assert out.blocking_window_id == "w_ns_bad"


@pytest.mark.unit
def test_single_weak_halt_window_blocks_pass() -> None:
    from smc.hedgerock.evolution.pass_evaluator import (
        evaluate_pass_xauusd_multi_window,
    )
    strong = [_pwm(f"w_strong_{i}") for i in range(5)]
    bad = _pwm("w_halt_bad", delta_halt_event_count=1)
    metrics = strong + [bad]
    out = evaluate_pass_xauusd_multi_window(
        coverage_report=_coverage_pass_report(tuple(m.window_id for m in metrics)),
        worst_summary=compute_worst_window_summary(metrics),
        per_window_metrics=metrics,
        mirror_consistency="PASS",
        exposure_class_violation=False,
        affects_halt_mode=False,
    )
    assert out.eligible_for_pass is False
    assert "halt" in out.abstain_reason
    assert out.blocking_window_id == "w_halt_bad"


@pytest.mark.unit
def test_high_absolute_candidate_dd_blocks_pass() -> None:
    """Even when delta_dd is small, absolute candidate DD above the
    worst-window ceiling must ABSTAIN."""
    from smc.hedgerock.evolution.pass_evaluator import (
        evaluate_pass_xauusd_multi_window,
    )
    strong = [_pwm(f"w_strong_{i}", candidate_max_dd_pct=0.005) for i in range(5)]
    bad = _pwm("w_abs_dd", candidate_max_dd_pct=0.05)   # 5pp >> 1pp ceiling
    metrics = strong + [bad]
    out = evaluate_pass_xauusd_multi_window(
        coverage_report=_coverage_pass_report(tuple(m.window_id for m in metrics)),
        worst_summary=compute_worst_window_summary(metrics),
        per_window_metrics=metrics,
        mirror_consistency="PASS",
        exposure_class_violation=False,
        affects_halt_mode=False,
    )
    assert out.eligible_for_pass is False
    assert "candidate_dd" in out.abstain_reason or "absolute" in out.abstain_reason \
           or "dd_above_ceiling" in out.abstain_reason
    assert out.blocking_window_id == "w_abs_dd"


# ---------------------------------------------------------------------------
# 5. Exposure-class violation blocks
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_exposure_class_violation_blocks() -> None:
    from smc.hedgerock.evolution.pass_evaluator import (
        evaluate_pass_xauusd_multi_window,
    )
    metrics = [_pwm(f"w{i}") for i in range(6)]
    out = evaluate_pass_xauusd_multi_window(
        coverage_report=_coverage_pass_report(tuple(m.window_id for m in metrics)),
        worst_summary=compute_worst_window_summary(metrics),
        per_window_metrics=metrics,
        mirror_consistency="PASS",
        exposure_class_violation=True,
        affects_halt_mode=False,
    )
    assert out.eligible_for_pass is False
    assert "exposure_class" in out.abstain_reason


# ---------------------------------------------------------------------------
# 6. min_pass_window_fraction enforcement (default 1.0)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_failing_window_fraction_recorded() -> None:
    from smc.hedgerock.evolution.pass_evaluator import (
        evaluate_pass_xauusd_multi_window,
    )
    metrics = [
        _pwm("w0", delta_pnl_pp=1.5),
        _pwm("w1", delta_pnl_pp=1.5),
        _pwm("w2", delta_pnl_pp=-2.0),  # fails delta_pnl floor
        _pwm("w3", delta_pnl_pp=-1.0),  # fails delta_pnl floor
        _pwm("w4", delta_pnl_pp=1.5),
        _pwm("w5", delta_pnl_pp=1.5),
    ]
    out = evaluate_pass_xauusd_multi_window(
        coverage_report=_coverage_pass_report(tuple(m.window_id for m in metrics)),
        worst_summary=compute_worst_window_summary(metrics),
        per_window_metrics=metrics,
        mirror_consistency="PASS",
        exposure_class_violation=False,
        affects_halt_mode=False,
    )
    assert out.eligible_for_pass is False
    assert out.n_windows_total == 6
    assert out.n_windows_passing == 4
    assert set(out.failing_window_ids) == {"w2", "w3"}


# ---------------------------------------------------------------------------
# 7. XAUUSD-only vocabulary invariant
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_module_no_legacy_single_symbol_text() -> None:
    """The new evaluator's docstrings/reasons stay XAUUSD-only.
    Legacy v1 text still exists in the dormant `evaluate_pass`
    function — that's expected. We only require that NO new line
    introduced for the multi-window evaluator emits `single_symbol`
    or `cross_symbol` blocker phrasing in its returned reasons."""
    src = (_ai_smc_home_p() / "src" / "smc" / "hedgerock" / "evolution" / "pass_evaluator.py").read_text(encoding="utf-8")
    # Locate the new function and its dataclasses; ensure their
    # surrounding block doesn't emit cross_symbol phrasing.
    assert "evaluate_pass_xauusd_multi_window" in src
    assert "MultiWindowPassThresholds" in src
    assert "MultiWindowPassEvaluation" in src

    # Ensure the literal v1 blocker string is NOT used in the new
    # function's body. We split the source and check that the new
    # function block doesn't contain it.
    marker = "def evaluate_pass_xauusd_multi_window"
    if marker in src:
        body = src[src.index(marker):]
        assert "single_symbol shadow window" not in body
        assert "cross_symbol" not in body


@pytest.mark.unit
def test_pass_evaluator_signature_uses_xauusd_only_terms() -> None:
    """Public signature MUST NOT take a `symbols` plural / cross
    parameter."""
    from smc.hedgerock.evolution.pass_evaluator import (
        evaluate_pass_xauusd_multi_window,
    )
    sig = inspect.signature(evaluate_pass_xauusd_multi_window)
    for name in sig.parameters:
        assert "symbols" not in name
        assert "cross_symbol" not in name
        assert "single_symbol" not in name

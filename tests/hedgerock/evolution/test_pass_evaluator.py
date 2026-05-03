"""Ticket 3 Step 6 — dormant PASS evaluator + multi-symbol skeleton.

Per R1 (Ticket 3 Plan v2): the PASS path is implemented but DORMANT
under v1's single-symbol lake. Tests verify that:

  - Single-symbol bundle → evaluator returns PassEvaluation with
    `eligible == False` AND reason = "single_symbol".
  - Even on a synthetic multi-symbol bundle with all metrics
    perfect, evaluator can return eligible == True ONLY when:
      * cross_symbol_count >= 2
      * each symbol's years_total >= 3
      * no negative_sign_years
      * halt_event_count >= min (when candidate affects halt)
      * delta_pnl > threshold AND delta_dd <= tolerance
      * near_stopout count not increased
      * exposure_class_violation == False
      * mirror_consistency_check == "PASS"
  - Lower any one criterion → eligible False with the relevant
    reason.
"""

from __future__ import annotations

import pytest

from smc.hedgerock.evolution.pass_evaluator import (
    PassEvaluation,
    PassThresholds,
    evaluate_pass,
)
from smc.hedgerock.evolution.shadow_artefact import ShadowMetrics


def _z_metrics(
    *, n_trades: int = 4, dd: float = 1.0, ret: float = 1.0,
    near_stopout: int = 0, halt: int = 1, n_bars: int = 100,
    max_open_lots: float = 0.1, max_grid_density: int = 1,
    final_equity: float | None = None,
) -> ShadowMetrics:
    fe = final_equity if final_equity is not None else 10_000.0 + (ret * 100.0)
    return ShadowMetrics(
        final_equity=fe,
        total_return_pct=ret,
        max_dd_pct=dd,
        near_stopout_count=near_stopout,
        n_trades=n_trades,
        max_open_lots=max_open_lots,
        max_grid_density=max_grid_density,
        halt_event_count=halt,
        n_bars_envelope_decided=n_bars,
    )


def _multi_sym_year_repl():
    return {
        "XAUUSD": {"years_total": 4, "years_passing": 4,
                    "negative_sign_years": ()},
        "EURUSD": {"years_total": 4, "years_passing": 4,
                    "negative_sign_years": ()},
    }


# ---------------------------------------------------------------------------
# 1. Single-symbol → DORMANT (eligible=False, reason single_symbol)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_single_symbol_blocks_pass_eligibility() -> None:
    baseline = _z_metrics(ret=1.0, dd=2.0)
    candidate = _z_metrics(ret=2.5, dd=1.5)  # better
    delta = ShadowMetrics(
        final_equity=candidate.final_equity,
        total_return_pct=candidate.total_return_pct - baseline.total_return_pct,
        max_dd_pct=candidate.max_dd_pct - baseline.max_dd_pct,
        near_stopout_count=candidate.near_stopout_count - baseline.near_stopout_count,
        n_trades=candidate.n_trades - baseline.n_trades,
        max_open_lots=candidate.max_open_lots - baseline.max_open_lots,
        max_grid_density=candidate.max_grid_density - baseline.max_grid_density,
        halt_event_count=candidate.halt_event_count - baseline.halt_event_count,
        n_bars_envelope_decided=0,
    )
    out = evaluate_pass(
        symbols=("XAUUSD",),  # single symbol
        year_replication={"XAUUSD": {"years_total": 4, "years_passing": 4,
                                       "negative_sign_years": ()}},
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        delta_metrics=delta,
        exposure_class_violation=False,
        mirror_consistency="PASS",
        affects_halt_mode=False,
    )
    assert isinstance(out, PassEvaluation)
    assert out.eligible_for_pass is False
    assert "single_symbol" in out.block_reason


# ---------------------------------------------------------------------------
# 2. Multi-symbol synthetic + clean → eligible == True
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_multi_symbol_clean_inputs_yield_eligible_pass() -> None:
    """The PASS branch must be REACHABLE in code (not ifdef'd out)
    given perfect synthetic inputs. Real lake never reaches here."""
    baseline = _z_metrics(ret=1.0, dd=2.0)
    candidate = _z_metrics(ret=2.5, dd=1.5)
    delta = ShadowMetrics(
        final_equity=candidate.final_equity,
        total_return_pct=1.5, max_dd_pct=-0.5,
        near_stopout_count=0, n_trades=0, max_open_lots=0.0,
        max_grid_density=0, halt_event_count=0,
        n_bars_envelope_decided=0,
    )
    out = evaluate_pass(
        symbols=("XAUUSD", "EURUSD"),
        year_replication=_multi_sym_year_repl(),
        baseline_metrics=baseline, candidate_metrics=candidate,
        delta_metrics=delta,
        exposure_class_violation=False, mirror_consistency="PASS",
        affects_halt_mode=False,
    )
    assert out.eligible_for_pass is True
    assert out.block_reason == ""


# ---------------------------------------------------------------------------
# 3. Each individual gate blocks PASS
# ---------------------------------------------------------------------------


def _base_kwargs():
    return dict(
        symbols=("XAUUSD", "EURUSD"),
        year_replication=_multi_sym_year_repl(),
        baseline_metrics=_z_metrics(ret=1.0, dd=2.0),
        candidate_metrics=_z_metrics(ret=2.5, dd=1.5),
        delta_metrics=ShadowMetrics(
            final_equity=10_250.0, total_return_pct=1.5, max_dd_pct=-0.5,
            near_stopout_count=0, n_trades=0, max_open_lots=0.0,
            max_grid_density=0, halt_event_count=0, n_bars_envelope_decided=0,
        ),
        exposure_class_violation=False,
        mirror_consistency="PASS",
        affects_halt_mode=False,
    )


@pytest.mark.unit
def test_negative_sign_year_blocks_pass() -> None:
    kwargs = _base_kwargs()
    kwargs["year_replication"] = {
        "XAUUSD": {"years_total": 4, "years_passing": 3,
                    "negative_sign_years": (2021,)},
        "EURUSD": {"years_total": 4, "years_passing": 4,
                    "negative_sign_years": ()},
    }
    out = evaluate_pass(**kwargs)
    assert out.eligible_for_pass is False
    assert "negative_sign_year" in out.block_reason or \
           "2021" in out.block_reason


@pytest.mark.unit
def test_insufficient_years_blocks_pass() -> None:
    kwargs = _base_kwargs()
    kwargs["year_replication"] = {
        "XAUUSD": {"years_total": 1, "years_passing": 1,
                    "negative_sign_years": ()},
        "EURUSD": {"years_total": 4, "years_passing": 4,
                    "negative_sign_years": ()},
    }
    out = evaluate_pass(**kwargs)
    assert out.eligible_for_pass is False
    assert "years" in out.block_reason.lower()


@pytest.mark.unit
def test_negative_delta_pnl_blocks_pass() -> None:
    kwargs = _base_kwargs()
    kwargs["delta_metrics"] = ShadowMetrics(
        final_equity=9_900.0, total_return_pct=-0.5, max_dd_pct=0.0,
        near_stopout_count=0, n_trades=0, max_open_lots=0.0,
        max_grid_density=0, halt_event_count=0, n_bars_envelope_decided=0,
    )
    out = evaluate_pass(**kwargs)
    assert out.eligible_for_pass is False
    assert "delta_pnl" in out.block_reason or "pnl" in out.block_reason.lower()


@pytest.mark.unit
def test_dd_above_tolerance_blocks_pass() -> None:
    kwargs = _base_kwargs()
    kwargs["delta_metrics"] = ShadowMetrics(
        final_equity=10_300.0, total_return_pct=3.0, max_dd_pct=2.0,
        near_stopout_count=0, n_trades=0, max_open_lots=0.0,
        max_grid_density=0, halt_event_count=0, n_bars_envelope_decided=0,
    )
    out = evaluate_pass(**kwargs)
    assert out.eligible_for_pass is False
    assert "delta_dd" in out.block_reason or "dd" in out.block_reason.lower()


@pytest.mark.unit
def test_near_stopout_increase_blocks_pass() -> None:
    kwargs = _base_kwargs()
    kwargs["delta_metrics"] = ShadowMetrics(
        final_equity=10_300.0, total_return_pct=3.0, max_dd_pct=-0.1,
        near_stopout_count=2, n_trades=0, max_open_lots=0.0,
        max_grid_density=0, halt_event_count=0, n_bars_envelope_decided=0,
    )
    out = evaluate_pass(**kwargs)
    assert out.eligible_for_pass is False
    assert "near_stopout" in out.block_reason


@pytest.mark.unit
def test_halt_increase_blocks_pass() -> None:
    kwargs = _base_kwargs()
    kwargs["delta_metrics"] = ShadowMetrics(
        final_equity=10_300.0, total_return_pct=3.0, max_dd_pct=-0.1,
        near_stopout_count=0, n_trades=0, max_open_lots=0.0,
        max_grid_density=0, halt_event_count=3, n_bars_envelope_decided=0,
    )
    out = evaluate_pass(**kwargs)
    assert out.eligible_for_pass is False
    assert "halt" in out.block_reason


@pytest.mark.unit
def test_exposure_class_violation_blocks_pass() -> None:
    kwargs = _base_kwargs()
    kwargs["exposure_class_violation"] = True
    out = evaluate_pass(**kwargs)
    assert out.eligible_for_pass is False
    assert "exposure" in out.block_reason


@pytest.mark.unit
def test_mirror_drift_blocks_pass() -> None:
    kwargs = _base_kwargs()
    kwargs["mirror_consistency"] = "FAIL"
    out = evaluate_pass(**kwargs)
    assert out.eligible_for_pass is False
    assert "mirror" in out.block_reason.lower()


@pytest.mark.unit
def test_halt_corpus_blocks_pass_when_candidate_affects_halt() -> None:
    kwargs = _base_kwargs()
    kwargs["affects_halt_mode"] = True
    kwargs["candidate_metrics"] = _z_metrics(ret=2.5, dd=1.5, halt=4)  # n=4 < 30
    out = evaluate_pass(**kwargs)
    assert out.eligible_for_pass is False
    assert "halt_corpus" in out.block_reason or "30" in out.block_reason


# ---------------------------------------------------------------------------
# 4. Threshold defaults are at least the values from Ticket 1+2 history
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_thresholds_align_with_existing_gate_config() -> None:
    """Defaults: delta_pnl threshold > 0, delta_dd tolerance ≥ 0,
    halt_min ≥ 30 (Ticket 1 G5 floor)."""
    t = PassThresholds()
    assert t.min_delta_pnl_pct > 0.0
    assert t.max_delta_dd_pct >= 0.0
    assert t.min_halt_events >= 30

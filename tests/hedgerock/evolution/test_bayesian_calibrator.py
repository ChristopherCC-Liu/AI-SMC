"""Tests for the Bayesian Beta–Bernoulli calibrator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smc.hedgerock.evolution.bayesian_calibrator import (
    BayesianCalibrator,
    DEFAULT_PARAMETER_CLASSES,
    ParameterClassPrior,
    load_calibrator,
)


# ---------------------------------------------------------------------------
# ParameterClassPrior arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_uninformative_prior_mean_is_half() -> None:
    p = ParameterClassPrior(parameter_class="x")
    assert p.mean == 0.5
    assert p.alpha == 1.0
    assert p.beta == 1.0


@pytest.mark.unit
def test_update_increases_alpha_on_success_only() -> None:
    p = ParameterClassPrior(parameter_class="x")
    p2 = p.update(n_success=3, n_failure=1)
    assert p2.alpha == pytest.approx(4.0)
    assert p2.beta == pytest.approx(2.0)
    assert p2.n_observations == 4


@pytest.mark.unit
def test_negative_counts_rejected() -> None:
    p = ParameterClassPrior(parameter_class="x")
    with pytest.raises(ValueError):
        p.update(n_success=-1, n_failure=0)


@pytest.mark.unit
def test_mean_converges_with_lots_of_successes() -> None:
    p = ParameterClassPrior(parameter_class="x")
    for _ in range(50):
        p = p.update(n_success=1, n_failure=0)
    assert p.mean > 0.9


@pytest.mark.unit
def test_mean_collapses_on_repeated_failures() -> None:
    p = ParameterClassPrior(parameter_class="x")
    for _ in range(50):
        p = p.update(n_success=0, n_failure=1)
    assert p.mean < 0.05


@pytest.mark.unit
def test_variance_non_negative_and_finite() -> None:
    p = ParameterClassPrior(parameter_class="x", alpha=10, beta=5)
    assert 0 <= p.variance < 1


@pytest.mark.unit
def test_dict_round_trip() -> None:
    p = ParameterClassPrior(parameter_class="z", alpha=3.0, beta=2.0,
                            n_observations=4, last_updated_at="2026-05-03T00:00:00+00:00")
    out = ParameterClassPrior.from_dict(p.to_dict())
    assert out == p


# ---------------------------------------------------------------------------
# Calibrator orchestration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_calibrator_seeds_known_parameter_classes() -> None:
    cal = BayesianCalibrator.with_uninformative_priors()
    for pc in DEFAULT_PARAMETER_CLASSES:
        assert cal.calibrated_confidence(pc) == 0.5


@pytest.mark.unit
def test_unknown_class_lazy_initialised_with_uninformative_prior() -> None:
    cal = BayesianCalibrator()
    assert cal.calibrated_confidence("never_seen_before") == 0.5


@pytest.mark.unit
def test_update_from_outcome_changes_posterior() -> None:
    cal = BayesianCalibrator.with_uninformative_priors()
    pc = "confidence_threshold_observe"
    cal.update_from_outcome(parameter_class=pc, success=True)
    assert cal.calibrated_confidence(pc) > 0.5
    cal.update_from_outcome(parameter_class=pc, success=False)
    # Two observations, one success one failure → 2/4 = 0.5
    assert cal.calibrated_confidence(pc) == pytest.approx(0.5, abs=1e-9)


@pytest.mark.unit
def test_paper_test_summary_success_path() -> None:
    cal = BayesianCalibrator.with_uninformative_priors()
    cal.update_from_paper_test_summary(
        parameter_class="confidence_threshold_observe",
        n_trades=20, pnl_sum=100.0, max_drawdown=-15.0,
        drawdown_floor=-50.0,
    )
    assert cal.calibrated_confidence("confidence_threshold_observe") > 0.5


@pytest.mark.unit
def test_paper_test_summary_failure_paths() -> None:
    cal = BayesianCalibrator.with_uninformative_priors()
    # No trades → failure
    cal.update_from_paper_test_summary(
        parameter_class="x", n_trades=0, pnl_sum=0.0, max_drawdown=0.0,
    )
    # Negative PnL → failure
    cal.update_from_paper_test_summary(
        parameter_class="x", n_trades=10, pnl_sum=-5.0, max_drawdown=-5.0,
    )
    # DD beyond floor → failure
    cal.update_from_paper_test_summary(
        parameter_class="x", n_trades=10, pnl_sum=100.0, max_drawdown=-200.0,
    )
    assert cal.calibrated_confidence("x") < 0.5


# ---------------------------------------------------------------------------
# JSON save/load
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_save_and_load_round_trips_state(tmp_path: Path) -> None:
    cal = BayesianCalibrator.with_uninformative_priors()
    cal.update_from_outcome(
        parameter_class="confidence_threshold_observe", success=True,
    )
    path = cal.save(tmp_path / "cal.json")
    parsed = json.loads(path.read_text())
    assert parsed["schema"] == "bayesian_calibrator/v1"

    loaded = BayesianCalibrator.load(path)
    assert (
        loaded.calibrated_confidence("confidence_threshold_observe")
        == cal.calibrated_confidence("confidence_threshold_observe")
    )


@pytest.mark.unit
def test_load_calibrator_missing_file_returns_uninformative(tmp_path: Path) -> None:
    cal = load_calibrator(tmp_path / "absent.json")
    for pc in DEFAULT_PARAMETER_CLASSES:
        assert cal.calibrated_confidence(pc) == 0.5


@pytest.mark.unit
def test_load_calibrator_none_returns_uninformative() -> None:
    cal = load_calibrator(None)
    assert cal.calibrated_confidence(DEFAULT_PARAMETER_CLASSES[0]) == 0.5


# ---------------------------------------------------------------------------
# Integration with candidate_generator (optional kwarg)
# ---------------------------------------------------------------------------


def _bundle():
    from smc.hedgerock.evolution.policy_manifest import EvidenceBundle
    return EvidenceBundle(
        bundle_id="<test>",
        bundle_hash_sha256="0" * 64,
        atlas_report_path="<test>",
        atlas_report_hash_sha256="0" * 64,
        data_availability_report_path="<test>",
        data_availability_report_hash_sha256="0" * 64,
        walk_forward_run_paths=("<test>",),
        year_replication={
            "XAUUSD": {"years_total": 5, "years_passing": 5,
                       "negative_sign_years": ()},
        },
        cross_symbol_count=1,
        halt_event_count=42,
        no_strategy_change=True,
    )


@pytest.mark.unit
def test_generator_populates_calibrated_confidence_when_calibrator_supplied() -> None:
    from smc.hedgerock.evolution.candidate_generator import (
        generate_candidate_proposals,
    )
    from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0

    cal = BayesianCalibrator.with_uninformative_priors()
    cal.update_from_outcome(
        parameter_class="confidence_threshold_observe", success=True,
    )
    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=_bundle(),
        gate_results_per_candidate={},
        blocking_reasons_per_candidate={},
        calibrator=cal,
    )
    by_id = {p.candidate_id: p for p in proposals}
    obs_proposal = by_id.get("c1-lower-observe-floor-0.50")
    assert obs_proposal is not None
    assert obs_proposal.calibrated_confidence is not None
    assert 0.0 < obs_proposal.calibrated_confidence < 1.0


@pytest.mark.unit
def test_generator_default_leaves_calibrated_confidence_none() -> None:
    from smc.hedgerock.evolution.candidate_generator import (
        generate_candidate_proposals,
    )
    from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0

    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=_bundle(),
        gate_results_per_candidate={},
        blocking_reasons_per_candidate={},
    )
    for p in proposals:
        assert p.calibrated_confidence is None

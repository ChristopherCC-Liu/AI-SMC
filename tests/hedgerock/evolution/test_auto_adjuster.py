"""Tests for src/smc/hedgerock/evolution/auto_adjuster.py."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from smc.hedgerock.evolution.auto_adjuster import (
    AdjustedParams,
    AdjustmentProposal,
    AutoAdjuster,
)
from smc.hedgerock.evolution.drift_detector import (
    DriftReport,
    DriftScore,
    FreshnessScore,
    QualityScore,
    StabilityScore,
)


# ---------------------------------------------------------------------------
# Helpers — synthesize a DriftReport directly with chosen severities.
# ---------------------------------------------------------------------------


def _report(
    *,
    regime_severity: str = "none",
    freshness_severity: str = "fresh",
    stability_severity: str = "stable",
    quality_severity: str = "good",
) -> DriftReport:
    return DriftReport(
        overall_severity="none",
        regime_baseline=DriftScore(
            score=0.0,
            severity=regime_severity,
            metric="regime_baseline",
            detail="synthetic",
        ),
        evidence_freshness=FreshnessScore(
            age_days=0.0,
            severity=freshness_severity,
            detail="synthetic",
        ),
        parameter_stability=StabilityScore(
            score=0.0,
            severity=stability_severity,
            diverged_keys=(),
            detail="synthetic",
        ),
        recommendation_quality=QualityScore(
            score=1.0,
            severity=quality_severity,
            detail="synthetic",
        ),
        generated_at="2026-05-03T00:00:00+00:00",
    )


def _midpoint(parameter: str) -> float:
    spec = AutoAdjuster.ADJUSTABLE_PARAMS[parameter]
    return (spec["floor"] + spec["ceiling"]) / 2.0


# ---------------------------------------------------------------------------
# 1. propose_adjustments — all healthy → no proposals
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_propose_adjustments_all_healthy_returns_empty():
    report = _report()
    proposals = AutoAdjuster.propose_adjustments(report)
    assert proposals == []


# ---------------------------------------------------------------------------
# 2. high regime drift → regime_vol_threshold proposal at full delta_max
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_high_regime_drift_full_delta_on_regime_vol_threshold():
    report = _report(regime_severity="high")
    proposals = AutoAdjuster.propose_adjustments(report)

    matches = [p for p in proposals if p.parameter == "regime_vol_threshold"]
    assert matches, "expected at least one regime_vol_threshold proposal"
    spec = AutoAdjuster.ADJUSTABLE_PARAMS["regime_vol_threshold"]
    assert matches[0].delta == pytest.approx(spec["delta_max"])
    assert matches[0].triggered_by == "regime_baseline"
    assert matches[0].severity == "high"


# ---------------------------------------------------------------------------
# 3. moderate regime drift → delta ≈ 0.5 * delta_max
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_moderate_regime_drift_half_delta():
    report = _report(regime_severity="moderate")
    proposals = AutoAdjuster.propose_adjustments(report)

    matches = [p for p in proposals if p.parameter == "regime_vol_threshold"]
    assert matches
    spec = AutoAdjuster.ADJUSTABLE_PARAMS["regime_vol_threshold"]
    assert matches[0].delta == pytest.approx(0.5 * spec["delta_max"])
    assert matches[0].severity == "moderate"


# ---------------------------------------------------------------------------
# 4. stale evidence freshness → confidence_prior_alpha + delta_max
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stale_freshness_full_delta_on_confidence_prior_alpha():
    report = _report(freshness_severity="stale")
    proposals = AutoAdjuster.propose_adjustments(report)

    matches = [p for p in proposals if p.parameter == "confidence_prior_alpha"]
    assert matches
    spec = AutoAdjuster.ADJUSTABLE_PARAMS["confidence_prior_alpha"]
    assert matches[0].delta == pytest.approx(spec["delta_max"])
    assert matches[0].delta > 0


# ---------------------------------------------------------------------------
# 5. unstable parameter_stability → anomaly_sensitivity_multiplier with negative delta
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unstable_stability_negative_delta_on_anomaly_sensitivity():
    report = _report(stability_severity="unstable")
    proposals = AutoAdjuster.propose_adjustments(report)

    matches = [
        p for p in proposals if p.parameter == "anomaly_sensitivity_multiplier"
    ]
    assert matches
    spec = AutoAdjuster.ADJUSTABLE_PARAMS["anomaly_sensitivity_multiplier"]
    assert matches[0].delta < 0
    assert matches[0].delta == pytest.approx(-spec["delta_max"])


# ---------------------------------------------------------------------------
# 6. degraded recommendation_quality → stop_atr_multiplier with positive delta
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_degraded_quality_positive_delta_on_stop_atr_multiplier():
    report = _report(quality_severity="degraded")
    proposals = AutoAdjuster.propose_adjustments(report)

    matches = [p for p in proposals if p.parameter == "stop_atr_multiplier"]
    assert matches
    spec = AutoAdjuster.ADJUSTABLE_PARAMS["stop_atr_multiplier"]
    assert matches[0].delta > 0
    assert matches[0].delta == pytest.approx(spec["delta_max"])


# ---------------------------------------------------------------------------
# 7. validate_adjustment: out-of-range vs in-range
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_adjustment_range_checks():
    spec = AutoAdjuster.ADJUSTABLE_PARAMS["regime_vol_threshold"]
    midpoint = (spec["floor"] + spec["ceiling"]) / 2.0

    in_range = AdjustmentProposal(
        parameter="regime_vol_threshold",
        current_value=midpoint,
        proposed_value=midpoint + spec["delta_max"],
        delta=spec["delta_max"],
        rationale="ok",
        severity="high",
        triggered_by="regime_baseline",
    )
    assert AutoAdjuster.validate_adjustment(in_range) is True

    # delta exceeds delta_max
    too_big_delta = AdjustmentProposal(
        parameter="regime_vol_threshold",
        current_value=midpoint,
        proposed_value=midpoint + spec["delta_max"] * 5,
        delta=spec["delta_max"] * 5,
        rationale="bad",
        severity="high",
        triggered_by="regime_baseline",
    )
    assert AutoAdjuster.validate_adjustment(too_big_delta) is False

    # proposed_value above ceiling
    above_ceiling = AdjustmentProposal(
        parameter="regime_vol_threshold",
        current_value=spec["ceiling"],
        proposed_value=spec["ceiling"] + spec["delta_max"] / 2.0,
        delta=spec["delta_max"] / 2.0,
        rationale="bad",
        severity="high",
        triggered_by="regime_baseline",
    )
    assert AutoAdjuster.validate_adjustment(above_ceiling) is False

    # proposed_value below floor
    below_floor = AdjustmentProposal(
        parameter="regime_vol_threshold",
        current_value=spec["floor"],
        proposed_value=spec["floor"] - spec["delta_max"] / 2.0,
        delta=-spec["delta_max"] / 2.0,
        rationale="bad",
        severity="high",
        triggered_by="parameter_stability",
    )
    assert AutoAdjuster.validate_adjustment(below_floor) is False


# ---------------------------------------------------------------------------
# 8. validate_adjustment: forbidden parameter → False
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_adjustment_forbidden_parameter_returns_false():
    forbidden = AdjustmentProposal(
        parameter="position_size",
        current_value=0.5,
        proposed_value=0.6,
        delta=0.1,
        rationale="should never reach here",
        severity="high",
        triggered_by="regime_baseline",
    )
    assert AutoAdjuster.validate_adjustment(forbidden) is False


# ---------------------------------------------------------------------------
# 9. apply_adjustments: forbidden parameter → ValueError
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_adjustments_forbidden_parameter_raises():
    forbidden = AdjustmentProposal(
        parameter="leverage",
        current_value=2.0,
        proposed_value=3.0,
        delta=1.0,
        rationale="malicious",
        severity="high",
        triggered_by="regime_baseline",
    )
    with pytest.raises(ValueError):
        AutoAdjuster.apply_adjustments([forbidden], {"leverage": 2.0})


# ---------------------------------------------------------------------------
# 10. apply_adjustments: out-of-bound proposal → rejected with audit reason
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_adjustments_out_of_bound_lands_in_rejected():
    spec = AutoAdjuster.ADJUSTABLE_PARAMS["regime_vol_threshold"]
    bad = AdjustmentProposal(
        parameter="regime_vol_threshold",
        current_value=spec["ceiling"],
        proposed_value=spec["ceiling"] + 10.0,  # way above ceiling
        delta=10.0,                              # way above delta_max
        rationale="oversized",
        severity="high",
        triggered_by="regime_baseline",
    )
    result = AutoAdjuster.apply_adjustments(
        [bad], {"regime_vol_threshold": spec["ceiling"]}
    )
    assert len(result.proposals_applied) == 0
    assert len(result.proposals_rejected) == 1
    rejected_proposal, reject_reason = result.proposals_rejected[0]
    assert rejected_proposal is bad
    assert reject_reason  # non-empty reason captured
    assert reject_reason != "applied"

    # Audit captures the rejection.
    assert len(result.audit_entries) == 1
    entry = result.audit_entries[0]
    assert entry["applied"] is False
    assert entry["parameter"] == "regime_vol_threshold"
    assert entry["reason"] == reject_reason
    assert entry["triggered_by"] == "regime_baseline"


# ---------------------------------------------------------------------------
# 11. apply_adjustments: current_params NOT mutated
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_adjustments_does_not_mutate_input():
    initial = {
        "regime_vol_threshold": _midpoint("regime_vol_threshold"),
        "stop_atr_multiplier": _midpoint("stop_atr_multiplier"),
    }
    snapshot = dict(initial)

    proposal = AdjustmentProposal(
        parameter="regime_vol_threshold",
        current_value=initial["regime_vol_threshold"],
        proposed_value=initial["regime_vol_threshold"] + 0.05,
        delta=0.05,
        rationale="ok",
        severity="high",
        triggered_by="regime_baseline",
    )
    result = AutoAdjuster.apply_adjustments([proposal], initial)

    # Input untouched.
    assert initial == snapshot
    # Output reflects the change.
    assert result.after["regime_vol_threshold"] == pytest.approx(
        snapshot["regime_vol_threshold"] + 0.05
    )
    # `before` is a faithful copy of the input.
    assert dict(result.before) == snapshot
    # Untouched key preserved.
    assert result.after["stop_atr_multiplier"] == snapshot["stop_atr_multiplier"]


# ---------------------------------------------------------------------------
# 12. circuit_breaker frozen → all rejected, audit reason captured, no raise
# ---------------------------------------------------------------------------


class _FrozenBreaker:
    def is_frozen(self) -> bool:
        return True


@pytest.mark.unit
def test_apply_adjustments_circuit_breaker_frozen_rejects_all():
    p1 = AdjustmentProposal(
        parameter="regime_vol_threshold",
        current_value=1.0,
        proposed_value=1.05,
        delta=0.05,
        rationale="r1",
        severity="high",
        triggered_by="regime_baseline",
    )
    p2 = AdjustmentProposal(
        parameter="stop_atr_multiplier",
        current_value=2.0,
        proposed_value=2.3,
        delta=0.3,
        rationale="r2",
        severity="high",
        triggered_by="recommendation_quality",
    )
    initial = {"regime_vol_threshold": 1.0, "stop_atr_multiplier": 2.0}

    result = AutoAdjuster.apply_adjustments(
        [p1, p2], initial, circuit_breaker=_FrozenBreaker()
    )

    assert len(result.proposals_applied) == 0
    assert len(result.proposals_rejected) == 2
    for _, reason in result.proposals_rejected:
        assert reason == "circuit_breaker_frozen"
    for entry in result.audit_entries:
        assert entry["applied"] is False
        assert entry["reason"] == "circuit_breaker_frozen"

    # Nothing applied → after equals before.
    assert dict(result.after) == initial


# ---------------------------------------------------------------------------
# 13. AdjustmentProposal / AdjustedParams are frozen dataclasses
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dataclasses_are_frozen():
    proposal = AdjustmentProposal(
        parameter="regime_vol_threshold",
        current_value=1.0,
        proposed_value=1.05,
        delta=0.05,
        rationale="r",
        severity="high",
        triggered_by="regime_baseline",
    )
    with pytest.raises(FrozenInstanceError):
        proposal.parameter = "leverage"  # type: ignore[misc]

    adjusted = AdjustedParams(
        before={},
        after={},
        proposals_applied=(),
        proposals_rejected=(),
        audit_entries=(),
        generated_at="2026-05-03T00:00:00+00:00",
    )
    with pytest.raises(FrozenInstanceError):
        adjusted.generated_at = "x"  # type: ignore[misc]

"""Tests for the explainability certificate."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from smc.hedgerock.evolution.candidate_generator import (
    CandidateProposal,
    DECISION_NO_RECOMMENDATION,
    DECISION_RECOMMEND,
    REASON_MARKET_ANOMALY,
)
from smc.hedgerock.evolution.explainability import (
    CausalFactor,
    CounterfactualPair,
    ExplainabilityCertificate,
    generate_certificate,
)
from smc.hedgerock.evolution.policy_manifest import EvidenceBundle
from smc.hedgerock.evolution.registry_audit import RegistryAuditState


def _bundle(*, violation: bool = False, audit_present: bool = True,
            years_passing: int = 5) -> EvidenceBundle:
    audit = RegistryAuditState(
        audit_log_path="/tmp/x", audit_log_present=audit_present,
        stale_v030_deleted_during_this_session=violation,
        lost_sha_count=4 if violation else 0, lost_sha256=(),
        registry_append_only_violation=violation,
    )
    return EvidenceBundle(
        bundle_id="<test>",
        bundle_hash_sha256="0" * 64,
        atlas_report_path="<test>",
        atlas_report_hash_sha256="0" * 64,
        data_availability_report_path="<test>",
        data_availability_report_hash_sha256="0" * 64,
        walk_forward_run_paths=("<test>",),
        year_replication={
            "XAUUSD": {
                "years_total": 5, "years_passing": years_passing,
                "negative_sign_years": (),
            },
        },
        cross_symbol_count=1,
        halt_event_count=42,
        no_strategy_change=True,
        registry_audit=audit,
    )


def _proposal(
    *, decision: str = DECISION_RECOMMEND,
    reason: str = "", triggered: tuple = ("g6_safety_bound_undefined",),
    parameter_class: str = "confidence_threshold_observe",
) -> CandidateProposal:
    return CandidateProposal(
        candidate_id="c1", parameter_target="t",
        parameter_class=parameter_class,
        baseline_value=0.55, proposed_value=0.50,
        triggered_by=triggered,
        expected_improvement="t", risks=(), next_validation=(),
        decision=decision, decision_reason=reason,
    )


# ---------------------------------------------------------------------------
# Causal factor weights normalise to 1
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_causal_factors_normalised_to_unit_sum() -> None:
    cert = generate_certificate(proposal=_proposal(), bundle=_bundle())
    total = sum(abs(f.weight) for f in cert.causal_factors)
    assert total == pytest.approx(1.0, abs=1e-3)


@pytest.mark.unit
def test_factors_sorted_by_descending_magnitude() -> None:
    cert = generate_certificate(proposal=_proposal(), bundle=_bundle())
    weights = [abs(f.weight) for f in cert.causal_factors]
    assert weights == sorted(weights, reverse=True)


# ---------------------------------------------------------------------------
# Audit + coverage paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_audit_violation_appears_with_negative_weight() -> None:
    cert = generate_certificate(
        proposal=_proposal(decision=DECISION_NO_RECOMMENDATION,
                           reason="evidence_chain_invalid"),
        bundle=_bundle(violation=True),
    )
    names = {f.name: f.weight for f in cert.causal_factors}
    assert "registry_append_only_violation" in names
    assert names["registry_append_only_violation"] < 0


@pytest.mark.unit
def test_xauusd_coverage_insufficient_negative() -> None:
    cert = generate_certificate(
        proposal=_proposal(),
        bundle=_bundle(years_passing=2),
    )
    names = {f.name: f.weight for f in cert.causal_factors}
    assert "xauusd_coverage_insufficient" in names
    assert names["xauusd_coverage_insufficient"] < 0


@pytest.mark.unit
def test_clean_audit_positive() -> None:
    cert = generate_certificate(proposal=_proposal(), bundle=_bundle())
    names = {f.name: f.weight for f in cert.causal_factors}
    assert "audit_log_clean" in names
    assert names["audit_log_clean"] > 0


# ---------------------------------------------------------------------------
# Stress + regime + anomaly + consensus
# ---------------------------------------------------------------------------


def _stress_results(*, n_survived: int, n_breached: int, n_partial: int = 0):
    from smc.hedgerock.evolution.stress_tester import (
        StressTestResult, VERDICT_BREACHED, VERDICT_PARTIAL, VERDICT_SURVIVED,
    )
    out = []
    idx = 0
    def make(verdict):
        nonlocal idx
        idx += 1
        return StressTestResult(
            scenario_id=f"s{idx}", scenario_name=f"s{idx}",
            candidate_id="c1", parameter_class="confidence_threshold_observe",
            survived=(verdict == VERDICT_SURVIVED),
            max_drawdown_pct=1.0, recovery_bars=None, pnl_pct=0.0,
            regime_transitions=(), anomaly_triggers=(), shield_actions=(),
            stop_adjustments=(), verdict=verdict, risk_factor=1.0,
            historical_max_drawdown_pct=1.0,
        )
    for _ in range(n_survived):
        out.append(make(VERDICT_SURVIVED))
    for _ in range(n_breached):
        out.append(make(VERDICT_BREACHED))
    for _ in range(n_partial):
        out.append(make(VERDICT_PARTIAL))
    return out


@pytest.mark.unit
def test_stress_pass_rate_recorded() -> None:
    cert = generate_certificate(
        proposal=_proposal(), bundle=_bundle(),
        stress_results=_stress_results(n_survived=4, n_breached=2),
    )
    assert cert.stress_test_pass_rate == pytest.approx(4 / 6)


@pytest.mark.unit
def test_stress_breach_emits_negative_factor() -> None:
    cert = generate_certificate(
        proposal=_proposal(), bundle=_bundle(),
        stress_results=_stress_results(n_survived=1, n_breached=5),
    )
    names = {f.name: f.weight for f in cert.causal_factors}
    assert "stress_test_breach" in names
    assert names["stress_test_breach"] < 0


@pytest.mark.unit
def test_anomaly_lockdown_appears_in_factors() -> None:
    from smc.hedgerock.evolution.anomaly_shield import (
        AnomalyLevel, AnomalyState,
    )
    state = AnomalyState(
        level=AnomalyLevel.LOCKDOWN, triggers=("gap",),
        short_window_vol=0.0, historical_vol_p90=0.0,
        historical_vol_p95=0.0, historical_vol_p99=0.0,
        max_gap_pct=0.0, n_bars_observed=60,
        last_anomaly_at=None, next_recovery_at=None, blocking_conditions=(),
    )
    cert = generate_certificate(
        proposal=_proposal(decision=DECISION_NO_RECOMMENDATION,
                           reason=REASON_MARKET_ANOMALY),
        bundle=_bundle(),
        anomaly_state=state,
    )
    names = {f.name: f.weight for f in cert.causal_factors}
    assert "anomaly_block" in names
    assert cert.anomaly_at_decision == "LOCKDOWN"


@pytest.mark.unit
def test_consensus_block_recorded() -> None:
    from smc.hedgerock.evolution.multi_timeframe_state import (
        TimeframeState, compute_consensus,
    )
    cons = compute_consensus(
        d1_state=TimeframeState.ACCUMULATING,
        h4_state=TimeframeState.READY, h1_state=TimeframeState.READY,
        m5_state=TimeframeState.READY,
    )
    cert = generate_certificate(
        proposal=_proposal(),
        bundle=_bundle(),
        consensus=cons,
    )
    names = {f.name: f.weight for f in cert.causal_factors}
    assert "timeframe_consensus_block" in names
    assert cert.can_recommend_at_decision is False


# ---------------------------------------------------------------------------
# Counterfactuals + explanation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_counterfactual_emitted_when_no_trigger() -> None:
    cert = generate_certificate(
        proposal=_proposal(triggered=()),
        bundle=_bundle(),
    )
    factors = {cf.factor for cf in cert.counterfactual_comparison}
    assert "trigger_evidence" in factors


@pytest.mark.unit
def test_explanation_text_non_empty() -> None:
    cert = generate_certificate(proposal=_proposal(), bundle=_bundle())
    assert cert.verdict_explanation
    assert "c1" in cert.verdict_explanation


# ---------------------------------------------------------------------------
# Frozen + JSON round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_certificate_is_frozen() -> None:
    cert = generate_certificate(proposal=_proposal(), bundle=_bundle())
    with pytest.raises(Exception):
        cert.decision = "OTHER"  # type: ignore[misc]


@pytest.mark.unit
def test_certificate_json_round_trip() -> None:
    cert = generate_certificate(proposal=_proposal(), bundle=_bundle())
    blob = json.dumps(cert.to_dict(), default=str)
    parsed = json.loads(blob)
    assert parsed["candidate_id"] == "c1"
    assert isinstance(parsed["causal_factors"], list)

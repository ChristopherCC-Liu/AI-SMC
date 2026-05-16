"""P1-2 — Explainability certificate.

For every recommendation, build a human + machine readable proof
of WHY the candidate landed where it did:

  * causal_factors — sorted (factor, weight) pairs whose absolute
    weights sum to 1 (renderer can quote "DD contributed 35%").
  * counterfactual_comparison — small list of "if X had been Y, the
    decision would have been Z" pairs, computed by perturbing one
    input at a time.
  * stress_test_pass_rate — share of scenarios that SURVIVED.
  * regime_at_decision / anomaly_at_decision / consensus_at_decision
    — snapshot of the upstream context the generator saw.
  * verdict_explanation — one-paragraph operator-readable sentence.

Pure data + arithmetic. No imports of ``rule_engine`` or the Tier-1
unsealed prod modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


__all__ = [
    "CausalFactor",
    "CounterfactualPair",
    "ExplainabilityCertificate",
    "generate_certificate",
]


@dataclass(frozen=True)
class CausalFactor:
    name: str
    weight: float  # signed contribution; abs across factors sums to 1
    detail: str = ""


@dataclass(frozen=True)
class CounterfactualPair:
    factor: str
    proposed_alternative: str
    would_change_to: str  # e.g. "RECOMMEND" / "NO_RECOMMENDATION"


@dataclass(frozen=True)
class ExplainabilityCertificate:
    candidate_id: str
    decision: str
    decision_reason: str
    causal_factors: tuple[CausalFactor, ...]
    counterfactual_comparison: tuple[CounterfactualPair, ...]
    stress_test_pass_rate: float | None
    regime_at_decision: str | None
    anomaly_at_decision: str | None
    consensus_at_decision: float | None
    can_recommend_at_decision: bool | None
    verdict_explanation: str
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Causal-factor scoring
# ---------------------------------------------------------------------------


def _score_factors(
    *,
    proposal,
    bundle,
    stress_results: Iterable | None,
    regime_snapshot,
    anomaly_state,
    consensus,
    stop_recommendation,
) -> list[CausalFactor]:
    """Compute a normalised causal-factor breakdown.

    Weights are signed: positive means "pushed toward RECOMMEND",
    negative means "pushed toward NO_RECOMMENDATION". Magnitudes
    across factors sum to 1 after normalisation.
    """
    raw: list[tuple[str, float, str]] = []

    # Audit + coverage are the headline gates.
    audit = getattr(bundle, "registry_audit", None)
    if audit is None:
        raw.append(("audit_log_present", 0.05, "no audit state attached"))
    else:
        if getattr(audit, "registry_append_only_violation", False):
            raw.append((
                "registry_append_only_violation", -0.4,
                "violation forces NO_RECOMMENDATION/evidence_chain_invalid",
            ))
        elif not getattr(audit, "audit_log_present", False):
            raw.append((
                "audit_log_absent", -0.2,
                "audit log not at resolved path; abstain until re-pointed",
            ))
        else:
            raw.append(("audit_log_clean", 0.2, "no append-only violation"))

    # XAUUSD coverage
    yr = getattr(bundle, "year_replication", {}) or {}
    xau = yr.get("XAUUSD", {}) if isinstance(yr, dict) else {}
    years_passing = int(xau.get("years_passing", 0)) if xau else 0
    if years_passing >= 4:
        raw.append((
            "xauusd_coverage_sufficient", 0.15,
            f"{years_passing} XAUUSD years pass replication",
        ))
    else:
        raw.append((
            "xauusd_coverage_insufficient", -0.25,
            f"only {years_passing} XAUUSD years pass; need ≥ 4",
        ))

    # Stress test pass rate
    pass_rate: float | None = None
    if stress_results:
        results = list(stress_results)
        if results:
            from smc.hedgerock.evolution.stress_tester import (
                VERDICT_BREACHED, VERDICT_SURVIVED,
            )
            n_total = len(results)
            n_survived = sum(1 for r in results if r.verdict == VERDICT_SURVIVED)
            n_breached = sum(1 for r in results if r.verdict == VERDICT_BREACHED)
            pass_rate = n_survived / n_total
            if n_breached > 0:
                raw.append((
                    "stress_test_breach", -0.30,
                    f"{n_breached}/{n_total} scenarios BREACHED",
                ))
            else:
                raw.append((
                    "stress_test_pass", 0.20,
                    f"{n_survived}/{n_total} scenarios SURVIVED",
                ))

    # Regime + anomaly + consensus weights (best-effort).
    if regime_snapshot is not None:
        regime_name = getattr(getattr(regime_snapshot, "regime", None), "value", None)
        if regime_name in ("EXTREME", "CRISIS"):
            raw.append((
                "regime_extreme", -0.15,
                f"market regime is {regime_name}; safety bias active",
            ))
        elif regime_name == "LOW_VOL":
            raw.append((
                "regime_low_vol", 0.05,
                "low-vol regime: more permissive",
            ))
    if anomaly_state is not None:
        lvl = getattr(getattr(anomaly_state, "level", None), "value", None)
        if lvl in ("CRITICAL", "LOCKDOWN"):
            raw.append((
                "anomaly_block", -0.40,
                f"anomaly_level={lvl} forces NO_RECOMMENDATION",
            ))
        elif lvl == "ELEVATED":
            raw.append((
                "anomaly_elevated", -0.05,
                "elevated anomaly tightens confidence floor",
            ))
    if consensus is not None:
        if getattr(consensus, "can_recommend", True) is False:
            raw.append((
                "timeframe_consensus_block", -0.20,
                "multi-TF consensus says do not recommend",
            ))
        else:
            raw.append((
                "timeframe_consensus_pass", 0.10,
                f"consensus_score={consensus.consensus_score:.2f} ≥ threshold",
            ))
    if stop_recommendation is not None:
        regime_name = getattr(
            getattr(stop_recommendation, "vol_regime", None), "value", None,
        )
        if regime_name == "EXTREME":
            raw.append((
                "vol_regime_extreme", -0.15,
                "EXTREME vol regime — aggressive_cap class blocked",
            ))

    # Trigger evidence — proposal having a trigger pushes it positive.
    if proposal.triggered_by:
        raw.append((
            "trigger_evidence", 0.10,
            "; ".join(proposal.triggered_by),
        ))
    else:
        raw.append((
            "no_trigger", -0.10, "no trigger fired for this candidate",
        ))

    # Normalise so |weights| sum to 1.
    total_abs = sum(abs(w) for _, w, _ in raw) or 1.0
    out: list[CausalFactor] = [
        CausalFactor(name=n, weight=round(w / total_abs, 6), detail=d)
        for n, w, d in raw
    ]
    out.sort(key=lambda f: -abs(f.weight))
    return out


def _build_counterfactuals(
    *, proposal, anomaly_state, consensus, stop_recommendation,
) -> list[CounterfactualPair]:
    """Tiny rule-based counterfactual list — what ONE-FACTOR change
    would flip the verdict?"""
    out: list[CounterfactualPair] = []
    if anomaly_state is not None and getattr(
        getattr(anomaly_state, "level", None), "value", None,
    ) in ("CRITICAL", "LOCKDOWN"):
        out.append(CounterfactualPair(
            factor="anomaly_state.level",
            proposed_alternative="NORMAL",
            would_change_to="(re-evaluation possible — may RECOMMEND)",
        ))
    if consensus is not None and not getattr(consensus, "can_recommend", True):
        out.append(CounterfactualPair(
            factor="timeframe_consensus.can_recommend",
            proposed_alternative="True",
            would_change_to="(re-evaluation possible — may RECOMMEND)",
        ))
    if stop_recommendation is not None and getattr(
        getattr(stop_recommendation, "vol_regime", None), "value", None,
    ) == "EXTREME" and proposal.parameter_class == "confidence_threshold_aggressive":
        out.append(CounterfactualPair(
            factor="stop_recommendation.vol_regime",
            proposed_alternative="NORMAL",
            would_change_to="(aggressive_cap proposal would be allowed)",
        ))
    if not proposal.triggered_by:
        out.append(CounterfactualPair(
            factor="trigger_evidence",
            proposed_alternative="any_g6_safety_bound_undefined",
            would_change_to="(would generate a fresh micro-tweak proposal)",
        ))
    return out


def _build_explanation(
    *, proposal, factors: list[CausalFactor],
    pass_rate: float | None,
) -> str:
    if not factors:
        return f"{proposal.candidate_id}: {proposal.decision}."
    top = factors[0]
    direction = "supports" if top.weight > 0 else "opposes"
    parts = [
        f"{proposal.candidate_id} → {proposal.decision}",
    ]
    if proposal.decision_reason:
        parts.append(f"({proposal.decision_reason})")
    parts.append(
        f"; top factor: `{top.name}` {direction} the verdict "
        f"({abs(top.weight):.2%})"
    )
    if pass_rate is not None:
        parts.append(f"; stress survival = {pass_rate:.0%}")
    return " ".join(parts) + "."


def generate_certificate(
    *,
    proposal,
    bundle,
    stress_results: Iterable | None = None,
    regime_snapshot=None,
    anomaly_state=None,
    consensus=None,
    stop_recommendation=None,
) -> ExplainabilityCertificate:
    """Build an :class:`ExplainabilityCertificate` from the inputs the
    candidate generator already had at decision time."""
    factors = _score_factors(
        proposal=proposal, bundle=bundle, stress_results=stress_results,
        regime_snapshot=regime_snapshot, anomaly_state=anomaly_state,
        consensus=consensus, stop_recommendation=stop_recommendation,
    )
    counterfactuals = _build_counterfactuals(
        proposal=proposal, anomaly_state=anomaly_state,
        consensus=consensus, stop_recommendation=stop_recommendation,
    )
    pass_rate: float | None = None
    if stress_results:
        from smc.hedgerock.evolution.stress_tester import VERDICT_SURVIVED
        results = list(stress_results)
        if results:
            pass_rate = sum(
                1 for r in results if r.verdict == VERDICT_SURVIVED
            ) / len(results)
    explanation = _build_explanation(
        proposal=proposal, factors=factors, pass_rate=pass_rate,
    )
    return ExplainabilityCertificate(
        candidate_id=proposal.candidate_id,
        decision=proposal.decision,
        decision_reason=proposal.decision_reason,
        causal_factors=tuple(factors),
        counterfactual_comparison=tuple(counterfactuals),
        stress_test_pass_rate=pass_rate,
        regime_at_decision=(
            getattr(getattr(regime_snapshot, "regime", None), "value", None)
            if regime_snapshot else None
        ),
        anomaly_at_decision=(
            getattr(getattr(anomaly_state, "level", None), "value", None)
            if anomaly_state else None
        ),
        consensus_at_decision=(
            float(consensus.consensus_score) if consensus else None
        ),
        can_recommend_at_decision=(
            bool(consensus.can_recommend) if consensus else None
        ),
        verdict_explanation=explanation,
    )

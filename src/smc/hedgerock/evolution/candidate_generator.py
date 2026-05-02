"""Stage 3 — Candidate Generator v0 (report-only, XAUUSD-only).

**Sidecar.** This module never imports the live trading runtime
(``smc.hedgerock.rule_engine`` / ``decision_server`` /
``phase_d_walk_forward``). It reads evidence-bundle metadata + gate
results + blocking reasons + per-candidate registry-audit state, and
emits a list of :class:`CandidateProposal` objects.

The generator does *not* promote anything. It does *not* write under
``policy_registry/approved/`` or modify ``policy_registry/pointer.json``.
Every numeric proposal is clamped against an RFC §5 safety table.

Public surface:

  * :class:`CandidateProposal`
  * :class:`SafetyClamp`
  * :data:`SAFETY_CLAMPS`
  * :data:`DECISION_RECOMMEND`, :data:`DECISION_NO_RECOMMENDATION`
  * Stable reason ids — ``REASON_*`` constants in this module.
  * :func:`generate_candidate_proposals`
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from smc.hedgerock.evolution.policy_manifest import (
    CandidateManifest,
    EvidenceBundle,
    GateStatus,
    PromotionGateResult,
)


__all__ = [
    "CandidateProposal",
    "DECISION_NO_RECOMMENDATION",
    "DECISION_RECOMMEND",
    "REASON_EVIDENCE_CHAIN_INVALID",
    "REASON_INSUFFICIENT_XAUUSD_COVERAGE",
    "REASON_NO_TRIGGER",
    "REASON_PARAMETER_CLASS_UNSUPPORTED",
    "REASON_PROPOSAL_OUTSIDE_SAFETY_CLAMP",
    "SAFETY_CLAMPS",
    "SafetyClamp",
    "generate_candidate_proposals",
]


# ---------------------------------------------------------------------------
# Decision constants
# ---------------------------------------------------------------------------

DECISION_RECOMMEND = "RECOMMEND"
DECISION_NO_RECOMMENDATION = "NO_RECOMMENDATION"

REASON_EVIDENCE_CHAIN_INVALID = "evidence_chain_invalid"
REASON_INSUFFICIENT_XAUUSD_COVERAGE = "insufficient_xauusd_coverage"
REASON_PROPOSAL_OUTSIDE_SAFETY_CLAMP = "proposal_outside_safety_clamp"
REASON_PARAMETER_CLASS_UNSUPPORTED = "parameter_class_unsupported"
REASON_NO_TRIGGER = "no_trigger"

_MIN_XAUUSD_YEARS_PASSING = 4


# ---------------------------------------------------------------------------
# Safety clamps — mirror RFC §5
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SafetyClamp:
    """Hard band the generator's proposed value MUST land inside.

    A proposal that escapes the band returns
    ``REASON_PROPOSAL_OUTSIDE_SAFETY_CLAMP`` — never a clipped value.
    Silent clipping would bury the bug.
    """

    lo: float
    hi: float


SAFETY_CLAMPS: Mapping[str, SafetyClamp] = {
    "confidence_threshold_observe": SafetyClamp(lo=0.40, hi=0.65),
    "confidence_threshold_aggressive": SafetyClamp(lo=0.70, hi=0.85),
    "confidence_threshold_range_2": SafetyClamp(lo=0.55, hi=0.80),
    "halt_expiry_observe_hours": SafetyClamp(lo=4.0, hi=48.0),
}


# Map dotted parameter targets → parameter class id.
_PARAMETER_CLASS_BY_TARGET: Mapping[str, str] = {
    "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR":
        "confidence_threshold_observe",
    "smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR":
        "confidence_threshold_aggressive",
    "smc.hedgerock.rule_engine._RANGE_2_CONFIDENCE_BASELINE":
        "confidence_threshold_range_2",
    "smc.hedgerock.phase_d_walk_forward._HALT_AUTO_EXPIRY_HOURS_OBSERVE":
        "halt_expiry_observe_hours",
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateProposal:
    """A single proposal produced by the generator.

    Always frozen; ``report_only=True`` is invariant. Promotion is a
    separate, manual step downstream.
    """

    candidate_id: str
    parameter_target: str
    parameter_class: str
    baseline_value: float
    proposed_value: float
    triggered_by: tuple[str, ...]
    expected_improvement: str
    risks: tuple[str, ...]
    next_validation: tuple[str, ...]
    decision: str  # DECISION_RECOMMEND | DECISION_NO_RECOMMENDATION
    decision_reason: str  # one of the REASON_* constants, or "" when RECOMMEND
    report_only: bool = True
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _xauusd_coverage_sufficient(bundle: EvidenceBundle) -> bool:
    yr = bundle.year_replication.get("XAUUSD")
    if yr is None:
        return False
    passing = int(yr.get("years_passing", 0))
    return passing >= _MIN_XAUUSD_YEARS_PASSING


def _has_evidence_chain_violation(bundle: EvidenceBundle) -> bool:
    audit = getattr(bundle, "registry_audit", None)
    if audit is None:
        return False
    return bool(getattr(audit, "registry_append_only_violation", False))


def _resolve_parameter_class(target: str) -> str | None:
    return _PARAMETER_CLASS_BY_TARGET.get(target)


def _is_g6_safety_undefined_for_target(
    *, target: str, gate_results: Mapping[str, PromotionGateResult],
) -> bool:
    g6 = gate_results.get("G6")
    if g6 is None or g6.status != GateStatus.FAIL:
        return False
    return "safety_bound_undefined" in g6.reason and target in g6.reason


def _propose_value_for_class(
    *, parameter_class: str, baseline: float,
) -> float:
    """Compute a micro-tweak proposal for a parameter class.

    v0 strategy — for confidence floors, lower by 5% (rounded to 2dp);
    for aggressive cap, raise by 3%; for halt expiry observe, extend
    by 50% to the next 2-hour boundary. Always inside the §5 clamp by
    construction; the safety net catches drift in this rule.
    """
    if parameter_class == "confidence_threshold_observe":
        return round(baseline - 0.05, 2)
    if parameter_class == "confidence_threshold_aggressive":
        return round(baseline + 0.03, 2)
    if parameter_class == "confidence_threshold_range_2":
        return round(baseline - 0.05, 2)
    if parameter_class == "halt_expiry_observe_hours":
        return float(int(round(baseline * 1.5 / 2.0)) * 2)
    # Should never reach here — class guard upstream.
    return baseline


def _no_recommendation(
    *,
    candidate: CandidateManifest,
    reason: str,
    parameter_class: str = "",
) -> CandidateProposal:
    baseline = float(candidate.diff.baseline_value or 0.0)
    return CandidateProposal(
        candidate_id=candidate.candidate_id,
        parameter_target=candidate.diff.target,
        parameter_class=parameter_class or _resolve_parameter_class(
            candidate.diff.target
        ) or "",
        baseline_value=baseline,
        proposed_value=baseline,
        triggered_by=(),
        expected_improvement="",
        risks=(),
        next_validation=(),
        decision=DECISION_NO_RECOMMENDATION,
        decision_reason=reason,
    )


def _evaluate_one(
    *,
    candidate: CandidateManifest,
    bundle: EvidenceBundle,
    gate_results: Mapping[str, PromotionGateResult],
    blocking_reasons: tuple[str, ...],
) -> CandidateProposal:
    # 1. Append-only audit violation short-circuits everything.
    if _has_evidence_chain_violation(bundle):
        return _no_recommendation(
            candidate=candidate, reason=REASON_EVIDENCE_CHAIN_INVALID,
        )

    # 2. XAUUSD coverage must be sufficient.
    if not _xauusd_coverage_sufficient(bundle):
        return _no_recommendation(
            candidate=candidate,
            reason=REASON_INSUFFICIENT_XAUUSD_COVERAGE,
        )

    # 3. Exposure-raising candidates are RFC §10.1 forbidden — they
    #    are not auto-recommended regardless of gate state.
    if candidate.diff.scope.raises_gross_exposure or \
       candidate.diff.scope.raises_leverage or \
       candidate.diff.scope.raises_max_open_positions or \
       candidate.diff.scope.raises_max_recovery_multiplier or \
       candidate.diff.scope.raises_max_grid_density:
        return _no_recommendation(
            candidate=candidate,
            reason=REASON_PARAMETER_CLASS_UNSUPPORTED,
        )

    # 4. Map dotted target → parameter class.
    parameter_class = _resolve_parameter_class(candidate.diff.target)
    if parameter_class is None or parameter_class not in SAFETY_CLAMPS:
        return _no_recommendation(
            candidate=candidate,
            reason=REASON_PARAMETER_CLASS_UNSUPPORTED,
        )

    # 5. Trigger detection — v0 fires when G6 reports
    #    safety_bound_undefined for the candidate's target, or when
    #    the candidate's blocking_reasons mentions the same.
    triggered: list[str] = []
    if _is_g6_safety_undefined_for_target(
        target=candidate.diff.target, gate_results=gate_results,
    ):
        triggered.append(
            f"G6_fail:safety_bound_undefined:{candidate.diff.target}"
        )
    for r in blocking_reasons:
        if "safety_bound_undefined" in r and candidate.diff.target in r:
            tag = f"blocking_reason:{r}"
            if tag not in triggered:
                triggered.append(tag)

    if not triggered:
        return _no_recommendation(
            candidate=candidate,
            reason=REASON_NO_TRIGGER,
            parameter_class=parameter_class,
        )

    # 6. Compute + clamp.
    baseline = float(candidate.diff.baseline_value or 0.0)
    proposed = _propose_value_for_class(
        parameter_class=parameter_class, baseline=baseline,
    )
    clamp = SAFETY_CLAMPS[parameter_class]
    if not (clamp.lo <= proposed <= clamp.hi):
        return _no_recommendation(
            candidate=candidate,
            reason=REASON_PROPOSAL_OUTSIDE_SAFETY_CLAMP,
            parameter_class=parameter_class,
        )

    return CandidateProposal(
        candidate_id=candidate.candidate_id,
        parameter_target=candidate.diff.target,
        parameter_class=parameter_class,
        baseline_value=baseline,
        proposed_value=proposed,
        triggered_by=tuple(triggered),
        expected_improvement=(
            f"micro-tweak {parameter_class} from {baseline} to {proposed} "
            "to address G6 safety_bound_undefined trigger"
        ),
        risks=(
            "value not yet shadow-tested",
            "baseline drift if Phase D evidence changes",
        ),
        next_validation=(
            "shadow_runner v0.3.0 multi-window run on XAUUSD",
            "G1-G8 re-evaluation under refreshed bundle",
            "human review before promotion",
        ),
        decision=DECISION_RECOMMEND,
        decision_reason="",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_candidate_proposals(
    *,
    candidate_menu: Iterable[CandidateManifest],
    bundle: EvidenceBundle,
    gate_results_per_candidate: Mapping[
        str, Mapping[str, PromotionGateResult]
    ],
    blocking_reasons_per_candidate: Mapping[str, Iterable[str]],
    output_dir: Path | None = None,
) -> list[CandidateProposal]:
    """Generate one :class:`CandidateProposal` per menu entry.

    The function NEVER touches live registry paths. When
    ``output_dir`` is provided, a sidecar JSON snapshot of the
    proposals is written under it for the recommendation CLI to
    consume — but the directory MUST be a tmp / report-only path,
    never under ``policy_registry/approved/`` and never
    ``policy_registry/pointer.json``.
    """
    proposals: list[CandidateProposal] = []
    for candidate in candidate_menu:
        gate_results = dict(
            gate_results_per_candidate.get(candidate.candidate_id, {})
        )
        blocking_reasons = tuple(
            blocking_reasons_per_candidate.get(candidate.candidate_id, ())
        )
        proposal = _evaluate_one(
            candidate=candidate, bundle=bundle,
            gate_results=gate_results,
            blocking_reasons=blocking_reasons,
        )
        proposals.append(proposal)

    if output_dir is not None:
        _emit_sidecar_snapshot(proposals=proposals, output_dir=output_dir)

    return proposals


# ---------------------------------------------------------------------------
# Sidecar snapshot — strictly under the operator-supplied tmp path
# ---------------------------------------------------------------------------


_FORBIDDEN_OUTPUT_FRAGMENTS = (
    "policy_registry/approved",
    "policy_registry/pointer.json",
)


def _emit_sidecar_snapshot(
    *,
    proposals: list[CandidateProposal],
    output_dir: Path,
) -> Path:
    out = Path(output_dir)
    text = str(out)
    for fragment in _FORBIDDEN_OUTPUT_FRAGMENTS:
        if fragment in text:
            raise ValueError(
                f"candidate_generator output_dir would write under "
                f"a forbidden registry path: {text!r} "
                f"(matched fragment {fragment!r})"
            )
    out.mkdir(parents=True, exist_ok=True)
    snapshot_path = out / "candidate_proposals.json"
    payload = {
        "schema": "candidate_proposals/v0",
        "report_only": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "proposals": [asdict(p) for p in proposals],
    }
    snapshot_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return snapshot_path

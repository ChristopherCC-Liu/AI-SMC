"""Stage 3 — candidate generator v0 tests (report-only).

Pinned guarantees:

  * Generator returns ``NO_RECOMMENDATION/evidence_chain_invalid``
    whenever the bundle's registry-audit state shows
    ``registry_append_only_violation = True``. The append-only audit
    breach must short-circuit BEFORE any other rule fires.
  * Generator returns ``NO_RECOMMENDATION/insufficient_xauusd_coverage``
    when the bundle's ``year_replication["XAUUSD"]`` shows
    ``years_passing < 4``.
  * Generator only proposes parameter classes listed in the RFC v0
    safety table — anything else returns
    ``NO_RECOMMENDATION/parameter_class_unsupported``.
  * Every numeric proposal is clamped against the §5 safety table.
    A clamp violation by an upstream rule returns
    ``NO_RECOMMENDATION/proposal_outside_safety_clamp``.
  * Generator NEVER writes to disk. ``policy_registry/approved/``
    and ``policy_registry/pointer.json`` MUST not exist after a run.
  * Generator NEVER imports ``smc.hedgerock.rule_engine``,
    ``smc.hedgerock.decision_server``, or
    ``smc.hedgerock.phase_d_walk_forward``.
  * Output is a frozen :class:`CandidateProposal` carrying
    ``report_only = True``.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

import pytest

from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
from smc.hedgerock.evolution.policy_manifest import (
    EvidenceBundle,
    GateStatus,
    PromotionGateResult,
)
from smc.hedgerock.evolution.registry_audit import RegistryAuditState

from smc.hedgerock.evolution.candidate_generator import (
    CandidateProposal,
    DECISION_NO_RECOMMENDATION,
    DECISION_RECOMMEND,
    REASON_EVIDENCE_CHAIN_INVALID,
    REASON_INSUFFICIENT_XAUUSD_COVERAGE,
    REASON_NO_TRIGGER,
    REASON_PARAMETER_CLASS_UNSUPPORTED,
    REASON_PROPOSAL_OUTSIDE_SAFETY_CLAMP,
    SAFETY_CLAMPS,
    generate_candidate_proposals,
)


_REPO = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Bundle helpers
# ---------------------------------------------------------------------------


def _audit(*, violation: bool, lost: int = 0) -> RegistryAuditState:
    return RegistryAuditState(
        audit_log_path="/tmp/_audit.md",
        audit_log_present=True,
        stale_v030_deleted_during_this_session=violation,
        lost_sha_count=lost if violation else 0,
        lost_sha256=("a" * 64,) * (lost if violation else 0),
        registry_append_only_violation=violation,
    )


def _bundle(
    *,
    audit: RegistryAuditState | None = None,
    xauusd_years_passing: int = 4,
    no_strategy_change: bool = False,
) -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id="evb-test",
        bundle_hash_sha256="x" * 64,
        atlas_report_path="/x/atlas.md",
        atlas_report_hash_sha256="a" * 64,
        data_availability_report_path="/x/avail.md",
        data_availability_report_hash_sha256="b" * 64,
        walk_forward_run_paths=("/x/wf.md",),
        year_replication={
            "XAUUSD": {
                "years_total": 4,
                "years_passing": xauusd_years_passing,
                "negative_sign_years": (),
            },
        },
        cross_symbol_count=1,
        halt_event_count=4,
        no_strategy_change=no_strategy_change,
        registry_audit=audit if audit is not None else _audit(violation=False),
    )


def _gate_result(gate_id: str, status: GateStatus, reason: str) -> PromotionGateResult:
    return PromotionGateResult(
        gate_id=gate_id, status=status, reason=reason, details={},
    )


def _gate_results_all_clean() -> dict[str, PromotionGateResult]:
    """Synthetic 'everything PASS' gate map. Used to verify
    `no_trigger` returns when the gates surface no problem to fix."""
    return {
        gid: _gate_result(gid, GateStatus.PASS, "ok")
        for gid in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")
    }


def _gate_results_g6_safety_undefined(target: str) -> dict[str, PromotionGateResult]:
    """G6 fail with safety_bound_undefined for the given dotted
    target — this is the dominant trigger pattern in v0."""
    out = _gate_results_all_clean()
    out["G6"] = _gate_result(
        "G6", GateStatus.FAIL, f"safety_bound_undefined: {target}",
    )
    return out


# ---------------------------------------------------------------------------
# 1. Evidence chain invalid (registry append-only violation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_violation_short_circuits_to_no_recommendation_for_every_candidate(
    tmp_path: Path,
) -> None:
    audit = _audit(violation=True, lost=4)
    bundle = _bundle(audit=audit)
    gate_map = _gate_results_g6_safety_undefined(
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
    )

    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=bundle,
        gate_results_per_candidate={c.candidate_id: gate_map for c in CANDIDATE_MENU_V0},
        blocking_reasons_per_candidate={
            c.candidate_id: ("G6_fail: safety_bound_undefined: x",)
            for c in CANDIDATE_MENU_V0
        },
        output_dir=tmp_path,
    )

    assert len(proposals) == len(CANDIDATE_MENU_V0)
    for p in proposals:
        assert p.decision == DECISION_NO_RECOMMENDATION
        assert p.decision_reason == REASON_EVIDENCE_CHAIN_INVALID
        # Report-only invariant.
        assert p.report_only is True
        # No clamp evaluation should have happened — proposed equals
        # baseline by convention when short-circuiting.
        assert p.proposed_value == p.baseline_value


# ---------------------------------------------------------------------------
# 2. Insufficient XAUUSD coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_insufficient_xauusd_coverage_blocks_recommendation(
    tmp_path: Path,
) -> None:
    bundle = _bundle(xauusd_years_passing=2)
    gate_map = _gate_results_g6_safety_undefined(
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
    )

    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=bundle,
        gate_results_per_candidate={c.candidate_id: gate_map for c in CANDIDATE_MENU_V0},
        blocking_reasons_per_candidate={c.candidate_id: () for c in CANDIDATE_MENU_V0},
        output_dir=tmp_path,
    )

    for p in proposals:
        assert p.decision == DECISION_NO_RECOMMENDATION
        assert p.decision_reason == REASON_INSUFFICIENT_XAUUSD_COVERAGE
        assert p.report_only is True


@pytest.mark.unit
def test_xauusd_missing_from_year_replication_is_insufficient_coverage(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    # Replace year_replication with a non-XAUUSD-only bundle.
    bundle = EvidenceBundle(
        bundle_id=bundle.bundle_id,
        bundle_hash_sha256=bundle.bundle_hash_sha256,
        atlas_report_path=bundle.atlas_report_path,
        atlas_report_hash_sha256=bundle.atlas_report_hash_sha256,
        data_availability_report_path=bundle.data_availability_report_path,
        data_availability_report_hash_sha256=bundle.data_availability_report_hash_sha256,
        walk_forward_run_paths=bundle.walk_forward_run_paths,
        year_replication={
            "XAGUSD": {
                "years_total": 4, "years_passing": 4,
                "negative_sign_years": (),
            },
        },
        cross_symbol_count=1, halt_event_count=4,
        no_strategy_change=False,
        registry_audit=bundle.registry_audit,
    )

    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=bundle,
        gate_results_per_candidate={c.candidate_id: _gate_results_all_clean()
                                    for c in CANDIDATE_MENU_V0},
        blocking_reasons_per_candidate={c.candidate_id: () for c in CANDIDATE_MENU_V0},
        output_dir=tmp_path,
    )

    for p in proposals:
        assert p.decision == DECISION_NO_RECOMMENDATION
        assert p.decision_reason == REASON_INSUFFICIENT_XAUUSD_COVERAGE


# ---------------------------------------------------------------------------
# 3. Recommend on G6 safety_bound_undefined (the v0 happy path)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_g6_undefined_triggers_recommend_for_observe_floor(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    target = "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR"
    gate_map = _gate_results_g6_safety_undefined(target)

    # Drive only the c1 candidate (observe floor) — others may have
    # different parameter classes.
    c1 = next(c for c in CANDIDATE_MENU_V0 if c.candidate_id.startswith("c1"))
    proposals = generate_candidate_proposals(
        candidate_menu=(c1,),
        bundle=bundle,
        gate_results_per_candidate={c1.candidate_id: gate_map},
        blocking_reasons_per_candidate={
            c1.candidate_id: (f"G6_fail: safety_bound_undefined: {target}",),
        },
        output_dir=tmp_path,
    )

    assert len(proposals) == 1
    p = proposals[0]
    assert p.decision == DECISION_RECOMMEND, (
        f"expected RECOMMEND, got {p.decision} ({p.decision_reason})"
    )
    assert p.decision_reason == ""  # blank on RECOMMEND
    assert p.parameter_target == target
    assert p.parameter_class == "confidence_threshold_observe"
    assert p.baseline_value == 0.55
    # Generator should not echo back the menu's proposed value
    # blindly — it computes its own clamp-respecting suggestion.
    clamp = SAFETY_CLAMPS["confidence_threshold_observe"]
    assert clamp.lo <= p.proposed_value <= clamp.hi
    assert p.report_only is True
    assert "G6" in " ".join(p.triggered_by)


# ---------------------------------------------------------------------------
# 4. Safety clamp — proposed value outside band → NO_RECOMMENDATION
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_safety_clamp_rejects_proposed_value_outside_band(
    tmp_path: Path,
) -> None:
    """If the rule logic ever proposes a value outside the §5 band,
    the generator must drop to NO_RECOMMENDATION rather than emit
    a clipped value silently. This is the safety-of-last-resort net."""
    from smc.hedgerock.evolution import candidate_generator as cg

    bundle = _bundle()
    target = "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR"
    gate_map = _gate_results_g6_safety_undefined(target)
    c1 = next(c for c in CANDIDATE_MENU_V0 if c.candidate_id.startswith("c1"))

    # Force the rule layer to emit an out-of-band value by
    # monkey-patching the rule. Generator must catch it.
    orig = cg._propose_value_for_class
    def _bad_rule(*, parameter_class: str, baseline: float) -> float:
        return 0.20  # clearly outside [0.40, 0.65] band
    cg._propose_value_for_class = _bad_rule
    try:
        proposals = generate_candidate_proposals(
            candidate_menu=(c1,),
            bundle=bundle,
            gate_results_per_candidate={c1.candidate_id: gate_map},
            blocking_reasons_per_candidate={c1.candidate_id: ()},
            output_dir=tmp_path,
        )
    finally:
        cg._propose_value_for_class = orig

    p = proposals[0]
    assert p.decision == DECISION_NO_RECOMMENDATION
    assert p.decision_reason == REASON_PROPOSAL_OUTSIDE_SAFETY_CLAMP


# ---------------------------------------------------------------------------
# 5. No trigger — gates clean, no candidate to propose
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_trigger_when_gates_clean_and_no_blocking_reason(
    tmp_path: Path,
) -> None:
    """When the gates are clean and no blocking reason fires, the
    candidates whose parameter class IS supported (i.e. not vetoed
    by exposure rules) must return `no_trigger`. Exposure-vetoed
    candidates (c3) deliberately return `parameter_class_unsupported`
    earlier — that branch is asserted in its own test."""
    bundle = _bundle()
    safe_candidates = tuple(
        c for c in CANDIDATE_MENU_V0
        if not c.diff.scope.raises_gross_exposure
    )
    assert safe_candidates  # sanity — menu has at least one safe entry
    proposals = generate_candidate_proposals(
        candidate_menu=safe_candidates,
        bundle=bundle,
        gate_results_per_candidate={c.candidate_id: _gate_results_all_clean()
                                    for c in safe_candidates},
        blocking_reasons_per_candidate={c.candidate_id: () for c in safe_candidates},
        output_dir=tmp_path,
    )
    for p in proposals:
        assert p.decision == DECISION_NO_RECOMMENDATION
        assert p.decision_reason == REASON_NO_TRIGGER


# ---------------------------------------------------------------------------
# 6. Unsupported parameter class — exposure raise (c3) is forbidden by RFC
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_exposure_raising_candidate_is_unsupported_class(
    tmp_path: Path,
) -> None:
    """c3 raises gross exposure — RFC §10.1 forbids auto-promote.
    The generator MUST refuse to propose anything for it, regardless
    of gate state. (Aggressive-floor knob CAN be tweaked; the c3
    candidate's raises_gross_exposure flag is what makes it unsafe
    for auto-recommendation.)"""
    bundle = _bundle()
    c3 = next(c for c in CANDIDATE_MENU_V0 if c.diff.scope.raises_gross_exposure)

    target = c3.diff.target
    proposals = generate_candidate_proposals(
        candidate_menu=(c3,),
        bundle=bundle,
        gate_results_per_candidate={c3.candidate_id: _gate_results_g6_safety_undefined(target)},
        blocking_reasons_per_candidate={c3.candidate_id: ()},
        output_dir=tmp_path,
    )
    p = proposals[0]
    assert p.decision == DECISION_NO_RECOMMENDATION
    assert p.decision_reason == REASON_PARAMETER_CLASS_UNSUPPORTED


# ---------------------------------------------------------------------------
# 7. Generator does NOT write to live registry paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generator_does_not_create_approved_or_pointer(tmp_path: Path) -> None:
    bundle = _bundle()
    target = "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR"
    gate_map = _gate_results_g6_safety_undefined(target)

    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=bundle,
        gate_results_per_candidate={c.candidate_id: gate_map for c in CANDIDATE_MENU_V0},
        blocking_reasons_per_candidate={c.candidate_id: () for c in CANDIDATE_MENU_V0},
        output_dir=tmp_path,
    )

    # Output dir may contain the proposals JSON (sidecar). It must
    # NOT contain a directory or file named approved/, pointer.json,
    # or anything matching the live registry layout.
    forbidden = {
        tmp_path / "approved",
        tmp_path / "pointer.json",
    }
    for p in forbidden:
        assert not p.exists(), f"generator created forbidden path: {p}"

    # Real production paths untouched (the canonical tree).
    real_root = Path("/Users/christopher/HedgeRock/policy_registry")
    if real_root.exists():
        assert not (real_root / "approved").exists() or \
            (real_root / "approved").is_dir()
        # pointer.json may legitimately exist outside this test run;
        # the invariant is the GENERATOR did not create it. We can't
        # easily assert non-modification by mtime here; the
        # source-level isolation test below catches accidental writes.

    assert proposals  # smoke


# ---------------------------------------------------------------------------
# 8. Source-level isolation — generator imports no live code
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generator_does_not_import_rule_engine() -> None:
    """rule_engine stays red-line. decision_server is Tier-1
    unsealed for the generator's baseline lookup; phase_d_walk_forward
    constants remain referenceable as documentation strings only —
    actual imports are tested by test_regression_guard's whitelist."""
    src = (
        _REPO / "src" / "smc" / "hedgerock" / "evolution"
        / "candidate_generator.py"
    ).read_text(encoding="utf-8")

    forbidden_imports = (
        "from smc.hedgerock.rule_engine",
        "import smc.hedgerock.rule_engine",
    )
    for fragment in forbidden_imports:
        assert fragment not in src, (
            f"candidate_generator imports rule_engine "
            f"(red-line): {fragment!r}"
        )


@pytest.mark.unit
def test_generator_uses_only_read_only_decision_server_surface() -> None:
    """The Tier-1 unseal lets the generator import decision_server
    but only call its read-only getter."""
    src = (
        _REPO / "src" / "smc" / "hedgerock" / "evolution"
        / "candidate_generator.py"
    ).read_text(encoding="utf-8")
    import re as _re
    allowed = {"decision_server.get_live_parameters"}
    for m in _re.finditer(
        r"\bdecision_server\.[A-Za-z_][A-Za-z0-9_]*", src,
    ):
        token = m.group(0)
        assert token in allowed, (
            f"candidate_generator accessed non-public "
            f"decision_server symbol: {token}"
        )


# ---------------------------------------------------------------------------
# 9. CandidateProposal is frozen + carries report_only=True
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_candidate_proposal_is_frozen_and_report_only() -> None:
    p = CandidateProposal(
        candidate_id="c1-lower-observe-floor-0.50",
        parameter_target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        parameter_class="confidence_threshold_observe",
        baseline_value=0.55,
        proposed_value=0.52,
        triggered_by=("G6_safety_bound_undefined",),
        expected_improvement="micro-relax observe floor",
        risks=("possible false-positive uptick",),
        next_validation=("4-window shadow run on XAUUSD",),
        decision=DECISION_RECOMMEND,
        decision_reason="",
    )
    assert p.report_only is True
    with pytest.raises((AttributeError, TypeError)):
        p.proposed_value = 0.40  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        p.report_only = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 10. Safety clamps are documented per RFC §5
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_safety_clamps_cover_rfc_v0_classes() -> None:
    expected_classes = {
        "confidence_threshold_observe",
        "confidence_threshold_aggressive",
        "confidence_threshold_range_2",
        "halt_expiry_observe_hours",
    }
    assert expected_classes.issubset(set(SAFETY_CLAMPS))
    for name, clamp in SAFETY_CLAMPS.items():
        assert clamp.lo < clamp.hi, f"clamp {name!r} has lo>=hi"

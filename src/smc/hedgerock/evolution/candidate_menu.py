"""Phase D-cont3 / Ticket 1 — hand-curated candidate menu.

**Pure data.** No FS, no imports from production runtime, no
auto-generation. Adding or removing a candidate is a code change
subject to normal review.

The menu mirrors Plan §4. Each entry is constructed in state
``draft`` with no evidence bundle attached — the report CLI attaches
the loaded bundle at runtime and runs the gates.
"""

from __future__ import annotations

from smc.hedgerock.evolution.policy_manifest import (
    CandidateDiff,
    CandidateDiffScope,
    CandidateManifest,
    CandidateState,
    MANIFEST_SCHEMA_VERSION,
    OverallResult,
)


__all__ = ["CANDIDATE_MENU_V0"]


_AUTHOR = "evolution_sidecar/v0 (hand-curated)"
_CREATED_AT = "2026-05-01T00:00:00+00:00"


def _candidate(
    *,
    candidate_id: str,
    title: str,
    target: str,
    baseline: float,
    proposed: float,
    affects_halt_mode: bool = False,
    raises_gross_exposure: bool = False,
) -> CandidateManifest:
    """Reduce boilerplate for the four entries below. All four are
    classifier/rule_engine-affecting threshold tweaks; only c2 affects
    halt mode and only c3 raises gross exposure."""
    return CandidateManifest(
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        candidate_id=candidate_id,
        title=title,
        author=_AUTHOR,
        created_at=_CREATED_AT,
        state=CandidateState.DRAFT,
        diff=CandidateDiff(
            kind="threshold_tweak",
            target=target,
            baseline_value=baseline,
            proposed_value=proposed,
            scope=CandidateDiffScope(
                regimes_affected=("range",) if not affects_halt_mode else ("halt",),
                affects_halt_mode=affects_halt_mode,
                affects_classifier_or_rule_engine=True,
                raises_gross_exposure=raises_gross_exposure,
                interfaces_touched=(),
            ),
        ),
        evidence_bundle=None,
        gates=(),
        result=OverallResult.PROMOTION_BLOCKED,
        blocking_reasons=(),
        next_data_needs=(),
        required_next_data_or_policy="",
        human_approval_required_for_state_transitions_above="tested",
        audit_trail=(),
    )


# Plan §4 — exactly four entries, each illustrating a different
# blocking pattern under the current Phase D evidence:
#   c1 — year-replication failure on classifier threshold
#   c2 — halt-corpus floor (n < 30)
#   c3 — exposure-class veto (raises gross exposure)
#   c4 — single-symbol abstain on classifier threshold
CANDIDATE_MENU_V0: tuple[CandidateManifest, ...] = (
    _candidate(
        candidate_id="c1-lower-observe-floor-0.50",
        title="Lower _CONFIDENCE_OBSERVE_FLOOR from 0.55 to 0.50 (range regime)",
        target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        baseline=0.55,
        proposed=0.50,
        affects_halt_mode=False,
        raises_gross_exposure=False,
    ),
    _candidate(
        candidate_id="c2-halt-expiry-observe-6h",
        title="Extend halt auto-expiry observe-only release from 4h to 6h",
        target="smc.hedgerock.phase_d_walk_forward._HALT_AUTO_EXPIRY_HOURS_OBSERVE",
        baseline=4.0,
        proposed=6.0,
        affects_halt_mode=True,
        raises_gross_exposure=False,
    ),
    _candidate(
        candidate_id="c3-aggressive-floor-0.78",
        title="Lower _CONFIDENCE_AGGRESSIVE_FLOOR from 0.80 to 0.78 (raises exposure)",
        target="smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR",
        baseline=0.80,
        proposed=0.78,
        affects_halt_mode=False,
        raises_gross_exposure=True,
    ),
    _candidate(
        candidate_id="c4-range2-conf-0.70",
        title="Raise range#2 confidence baseline from 0.65 to 0.70",
        target="smc.hedgerock.rule_engine._RANGE_2_CONFIDENCE_BASELINE",
        baseline=0.65,
        proposed=0.70,
        affects_halt_mode=False,
        raises_gross_exposure=False,
    ),
)

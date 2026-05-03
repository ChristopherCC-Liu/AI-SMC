"""Phase D-cont3 / Ticket 1 — promotion gate tests.

Each gate G1–G8 is parametrically covered for every status outcome
(PASS / FAIL / ABSTAIN / NOT_RUN) per Plan §3.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smc.hedgerock.evolution.policy_manifest import (
    CandidateDiff,
    CandidateDiffScope,
    CandidateManifest,
    CandidateState,
    EvidenceBundle,
    GateStatus,
    OverallResult,
    MANIFEST_SCHEMA_VERSION,
)
from smc.hedgerock.evolution.promotion_gates import (
    GATE_IDS,
    GateConfig,
    SafetyBoundsConfig,
    compute_overall_result,
    evaluate_all_gates,
    g1_min_evidence,
    g2_year_replication,
    g3_cross_symbol,
    g4_no_negative_sign_years,
    g5_halt_event_corpus,
    g6_safety_bounds,
    g7_interface_stability,
    g8_shadow_comparison,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bundle(
    *,
    year_replication: dict | None = None,
    cross_symbol_count: int = 1,
    halt_event_count: int = 4,
    no_strategy_change: bool = True,
    atlas_hash: str = "a" * 64,
    avail_hash: str = "b" * 64,
    bundle_hash: str = "c" * 64,
    wf_paths: tuple = ("/x/wf.md",),
) -> EvidenceBundle:
    if year_replication is None:
        year_replication = {
            "XAUUSD": {
                "years_total": 4,
                "years_passing": 2,
                "negative_sign_years": (2021,),
            },
        }
    return EvidenceBundle(
        bundle_id="evb",
        bundle_hash_sha256=bundle_hash,
        atlas_report_path="/x/atlas.md",
        atlas_report_hash_sha256=atlas_hash,
        data_availability_report_path="/x/availability.md",
        data_availability_report_hash_sha256=avail_hash,
        walk_forward_run_paths=wf_paths,
        year_replication=year_replication,
        cross_symbol_count=cross_symbol_count,
        halt_event_count=halt_event_count,
        no_strategy_change=no_strategy_change,
    )


def _diff(
    *,
    target: str = "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
    proposed_value: float | int | None = 0.50,
    affects_halt_mode: bool = False,
    affects_classifier_or_rule_engine: bool = True,
    raises_gross_exposure: bool = False,
    interfaces_touched: tuple = (),
) -> CandidateDiff:
    return CandidateDiff(
        kind="threshold_tweak",
        target=target,
        baseline_value=0.55,
        proposed_value=proposed_value,
        scope=CandidateDiffScope(
            regimes_affected=("range",),
            affects_halt_mode=affects_halt_mode,
            affects_classifier_or_rule_engine=affects_classifier_or_rule_engine,
            raises_gross_exposure=raises_gross_exposure,
            interfaces_touched=interfaces_touched,
        ),
    )


def _candidate(
    *,
    candidate_id: str = "c-test",
    diff: CandidateDiff | None = None,
    bundle: EvidenceBundle | None = None,
) -> CandidateManifest:
    return CandidateManifest(
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        candidate_id=candidate_id,
        title="t",
        author="test",
        created_at="2026-05-01T12:00:00+00:00",
        state=CandidateState.DRAFT,
        diff=diff or _diff(),
        evidence_bundle=bundle,
        gates=(),
        result=OverallResult.PROMOTION_BLOCKED,
        blocking_reasons=(),
        next_data_needs=(),
        required_next_data_or_policy="",
        human_approval_required_for_state_transitions_above="tested",
        audit_trail=(),
    )


# ---------------------------------------------------------------------------
# G1 — minimum evidence
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_g1_pass_with_full_bundle() -> None:
    r = g1_min_evidence(candidate=_candidate(), bundle=_bundle())
    assert r.status == GateStatus.PASS


@pytest.mark.unit
def test_g1_fail_when_no_bundle() -> None:
    r = g1_min_evidence(candidate=_candidate(), bundle=None)
    assert r.status == GateStatus.FAIL


@pytest.mark.unit
def test_g1_fail_when_no_walk_forward_runs() -> None:
    bundle = _bundle(wf_paths=())
    r = g1_min_evidence(candidate=_candidate(), bundle=bundle)
    assert r.status == GateStatus.FAIL


# ---------------------------------------------------------------------------
# G2 — year replication
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_g2_pass_at_threshold() -> None:
    bundle = _bundle(year_replication={
        "XAUUSD": {"years_total": 4, "years_passing": 3, "negative_sign_years": ()},
    })
    cfg = GateConfig(g2_pass_rate_threshold=0.75)
    r = g2_year_replication(candidate=_candidate(), bundle=bundle, gate_config=cfg)
    assert r.status == GateStatus.PASS


@pytest.mark.unit
def test_g2_fail_below_threshold() -> None:
    bundle = _bundle()  # default 2/4 = 0.5
    r = g2_year_replication(candidate=_candidate(), bundle=bundle)
    assert r.status == GateStatus.FAIL
    assert "2/4" in r.reason


@pytest.mark.unit
def test_g2_abstain_when_diff_does_not_affect_rule_engine() -> None:
    diff = _diff(affects_classifier_or_rule_engine=False)
    r = g2_year_replication(candidate=_candidate(diff=diff), bundle=_bundle())
    assert r.status == GateStatus.ABSTAIN


@pytest.mark.unit
def test_g2_not_run_when_year_replication_empty() -> None:
    bundle = _bundle(year_replication={})
    r = g2_year_replication(candidate=_candidate(), bundle=bundle)
    assert r.status == GateStatus.NOT_RUN


# ---------------------------------------------------------------------------
# G3 — cross-symbol replication
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_g3_abstain_on_single_symbol() -> None:
    r = g3_cross_symbol(candidate=_candidate(), bundle=_bundle(cross_symbol_count=1))
    assert r.status == GateStatus.ABSTAIN


@pytest.mark.unit
def test_g3_pass_when_two_symbols_pass_g2() -> None:
    bundle = _bundle(
        cross_symbol_count=2,
        year_replication={
            "XAUUSD": {"years_total": 4, "years_passing": 4, "negative_sign_years": ()},
            "EURUSD": {"years_total": 5, "years_passing": 4, "negative_sign_years": ()},
        },
    )
    r = g3_cross_symbol(candidate=_candidate(), bundle=bundle)
    assert r.status == GateStatus.PASS


@pytest.mark.unit
def test_g3_fail_when_one_symbol_regresses() -> None:
    bundle = _bundle(
        cross_symbol_count=2,
        year_replication={
            "XAUUSD": {"years_total": 4, "years_passing": 4, "negative_sign_years": ()},
            "EURUSD": {"years_total": 5, "years_passing": 1, "negative_sign_years": ()},
        },
    )
    r = g3_cross_symbol(candidate=_candidate(), bundle=bundle)
    assert r.status == GateStatus.FAIL


# ---------------------------------------------------------------------------
# G4 — no negative-sign years
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_g4_pass_when_no_negatives() -> None:
    bundle = _bundle(year_replication={
        "XAUUSD": {"years_total": 4, "years_passing": 4, "negative_sign_years": ()},
    })
    r = g4_no_negative_sign_years(candidate=_candidate(), bundle=bundle)
    assert r.status == GateStatus.PASS


@pytest.mark.unit
def test_g4_fail_with_2021_reverse_must_name_year() -> None:
    """Phase D-cont3 lesson encoded — reason MUST cite the year."""
    r = g4_no_negative_sign_years(candidate=_candidate(), bundle=_bundle())
    assert r.status == GateStatus.FAIL
    assert "2021" in r.reason


# ---------------------------------------------------------------------------
# G5 — halt event corpus
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_g5_abstain_when_candidate_does_not_affect_halt() -> None:
    diff = _diff(affects_halt_mode=False)
    r = g5_halt_event_corpus(candidate=_candidate(diff=diff), bundle=_bundle())
    assert r.status == GateStatus.ABSTAIN


@pytest.mark.unit
def test_g5_fail_when_corpus_below_floor() -> None:
    diff = _diff(affects_halt_mode=True)
    r = g5_halt_event_corpus(candidate=_candidate(diff=diff), bundle=_bundle(halt_event_count=4))
    assert r.status == GateStatus.FAIL
    assert "4" in r.reason and "30" in r.reason


@pytest.mark.unit
def test_g5_pass_with_n_above_floor() -> None:
    diff = _diff(affects_halt_mode=True)
    r = g5_halt_event_corpus(
        candidate=_candidate(diff=diff),
        bundle=_bundle(halt_event_count=42),
    )
    assert r.status == GateStatus.PASS


# ---------------------------------------------------------------------------
# G6 — safety bounds
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_g6_pass_within_band() -> None:
    bounds = SafetyBoundsConfig(bounds={
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR": (0.45, 0.65),
    })
    r = g6_safety_bounds(candidate=_candidate(), bounds=bounds)
    assert r.status == GateStatus.PASS


@pytest.mark.unit
def test_g6_fail_outside_band_names_field_value_bound() -> None:
    bounds = SafetyBoundsConfig(bounds={
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR": (0.55, 0.65),
    })
    r = g6_safety_bounds(candidate=_candidate(diff=_diff(proposed_value=0.30)), bounds=bounds)
    assert r.status == GateStatus.FAIL
    assert "_CONFIDENCE_OBSERVE_FLOOR" in r.reason
    assert "0.30" in r.reason or "0.3" in r.reason
    assert "0.55" in r.reason


@pytest.mark.unit
def test_g6_fail_with_safety_bound_undefined_when_target_missing() -> None:
    """Plan §3 G6 (b): undefined bound → FAIL with safety_bound_undefined."""
    bounds = SafetyBoundsConfig(bounds={})  # empty: target unknown
    r = g6_safety_bounds(candidate=_candidate(), bounds=bounds)
    assert r.status == GateStatus.FAIL
    assert "safety_bound_undefined" in r.reason
    # The reason must name the offending knob.
    assert "_CONFIDENCE_OBSERVE_FLOOR" in r.reason


@pytest.mark.unit
def test_g6_abstain_when_no_numeric_proposed_value() -> None:
    diff = _diff(proposed_value=None)
    r = g6_safety_bounds(
        candidate=_candidate(diff=diff),
        bounds=SafetyBoundsConfig(bounds={}),
    )
    assert r.status == GateStatus.ABSTAIN


# ---------------------------------------------------------------------------
# G7 — interface stability
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_g7_pass_with_no_interfaces_touched() -> None:
    r = g7_interface_stability(candidate=_candidate())
    assert r.status == GateStatus.PASS


@pytest.mark.unit
def test_g7_fail_with_interfaces_touched() -> None:
    diff = _diff(interfaces_touched=("DynamicParams.mode",))
    r = g7_interface_stability(candidate=_candidate(diff=diff))
    assert r.status == GateStatus.FAIL
    assert "DynamicParams.mode" in r.reason


# ---------------------------------------------------------------------------
# G8 — shadow comparison (always NOT_RUN in MVP)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_g8_default_not_run_when_bundle_has_no_shadow_artefact() -> None:
    """Ticket 2 Step 6: G8 returns NOT_RUN by default when the
    evidence bundle does not carry a shadow artefact path. Legacy
    test (was 'MVP+1') updated for the new evidence-driven path.
    """
    r = g8_shadow_comparison(candidate=_candidate(), bundle=_bundle())
    assert r.status == GateStatus.NOT_RUN
    assert "no shadow artefact" in r.reason.lower()


# ---------------------------------------------------------------------------
# evaluate_all_gates + compute_overall_result
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_evaluate_all_gates_returns_dict_keyed_by_gate_id() -> None:
    results = evaluate_all_gates(
        candidate=_candidate(), bundle=_bundle(),
        bounds=SafetyBoundsConfig(bounds={
            "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR": (0.45, 0.65),
        }),
    )
    assert set(results.keys()) == set(GATE_IDS)
    for gid, r in results.items():
        assert r.gate_id == gid


@pytest.mark.unit
def test_overall_result_blocked_when_any_fail() -> None:
    results = evaluate_all_gates(
        candidate=_candidate(),
        bundle=_bundle(),
        bounds=SafetyBoundsConfig(bounds={
            "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR": (0.45, 0.65),
        }),
    )
    overall, reasons = compute_overall_result(
        gate_results=results, candidate=_candidate(), bundle=_bundle(),
    )
    assert overall == OverallResult.PROMOTION_BLOCKED
    assert any("G2_fail" in r for r in reasons)
    assert any("G4_fail" in r for r in reasons)


@pytest.mark.unit
def test_overall_result_manifest_invalid_when_safety_bound_undefined() -> None:
    """Plan §3 G6 (b): undefined bound → PROMOTION_BLOCKED / manifest_invalid."""
    results = evaluate_all_gates(
        candidate=_candidate(),
        bundle=_bundle(),
        bounds=SafetyBoundsConfig(bounds={}),  # no bands at all
    )
    overall, reasons = compute_overall_result(
        gate_results=results, candidate=_candidate(), bundle=_bundle(),
    )
    assert overall == OverallResult.PROMOTION_BLOCKED_MANIFEST_INVALID
    assert any("safety_bound_undefined" in r for r in reasons)


@pytest.mark.unit
def test_overall_result_blocks_exposure_class_even_when_gates_pass() -> None:
    # Synthetic "perfect" bundle: 4/4 years, 2 symbols, no negatives,
    # halt_corpus high, no_strategy_change=False.
    perfect_bundle = _bundle(
        cross_symbol_count=2,
        year_replication={
            "XAUUSD": {"years_total": 4, "years_passing": 4, "negative_sign_years": ()},
            "EURUSD": {"years_total": 4, "years_passing": 4, "negative_sign_years": ()},
        },
        halt_event_count=42,
        no_strategy_change=False,
    )
    bounds = SafetyBoundsConfig(bounds={
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR": (0.45, 0.65),
    })
    diff = _diff(raises_gross_exposure=True)
    cand = _candidate(diff=diff, bundle=perfect_bundle)
    results = evaluate_all_gates(candidate=cand, bundle=perfect_bundle, bounds=bounds)
    overall, reasons = compute_overall_result(
        gate_results=results, candidate=cand, bundle=perfect_bundle,
    )
    assert overall == OverallResult.PROMOTION_BLOCKED
    assert any("exposure_class_human_only" in r for r in reasons)


@pytest.mark.unit
def test_overall_result_blocks_when_no_strategy_change_true_even_if_gates_pass() -> None:
    """Plan §3 + RFC §11: NO_STRATEGY_CHANGE: true short-circuits."""
    perfect_bundle = _bundle(
        cross_symbol_count=2,
        year_replication={
            "XAUUSD": {"years_total": 4, "years_passing": 4, "negative_sign_years": ()},
            "EURUSD": {"years_total": 4, "years_passing": 4, "negative_sign_years": ()},
        },
        halt_event_count=42,
        no_strategy_change=True,
    )
    bounds = SafetyBoundsConfig(bounds={
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR": (0.45, 0.65),
    })
    cand = _candidate(bundle=perfect_bundle)
    results = evaluate_all_gates(candidate=cand, bundle=perfect_bundle, bounds=bounds)
    overall, reasons = compute_overall_result(
        gate_results=results, candidate=cand, bundle=perfect_bundle,
    )
    assert overall == OverallResult.PROMOTION_BLOCKED
    assert any("data_availability_action_gate_blocks_all" in r for r in reasons)


@pytest.mark.unit
def test_overall_result_ready_for_tested_only_when_all_clean() -> None:
    perfect_bundle = _bundle(
        cross_symbol_count=2,
        year_replication={
            "XAUUSD": {"years_total": 4, "years_passing": 4, "negative_sign_years": ()},
            "EURUSD": {"years_total": 4, "years_passing": 4, "negative_sign_years": ()},
        },
        halt_event_count=42,
        no_strategy_change=False,  # clear short-circuit
    )
    bounds = SafetyBoundsConfig(bounds={
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR": (0.45, 0.65),
    })
    cand = _candidate(bundle=perfect_bundle)
    results = evaluate_all_gates(candidate=cand, bundle=perfect_bundle, bounds=bounds)
    overall, reasons = compute_overall_result(
        gate_results=results, candidate=cand, bundle=perfect_bundle,
    )
    # G8 stays NOT_RUN; READY_FOR_TESTED is a label, not a state change.
    assert overall == OverallResult.READY_FOR_TESTED
    assert reasons == ()

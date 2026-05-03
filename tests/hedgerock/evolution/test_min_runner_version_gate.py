"""Ticket 3 Step 1 — old-artefact runner-version PASS gate tests.

Pinned guarantees (per R2 of Ticket 3 Plan v2):
  - MIN_RUNNER_VERSION_FOR_PASS constant exists and is at least
    "shadow_runner-0.2.0" (the Ticket 3 runner version).
  - parse_runner_version returns a comparable tuple.
  - version_lt is a total order on (semver_major, minor, patch).
  - An artefact with runner_version < MIN can NEVER receive G8 PASS:
      * old artefact + clean envelope + clean metrics → ABSTAIN
      * old artefact + any FAIL-class condition → FAIL still wins
        (fail-closed not bypassed by "you're old")
  - An unparseable runner_version → FAIL: shadow_artefact_runner_version_unparseable.
  - The constant is NOT modifiable by the sidecar — defending-in-depth
    we just check it's a module-level constant in promotion_gates,
    not configurable at runtime.
"""

from __future__ import annotations

import pytest

from smc.hedgerock.evolution.promotion_gates import (
    MIN_RUNNER_VERSION_FOR_PASS,
    parse_runner_version,
    version_lt,
)


# ---------------------------------------------------------------------------
# 1. Constant existence + minimum value
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_min_runner_version_constant_is_at_least_0_2_0() -> None:
    """MIN must be the Ticket 3 runner version or higher. Old
    Ticket 2 zero-trade runner ('shadow_runner-0.1.0') is below."""
    assert isinstance(MIN_RUNNER_VERSION_FOR_PASS, str)
    assert MIN_RUNNER_VERSION_FOR_PASS.startswith("shadow_runner-")
    minor = parse_runner_version(MIN_RUNNER_VERSION_FOR_PASS)
    assert minor >= (0, 2, 0)


@pytest.mark.unit
def test_min_runner_version_blocks_ticket_2_zero_trade() -> None:
    """Ticket 2's runner_version = 'shadow_runner-0.1.0' → must be lt MIN."""
    assert version_lt("shadow_runner-0.1.0", MIN_RUNNER_VERSION_FOR_PASS)


# ---------------------------------------------------------------------------
# 2. parse_runner_version + version_lt semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_runner_version_basic() -> None:
    assert parse_runner_version("shadow_runner-0.2.0") == (0, 2, 0)
    assert parse_runner_version("shadow_runner-1.0.0") == (1, 0, 0)
    assert parse_runner_version("shadow_runner-0.10.5") == (0, 10, 5)


@pytest.mark.unit
def test_parse_runner_version_unparseable_raises() -> None:
    with pytest.raises(ValueError):
        parse_runner_version("not-a-runner")
    with pytest.raises(ValueError):
        parse_runner_version("shadow_runner-not.a.version")
    with pytest.raises(ValueError):
        parse_runner_version("")


@pytest.mark.unit
def test_version_lt_is_a_total_order() -> None:
    assert version_lt("shadow_runner-0.1.0", "shadow_runner-0.2.0")
    assert version_lt("shadow_runner-0.1.9", "shadow_runner-0.2.0")
    assert not version_lt("shadow_runner-0.2.0", "shadow_runner-0.2.0")
    assert not version_lt("shadow_runner-0.3.0", "shadow_runner-0.2.0")


# ---------------------------------------------------------------------------
# 3. G8 behaviour with an old (pre-Ticket-3) artefact
# ---------------------------------------------------------------------------


def _candidate_c1():
    from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
    return next(c for c in CANDIDATE_MENU_V0
                if c.candidate_id == "c1-lower-observe-floor-0.50")


def _build_old_artefact_with_pass_verdict(tmp_path):
    """Construct an old (Ticket-2-style) artefact whose runner_version
    is 0.1.0 but whose internal verdict is PASS — the kind of thing
    that would historically have been written before Ticket 3. The
    G8 gate must REFUSE PASS even though the artefact self-reports
    PASS, because the runner version is too old."""
    from smc.hedgerock.evolution.shadow_artefact import (
        SHADOW_ARTEFACT_SCHEMA_VERSION,
        CandidateDiffSnapshot, DataSliceIdentity, NoLiveEvidence,
        NoLookaheadAudit, ReplayInvariants, ShadowArtefact, ShadowMetrics,
        ShadowVerdict, SidecarModuleHashes, dump_shadow_artefact,
    )
    from smc.hedgerock.evolution.policy_manifest import (
        compute_canonical_candidate_hash,
    )

    cand = _candidate_c1()
    z = ShadowMetrics(
        final_equity=10000.0, total_return_pct=0.0, max_dd_pct=0.0,
        near_stopout_count=0, n_trades=0, max_open_lots=0.0,
        max_grid_density=0, halt_event_count=0, n_bars_envelope_decided=0,
    )
    art = ShadowArtefact(
        artefact_schema_version=SHADOW_ARTEFACT_SCHEMA_VERSION,
        artefact_id="evb-old",
        generated_at="2026-04-01T00:00:00+00:00",
        candidate_id=cand.candidate_id,
        candidate_manifest_content_hash=compute_canonical_candidate_hash(cand),
        candidate_diff=CandidateDiffSnapshot(
            target=cand.diff.target,
            proposed_value=cand.diff.proposed_value,
            baseline_value=cand.diff.baseline_value,
        ),
        candidate_diff_hash="0" * 64,
        baseline_policy_id="phase_d_walk_forward_baseline_v1",
        baseline_policy_hash="0" * 64,
        candidate_overlay_id="0" * 64,
        data_slice=DataSliceIdentity(
            symbols=("XAUUSD", "EURUSD"),  # multi-symbol — to defeat single_symbol abstain
            time_range_start="2021-01-01",
            time_range_end="2025-01-01",
            timeframes=("H1", "H4", "D1"),
            closed_bar_rule_version="phase_d_strict_prior_v1",
            lake_snapshot_hash="0" * 64,
            lake_snapshot_row_counts={"H1": 1000, "H4": 250, "D1": 50},
        ),
        runner_version="shadow_runner-0.1.0",  # OLD
        mirror_version="0" * 64,
        metric_schema_version="0" * 64,
        sidecar_module_hashes=SidecarModuleHashes(
            policy_overlay="0" * 64, rule_engine_mirror="0" * 64,
            replay_constant_mirror="0" * 64, shadow_runner="0" * 64,
            shadow_metrics="0" * 64,
        ),
        baseline_metrics=z, candidate_metrics=z, delta_metrics=z,
        replay_invariants=ReplayInvariants(
            same_bar_set_used=True, same_transition_lock_state_machine=True,
            same_cooldown_carryover=True,
            decision_only_uses_strictly_prior_data=True,
            h4_partial_bar_in_window=False, d1_partial_bar_in_window=False,
            decision_uses_data_with_ts_lt_trade_bar_ts=True,
        ),
        no_live_evidence=NoLiveEvidence(
            decision_server_routes_unchanged_hash="0" * 64,
            rule_engine_constants_unchanged_hash="0" * 64,
            http_calls_made_count=0, broker_api_calls_made_count=0,
            files_written_under_src_or_config_or_mq5_count=0,
            files_written_under_approved_or_pointer_count=0,
        ),
        mirror_consistency_check="PASS",
        exposure_class_violation=False,
        no_lookahead_audit=NoLookaheadAudit(
            decision_uses_only_prior_closed_bars=True,
            partial_bar_violation_count=0,
        ),
        verdict=ShadowVerdict.PASS,  # claims PASS — must be refused
        verdict_reason="(synthetic; old runner_version)",
    )
    p = tmp_path / "old.json"
    dump_shadow_artefact(art, p)
    return cand, p


@pytest.mark.unit
def test_g8_refuses_pass_when_runner_version_is_old(tmp_path) -> None:
    """Even with PASS in the artefact, multi-symbol slice, clean
    metrics, etc — old runner_version blocks PASS at the G8 layer."""
    from smc.hedgerock.evolution.evidence_bundle import compute_file_sha256
    from smc.hedgerock.evolution.policy_manifest import EvidenceBundle, GateStatus
    from smc.hedgerock.evolution.promotion_gates import g8_shadow_comparison

    cand, art_path = _build_old_artefact_with_pass_verdict(tmp_path)
    bundle = EvidenceBundle(
        bundle_id="evb", bundle_hash_sha256="x" * 64,
        atlas_report_path="/x/atlas.md", atlas_report_hash_sha256="a" * 64,
        data_availability_report_path="/x/avail.md",
        data_availability_report_hash_sha256="b" * 64,
        walk_forward_run_paths=("/x/wf.md",),
        year_replication={"XAUUSD": {"years_total": 4, "years_passing": 4,
                                       "negative_sign_years": ()}},
        cross_symbol_count=1, halt_event_count=4,
        no_strategy_change=False,
        shadow_artefact_path=str(art_path),
        shadow_artefact_hash_sha256=compute_file_sha256(art_path),
    )
    r = g8_shadow_comparison(candidate=cand, bundle=bundle)
    assert r.status == GateStatus.ABSTAIN, (
        f"old artefact must yield ABSTAIN (not {r.status.value}); reason={r.reason}"
    )
    assert "runner_version_too_old" in r.reason.lower() or \
           "too old" in r.reason.lower() or \
           "0.1.0" in r.reason


@pytest.mark.unit
def test_g8_old_artefact_still_FAIL_when_integrity_breaks(tmp_path) -> None:
    """fail-closed: tamper the inner verdict_reason (without refreshing
    content_sha256). Even though the artefact is old, integrity-FAIL
    wins over the runner-version ABSTAIN — old artefact does not get
    any pass on integrity errors."""
    from smc.hedgerock.evolution.policy_manifest import EvidenceBundle, GateStatus
    from smc.hedgerock.evolution.promotion_gates import g8_shadow_comparison
    import json

    cand, art_path = _build_old_artefact_with_pass_verdict(tmp_path)
    art_path.chmod(0o644)
    raw = json.loads(art_path.read_text())
    raw["artefact"]["verdict_reason"] = "TAMPERED"
    art_path.write_text(json.dumps(raw, indent=2, sort_keys=True))
    # We deliberately compute the recorded hash on the TAMPERED file.
    from smc.hedgerock.evolution.evidence_bundle import compute_file_sha256
    bundle = EvidenceBundle(
        bundle_id="evb", bundle_hash_sha256="x" * 64,
        atlas_report_path="/x/atlas.md", atlas_report_hash_sha256="a" * 64,
        data_availability_report_path="/x/avail.md",
        data_availability_report_hash_sha256="b" * 64,
        walk_forward_run_paths=("/x/wf.md",),
        year_replication={"XAUUSD": {"years_total": 4, "years_passing": 4,
                                       "negative_sign_years": ()}},
        cross_symbol_count=1, halt_event_count=4,
        no_strategy_change=False,
        shadow_artefact_path=str(art_path),
        shadow_artefact_hash_sha256=compute_file_sha256(art_path),
    )
    r = g8_shadow_comparison(candidate=cand, bundle=bundle)
    # Envelope hash now matches disk (we recomputed). But INNER hash
    # (content_sha256 vs recomputed) doesn't match. So FAIL via
    # integrity, not ABSTAIN via runner_version.
    assert r.status == GateStatus.FAIL


@pytest.mark.unit
def test_g8_FAIL_when_runner_version_unparseable(tmp_path) -> None:
    """Garbage runner_version → FAIL: shadow_artefact_runner_version_unparseable."""
    from smc.hedgerock.evolution.shadow_artefact import (
        SHADOW_ARTEFACT_SCHEMA_VERSION,
        CandidateDiffSnapshot, DataSliceIdentity, NoLiveEvidence,
        NoLookaheadAudit, ReplayInvariants, ShadowArtefact, ShadowMetrics,
        ShadowVerdict, SidecarModuleHashes, dump_shadow_artefact,
    )
    from smc.hedgerock.evolution.policy_manifest import (
        EvidenceBundle, GateStatus, compute_canonical_candidate_hash,
    )
    from smc.hedgerock.evolution.promotion_gates import g8_shadow_comparison
    from smc.hedgerock.evolution.evidence_bundle import compute_file_sha256

    cand = _candidate_c1()
    z = ShadowMetrics(
        final_equity=10000.0, total_return_pct=0.0, max_dd_pct=0.0,
        near_stopout_count=0, n_trades=0, max_open_lots=0.0,
        max_grid_density=0, halt_event_count=0, n_bars_envelope_decided=0,
    )
    art = ShadowArtefact(
        artefact_schema_version=SHADOW_ARTEFACT_SCHEMA_VERSION,
        artefact_id="evb-bad-version",
        generated_at="2026-04-01T00:00:00+00:00",
        candidate_id=cand.candidate_id,
        candidate_manifest_content_hash=compute_canonical_candidate_hash(cand),
        candidate_diff=CandidateDiffSnapshot(
            target=cand.diff.target,
            proposed_value=cand.diff.proposed_value,
            baseline_value=cand.diff.baseline_value,
        ),
        candidate_diff_hash="0" * 64,
        baseline_policy_id="x", baseline_policy_hash="0" * 64,
        candidate_overlay_id="0" * 64,
        data_slice=DataSliceIdentity(
            symbols=("XAUUSD",), time_range_start="2024-01-01",
            time_range_end="2024-01-31", timeframes=("H1", "H4", "D1"),
            closed_bar_rule_version="phase_d_strict_prior_v1",
            lake_snapshot_hash="0" * 64,
            lake_snapshot_row_counts={"H1": 100, "H4": 25, "D1": 5},
        ),
        runner_version="garbage-not-a-version",
        mirror_version="0" * 64, metric_schema_version="0" * 64,
        sidecar_module_hashes=SidecarModuleHashes(
            policy_overlay="0" * 64, rule_engine_mirror="0" * 64,
            replay_constant_mirror="0" * 64, shadow_runner="0" * 64,
            shadow_metrics="0" * 64,
        ),
        baseline_metrics=z, candidate_metrics=z, delta_metrics=z,
        replay_invariants=ReplayInvariants(
            same_bar_set_used=True, same_transition_lock_state_machine=True,
            same_cooldown_carryover=True,
            decision_only_uses_strictly_prior_data=True,
            h4_partial_bar_in_window=False, d1_partial_bar_in_window=False,
            decision_uses_data_with_ts_lt_trade_bar_ts=True,
        ),
        no_live_evidence=NoLiveEvidence(
            decision_server_routes_unchanged_hash="0" * 64,
            rule_engine_constants_unchanged_hash="0" * 64,
            http_calls_made_count=0, broker_api_calls_made_count=0,
            files_written_under_src_or_config_or_mq5_count=0,
            files_written_under_approved_or_pointer_count=0,
        ),
        mirror_consistency_check="PASS",
        exposure_class_violation=False,
        no_lookahead_audit=NoLookaheadAudit(
            decision_uses_only_prior_closed_bars=True,
            partial_bar_violation_count=0,
        ),
        verdict=ShadowVerdict.PASS,
        verdict_reason="(synthetic; garbage runner_version)",
    )
    p = tmp_path / "bad-version.json"
    dump_shadow_artefact(art, p)
    bundle = EvidenceBundle(
        bundle_id="evb", bundle_hash_sha256="x" * 64,
        atlas_report_path="/x/atlas.md", atlas_report_hash_sha256="a" * 64,
        data_availability_report_path="/x/avail.md",
        data_availability_report_hash_sha256="b" * 64,
        walk_forward_run_paths=("/x/wf.md",),
        year_replication={"XAUUSD": {"years_total": 4, "years_passing": 2,
                                       "negative_sign_years": (2021,)}},
        cross_symbol_count=1, halt_event_count=4,
        no_strategy_change=False,
        shadow_artefact_path=str(p),
        shadow_artefact_hash_sha256=compute_file_sha256(p),
    )
    r = g8_shadow_comparison(candidate=cand, bundle=bundle)
    assert r.status == GateStatus.FAIL
    assert "unparseable" in r.reason.lower() or \
           "runner_version" in r.reason.lower()

"""Ticket 2 Step 1 — ShadowArtefact schema + hash/verify tests.

Pinned guarantees (per R2 + R6 step 1):
  - All v1 schema fields are required; missing → ShadowArtefactIntegrityError.
  - Wrapped envelope (`content_sha256` + `artefact`); bare layout rejected.
  - Tamper of any inner field → load raises ShadowArtefactIntegrityError.
  - Tamper of recorded content_sha256 → load raises.
  - Files dump as chmod 0444; second dump to same path raises FileExistsError.
  - Round-trip is byte-identical (deterministic JSON).
  - runner_version / mirror_version / metric_schema_version absence → fail.
"""

from __future__ import annotations

import json
import stat
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from smc.hedgerock.evolution.shadow_artefact import (
    SHADOW_ARTEFACT_SCHEMA_VERSION,
    CandidateDiffSnapshot,
    DataSliceIdentity,
    NoLiveEvidence,
    NoLookaheadAudit,
    ReplayInvariants,
    ShadowArtefact,
    ShadowArtefactIntegrityError,
    ShadowMetrics,
    ShadowVerdict,
    SidecarModuleHashes,
    artefact_to_dict,
    artefact_from_dict,
    dump_shadow_artefact,
    load_shadow_artefact,
    verify_shadow_artefact_unchanged,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zero_metrics() -> ShadowMetrics:
    return ShadowMetrics(
        final_equity=10_000.0,
        total_return_pct=0.0,
        max_dd_pct=0.0,
        near_stopout_count=0,
        n_trades=0,
        max_open_lots=0.0,
        max_grid_density=0,
        halt_event_count=0,
        n_bars_envelope_decided=0,
    )


def _bare_artefact(*, verdict: ShadowVerdict = ShadowVerdict.PASS) -> ShadowArtefact:
    """Minimal valid ShadowArtefact for round-trip / immutability tests."""
    return ShadowArtefact(
        artefact_schema_version=SHADOW_ARTEFACT_SCHEMA_VERSION,
        artefact_id="evb-shadow-test-0001",
        generated_at="2026-05-01T12:00:00+00:00",
        candidate_id="c1-lower-observe-floor-0.50",
        candidate_manifest_content_hash="a" * 64,
        candidate_diff=CandidateDiffSnapshot(
            target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
            proposed_value=0.50,
            baseline_value=0.55,
        ),
        candidate_diff_hash="b" * 64,
        baseline_policy_id="phase_d_walk_forward_baseline_v1",
        baseline_policy_hash="c" * 64,
        candidate_overlay_id="d" * 64,
        data_slice=DataSliceIdentity(
            symbols=("XAUUSD",),
            time_range_start="2021-01-01",
            time_range_end="2025-01-01",
            timeframes=("H1", "H4", "D1"),
            closed_bar_rule_version="phase_d_strict_prior_v1",
            lake_snapshot_hash="e" * 64,
            lake_snapshot_row_counts={"H1": 28798, "H4": 7962, "D1": 1554},
        ),
        runner_version="shadow_runner-0.1.0",
        mirror_version="f" * 64,
        metric_schema_version="g" * 64,
        sidecar_module_hashes=SidecarModuleHashes(
            policy_overlay="h" * 64,
            rule_engine_mirror="i" * 64,
            replay_constant_mirror="j" * 64,
            shadow_runner="k" * 64,
            shadow_metrics="l" * 64,
        ),
        baseline_metrics=_zero_metrics(),
        candidate_metrics=_zero_metrics(),
        delta_metrics=_zero_metrics(),
        replay_invariants=ReplayInvariants(
            same_bar_set_used=True,
            same_transition_lock_state_machine=True,
            same_cooldown_carryover=True,
            decision_only_uses_strictly_prior_data=True,
            h4_partial_bar_in_window=False,
            d1_partial_bar_in_window=False,
            decision_uses_data_with_ts_lt_trade_bar_ts=True,
        ),
        no_live_evidence=NoLiveEvidence(
            decision_server_routes_unchanged_hash="m" * 64,
            rule_engine_constants_unchanged_hash="n" * 64,
            http_calls_made_count=0,
            broker_api_calls_made_count=0,
            files_written_under_src_or_config_or_mq5_count=0,
            files_written_under_approved_or_pointer_count=0,
        ),
        mirror_consistency_check="PASS",
        exposure_class_violation=False,
        no_lookahead_audit=NoLookaheadAudit(
            decision_uses_only_prior_closed_bars=True,
            partial_bar_violation_count=0,
        ),
        verdict=verdict,
        verdict_reason="shadow comparison passed all thresholds",
    )


# ===========================================================================
# 1. Schema & freeze
# ===========================================================================


@pytest.mark.unit
def test_schema_version_constant_is_v1() -> None:
    assert SHADOW_ARTEFACT_SCHEMA_VERSION == "1.0.0"


@pytest.mark.unit
def test_artefact_dataclass_is_frozen() -> None:
    a = _bare_artefact()
    with pytest.raises(FrozenInstanceError):
        a.candidate_id = "mutated"  # type: ignore[misc]


@pytest.mark.unit
def test_substructure_dataclasses_are_frozen() -> None:
    a = _bare_artefact()
    with pytest.raises(FrozenInstanceError):
        a.data_slice.symbols = ("EURUSD",)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        a.replay_invariants.same_bar_set_used = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        a.no_live_evidence.http_calls_made_count = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        a.candidate_diff.proposed_value = 0.99  # type: ignore[misc]


@pytest.mark.unit
def test_shadow_verdict_enum_constrained() -> None:
    """v1 verdict labels: PASS / FAIL / ABSTAIN / NOT_RUN."""
    expected = {"PASS", "FAIL", "ABSTAIN", "NOT_RUN"}
    actual = {v.value for v in ShadowVerdict}
    assert actual == expected


# ===========================================================================
# 2. JSON IO round-trip
# ===========================================================================


@pytest.mark.unit
def test_dump_load_round_trip_byte_identical(tmp_path: Path) -> None:
    a = _bare_artefact()
    p1 = tmp_path / "a.json"
    dump_shadow_artefact(a, p1)
    loaded = load_shadow_artefact(p1)
    p2 = tmp_path / "b.json"
    dump_shadow_artefact(loaded, p2)
    assert p1.read_bytes() == p2.read_bytes()


@pytest.mark.unit
def test_dump_writes_mode_0444(tmp_path: Path) -> None:
    a = _bare_artefact()
    p = tmp_path / "c.json"
    dump_shadow_artefact(a, p)
    mode = stat.S_IMODE(p.stat().st_mode)
    assert mode == 0o444, f"expected 0o444, got {oct(mode)}"


@pytest.mark.unit
def test_dump_does_not_overwrite_existing(tmp_path: Path) -> None:
    """Artefacts are immutable once written."""
    a = _bare_artefact()
    p = tmp_path / "d.json"
    dump_shadow_artefact(a, p)
    with pytest.raises(FileExistsError):
        dump_shadow_artefact(a, p)


# ===========================================================================
# 3. Strict load — envelope + content_sha256
# ===========================================================================


@pytest.mark.unit
def test_integrity_error_is_value_error_subclass() -> None:
    assert issubclass(ShadowArtefactIntegrityError, ValueError)


@pytest.mark.unit
def test_load_rejects_tampered_inner_verdict_reason(tmp_path: Path) -> None:
    a = _bare_artefact()
    p = tmp_path / "tamper.json"
    dump_shadow_artefact(a, p)
    p.chmod(0o644)
    raw = json.loads(p.read_text())
    raw["artefact"]["verdict_reason"] = "TAMPERED"
    p.write_text(json.dumps(raw, indent=2, sort_keys=True))
    with pytest.raises(ShadowArtefactIntegrityError):
        load_shadow_artefact(p)


@pytest.mark.unit
def test_load_rejects_tampered_inner_lake_snapshot_hash(tmp_path: Path) -> None:
    a = _bare_artefact()
    p = tmp_path / "tamper-data-slice.json"
    dump_shadow_artefact(a, p)
    p.chmod(0o644)
    raw = json.loads(p.read_text())
    raw["artefact"]["data_slice"]["lake_snapshot_hash"] = "z" * 64
    p.write_text(json.dumps(raw, indent=2, sort_keys=True))
    with pytest.raises(ShadowArtefactIntegrityError):
        load_shadow_artefact(p)


@pytest.mark.unit
def test_load_rejects_tampered_content_sha256(tmp_path: Path) -> None:
    a = _bare_artefact()
    p = tmp_path / "tamper-hash.json"
    dump_shadow_artefact(a, p)
    p.chmod(0o644)
    raw = json.loads(p.read_text())
    raw["content_sha256"] = "0" * 64
    p.write_text(json.dumps(raw, indent=2, sort_keys=True))
    with pytest.raises(ShadowArtefactIntegrityError):
        load_shadow_artefact(p)


@pytest.mark.unit
def test_load_rejects_bare_layout(tmp_path: Path) -> None:
    """Hand-authored bare artefact (no envelope) must not load."""
    a = _bare_artefact()
    bare = artefact_to_dict(a)
    p = tmp_path / "bare.json"
    p.write_text(json.dumps(bare, indent=2, sort_keys=True))
    with pytest.raises(ShadowArtefactIntegrityError):
        load_shadow_artefact(p)


@pytest.mark.unit
def test_load_rejects_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "broken.json"
    p.write_text("{not valid json")
    with pytest.raises(ShadowArtefactIntegrityError):
        load_shadow_artefact(p)


# ===========================================================================
# 4. Required-field gating — every R2 field must be present
# ===========================================================================


_REQUIRED_TOP_LEVEL_FIELDS = (
    "artefact_schema_version",
    "artefact_id",
    "generated_at",
    "candidate_id",
    "candidate_manifest_content_hash",
    "candidate_diff",
    "candidate_diff_hash",
    "baseline_policy_id",
    "baseline_policy_hash",
    "candidate_overlay_id",
    "data_slice",
    "runner_version",
    "mirror_version",
    "metric_schema_version",
    "sidecar_module_hashes",
    "baseline_metrics",
    "candidate_metrics",
    "delta_metrics",
    "replay_invariants",
    "no_live_evidence",
    "mirror_consistency_check",
    "exposure_class_violation",
    "no_lookahead_audit",
    "verdict",
    "verdict_reason",
)


@pytest.mark.unit
@pytest.mark.parametrize("missing_field", _REQUIRED_TOP_LEVEL_FIELDS)
def test_load_rejects_missing_top_level_field(
    tmp_path: Path, missing_field: str,
) -> None:
    """Each R2 v1 top-level field is mandatory; absence → load fails."""
    a = _bare_artefact()
    p = tmp_path / f"missing-{missing_field}.json"
    dump_shadow_artefact(a, p)
    p.chmod(0o644)
    raw = json.loads(p.read_text())
    del raw["artefact"][missing_field]
    p.write_text(json.dumps(raw, indent=2, sort_keys=True))
    with pytest.raises(ShadowArtefactIntegrityError):
        load_shadow_artefact(p)


@pytest.mark.unit
def test_load_rejects_unknown_schema_major(tmp_path: Path) -> None:
    """v2 major is refused by v1 loader."""
    a = _bare_artefact()
    p = tmp_path / "v2.json"
    dump_shadow_artefact(a, p)
    p.chmod(0o644)
    raw = json.loads(p.read_text())
    raw["artefact"]["artefact_schema_version"] = "2.0.0"
    p.write_text(json.dumps(raw, indent=2, sort_keys=True))
    with pytest.raises(ShadowArtefactIntegrityError):
        load_shadow_artefact(p)


# ===========================================================================
# 5. verify_shadow_artefact_unchanged
# ===========================================================================


@pytest.mark.unit
def test_verify_unchanged_pass_for_clean_dump(tmp_path: Path) -> None:
    a = _bare_artefact()
    p = tmp_path / "ok.json"
    dump_shadow_artefact(a, p)
    assert verify_shadow_artefact_unchanged(p) is True


@pytest.mark.unit
def test_verify_unchanged_false_after_inner_tamper(tmp_path: Path) -> None:
    a = _bare_artefact()
    p = tmp_path / "tamper-verify.json"
    dump_shadow_artefact(a, p)
    p.chmod(0o644)
    raw = json.loads(p.read_text())
    raw["artefact"]["verdict_reason"] = "TAMPER"
    p.write_text(json.dumps(raw, indent=2, sort_keys=True))
    assert verify_shadow_artefact_unchanged(p) is False


@pytest.mark.unit
def test_verify_unchanged_false_for_missing_file(tmp_path: Path) -> None:
    assert verify_shadow_artefact_unchanged(tmp_path / "missing.json") is False


# ===========================================================================
# 6. artefact_from_dict (test fixture escape hatch)
# ===========================================================================


@pytest.mark.unit
def test_artefact_from_dict_round_trip_bare() -> None:
    a = _bare_artefact()
    rebuilt = artefact_from_dict(artefact_to_dict(a))
    assert rebuilt == a


@pytest.mark.unit
def test_load_rejects_non_string_content_hash(tmp_path: Path) -> None:
    a = _bare_artefact()
    p = tmp_path / "bad-hash-type.json"
    dump_shadow_artefact(a, p)
    p.chmod(0o644)
    raw = json.loads(p.read_text())
    raw["content_sha256"] = 12345  # not a string
    p.write_text(json.dumps(raw))
    with pytest.raises(ShadowArtefactIntegrityError):
        load_shadow_artefact(p)

"""Phase D-cont3 / Ticket 1 — policy_manifest tests.

Pinned guarantees:
  - Frozen dataclasses — mutation raises.
  - JSON round-trip is byte-identical.
  - dump writes mode 0444 (read-only).
  - Schema-version is required and rejected on unknown major.
  - State enum values are constrained.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from smc.hedgerock.evolution.policy_manifest import (
    CandidateDiff,
    CandidateDiffScope,
    CandidateManifest,
    CandidateState,
    EvidenceBundle,
    GateStatus,
    ManifestIntegrityError,
    OverallResult,
    PromotionGateResult,
    dump_manifest,
    load_manifest,
    manifest_from_dict,
    manifest_to_dict,
    verify_manifest_unchanged,
    MANIFEST_SCHEMA_VERSION,
)


def _bare_manifest(state: CandidateState = CandidateState.DRAFT) -> CandidateManifest:
    diff = CandidateDiff(
        kind="threshold_tweak",
        target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        baseline_value=0.55,
        proposed_value=0.50,
        scope=CandidateDiffScope(
            regimes_affected=("range",),
            affects_halt_mode=False,
            affects_classifier_or_rule_engine=True,
            raises_gross_exposure=False,
            interfaces_touched=(),
        ),
    )
    return CandidateManifest(
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        candidate_id="c-test",
        title="test candidate",
        author="test",
        created_at="2026-05-01T12:00:00+00:00",
        state=state,
        diff=diff,
        evidence_bundle=None,
        gates=(),
        result=OverallResult.PROMOTION_BLOCKED,
        blocking_reasons=("test_block",),
        next_data_needs=(),
        required_next_data_or_policy="",
        human_approval_required_for_state_transitions_above="tested",
        audit_trail=(),
    )


@pytest.mark.unit
def test_dataclass_is_frozen() -> None:
    m = _bare_manifest()
    with pytest.raises(FrozenInstanceError):
        m.candidate_id = "mutated"  # type: ignore[misc]


@pytest.mark.unit
def test_state_enum_constrained() -> None:
    """Only the documented states exist."""
    expected = {"draft", "tested", "shadow_validated",
                "canary", "approved", "quarantined"}
    actual = {s.value for s in CandidateState}
    assert actual == expected


@pytest.mark.unit
def test_gate_status_enum_constrained() -> None:
    expected = {"PASS", "FAIL", "ABSTAIN", "NOT_RUN"}
    actual = {s.value for s in GateStatus}
    assert actual == expected


@pytest.mark.unit
def test_overall_result_enum_includes_manifest_invalid() -> None:
    """RFC §10.1 / Plan §3 G6 (b): undefined bound -> manifest_invalid."""
    assert "PROMOTION_BLOCKED_MANIFEST_INVALID" in {r.name for r in OverallResult}


@pytest.mark.unit
def test_dump_load_round_trip(tmp_path: Path) -> None:
    """Load → dump → byte-identical round trip."""
    m = _bare_manifest()
    p1 = tmp_path / "a.json"
    dump_manifest(m, p1)
    loaded = load_manifest(p1)
    p2 = tmp_path / "b.json"
    # Allow the loaded manifest to be re-written; clear chmod to avoid
    # write failure on the second dump (the file is created fresh).
    dump_manifest(loaded, p2)
    assert p1.read_bytes() == p2.read_bytes()


@pytest.mark.unit
def test_dump_writes_mode_0444(tmp_path: Path) -> None:
    """Files are written as read-only — overwrite must fail."""
    m = _bare_manifest()
    p = tmp_path / "c.json"
    dump_manifest(m, p)
    mode = stat.S_IMODE(p.stat().st_mode)
    # 0o444 = read-only for owner/group/other
    assert mode == 0o444, f"expected 0o444, got {oct(mode)}"


@pytest.mark.unit
def test_dump_does_not_overwrite_existing(tmp_path: Path) -> None:
    """A second dump to the same path must raise — manifests are
    immutable once written."""
    m = _bare_manifest()
    p = tmp_path / "d.json"
    dump_manifest(m, p)
    with pytest.raises(FileExistsError):
        dump_manifest(m, p)


@pytest.mark.unit
def test_load_rejects_missing_schema_version(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"candidate_id": "x"}))
    with pytest.raises((ValueError, KeyError)):
        load_manifest(p)


@pytest.mark.unit
def test_load_rejects_unknown_major_version(tmp_path: Path) -> None:
    """Anything outside the v1 major is refused."""
    m = _bare_manifest()
    p = tmp_path / "v2.json"
    dump_manifest(m, p)
    # Mutate the on-disk INNER manifest payload to claim a v2 major.
    # (Tampering also invalidates content_sha256 — verify_manifest_unchanged
    # would return False — but the load() path itself must refuse the
    # version major regardless of hash check.)
    raw = json.loads(p.read_text())
    raw["manifest"]["manifest_schema_version"] = "2.0.0"
    p.chmod(0o644)
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_manifest(p)


@pytest.mark.unit
def test_verify_manifest_unchanged_detects_tamper(tmp_path: Path) -> None:
    m = _bare_manifest()
    p = tmp_path / "e.json"
    dump_manifest(m, p)
    assert verify_manifest_unchanged(p) is True
    # tamper
    p.chmod(0o644)
    raw = p.read_text()
    p.write_text(raw.replace("test candidate", "tampered"))
    assert verify_manifest_unchanged(p) is False


@pytest.mark.unit
def test_promotion_gate_result_dataclass() -> None:
    r = PromotionGateResult(
        gate_id="G1",
        status=GateStatus.PASS,
        reason="all artefacts present",
        details={},
    )
    assert r.gate_id == "G1"
    with pytest.raises(FrozenInstanceError):
        r.status = GateStatus.FAIL  # type: ignore[misc]


# ===========================================================================
# Ticket 1-closeout — Finding 1 [P1]: load_manifest() must verify
# content_sha256 (not just verify_manifest_unchanged)
# ===========================================================================


@pytest.mark.unit
def test_manifest_integrity_error_is_value_error_subclass() -> None:
    """Callers may catch ValueError if they don't care about the
    specific integrity-failure shape. Subclassing keeps that
    contract."""
    assert issubclass(ManifestIntegrityError, ValueError)


@pytest.mark.unit
def test_load_manifest_rejects_tampered_inner_title(tmp_path: Path) -> None:
    """Mutating any inner-payload field after dump must break the
    envelope hash and cause load_manifest to refuse the file."""
    m = _bare_manifest()
    p = tmp_path / "tamper-title.json"
    dump_manifest(m, p)
    p.chmod(0o644)
    raw = json.loads(p.read_text())
    raw["manifest"]["title"] = "TAMPERED"
    p.write_text(json.dumps(raw, indent=2, sort_keys=True))
    with pytest.raises(ManifestIntegrityError):
        load_manifest(p)


@pytest.mark.unit
def test_load_manifest_rejects_tampered_inner_result(tmp_path: Path) -> None:
    """Same invariant for a different field — value-level tamper on
    `result` must also be detected, not just structural tamper."""
    m = _bare_manifest()
    p = tmp_path / "tamper-result.json"
    dump_manifest(m, p)
    p.chmod(0o644)
    raw = json.loads(p.read_text())
    raw["manifest"]["result"] = "READY_FOR_TESTED"
    p.write_text(json.dumps(raw, indent=2, sort_keys=True))
    with pytest.raises(ManifestIntegrityError):
        load_manifest(p)


@pytest.mark.unit
def test_load_manifest_rejects_tampered_content_sha256(tmp_path: Path) -> None:
    """Even if someone keeps the inner payload intact and only
    changes the recorded hash, the recomputed hash will not match."""
    m = _bare_manifest()
    p = tmp_path / "tamper-hash.json"
    dump_manifest(m, p)
    p.chmod(0o644)
    raw = json.loads(p.read_text())
    raw["content_sha256"] = "0" * 64
    p.write_text(json.dumps(raw, indent=2, sort_keys=True))
    with pytest.raises(ManifestIntegrityError):
        load_manifest(p)


@pytest.mark.unit
def test_load_manifest_rejects_bare_layout(tmp_path: Path) -> None:
    """A hand-authored bare manifest (no envelope) must be refused by
    the strict loader — registry-managed files are always wrapped."""
    m = _bare_manifest()
    bare_dict = manifest_to_dict(m)
    p = tmp_path / "bare.json"
    p.write_text(json.dumps(bare_dict, indent=2, sort_keys=True))
    with pytest.raises(ManifestIntegrityError):
        load_manifest(p)


@pytest.mark.unit
def test_manifest_from_dict_still_works_for_test_fixtures() -> None:
    """For test fixtures that need the bare path, manifest_from_dict()
    is the explicit escape hatch — load_manifest() refuses bare layouts
    on disk so registry-managed manifests stay strict, but
    manifest_from_dict() can still be called with a raw dict."""
    m = _bare_manifest()
    bare_dict = manifest_to_dict(m)
    rebuilt = manifest_from_dict(bare_dict)
    assert rebuilt.candidate_id == m.candidate_id
    assert rebuilt.title == m.title


@pytest.mark.unit
def test_load_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "broken.json"
    p.write_text("{not valid json")
    with pytest.raises(ManifestIntegrityError):
        load_manifest(p)


@pytest.mark.unit
def test_load_manifest_rejects_non_string_content_hash(tmp_path: Path) -> None:
    m = _bare_manifest()
    p = tmp_path / "bad-hash-type.json"
    dump_manifest(m, p)
    p.chmod(0o644)
    raw = json.loads(p.read_text())
    raw["content_sha256"] = 12345  # not a string
    p.write_text(json.dumps(raw))
    with pytest.raises(ManifestIntegrityError):
        load_manifest(p)


@pytest.mark.unit
def test_dump_then_load_round_trip_passes_integrity_check(tmp_path: Path) -> None:
    """Sanity: an untampered dump+load cycle returns the original
    manifest with no integrity error."""
    m = _bare_manifest()
    p = tmp_path / "ok.json"
    dump_manifest(m, p)
    loaded = load_manifest(p)
    assert loaded.candidate_id == m.candidate_id
    assert loaded.title == m.title
    assert loaded.result == m.result


@pytest.mark.unit
def test_evidence_bundle_dataclass_is_frozen() -> None:
    eb = EvidenceBundle(
        bundle_id="evb-test",
        bundle_hash_sha256="0" * 64,
        atlas_report_path="docs/atlas.md",
        atlas_report_hash_sha256="1" * 64,
        data_availability_report_path="docs/availability.md",
        data_availability_report_hash_sha256="2" * 64,
        walk_forward_run_paths=(),
        year_replication={"XAUUSD": {"years_total": 4, "years_passing": 2,
                                      "negative_sign_years": (2021,)}},
        cross_symbol_count=1,
        halt_event_count=4,
        no_strategy_change=True,
    )
    with pytest.raises(FrozenInstanceError):
        eb.cross_symbol_count = 99  # type: ignore[misc]

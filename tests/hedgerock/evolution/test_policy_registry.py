"""Phase D-cont3 / Ticket 1 — policy_registry tests.

The registry is the FS layer that the sidecar uses to persist
candidate manifests + audit log entries. Tests pin the hard-boundary
invariants: no writes under approved/, pointer.json, src/, or
config/.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
from smc.hedgerock.evolution.policy_manifest import (

    CandidateManifest,
    CandidateState,
)
from smc.hedgerock.evolution.policy_manifest import (
    ManifestIntegrityError,
    dump_manifest,
    manifest_to_dict,
)
from smc.hedgerock.evolution.policy_registry import (
    IllegalCandidateState,
    InvalidCandidateId,
    PolicyRegistry,
    RegistryWriteForbidden,
    validate_candidate_id,
)


from tests.hedgerock.evolution._paths import (
    ai_smc_home as _ai_smc_home_p,
    hedgerock_home as _hedgerock_home_p,
    real_audit_log as _real_audit_log_p,
    real_registry_root as _real_registry_p,
    real_shadow_artefacts_root as _real_shadow_p,
    scripts_dir as _scripts_dir_p,
)

@pytest.fixture
def registry(tmp_path: Path) -> PolicyRegistry:
    return PolicyRegistry(root=tmp_path / "policy_registry")


# ---------------------------------------------------------------------------
# Candidate writes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_candidate_creates_file_under_candidates_dir(registry: PolicyRegistry) -> None:
    cand = CANDIDATE_MENU_V0[0]
    p = registry.write_candidate(cand)
    assert p.parent == registry.candidates_dir
    assert p.exists()


@pytest.mark.unit
def test_write_candidate_refuses_overwrite(registry: PolicyRegistry) -> None:
    cand = CANDIDATE_MENU_V0[0]
    registry.write_candidate(cand)
    with pytest.raises(FileExistsError):
        registry.write_candidate(cand)


@pytest.mark.unit
def test_write_candidate_refuses_higher_state(registry: PolicyRegistry) -> None:
    """Plan §6.3: MVP only writes draft/tested. shadow_validated and
    above are out of scope."""
    cand = CANDIDATE_MENU_V0[0]
    promoted = CandidateManifest(
        manifest_schema_version=cand.manifest_schema_version,
        candidate_id=cand.candidate_id + "-promoted",
        title=cand.title,
        author=cand.author,
        created_at=cand.created_at,
        state=CandidateState.SHADOW_VALIDATED,
        diff=cand.diff,
        evidence_bundle=cand.evidence_bundle,
        gates=cand.gates,
        result=cand.result,
        blocking_reasons=cand.blocking_reasons,
        next_data_needs=cand.next_data_needs,
        required_next_data_or_policy=cand.required_next_data_or_policy,
        human_approval_required_for_state_transitions_above=cand.human_approval_required_for_state_transitions_above,
        audit_trail=cand.audit_trail,
    )
    with pytest.raises(IllegalCandidateState):
        registry.write_candidate(promoted)


# ---------------------------------------------------------------------------
# Approved / pointer writes are forbidden
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_set_pointer_raises_not_implemented(registry: PolicyRegistry) -> None:
    """Plan §6.3 + RFC §11: pointer flip is human-only / MVP+1."""
    with pytest.raises(NotImplementedError):
        registry.set_pointer("any-id")


@pytest.mark.unit
def test_get_pointer_returns_none_for_empty_registry(registry: PolicyRegistry) -> None:
    assert registry.get_pointer() is None


@pytest.mark.unit
def test_write_approved_raises_forbidden(registry: PolicyRegistry) -> None:
    cand = CANDIDATE_MENU_V0[0]
    with pytest.raises(RegistryWriteForbidden):
        registry.write_to_path(
            registry.approved_dir / "x.json",
            cand,
        )


@pytest.mark.unit
def test_write_under_src_raises_forbidden(registry: PolicyRegistry) -> None:
    cand = CANDIDATE_MENU_V0[0]
    src_path = (_ai_smc_home_p() / 'src' / 'smc' / 'hedgerock' / 'x.json')
    with pytest.raises(RegistryWriteForbidden):
        registry.write_to_path(src_path, cand)


@pytest.mark.unit
def test_write_under_config_raises_forbidden(registry: PolicyRegistry) -> None:
    cand = CANDIDATE_MENU_V0[0]
    cfg_path = (_ai_smc_home_p() / 'config' / 'safety_bounds.yaml')
    with pytest.raises(RegistryWriteForbidden):
        registry.write_to_path(cfg_path, cand)


@pytest.mark.unit
def test_write_under_mq5_raises_forbidden(registry: PolicyRegistry) -> None:
    cand = CANDIDATE_MENU_V0[0]
    mq5_path = (_hedgerock_home_p() / 'mql5' / 'HedgeRock.mq5')
    with pytest.raises(RegistryWriteForbidden):
        registry.write_to_path(mq5_path, cand)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_append_audit_writes_under_audit_dir(registry: PolicyRegistry) -> None:
    p = registry.append_audit({
        "action": "write_candidate",
        "candidate_id": "c-test",
    })
    assert p.parent == registry.audit_dir
    assert p.exists()


@pytest.mark.unit
def test_append_audit_refuses_to_overwrite(registry: PolicyRegistry, monkeypatch) -> None:
    """Audit entries are append-only — writing twice with the same
    timestamp must raise."""
    # Force the same timestamp twice by stubbing the timestamp source.
    fixed_ts = "2026-05-01T12-00-00"
    monkeypatch.setattr(
        "smc.hedgerock.evolution.policy_registry._audit_timestamp",
        lambda: fixed_ts,
    )
    registry.append_audit({"action": "x"})
    with pytest.raises(FileExistsError):
        registry.append_audit({"action": "x"})


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_list_candidates_returns_written_ids(registry: PolicyRegistry) -> None:
    for c in CANDIDATE_MENU_V0[:2]:
        registry.write_candidate(c)
    listed = registry.list_candidates()
    assert set(listed) == {c.candidate_id for c in CANDIDATE_MENU_V0[:2]}


@pytest.mark.unit
def test_get_candidate_round_trip(registry: PolicyRegistry) -> None:
    cand = CANDIDATE_MENU_V0[0]
    registry.write_candidate(cand)
    loaded = registry.get_candidate(cand.candidate_id)
    assert loaded.candidate_id == cand.candidate_id
    assert loaded.diff.target == cand.diff.target


# ---------------------------------------------------------------------------
# Static-scan: no module imports anything that would suggest production writes
# ---------------------------------------------------------------------------


# ===========================================================================
# Ticket 1-closeout — Finding 2 [P2]: candidate_id path-injection defence
# ===========================================================================


@pytest.mark.unit
def test_validate_candidate_id_accepts_menu_v0_ids() -> None:
    """All four hand-curated menu entries must pass — these are the
    canonical examples of valid IDs."""
    for cand in CANDIDATE_MENU_V0:
        # Must not raise
        validate_candidate_id(cand.candidate_id)


@pytest.mark.unit
@pytest.mark.parametrize("evil_id", [
    "../audit/evil",
    "../../outside",
    "../../../etc/passwd",
    "/absolute/path",
    "has space",
    "has\ttab",
    "slash/name",
    "back\\slash",
    ".hidden",
    "-leading-dash",
    "_leading-underscore",
    "",
    "name with newline\n",
    "name\x00null",
])
def test_validate_candidate_id_rejects_path_injection_attempts(evil_id: str) -> None:
    with pytest.raises(InvalidCandidateId):
        validate_candidate_id(evil_id)


@pytest.mark.unit
def test_validate_candidate_id_rejects_non_string() -> None:
    for evil in (None, 123, 3.14, b"bytes", ["list"]):
        with pytest.raises(InvalidCandidateId):
            validate_candidate_id(evil)  # type: ignore[arg-type]


@pytest.mark.unit
def test_validate_candidate_id_rejects_oversized() -> None:
    with pytest.raises(InvalidCandidateId):
        validate_candidate_id("a" * 200)


@pytest.mark.unit
def test_write_candidate_rejects_evil_id_via_validator(
    registry: PolicyRegistry,
) -> None:
    """write_candidate must refuse a malicious id BEFORE any path is
    built — InvalidCandidateId, not a successful write into the audit
    subtree (which the path-allowlist would have permitted)."""
    cand = CANDIDATE_MENU_V0[0]
    evil = CandidateManifest(
        manifest_schema_version=cand.manifest_schema_version,
        candidate_id="../audit/evil",
        title=cand.title,
        author=cand.author,
        created_at=cand.created_at,
        state=cand.state,
        diff=cand.diff,
        evidence_bundle=cand.evidence_bundle,
        gates=cand.gates,
        result=cand.result,
        blocking_reasons=cand.blocking_reasons,
        next_data_needs=cand.next_data_needs,
        required_next_data_or_policy=cand.required_next_data_or_policy,
        human_approval_required_for_state_transitions_above=cand.human_approval_required_for_state_transitions_above,
        audit_trail=cand.audit_trail,
    )
    with pytest.raises(InvalidCandidateId):
        registry.write_candidate(evil)
    # And — defence-in-depth — no file with that name landed in audit.
    assert not (registry.audit_dir / "evil.json").exists()


@pytest.mark.unit
@pytest.mark.parametrize("evil_id", [
    "../audit/evil",
    "../../outside",
    "has space",
    "slash/name",
])
def test_get_candidate_rejects_evil_id_pre_load(
    registry: PolicyRegistry, evil_id: str,
) -> None:
    with pytest.raises(InvalidCandidateId):
        registry.get_candidate(evil_id)


# ===========================================================================
# Ticket 1-closeout — Finding 1 cross-check at the registry layer
# ===========================================================================


@pytest.mark.unit
def test_registry_get_candidate_rejects_bare_layout(
    registry: PolicyRegistry, tmp_path: Path,
) -> None:
    """A bare manifest planted under candidates/ must NOT be
    loadable via registry.get_candidate — the strict loader refuses
    bare layouts."""
    cand = CANDIDATE_MENU_V0[0]
    registry.candidates_dir.mkdir(parents=True, exist_ok=True)
    bare_path = registry.candidates_dir / f"{cand.candidate_id}.json"
    bare_dict = manifest_to_dict(cand)
    import json as _json
    bare_path.write_text(_json.dumps(bare_dict, indent=2, sort_keys=True))
    with pytest.raises(ManifestIntegrityError):
        registry.get_candidate(cand.candidate_id)


# ===========================================================================
# Ticket 1-closeout-2 — symlink fail-closed on registry directories
# ===========================================================================


@pytest.mark.unit
def test_write_candidate_fails_closed_when_candidates_dir_is_symlink(
    tmp_path: Path,
) -> None:
    """If `candidates/` is a symlink (e.g. an attacker plants one to
    redirect writes), the registry MUST fail-closed rather than write
    through the link."""
    real_target = tmp_path / "real_target"
    real_target.mkdir()
    fake_root = tmp_path / "fake_registry"
    fake_root.mkdir()
    (fake_root / "candidates").symlink_to(real_target)

    registry = PolicyRegistry(root=fake_root)
    cand = CANDIDATE_MENU_V0[0]
    with pytest.raises(RegistryWriteForbidden):
        registry.write_candidate(cand)
    # The symlinked target stays empty — no write leaked through.
    assert list(real_target.iterdir()) == []


@pytest.mark.unit
def test_append_audit_fails_closed_when_audit_dir_is_symlink(
    tmp_path: Path,
) -> None:
    real_target = tmp_path / "audit_target"
    real_target.mkdir()
    fake_root = tmp_path / "fake_registry"
    fake_root.mkdir()
    (fake_root / "audit").symlink_to(real_target)

    registry = PolicyRegistry(root=fake_root)
    with pytest.raises(RegistryWriteForbidden):
        registry.append_audit({"action": "x"})
    assert list(real_target.iterdir()) == []


@pytest.mark.unit
def test_real_candidates_and_audit_dirs_pass(tmp_path: Path) -> None:
    """Sanity counterpart: when the directories exist as real (non-
    symlink) directories, writes succeed normally."""
    registry = PolicyRegistry(root=tmp_path / "real_registry")
    cand = CANDIDATE_MENU_V0[0]
    registry.write_candidate(cand)
    registry.append_audit({"action": "ok"})
    assert (registry.candidates_dir / f"{cand.candidate_id}.json").exists()
    assert any(registry.audit_dir.glob("*.json"))


# ===========================================================================
# Ticket 1-closeout-2 — static check: no destructive filesystem calls
# in the evolution sidecar or CLI source
# ===========================================================================


@pytest.mark.unit
def test_no_destructive_fs_calls_in_evolution_or_cli() -> None:
    """The sidecar source files and the report CLI MUST NOT contain
    code or recommendation text that deletes registry files. The
    canonical re-run pattern is "use a fresh registry root", not
    "delete the existing one".

    Specifically forbidden:
      - any AST `Call` whose attribute name is rmtree / unlink /
        remove / rmdir, OR whose function name is one of those.
      - any string literal containing "rm -rf" (catches subprocess
        args + comment-style text).

    Test files are exempt by construction: this test scopes only to
    `src/smc/hedgerock/evolution/*.py` + `scripts/hedgerock_evolution_report.py`.
    """
    import ast

    repo_root = Path(__file__).resolve().parents[3]
    paths = list((repo_root / "src" / "smc" / "hedgerock" / "evolution").glob("*.py"))
    paths.append(repo_root / "scripts" / "hedgerock_evolution_report.py")

    forbidden_call_names = {"rmtree", "unlink", "remove", "rmdir"}
    forbidden_substrings = ("rm -rf", "rm-rf")

    for p in paths:
        text = p.read_text(encoding="utf-8")
        for needle in forbidden_substrings:
            assert needle not in text, (
                f"{p.name}: forbidden token {needle!r} present — see "
                "Ticket 1-closeout-2 (no destructive filesystem suggestions)"
            )
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            raise AssertionError(f"{p.name}: failed to parse: {e}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in forbidden_call_names:
                    raise AssertionError(
                        f"{p.name}:{node.lineno}: forbidden destructive call "
                        f".{node.func.attr}() — registry history is "
                        "append-only; do not delete"
                    )
            elif isinstance(node.func, ast.Name):
                if node.func.id in forbidden_call_names:
                    raise AssertionError(
                        f"{p.name}:{node.lineno}: forbidden destructive call "
                        f"{node.func.id}() — registry history is append-only"
                    )


@pytest.mark.unit
def test_registry_get_candidate_rejects_tampered_wrapped(
    registry: PolicyRegistry,
) -> None:
    """A wrapped manifest written via dump_manifest, then mutated in
    place, must fail registry load with ManifestIntegrityError."""
    cand = CANDIDATE_MENU_V0[0]
    written = registry.write_candidate(cand)
    # Tamper inner field.
    written.chmod(0o644)
    import json as _json
    raw = _json.loads(written.read_text())
    raw["manifest"]["title"] = "INJECTED"
    written.write_text(_json.dumps(raw, indent=2, sort_keys=True))
    with pytest.raises(ManifestIntegrityError):
        registry.get_candidate(cand.candidate_id)


@pytest.mark.unit
def test_evolution_modules_do_not_import_production_runtime() -> None:
    """The sidecar must not import rule_engine or decision_server. A
    static check on the module source catches accidental imports."""
    import smc.hedgerock.evolution as evolution_pkg
    pkg_root = Path(evolution_pkg.__file__).parent
    forbidden = (
        "from smc.hedgerock.rule_engine",
        "import smc.hedgerock.rule_engine",
        "from smc.hedgerock.decision_server",
        "import smc.hedgerock.decision_server",
    )
    # Tier-1 read-only unseal — dynamic_replay imports rule_engine +
    # decision_server.MarketFeatures for closed-bar replay only.
    # Authorised by _RULE_ENGINE_REPLAY_WHITELIST in
    # test_regression_guard.py. Read-only enforcement is pinned by
    # test_dynamic_replay.py.
    tier1_unseal_files = {"dynamic_replay.py"}
    for py in pkg_root.glob("*.py"):
        if py.name in tier1_unseal_files:
            continue
        text = py.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, (
                f"{py.name} imports production runtime via '{needle}'"
            )


@pytest.mark.unit
def test_evolution_modules_do_not_open_safety_bounds_for_write() -> None:
    """No write-mode open of safety_bounds.yaml anywhere in the
    evolution package or its CLI."""
    import smc.hedgerock.evolution as evolution_pkg
    pkg_root = Path(evolution_pkg.__file__).parent
    cli = (_scripts_dir_p() / 'hedgerock_evolution_report.py')
    files = list(pkg_root.glob("*.py"))
    if cli.exists():
        files.append(cli)
    forbidden_substrings = (
        'safety_bounds.yaml", "w"',
        "safety_bounds.yaml', 'w'",
        '"config/safety_bounds.yaml"',
    )
    # The "open ... w" patterns are the most direct form. We allow the
    # path string to APPEAR for read use; we forbid the write-mode form.
    for f in files:
        text = f.read_text(encoding="utf-8")
        # Forbid write-mode against the bounds file.
        assert ", \"w\")" not in text or "safety_bounds" not in text, (
            f"{f.name} has potential write to safety_bounds.yaml"
        )
        # Forbid using shutil.copy* with safety_bounds destination.
        assert "shutil.copy" not in text or "safety_bounds" not in text, (
            f"{f.name} may copy onto safety_bounds.yaml"
        )

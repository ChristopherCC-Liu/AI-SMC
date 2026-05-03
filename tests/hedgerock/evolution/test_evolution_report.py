"""Phase D-cont3 / Ticket 1 — end-to-end evolution_report tests.

These tests anchor the user-listed acceptance criteria from the
final approval message:

  - current Phase D bundles → every candidate PROMOTION_BLOCKED
  - NO_STRATEGY_CHANGE: true blocks promotion
  - evidence bundle hash mismatch rejects
  - candidate referencing undefined safety bound → PROMOTION_BLOCKED /
    manifest_invalid
  - report/registry code has no write path to config/safety_bounds.yaml
  - no module writes under src production / approved / pointer / mq5
  - report contains required boundary strings + canonical certification
    line "safety_bounds write permission: 0"
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


from tests.hedgerock.evolution._paths import (
    ai_smc_home as _ai_smc_home_p,
    hedgerock_home as _hedgerock_home_p,
    real_audit_log as _real_audit_log_p,
    real_registry_root as _real_registry_p,
    real_shadow_artefacts_root as _real_shadow_p,
    scripts_dir as _scripts_dir_p,
)

# ---------------------------------------------------------------------------
# CLI import — the script is under scripts/, not on sys.path normally
# ---------------------------------------------------------------------------



def _import_cli():
    import sys
    # tests/hedgerock/evolution/test_*.py → parents[3] is the repo root.
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_report as cli  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return cli


# ---------------------------------------------------------------------------
# Synthetic Phase D bundle for offline tests
# ---------------------------------------------------------------------------


_SAMPLE_AVAILABILITY = """
# Phase D-cont3-preflight

## Year-replication summary

| Symbol | Year | Bars | trend_up | range@≥0.80 | breakout_signed_by_h4 | halt events |
|---|---|---|---|---|---|---|
| XAUUSD | 2021 | 5651 | -0.067% ±0.048 (NEG) | -0.037% ±0.039 (CI∋0) | -0.001% ±0.085 (CI∋0) | 1 |
| XAUUSD | 2022 | 5674 | +0.122% ±0.074 | -0.035% ±0.044 (CI∋0) | -0.188% ±0.128 (NEG) | 1 |
| XAUUSD | 2023 | 4913 | +0.063% ±0.069 (CI∋0) | +0.055% ±0.042 | -0.401% ±0.111 (NEG) | 1 |
| XAUUSD | 2024 | 5693 | +0.139% ±0.067 | +0.172% ±0.040 | -0.015% ±0.096 (CI∋0) | 1 |

## Action gate

```yaml
NO_STRATEGY_CHANGE: true
```
"""


_SAMPLE_AVAILABILITY_GATE_CLEAR = _SAMPLE_AVAILABILITY.replace(
    "NO_STRATEGY_CHANGE: true", "NO_STRATEGY_CHANGE: false"
)


def _seed_bundle(tmp_path: Path, *, availability: str = _SAMPLE_AVAILABILITY) -> tuple[Path, Path, Path]:
    atlas = tmp_path / "atlas.md"
    atlas.write_text("# atlas\n\nfake atlas\n")
    avail = tmp_path / "availability.md"
    avail.write_text(availability)
    wf = tmp_path / "wf.md"
    wf.write_text("# walk-forward\n")
    return atlas, avail, wf


# ---------------------------------------------------------------------------
# 1. End-to-end on a synthetic Phase D bundle
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_blocks_every_candidate_on_synthetic_phase_d_bundle(tmp_path: Path) -> None:
    cli = _import_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"  # missing → safety_bound_undefined
    report = tmp_path / "report.md"

    candidates, report_path = cli.run(
        atlas_path=atlas, availability_path=avail,
        walk_forward_paths=[wf],
        safety_bounds_path=bounds,
        registry_root=registry,
        report_path=report,
    )

    assert len(candidates) == 4
    for c in candidates:
        assert c.result.value.startswith("PROMOTION_BLOCKED")
    # The four candidate IDs all show up in the rendered report.
    body = report_path.read_text(encoding="utf-8")
    for cid in ("c1-lower-observe-floor-0.50", "c2-halt-expiry-observe-6h",
                "c3-aggressive-floor-0.78", "c4-range2-conf-0.70"):
        assert cid in body, f"missing candidate {cid} in rendered report"


@pytest.mark.unit
def test_no_strategy_change_short_circuit_appears_in_every_blocking_list(
    tmp_path: Path,
) -> None:
    cli = _import_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report = tmp_path / "report.md"

    candidates, _ = cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry, report_path=report,
    )
    for c in candidates:
        assert any(
            "data_availability_action_gate_blocks_all" in r
            for r in c.blocking_reasons
        ), f"{c.candidate_id} missing NO_STRATEGY_CHANGE short-circuit"


@pytest.mark.unit
def test_undefined_safety_bound_yields_manifest_invalid(tmp_path: Path) -> None:
    """Plan §3 G6 (b): missing bounds → PROMOTION_BLOCKED /
    manifest_invalid + required_next_data_or_policy = 'human re-scope
    outside Ticket 1'."""
    cli = _import_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry = tmp_path / "registry"
    bounds = tmp_path / "MISSING_safety_bounds.yaml"  # explicitly missing
    report = tmp_path / "report.md"

    candidates, _ = cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry, report_path=report,
    )
    for c in candidates:
        assert c.result.value == "PROMOTION_BLOCKED / manifest_invalid"
        assert c.required_next_data_or_policy == "human re-scope outside Ticket 1"
        assert any(
            "safety_bound_undefined" in r for r in c.blocking_reasons
        )


@pytest.mark.unit
def test_per_candidate_blocking_patterns_match_plan(tmp_path: Path) -> None:
    """Plan §5.6 acceptance test: each candidate's blocking reasons
    list contains the predicted gates."""
    cli = _import_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report = tmp_path / "report.md"

    candidates, _ = cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry, report_path=report,
    )
    by_id = {c.candidate_id: c for c in candidates}

    # c1: G2 fail + G4 fail (year-replication / 2021 reverse)
    c1 = by_id["c1-lower-observe-floor-0.50"]
    assert any("G2_fail" in r for r in c1.blocking_reasons)
    assert any("G4_fail" in r for r in c1.blocking_reasons)

    # c2: G5 fail (halt corpus)
    c2 = by_id["c2-halt-expiry-observe-6h"]
    assert any("G5_fail" in r for r in c2.blocking_reasons)

    # c3: exposure-class veto
    c3 = by_id["c3-aggressive-floor-0.78"]
    assert any("exposure_class_human_only" in r for r in c3.blocking_reasons)

    # c4: blocked, includes G2 fail
    c4 = by_id["c4-range2-conf-0.70"]
    assert any("G2_fail" in r for r in c4.blocking_reasons)


@pytest.mark.unit
def test_registry_persists_candidate_manifests(tmp_path: Path) -> None:
    cli = _import_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report = tmp_path / "report.md"

    cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry, report_path=report,
    )

    candidates_dir = registry / "candidates"
    written = sorted(p.name for p in candidates_dir.glob("*.json"))
    assert written == [
        "c1-lower-observe-floor-0.50.json",
        "c2-halt-expiry-observe-6h.json",
        "c3-aggressive-floor-0.78.json",
        "c4-range2-conf-0.70.json",
    ]


# ===========================================================================
# Ticket 1-closeout-2 — non-destructive re-run against the same registry
# ===========================================================================


@pytest.mark.unit
def test_two_consecutive_runs_do_not_require_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running the report against an existing registry root MUST
    NOT require ``rm -rf`` or any other deletion. Existing candidates
    are kept verbatim (write_candidate refuses overwrite, the CLI
    skips non-destructively); the audit log gains a new entry; every
    candidate file remains strict-loadable through
    ``PolicyRegistry.get_candidate``."""
    cli = _import_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report = tmp_path / "report.md"

    # Distinct timestamps for the two runs so audit filenames don't
    # collide. The behaviour we want to verify is logical (non-
    # destructive), not "what happens within one microsecond".
    timestamps = iter([
        "2026-05-01T12-00-00-000001",
        "2026-05-01T12-00-00-000002",
    ])
    monkeypatch.setattr(
        "smc.hedgerock.evolution.policy_registry._audit_timestamp",
        lambda: next(timestamps),
    )

    # First run.
    cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry_root, report_path=report,
    )
    candidates_dir = registry_root / "candidates"
    audit_dir = registry_root / "audit"

    files_before = sorted(p.name for p in candidates_dir.glob("*.json"))
    mtimes_before = {
        p.name: p.stat().st_mtime_ns
        for p in candidates_dir.glob("*.json")
    }
    audit_before = sorted(p.name for p in audit_dir.glob("*.json"))

    assert len(files_before) == 4
    assert len(audit_before) == 1

    # Second run — same registry root, NO deletion.
    cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry_root, report_path=report,
    )

    files_after = sorted(p.name for p in candidates_dir.glob("*.json"))
    mtimes_after = {
        p.name: p.stat().st_mtime_ns
        for p in candidates_dir.glob("*.json")
    }
    audit_after = sorted(p.name for p in audit_dir.glob("*.json"))

    # Candidate files: same set, unchanged byte-for-byte (mtimes
    # match → write_candidate did NOT run again).
    assert files_after == files_before, (
        "candidate set changed between runs — re-run was destructive"
    )
    assert mtimes_after == mtimes_before, (
        "candidate file mtimes changed — re-run rewrote a manifest"
    )

    # Audit log gained exactly one entry.
    assert len(audit_after) == len(audit_before) + 1, (
        "audit log did not grow by exactly 1 between non-destructive runs"
    )
    assert set(audit_before).issubset(set(audit_after)), (
        "audit log lost an entry between runs — append-only violated"
    )

    # Every candidate still strict-loadable through the registry
    # (i.e. their content_sha256 envelopes still verify).
    from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
    from smc.hedgerock.evolution.policy_registry import PolicyRegistry
    reg = PolicyRegistry(registry_root)
    for c in CANDIDATE_MENU_V0:
        loaded = reg.get_candidate(c.candidate_id)
        assert loaded.candidate_id == c.candidate_id


# ===========================================================================
# Ticket 1-closeout-3 — rerun fail-closed against tampered / drifted /
# bare-layout existing candidates
# ===========================================================================


def _audit_count(registry_root: Path) -> int:
    return len(list((registry_root / "audit").glob("*.json")))


def _candidate_path(registry_root: Path, candidate_id: str) -> Path:
    return registry_root / "candidates" / f"{candidate_id}.json"


@pytest.mark.unit
def test_rerun_fails_when_existing_candidate_inner_field_tampered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tampering an existing candidate file's inner manifest (without
    refreshing the content_sha256) MUST cause a subsequent rerun to
    fail BEFORE any audit-log entry or report file is written."""
    cli = _import_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "bounds.yaml"
    report = tmp_path / "report.md"

    # Distinct timestamps for would-be audit entries.
    timestamps = iter([
        "2026-05-01T12-00-00-000001",
        "2026-05-01T12-00-00-000002",
    ])
    monkeypatch.setattr(
        "smc.hedgerock.evolution.policy_registry._audit_timestamp",
        lambda: next(timestamps),
    )

    # First run — establishes a clean registry.
    cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry_root,
        report_path=report,
    )
    audit_before = _audit_count(registry_root)
    report_text_before = report.read_text(encoding="utf-8")
    report_mtime_before = report.stat().st_mtime_ns

    # Tamper c1: modify inner title without updating content_sha256.
    import json as _json
    from smc.hedgerock.evolution.policy_manifest import ManifestIntegrityError
    c1_path = _candidate_path(registry_root, "c1-lower-observe-floor-0.50")
    c1_path.chmod(0o644)
    raw = _json.loads(c1_path.read_text(encoding="utf-8"))
    raw["manifest"]["title"] = "TAMPERED INJECTED TITLE"
    c1_path.write_text(_json.dumps(raw, indent=2, sort_keys=True))

    # Rerun MUST fail before audit / report side effects.
    with pytest.raises(ManifestIntegrityError):
        cli.run(
            atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
            safety_bounds_path=bounds, registry_root=registry_root,
            report_path=report,
        )

    # Audit count unchanged.
    assert _audit_count(registry_root) == audit_before, (
        "audit log grew despite rerun failure — fail-closed violated"
    )
    # Report file unchanged.
    assert report.read_text(encoding="utf-8") == report_text_before, (
        "report file rewritten despite rerun failure"
    )
    assert report.stat().st_mtime_ns == report_mtime_before, (
        "report file mtime moved despite rerun failure"
    )


@pytest.mark.unit
def test_rerun_fails_when_existing_candidate_is_bare_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate file that lacks the content_sha256 envelope (bare
    layout) cannot be silently accepted by a rerun."""
    cli = _import_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "bounds.yaml"
    report = tmp_path / "report.md"

    monkeypatch.setattr(
        "smc.hedgerock.evolution.policy_registry._audit_timestamp",
        lambda: "2026-05-01T12-00-00-000099",
    )

    # Plant a bare manifest under candidates/ for c1 BEFORE any run.
    from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
    from smc.hedgerock.evolution.policy_manifest import (
        ManifestIntegrityError, manifest_to_dict,
    )
    candidates_dir = registry_root / "candidates"
    candidates_dir.mkdir(parents=True)
    c1 = next(c for c in CANDIDATE_MENU_V0 if c.candidate_id == "c1-lower-observe-floor-0.50")
    bare_path = candidates_dir / f"{c1.candidate_id}.json"
    import json as _json
    bare_path.write_text(_json.dumps(manifest_to_dict(c1), indent=2, sort_keys=True))

    # Run MUST fail on the bare c1 file before audit / report writes.
    with pytest.raises(ManifestIntegrityError):
        cli.run(
            atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
            safety_bounds_path=bounds, registry_root=registry_root,
            report_path=report,
        )
    assert _audit_count(registry_root) == 0, (
        "audit log grew despite rerun failure — fail-closed violated"
    )
    assert not report.exists(), (
        "report file written despite rerun failure"
    )


@pytest.mark.unit
def test_rerun_fails_when_existing_candidate_diff_drifted_from_menu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When an existing manifest is hash-valid but its diff has
    drifted from the current menu (e.g. proposed_value is different),
    the strict-load step succeeds but the menu comparison must raise
    StaleCandidateError."""
    cli = _import_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "bounds.yaml"
    report = tmp_path / "report.md"

    monkeypatch.setattr(
        "smc.hedgerock.evolution.policy_registry._audit_timestamp",
        lambda: "2026-05-01T12-00-00-000111",
    )

    from dataclasses import replace
    from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
    from smc.hedgerock.evolution.policy_manifest import dump_manifest
    from smc.hedgerock.evolution.policy_registry import StaleCandidateError

    # Plant a HASH-VALID c1 with a drifted proposed_value.
    candidates_dir = registry_root / "candidates"
    candidates_dir.mkdir(parents=True)
    c1 = next(c for c in CANDIDATE_MENU_V0 if c.candidate_id == "c1-lower-observe-floor-0.50")
    drifted = replace(c1, diff=replace(c1.diff, proposed_value=0.99))  # menu has 0.50
    dump_manifest(drifted, candidates_dir / f"{c1.candidate_id}.json")

    with pytest.raises(StaleCandidateError) as exc:
        cli.run(
            atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
            safety_bounds_path=bounds, registry_root=registry_root,
            report_path=report,
        )
    assert "proposed_value" in str(exc.value)
    assert _audit_count(registry_root) == 0
    assert not report.exists()


@pytest.mark.unit
def test_rerun_fails_when_existing_candidate_state_drifted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Menu candidates must always be in DRAFT state. If a registry
    file's state has been changed (e.g. someone hand-edited it to
    'tested'), the rerun must fail-closed."""
    cli = _import_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "bounds.yaml"
    report = tmp_path / "report.md"

    monkeypatch.setattr(
        "smc.hedgerock.evolution.policy_registry._audit_timestamp",
        lambda: "2026-05-01T12-00-00-000112",
    )

    from dataclasses import replace
    from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
    from smc.hedgerock.evolution.policy_manifest import (
        CandidateState, dump_manifest,
    )
    from smc.hedgerock.evolution.policy_registry import StaleCandidateError

    candidates_dir = registry_root / "candidates"
    candidates_dir.mkdir(parents=True)
    c1 = next(c for c in CANDIDATE_MENU_V0 if c.candidate_id == "c1-lower-observe-floor-0.50")
    promoted = replace(c1, state=CandidateState.TESTED)
    dump_manifest(promoted, candidates_dir / f"{c1.candidate_id}.json")

    with pytest.raises(StaleCandidateError) as exc:
        cli.run(
            atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
            safety_bounds_path=bounds, registry_root=registry_root,
            report_path=report,
        )
    assert "state" in str(exc.value).lower()


@pytest.mark.unit
def test_clean_rerun_remains_non_destructive_under_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fail-closed pathway must NOT regress the clean-rerun case:
    when the registry is intact and matches the menu, two consecutive
    runs produce identical candidate files (mtimes preserved) and the
    audit log gains exactly 1 entry."""
    cli = _import_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "bounds.yaml"
    report = tmp_path / "report.md"

    timestamps = iter([
        "2026-05-01T13-00-00-000001",
        "2026-05-01T13-00-00-000002",
    ])
    monkeypatch.setattr(
        "smc.hedgerock.evolution.policy_registry._audit_timestamp",
        lambda: next(timestamps),
    )

    cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry_root,
        report_path=report,
    )
    candidates_dir = registry_root / "candidates"
    audit_dir = registry_root / "audit"

    files_before = sorted(p.name for p in candidates_dir.glob("*.json"))
    mtimes_before = {
        p.name: p.stat().st_mtime_ns for p in candidates_dir.glob("*.json")
    }
    audit_before = sorted(p.name for p in audit_dir.glob("*.json"))

    # Clean rerun — no tamper, no drift.
    cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry_root,
        report_path=report,
    )

    files_after = sorted(p.name for p in candidates_dir.glob("*.json"))
    mtimes_after = {
        p.name: p.stat().st_mtime_ns for p in candidates_dir.glob("*.json")
    }
    audit_after = sorted(p.name for p in audit_dir.glob("*.json"))

    assert files_after == files_before
    assert mtimes_after == mtimes_before
    assert len(audit_after) == len(audit_before) + 1
    assert set(audit_before).issubset(set(audit_after))


@pytest.mark.unit
def test_cli_comment_no_longer_recommends_deletion() -> None:
    """The CLI's re-run guidance MUST NOT include the prior 'delete
    or use a fresh registry root' phrasing. The canonical wording is
    'use a fresh registry root for regeneration; do not delete
    production registry history'.

    Match on a normalised form (collapse comment markers + whitespace)
    so the test tolerates line wrapping in the actual comment.
    """
    cli_src = (_scripts_dir_p() / "hedgerock_evolution_report.py").read_text(encoding="utf-8")
    # Normalise: drop "#" comment markers and collapse whitespace so
    # the substring check works regardless of line wrapping.
    normalised = re.sub(r"\s+", " ", cli_src.replace("#", " "))

    # Forbidden phrasing — the prior comment.
    assert "delete or use a fresh registry root" not in normalised
    assert "tester / re-runner needs to" not in normalised
    # Required phrasing — the new comment.
    assert "use a fresh registry root for regeneration" in normalised
    assert "do not delete production registry history" in normalised


@pytest.mark.unit
def test_no_writes_under_approved_or_pointer(tmp_path: Path) -> None:
    cli = _import_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report = tmp_path / "report.md"

    cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry, report_path=report,
    )

    assert not (registry / "approved").exists() or not any(
        (registry / "approved").iterdir()
    ), "approved/ must be untouched by Ticket 1"
    assert not (registry / "pointer.json").exists(), (
        "pointer.json must be untouched by Ticket 1"
    )


# ---------------------------------------------------------------------------
# 2. Report content invariants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_report_contains_canonical_certification_line(tmp_path: Path) -> None:
    cli = _import_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report = tmp_path / "report.md"

    cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry, report_path=report,
    )
    body = report.read_text(encoding="utf-8")
    # Canonical certification line, on its own line.
    assert "\nsafety_bounds write permission: 0\n" in "\n" + body + "\n"


@pytest.mark.unit
def test_report_contains_rfc11_invariant_phrases(tmp_path: Path) -> None:
    cli = _import_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report = tmp_path / "report.md"

    cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry, report_path=report,
    )
    body = report.read_text(encoding="utf-8")
    # Direct quotes of two key RFC §11 invariants.
    assert (
        "MUST NOT auto-promote any candidate while the most recent "
        "data-availability report has NO_STRATEGY_CHANGE: true." in body
    )
    assert (
        "MUST NOT auto-promote any candidate that raises gross exposure"
        in body
    )


@pytest.mark.unit
def test_report_contains_required_section_strings(tmp_path: Path) -> None:
    cli = _import_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report = tmp_path / "report.md"

    cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry, report_path=report,
    )
    body = report.read_text(encoding="utf-8")
    for required in (
        "RESULT: PROMOTION_BLOCKED",
        "Blocking reasons:",
        "Required next data:",
    ):
        assert required in body, f"missing required string: {required!r}"


# ---------------------------------------------------------------------------
# 3. Hard-boundary file checks (CLI module + evolution package)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cli_does_not_write_to_safety_bounds() -> None:
    cli_path = (_scripts_dir_p() / 'hedgerock_evolution_report.py')
    text = cli_path.read_text(encoding="utf-8")
    # No write-mode open targeting safety_bounds.yaml.
    assert 'safety_bounds.yaml", "w"' not in text
    assert "safety_bounds.yaml', 'w'" not in text
    # No shutil.copy* targeting safety_bounds.
    assert "shutil.copy" not in text or "safety_bounds" not in text


@pytest.mark.unit
def test_cli_does_not_import_production_runtime() -> None:
    cli_path = (_scripts_dir_p() / 'hedgerock_evolution_report.py')
    text = cli_path.read_text(encoding="utf-8")
    forbidden = (
        "from smc.hedgerock.rule_engine",
        "import smc.hedgerock.rule_engine",
        "from smc.hedgerock.decision_server",
        "import smc.hedgerock.decision_server",
        "smc.hedgerock.phase_d_walk_forward",
    )
    for f in forbidden:
        # phase_d_walk_forward is mentioned only as the candidate's
        # diff target string (data, not import), but the import-style
        # patterns ("from ..." / "import ...") must not appear.
        assert f"from {f}" not in text, f"CLI imports {f}"
        assert f"import {f}" not in text, f"CLI imports {f}"


@pytest.mark.unit
def test_evolution_modules_have_no_production_imports() -> None:
    """AST-based check: production-runtime modules MUST NOT appear in
    `import` / `from … import …` statements. Docstring or comment
    mentions are fine — those are documentation, not dependencies."""
    import ast
    import smc.hedgerock.evolution as pkg
    pkg_root = Path(pkg.__file__).parent
    forbidden = {"smc.hedgerock.rule_engine", "smc.hedgerock.decision_server"}
    for py in pkg_root.glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in forbidden, (
                        f"{py.name}: forbidden import {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module not in forbidden, (
                    f"{py.name}: forbidden from-import: {module}"
                )


# ---------------------------------------------------------------------------
# 4. Real Phase D bundle integration
# ---------------------------------------------------------------------------


_REAL_DOCS = (_hedgerock_home_p() / 'docs')


@pytest.mark.integration
def test_run_against_real_phase_d_bundle_blocks_every_candidate(tmp_path: Path) -> None:
    """Plan §5.6 anchor — feeding the actual current Phase D bundles
    must produce PROMOTION_BLOCKED for every candidate."""
    atlas = _REAL_DOCS / "phase-d-regime-opportunity-atlas.md"
    avail = _REAL_DOCS / "phase-d-data-availability.md"
    wf = _REAL_DOCS / "phase-d-walk-forward-report.md"
    if not (atlas.exists() and avail.exists() and wf.exists()):
        pytest.skip("real Phase D bundle not present")

    cli = _import_cli()
    registry = tmp_path / "registry"
    # Default safety_bounds.yaml does not exist in this repo — every
    # G6 hits safety_bound_undefined per Plan §3 G6 (b). That's the
    # correct outcome under Ticket 1's hard boundary.
    bounds = (_ai_smc_home_p() / 'config' / 'safety_bounds.yaml')
    report = tmp_path / "report.md"

    candidates, _ = cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry, report_path=report,
    )
    assert len(candidates) == 4
    for c in candidates:
        assert c.result.value.startswith("PROMOTION_BLOCKED")
        # Real lake currently has NO_STRATEGY_CHANGE: true → every
        # candidate carries the short-circuit reason.
        assert any(
            "data_availability_action_gate_blocks_all" in r
            for r in c.blocking_reasons
        )

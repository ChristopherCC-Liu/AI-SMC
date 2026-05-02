"""Ticket 4 v2 follow-on T4-F3 — operator dry-run validation of the
report CLI's argv path.

T4-F2 pinned the library-level ``run()`` contract for the
registry-audit wiring. T4-F3 closes the operator-facing leg:

  * Drive the CLI through ``main(argv)`` exactly the way the
    operator runbook will. Capture stdout. Verify the printed
    summary names every blocked candidate.
  * Use a tmp ``--registry-root`` and tmp ``--report-path`` so the
    real
    ``/Users/christopher/HedgeRock/policy_registry/shadow_artefacts``
    is never written to.
  * Use the **real** ``_audit.md`` (or a verbatim tmp copy of it)
    as the audit-log source so the dry-run exercises a real-world
    violation signal, not a synthetic fixture.
  * Pin the per-candidate G8 reason: it must contain the string
    ``registry_append_only_violation``, the resolved audit-log
    path, and ``lost_sha_count=4``. The rendered report must show
    the same triplet.
  * Pin the no-audit-log scenario: the CLI must NOT raise, must
    NOT mark any G8 as a registry-violation block, and the report
    must explicitly carry ``audit_log_present=false`` so an
    operator never reads the absence as "checked clean".
  * Pin the default audit-log resolution against argv only —
    ``--registry-audit-log`` omitted → the CLI derives
    ``<registry_root>/shadow_artefacts/_audit.md``.

These tests cement the operator runbook
``docs/hedgerock-evolution-report-runbook.md`` against drift: any
flag rename or default change that breaks the runbook breaks
this suite.
"""

from __future__ import annotations

import io
import re
import shutil
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[3]
_REAL_REGISTRY_ROOT = Path("/Users/christopher/HedgeRock/policy_registry")
_REAL_AUDIT_LOG = _REAL_REGISTRY_ROOT / "shadow_artefacts" / "_audit.md"
_LOST_SHA_COUNT = 4


def _import_cli():
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_report as cli  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return cli


_AVAILABILITY = """
# Phase D-cont3-preflight (T4-F3 dry-run fixture)

## Year-replication summary

| Symbol | Year | Bars | trend_up | range@≥0.80 | breakout_signed_by_h4 | halt events |
|---|---|---|---|---|---|---|
| XAUUSD | 2021 | 5651 | +0.067% ±0.048 | -0.037% ±0.039 (CI∋0) | -0.001% ±0.085 (CI∋0) | 1 |
| XAUUSD | 2022 | 5674 | +0.122% ±0.074 | -0.035% ±0.044 (CI∋0) | -0.188% ±0.128 | 1 |
| XAUUSD | 2023 | 4913 | +0.063% ±0.069 (CI∋0) | +0.055% ±0.042 | -0.401% ±0.111 | 1 |
| XAUUSD | 2024 | 5693 | +0.139% ±0.067 | +0.172% ±0.040 | -0.015% ±0.096 (CI∋0) | 1 |

## Action gate

```yaml
NO_STRATEGY_CHANGE: false
```
"""


def _seed_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    atlas = tmp_path / "atlas.md"
    atlas.write_text("# atlas\n\nfake atlas\n", encoding="utf-8")
    avail = tmp_path / "availability.md"
    avail.write_text(_AVAILABILITY, encoding="utf-8")
    wf = tmp_path / "wf.md"
    wf.write_text("# walk-forward\n", encoding="utf-8")
    return atlas, avail, wf


def _real_registry_json_count() -> int:
    if not _REAL_REGISTRY_ROOT.exists():
        return 0
    return sum(1 for _ in _REAL_REGISTRY_ROOT.rglob("*.json"))


def _argv_for_dry_run(
    *,
    atlas: Path,
    avail: Path,
    wf: Path,
    bounds: Path,
    registry_root: Path,
    report_path: Path,
    audit_log_override: Path | None = None,
) -> list[str]:
    argv = [
        "--atlas-report", str(atlas),
        "--data-availability-report", str(avail),
        "--walk-forward-report", str(wf),
        "--safety-bounds", str(bounds),
        "--registry-root", str(registry_root),
        "--report-path", str(report_path),
    ]
    if audit_log_override is not None:
        argv += ["--registry-audit-log", str(audit_log_override)]
    return argv


# ---------------------------------------------------------------------------
# 1. main(argv) dry-run with REAL audit log → stdout + report both
#    surface registry_append_only_violation per candidate.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_main_argv_dry_run_with_real_audit_log_blocks_every_candidate(
    tmp_path: Path,
) -> None:
    if not _REAL_AUDIT_LOG.exists():
        pytest.skip(
            f"real audit log not present at {_REAL_AUDIT_LOG}; "
            "operator dry-run requires the production audit log to exist"
        )

    cli = _import_cli()
    real_pre = _real_registry_json_count()

    atlas, avail, wf = _seed_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report_path = tmp_path / "phase-d-evolution-report.md"

    argv = _argv_for_dry_run(
        atlas=atlas, avail=avail, wf=wf, bounds=bounds,
        registry_root=registry_root, report_path=report_path,
        audit_log_override=_REAL_AUDIT_LOG,
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = cli.main(argv)

    assert rc == 0, f"main returned {rc}; stderr={stderr.getvalue()!r}"

    body = report_path.read_text(encoding="utf-8")
    out = stdout.getvalue()

    # Operator stdout summary surfaces the headline numbers — 4
    # candidates evaluated, 4 PROMOTION_BLOCKED. Operators read
    # the report body for per-gate detail; the violation triplet
    # lives in the body, not on stdout.
    assert "evaluated 4 candidates" in out
    assert out.count("PROMOTION_BLOCKED") >= 4

    # Registry-audit summary block — surfaces the True flag plus
    # the warning the renderer attaches.
    assert "`registry_append_only_violation`: **True**" in body, (
        "report header should show the violation flag = True"
    )
    assert "Registry append-only contract violated this session" in body
    assert f"`lost_sha_count`: **{_LOST_SHA_COUNT}**" in body
    assert str(_REAL_AUDIT_LOG) in body, (
        "real audit log path must surface in the rendered report"
    )

    # Per-candidate G8 line: ABSTAIN + violation marker + lost
    # count + audit log path all on one rendered line.
    g8_lines = [
        ln for ln in body.splitlines()
        if ln.strip().startswith("- G8:")
    ]
    assert len(g8_lines) == 4, (
        f"expected 4 G8 lines (one per candidate), got {len(g8_lines)}: "
        f"{g8_lines!r}"
    )
    for ln in g8_lines:
        assert "**ABSTAIN**" in ln, f"G8 not ABSTAIN: {ln!r}"
        assert "registry_append_only_violation" in ln, (
            f"G8 reason missing violation marker: {ln!r}"
        )
        assert f"lost_sha_count={_LOST_SHA_COUNT}" in ln, (
            f"G8 reason missing lost_sha_count={_LOST_SHA_COUNT}: {ln!r}"
        )
        assert str(_REAL_AUDIT_LOG) in ln, (
            f"G8 reason missing audit log path: {ln!r}"
        )

    # CLI wrote candidate manifests under the tmp registry only.
    candidate_files = sorted(
        (registry_root / "candidates").glob("*.json")
    )
    assert candidate_files, "CLI did not write any candidate manifests"

    # Real production registry untouched.
    assert _real_registry_json_count() == real_pre, (
        "main(argv) dry-run wrote into the real registry"
    )


# ---------------------------------------------------------------------------
# 2. main(argv) dry-run with a tmp COPY of the real audit log —
#    operator-runbook variant for sandboxed dry-runs that don't even
#    need read access to the production registry.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_main_argv_dry_run_with_tmp_audit_log_copy_blocks_every_candidate(
    tmp_path: Path,
) -> None:
    if not _REAL_AUDIT_LOG.exists():
        pytest.skip(
            f"real audit log not present at {_REAL_AUDIT_LOG}; "
            "tmp-copy dry-run uses the real log as ground truth"
        )

    cli = _import_cli()
    real_pre = _real_registry_json_count()

    atlas, avail, wf = _seed_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report_path = tmp_path / "phase-d-evolution-report.md"

    audit_copy = tmp_path / "audit-copy" / "_audit.md"
    audit_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_REAL_AUDIT_LOG, audit_copy)

    argv = _argv_for_dry_run(
        atlas=atlas, avail=avail, wf=wf, bounds=bounds,
        registry_root=registry_root, report_path=report_path,
        audit_log_override=audit_copy,
    )
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        rc = cli.main(argv)

    assert rc == 0
    body = report_path.read_text(encoding="utf-8")

    # Tmp copy path (NOT real path) must surface in the report.
    assert str(audit_copy) in body
    assert str(_REAL_AUDIT_LOG) not in body, (
        "report leaked the real audit-log path even though the operator "
        "passed --registry-audit-log to a tmp copy"
    )
    assert "`registry_append_only_violation`: **True**" in body
    assert f"`lost_sha_count`: **{_LOST_SHA_COUNT}**" in body

    g8_lines = [
        ln for ln in body.splitlines()
        if ln.strip().startswith("- G8:")
    ]
    assert len(g8_lines) == 4
    for ln in g8_lines:
        assert "**ABSTAIN**" in ln
        assert "registry_append_only_violation" in ln
        assert f"lost_sha_count={_LOST_SHA_COUNT}" in ln
        assert str(audit_copy) in ln, (
            f"G8 reason should reference tmp audit copy, got {ln!r}"
        )

    # Stdout summary names every menu candidate id.
    out = stdout.getvalue()
    assert "evaluated" in out
    for cand_id in (
        "c1-lower-observe-floor-0.50",
        "c2-halt-expiry-observe-6h",
        "c3-aggressive-floor-0.78",
        "c4-range2-conf-0.70",
    ):
        assert cand_id in out, (
            f"candidate {cand_id} missing from stdout summary"
        )

    assert _real_registry_json_count() == real_pre


# ---------------------------------------------------------------------------
# 3. No audit log → no false alarm, but report shows audit_log_present
#    = false. Confirms the operator runbook's "log absence ≠ log clean"
#    invariant survives the argv path.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_main_argv_dry_run_no_audit_log_does_not_block_but_surfaces_absence(
    tmp_path: Path,
) -> None:
    cli = _import_cli()
    real_pre = _real_registry_json_count()

    atlas, avail, wf = _seed_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report_path = tmp_path / "phase-d-evolution-report.md"

    # Override audit log to a path that explicitly does NOT exist.
    missing_audit = tmp_path / "does-not-exist" / "_audit.md"
    assert not missing_audit.exists()

    argv = _argv_for_dry_run(
        atlas=atlas, avail=avail, wf=wf, bounds=bounds,
        registry_root=registry_root, report_path=report_path,
        audit_log_override=missing_audit,
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = cli.main(argv)

    assert rc == 0, (
        f"main(argv) raised on missing audit log; "
        f"stderr={stderr.getvalue()!r}"
    )
    body = report_path.read_text(encoding="utf-8")

    # The `registry_append_only_violation` flag is always rendered;
    # absence of the log means the flag MUST read False (not True).
    assert "`registry_append_only_violation`: **False**" in body
    # Warning block fires only on True — must NOT appear.
    assert (
        "Registry append-only contract violated this session"
        not in body
    )
    # G8 lines must NOT be ABSTAIN with violation marker.
    g8_lines = [
        ln for ln in body.splitlines()
        if ln.strip().startswith("- G8:")
    ]
    assert g8_lines, "report missing G8 lines"
    for ln in g8_lines:
        assert "registry_append_only_violation" not in ln, (
            f"G8 spuriously flagged violation when audit log absent: {ln!r}"
        )

    # ...but the absence MUST be surfaced explicitly.
    assert "`audit_log_present`: **False**" in body
    assert str(missing_audit) in body, (
        "the absent audit-log path must still surface so the operator "
        "can decide whether to re-point the flag"
    )

    # Real production registry untouched.
    assert _real_registry_json_count() == real_pre


# ---------------------------------------------------------------------------
# 4. argv default — no --registry-audit-log → CLI derives
#    <registry_root>/shadow_artefacts/_audit.md and surfaces it.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_main_argv_default_audit_log_resolves_to_registry_relative_path(
    tmp_path: Path,
) -> None:
    cli = _import_cli()
    real_pre = _real_registry_json_count()

    atlas, avail, wf = _seed_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report_path = tmp_path / "phase-d-evolution-report.md"

    argv = _argv_for_dry_run(
        atlas=atlas, avail=avail, wf=wf, bounds=bounds,
        registry_root=registry_root, report_path=report_path,
        # Intentionally omit --registry-audit-log.
    )
    with redirect_stdout(io.StringIO()):
        rc = cli.main(argv)

    assert rc == 0
    body = report_path.read_text(encoding="utf-8")

    expected_default = registry_root / "shadow_artefacts" / "_audit.md"
    assert str(expected_default) in body, (
        "default audit-log path was not surfaced in the report; "
        "operator runbook 'no-flag = registry-relative' contract broken"
    )

    # The real production audit-log path must NOT leak when the
    # operator never asked for it.
    assert str(_REAL_AUDIT_LOG) not in body

    # Real registry untouched.
    assert _real_registry_json_count() == real_pre


# ---------------------------------------------------------------------------
# 5. Real-registry isolation sentinel — every test path used here is
#    under tmp_path or the read-only real audit log.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_t4f3_paths_stay_inside_tmp_or_readonly_audit_log(
    tmp_path: Path,
) -> None:
    """All write targets named by the T4-F3 argv tests must be either
    under ``tmp_path`` or the canonical read-only real audit log.
    Catches a future copy-paste that points an output at a real
    registry directory."""
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report_path = tmp_path / "phase-d-evolution-report.md"

    write_targets = (
        atlas, avail, wf, bounds, registry_root, report_path,
    )
    for p in write_targets:
        # Every write target must descend from tmp_path. The real
        # registry is allowed only as a READ source via
        # ``--registry-audit-log``.
        assert tmp_path in p.parents or p == tmp_path or p.parent == tmp_path
        assert _REAL_REGISTRY_ROOT not in p.parents
        assert not str(p).startswith(str(_REAL_REGISTRY_ROOT))


# ---------------------------------------------------------------------------
# 6. CLI exit-code contract — bad argv → non-zero, but never partial
#    write into a real path.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_main_argv_missing_atlas_returns_nonzero_without_touching_real_registry(
    tmp_path: Path,
) -> None:
    cli = _import_cli()
    real_pre = _real_registry_json_count()

    registry_root = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report_path = tmp_path / "phase-d-evolution-report.md"

    # Atlas path that doesn't exist — load_evidence_bundle will raise.
    missing_atlas = tmp_path / "missing-atlas.md"
    avail = tmp_path / "availability.md"
    avail.write_text(_AVAILABILITY, encoding="utf-8")
    wf = tmp_path / "wf.md"
    wf.write_text("# wf\n", encoding="utf-8")

    argv = _argv_for_dry_run(
        atlas=missing_atlas, avail=avail, wf=wf, bounds=bounds,
        registry_root=registry_root, report_path=report_path,
    )
    stderr = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
        rc = cli.main(argv)

    assert rc != 0
    err = stderr.getvalue()
    assert err  # non-empty failure message

    # Real registry still untouched.
    assert _real_registry_json_count() == real_pre
    # Tmp report should not have been written either.
    assert not report_path.exists()

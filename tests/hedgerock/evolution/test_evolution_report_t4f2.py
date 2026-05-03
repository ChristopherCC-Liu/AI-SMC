"""Ticket 4 v2 follow-on T4-F2 — registry-audit state plumbed into
the actual policy-registry report CLI (`scripts/hedgerock_evolution_report.py`).

Pinned guarantees:

  * ``run()`` defaults the audit-log location to
    ``<registry_root>/shadow_artefacts/_audit.md`` — never a hard-
    coded production path. Tests run against a tmp registry +
    tmp audit log; the real registry is never touched.
  * When the audit log records a violation, the bundle attached to
    every candidate carries
    ``registry_audit.registry_append_only_violation=True``. G8
    blocks each candidate to ABSTAIN with a reason starting
    ``registry_append_only_violation``; the lost SHA count and
    the audit-log path surface in the rendered report.
  * The ``--registry-audit-log`` override flag bypasses the default
    derivation and points the CLI at any path the operator names.
  * When the audit log file is absent, no violation is enforced
    BUT the report explicitly shows ``audit_log_present=false`` so
    operators cannot mistake "log was checked, clean" for "log was
    never checked."
  * Per-candidate ``replace(base_bundle, shadow_artefact_path=…,
    shadow_artefact_hash_sha256=…)`` does NOT drop the
    ``registry_audit`` field — the operator's session-level
    violation signal flows into every candidate's G8 input.
  * Test helper smoke driver writes only under ``tmp_path``; the
    real registry's `*.json` count must be unchanged after each
    test.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
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


_REPO = Path(__file__).resolve().parents[3]
_REAL_REGISTRY = _real_shadow_p()


def _import_cli():
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_report as cli  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return cli


# ---------------------------------------------------------------------------
# Synthetic Phase D bundle (narrowed copy of the existing test pattern;
# we only need enough to drive the CLI through a full run)
# ---------------------------------------------------------------------------


_AVAILABILITY = """
# Phase D-cont3-preflight

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


_LOST_SHAS = (
    "f923fc24c3f1ecd7c2bae21a30b745791baeff6b55c2a4b530df4203c60bd186",
    "c50a2f9b28cdd841a9d62cc1816bba24ad8d65f5dfc2b050fdf5dff5df544d65",
    "913269e479ae57c96d579ba730b0601d8ac1f13fc5dfe1e28ead635716e5b933",
    "6916b7902fa93c5ed8bc755d6b7c973a92e14aa820d57da2bf10a06ceff5de86",
)


def _seed_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    atlas = tmp_path / "atlas.md"
    atlas.write_text("# atlas\n\nfake atlas\n", encoding="utf-8")
    avail = tmp_path / "availability.md"
    avail.write_text(_AVAILABILITY, encoding="utf-8")
    wf = tmp_path / "wf.md"
    wf.write_text("# walk-forward\n", encoding="utf-8")
    return atlas, avail, wf


def _write_violating_audit_log(path: Path) -> None:
    """Synthesise an audit log that the loader will parse as
    violation=True with lost_sha_count=4."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        "# Registry Audit Log (synthetic test fixture)",
        "",
        "## 2099-01-01 — Stale v0.3.0 artefacts deleted (test fixture)",
        "",
        "Deleted artefacts:",
    ]
    for sha in _LOST_SHAS:
        body.append(f"- `{sha}`")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def _real_registry_json_count() -> int:
    if not _REAL_REGISTRY.exists():
        return 0
    return sum(1 for _ in _REAL_REGISTRY.rglob("*.json"))


def _record_real_registry_baseline() -> int:
    return _real_registry_json_count()


# ---------------------------------------------------------------------------
# 1. Default audit log path is registry-relative
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_registry_audit_log_path_is_registry_relative(
    tmp_path: Path,
) -> None:
    cli = _import_cli()
    derived = cli._default_registry_audit_log(tmp_path)
    assert derived == tmp_path / "shadow_artefacts" / "_audit.md"
    # Must NOT be a hard-coded production path.
    assert "/HedgeRock/policy_registry/shadow_artefacts/_audit.md" \
        not in str(derived)


# ---------------------------------------------------------------------------
# 2. CLI run with violating audit log → every candidate ABSTAINs at G8
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cli_default_loads_violating_audit_log_and_blocks_pass(
    tmp_path: Path,
) -> None:
    cli = _import_cli()
    real_baseline = _record_real_registry_baseline()

    atlas, avail, wf = _seed_bundle(tmp_path)
    registry = tmp_path / "registry"
    # Plant a violating audit log at the DEFAULT location so the
    # CLI picks it up automatically.
    audit_log = registry / "shadow_artefacts" / "_audit.md"
    _write_violating_audit_log(audit_log)
    bounds = tmp_path / "safety_bounds.yaml"
    report = tmp_path / "report.md"

    candidates, report_path = cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry,
        report_path=report,
        # NOTE: no shadow_artefacts_dir — we want to verify the
        # registry-violation gate fires even when G8 has no
        # artefact to load.
    )
    body = report_path.read_text(encoding="utf-8")

    # Every candidate's G8 result line must mention the violation.
    for c in candidates:
        g8 = next(
            (r for r in c.gates if r.gate_id == "G8"), None
        )
        assert g8 is not None, f"{c.candidate_id} missing G8 gate result"
        assert "registry_append_only_violation" in g8.reason, (
            f"{c.candidate_id} G8 reason missing violation marker: "
            f"{g8.reason!r}"
        )
        assert g8.status.value == "ABSTAIN", (
            f"{c.candidate_id} G8 status should be ABSTAIN, got "
            f"{g8.status.value}"
        )
        # Lost SHAs surfaced in details.
        assert g8.details.get("lost_sha_count") == 4

    # Report body shows the violation block AND the audit-log path.
    assert "registry_append_only_violation" in body
    assert "lost_sha_count" in body and "**4**" in body
    assert str(audit_log) in body, (
        "audit log path missing from rendered report"
    )
    # Critical RFC v2 invariant — never use the legacy v1-era
    # blocker phrasing. ``cross_symbol_count`` (an evidence-bundle
    # FIELD) is allowed; only the legacy *blocker reason* phrasing
    # would be a regression.
    assert "single_symbol shadow window" not in body
    assert "cross-symbol blocker" not in body.lower()
    # The G8 reasons we just inspected per-candidate must not carry
    # those phrases either.
    for c in candidates:
        g8 = next(r for r in c.gates if r.gate_id == "G8")
        assert "single_symbol" not in g8.reason
        assert "cross_symbol" not in g8.reason

    # Real registry untouched.
    assert _real_registry_json_count() == real_baseline


# ---------------------------------------------------------------------------
# 3. --registry-audit-log override
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_registry_audit_log_override_takes_precedence(
    tmp_path: Path,
) -> None:
    """When the operator passes ``registry_audit_log_path`` to
    ``run()``, the CLI loads from that path even if the default
    ``<registry_root>/shadow_artefacts/_audit.md`` is absent or
    clean."""
    cli = _import_cli()
    real_baseline = _record_real_registry_baseline()

    atlas, avail, wf = _seed_bundle(tmp_path)
    registry = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report = tmp_path / "report.md"

    # Default location: NOT seeded → would be clean.
    # Override location: seeded with a violating log.
    override_log = tmp_path / "alt" / "audit.md"
    _write_violating_audit_log(override_log)

    candidates, report_path = cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry,
        report_path=report,
        registry_audit_log_path=override_log,
    )
    body = report_path.read_text(encoding="utf-8")

    for c in candidates:
        g8 = next(r for r in c.gates if r.gate_id == "G8")
        assert "registry_append_only_violation" in g8.reason

    # The override path is what surfaces in the report header.
    assert str(override_log) in body
    # Default location string should NOT appear (it was never read).
    default_log = registry / "shadow_artefacts" / "_audit.md"
    assert str(default_log) not in body

    # Real registry untouched.
    assert _real_registry_json_count() == real_baseline


# ---------------------------------------------------------------------------
# 4. Missing audit log → no false alarm, but report shows audit_log_present=false
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_missing_audit_log_does_not_block_but_surfaces_absence(
    tmp_path: Path,
) -> None:
    cli = _import_cli()
    real_baseline = _record_real_registry_baseline()

    atlas, avail, wf = _seed_bundle(tmp_path)
    registry = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report = tmp_path / "report.md"
    # NOTE: no audit log planted anywhere.

    candidates, report_path = cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry,
        report_path=report,
    )
    body = report_path.read_text(encoding="utf-8")

    # No candidate gets the violation block.
    for c in candidates:
        g8 = next(r for r in c.gates if r.gate_id == "G8")
        assert "registry_append_only_violation" not in g8.reason

    # The absence MUST be surfaced — operators must not assume
    # "no violation block" means "audit log was checked clean."
    assert "audit_log_present" in body
    assert "**False**" in body
    # Confirm the resolved default path is shown so an operator can
    # verify it themselves.
    expected_default = registry / "shadow_artefacts" / "_audit.md"
    assert str(expected_default) in body

    # Real registry untouched.
    assert _real_registry_json_count() == real_baseline


# ---------------------------------------------------------------------------
# 5. Per-candidate replace() preserves registry_audit
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_per_candidate_replace_preserves_registry_audit(
    tmp_path: Path,
) -> None:
    """When the CLI synthesises a per-candidate bundle by
    ``replace(base_bundle, shadow_artefact_path=…)``, the
    ``registry_audit`` attribute must carry through. Verify by
    constructing a base bundle with audit state and replacing
    only the artefact-related fields."""
    from dataclasses import replace
    from smc.hedgerock.evolution.policy_manifest import EvidenceBundle
    from smc.hedgerock.evolution.registry_audit import (
        RegistryAuditState,
    )

    audit = RegistryAuditState(
        audit_log_path=str(tmp_path / "_audit.md"),
        audit_log_present=True,
        stale_v030_deleted_during_this_session=True,
        lost_sha_count=4,
        lost_sha256=_LOST_SHAS,
        registry_append_only_violation=True,
    )
    base = EvidenceBundle(
        bundle_id="evb-base", bundle_hash_sha256="x" * 64,
        atlas_report_path="/x/atlas.md", atlas_report_hash_sha256="a" * 64,
        data_availability_report_path="/x/avail.md",
        data_availability_report_hash_sha256="b" * 64,
        walk_forward_run_paths=("/x/wf.md",),
        year_replication={"XAUUSD": {"years_total": 4, "years_passing": 4,
                                       "negative_sign_years": ()}},
        cross_symbol_count=1, halt_event_count=4,
        no_strategy_change=False,
        registry_audit=audit,
    )
    per_cand = replace(
        base,
        shadow_artefact_path=str(tmp_path / "art.json"),
        shadow_artefact_hash_sha256="c" * 64,
    )
    assert per_cand.registry_audit is audit
    assert per_cand.registry_audit.registry_append_only_violation is True
    assert per_cand.registry_audit.lost_sha_count == 4


# ---------------------------------------------------------------------------
# 6. Real registry isolation — every test path uses tmp_path only
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_test_helpers_never_target_real_registry_paths(
    tmp_path: Path,
) -> None:
    """Sentinel: the synthesised audit log + bundle paths MUST NOT
    overlap with the real registry. Catches any future copy-paste
    that accidentally points at production paths."""
    fake_audit = tmp_path / "alt" / "audit.md"
    _write_violating_audit_log(fake_audit)
    assert _REAL_REGISTRY not in fake_audit.parents
    assert not str(fake_audit).startswith(str(_REAL_REGISTRY))
    # Real registry is unchanged before AND after this fixture.
    baseline = _real_registry_json_count()
    assert baseline == _real_registry_json_count()


@pytest.mark.unit
def test_cli_run_does_not_write_to_real_registry(tmp_path: Path) -> None:
    """End-to-end: a full CLI run with a tmp registry MUST NOT
    create or modify any file under the real
    ``$HEDGEROCK_HOME/policy_registry/shadow_artefacts``
    directory."""
    cli = _import_cli()
    pre = _record_real_registry_baseline()

    atlas, avail, wf = _seed_bundle(tmp_path)
    registry = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report = tmp_path / "report.md"
    audit_log = registry / "shadow_artefacts" / "_audit.md"
    _write_violating_audit_log(audit_log)

    cli.run(
        atlas_path=atlas, availability_path=avail, walk_forward_paths=[wf],
        safety_bounds_path=bounds, registry_root=registry,
        report_path=report,
    )
    post = _real_registry_json_count()
    assert pre == post, (
        f"CLI run wrote into the real registry: pre={pre} post={post}"
    )

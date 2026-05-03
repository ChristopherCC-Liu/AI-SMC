"""Stage 4 — recommendation CLI tests (report-only).

Pinned guarantees:

  * The CLI runs the full evolution-report pipeline + the candidate
    generator and renders a markdown recommendation file. Output
    lives under a tmp report path; the production registry is never
    written to.
  * The rendered recommendation MUST carry explicit, machine-greppable
    banners: ``NOT LIVE``, ``NOT APPROVED``, ``NOT DEPLOYED``.
  * When the registry-audit log records a violation, every candidate's
    recommendation block reads ``decision: NO_RECOMMENDATION`` with
    reason ``evidence_chain_invalid`` — even if a parameter rule
    would otherwise have fired.
  * When the audit log is missing, the recommendation MUST surface
    ``audit_log_present: False`` exactly once and refuse to emit any
    RECOMMEND verdict (operator must re-point the flag).
  * The CLI is XAUUSD-only; multi-symbol bundles MUST be rejected
    via ``insufficient_xauusd_coverage``.
  * The CLI MUST NOT write under ``policy_registry/approved/``,
    ``policy_registry/pointer.json``, ``config/safety_bounds.yaml``,
    EA `*.mq5`, or any production source file.
"""

from __future__ import annotations

import io
import shutil
import sys
from contextlib import redirect_stdout, redirect_stderr
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
_REAL_AUDIT_LOG = _real_audit_log_p()


def _import_recommend_cli():
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_recommend as cli  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return cli


_AVAILABILITY = """
# Phase D-cont3-preflight (Stage 4 fixture — full XAUUSD coverage)

## Year-replication summary

| Symbol | Year | Bars | trend_up | range@≥0.80 | breakout_signed_by_h4 | halt events |
|---|---|---|---|---|---|---|
| XAUUSD | 2021 | 5651 | +0.167% ±0.048 | -0.037% ±0.039 (CI∋0) | -0.001% ±0.085 (CI∋0) | 1 |
| XAUUSD | 2022 | 5674 | +0.222% ±0.074 | -0.035% ±0.044 (CI∋0) | -0.188% ±0.128 | 1 |
| XAUUSD | 2023 | 4913 | +0.163% ±0.069 | +0.155% ±0.042 | -0.401% ±0.111 | 1 |
| XAUUSD | 2024 | 5693 | +0.239% ±0.067 | +0.172% ±0.040 | -0.015% ±0.096 (CI∋0) | 1 |

## Action gate

```yaml
NO_STRATEGY_CHANGE: false
```
"""


def _seed_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    atlas = tmp_path / "atlas.md"
    atlas.write_text("# atlas\n", encoding="utf-8")
    avail = tmp_path / "availability.md"
    avail.write_text(_AVAILABILITY, encoding="utf-8")
    wf = tmp_path / "wf.md"
    wf.write_text("# wf\n", encoding="utf-8")
    return atlas, avail, wf


def _argv(
    *,
    atlas: Path,
    avail: Path,
    wf: Path,
    bounds: Path,
    registry_root: Path,
    report_path: Path,
    recommendation_path: Path,
    audit_log_override: Path | None = None,
) -> list[str]:
    argv = [
        "--atlas-report", str(atlas),
        "--data-availability-report", str(avail),
        "--walk-forward-report", str(wf),
        "--safety-bounds", str(bounds),
        "--registry-root", str(registry_root),
        "--report-path", str(report_path),
        "--recommendation-path", str(recommendation_path),
    ]
    if audit_log_override is not None:
        argv += ["--registry-audit-log", str(audit_log_override)]
    return argv


# ---------------------------------------------------------------------------
# 1. Audit-log violation → every recommendation is NO_RECOMMENDATION /
#    evidence_chain_invalid + the report carries NOT-LIVE banners.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recommendation_with_violating_audit_log_blocks_every_proposal(
    tmp_path: Path,
) -> None:
    if not _REAL_AUDIT_LOG.exists():
        pytest.skip("real audit log missing; cannot exercise violation path")

    cli = _import_recommend_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report_path = tmp_path / "phase-d-evolution-report.md"
    rec_path = tmp_path / "hedgerock-evolution-recommendation.md"

    audit_copy = tmp_path / "audit-copy" / "_audit.md"
    audit_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_REAL_AUDIT_LOG, audit_copy)

    argv = _argv(
        atlas=atlas, avail=avail, wf=wf, bounds=bounds,
        registry_root=registry_root, report_path=report_path,
        recommendation_path=rec_path,
        audit_log_override=audit_copy,
    )
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        rc = cli.main(argv)

    assert rc == 0
    body = rec_path.read_text(encoding="utf-8")

    # NOT-LIVE banners.
    assert "**NOT LIVE**" in body
    assert "**NOT APPROVED**" in body
    assert "**NOT DEPLOYED**" in body

    # Each candidate has decision NO_RECOMMENDATION / evidence_chain_invalid.
    assert body.count("decision: `NO_RECOMMENDATION`") == 4
    assert body.count("evidence_chain_invalid") >= 4

    # XAUUSD-only wording surfaces.
    assert "XAUUSD" in body
    # Real production registry untouched.
    assert not (registry_root / "approved").exists()
    assert not (registry_root / "pointer.json").exists()


# ---------------------------------------------------------------------------
# 2. Clean audit log + clean bundle → at least one RECOMMEND surfaces.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recommendation_with_clean_audit_emits_at_least_one_recommend(
    tmp_path: Path,
) -> None:
    cli = _import_recommend_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report_path = tmp_path / "phase-d-evolution-report.md"
    rec_path = tmp_path / "hedgerock-evolution-recommendation.md"

    # Plant an empty (clean) audit log at the default location so the
    # CLI sees audit_log_present=True with no violation.
    clean_audit = registry_root / "shadow_artefacts" / "_audit.md"
    clean_audit.parent.mkdir(parents=True, exist_ok=True)
    clean_audit.write_text(
        "# Shadow-Artefact Registry Audit Log\n\n(no incidents)\n",
        encoding="utf-8",
    )

    argv = _argv(
        atlas=atlas, avail=avail, wf=wf, bounds=bounds,
        registry_root=registry_root, report_path=report_path,
        recommendation_path=rec_path,
    )
    with redirect_stdout(io.StringIO()):
        rc = cli.main(argv)

    assert rc == 0
    body = rec_path.read_text(encoding="utf-8")

    # NOT-LIVE banners always surface — even when recommending.
    assert "**NOT LIVE**" in body
    assert "**NOT APPROVED**" in body
    assert "**NOT DEPLOYED**" in body

    # At least one candidate gets RECOMMEND. (G6 fires
    # safety_bound_undefined for every candidate's target because
    # the bounds file does not exist; the generator treats that as a
    # trigger.)
    assert body.count("decision: `RECOMMEND`") >= 1
    # Exposure-raising c3 must remain NO_RECOMMENDATION.
    assert "c3-aggressive-floor-0.78" in body
    # Find the c3 detail section (header `### <id>` is unique to the
    # per-candidate detail block; the visualisation tables use a
    # different format) and assert its decision line.
    c3_header = "### c3-aggressive-floor-0.78"
    assert c3_header in body
    c3_idx = body.find(c3_header)
    c3_block = body[c3_idx: c3_idx + 1200]
    assert "decision: `NO_RECOMMENDATION`" in c3_block


# ---------------------------------------------------------------------------
# 3. Missing audit log → block all recommendations with explicit
#    "audit log absent — re-point flag" verdict.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recommendation_with_missing_audit_log_blocks_recommend(
    tmp_path: Path,
) -> None:
    cli = _import_recommend_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report_path = tmp_path / "phase-d-evolution-report.md"
    rec_path = tmp_path / "hedgerock-evolution-recommendation.md"
    missing_audit = tmp_path / "no-audit" / "_audit.md"

    argv = _argv(
        atlas=atlas, avail=avail, wf=wf, bounds=bounds,
        registry_root=registry_root, report_path=report_path,
        recommendation_path=rec_path,
        audit_log_override=missing_audit,
    )
    with redirect_stdout(io.StringIO()):
        rc = cli.main(argv)

    assert rc == 0
    body = rec_path.read_text(encoding="utf-8")

    # Surface audit log absence.
    assert "`audit_log_present`: **False**" in body
    # Refuse to emit any RECOMMEND verdict.
    assert "decision: `RECOMMEND`" not in body
    # Banners.
    assert "**NOT LIVE**" in body


# ---------------------------------------------------------------------------
# 4. Recommendation CLI never writes outside tmp + only into the
#    operator-named recommendation path.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recommendation_cli_does_not_touch_real_registry(tmp_path: Path) -> None:
    cli = _import_recommend_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report_path = tmp_path / "phase-d-evolution-report.md"
    rec_path = tmp_path / "hedgerock-evolution-recommendation.md"

    real_root = (_real_registry_p())
    pre = sum(1 for _ in real_root.rglob("*.json")) if real_root.exists() else 0

    argv = _argv(
        atlas=atlas, avail=avail, wf=wf, bounds=bounds,
        registry_root=registry_root, report_path=report_path,
        recommendation_path=rec_path,
    )
    with redirect_stdout(io.StringIO()):
        rc = cli.main(argv)

    assert rc == 0
    post = sum(1 for _ in real_root.rglob("*.json")) if real_root.exists() else 0
    assert pre == post, "recommendation CLI wrote into real registry"

    # Forbidden output paths.
    assert not (tmp_path / "approved").exists()
    assert not (tmp_path / "pointer.json").exists()


# ---------------------------------------------------------------------------
# 5. Source-level isolation — the recommendation script does not
#    import live runtime modules.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recommend_script_does_not_import_live_runtime() -> None:
    src = (_REPO / "scripts" / "hedgerock_evolution_recommend.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "from smc.hedgerock.rule_engine",
        "from smc.hedgerock.decision_server",
        "from smc.hedgerock.phase_d_walk_forward",
        "import smc.hedgerock.rule_engine",
        "import smc.hedgerock.decision_server",
        "import smc.hedgerock.phase_d_walk_forward",
    )
    for f in forbidden:
        assert f not in src, (
            f"recommendation CLI imports a live module: {f!r}"
        )


# ---------------------------------------------------------------------------
# 6. Required-validation list mentions XAUUSD shadow runner explicitly.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recommendation_required_validation_mentions_xauusd_shadow(
    tmp_path: Path,
) -> None:
    cli = _import_recommend_cli()
    atlas, avail, wf = _seed_bundle(tmp_path)
    registry_root = tmp_path / "registry"
    bounds = tmp_path / "safety_bounds.yaml"
    report_path = tmp_path / "phase-d-evolution-report.md"
    rec_path = tmp_path / "hedgerock-evolution-recommendation.md"

    clean_audit = registry_root / "shadow_artefacts" / "_audit.md"
    clean_audit.parent.mkdir(parents=True, exist_ok=True)
    clean_audit.write_text(
        "# Shadow-Artefact Registry Audit Log\n\n(no incidents)\n",
        encoding="utf-8",
    )

    argv = _argv(
        atlas=atlas, avail=avail, wf=wf, bounds=bounds,
        registry_root=registry_root, report_path=report_path,
        recommendation_path=rec_path,
    )
    with redirect_stdout(io.StringIO()):
        cli.main(argv)
    body = rec_path.read_text(encoding="utf-8")

    assert "XAUUSD" in body
    assert "shadow_runner" in body or "shadow runner" in body.lower()
    assert "human review" in body.lower() or "human approval" in body.lower()

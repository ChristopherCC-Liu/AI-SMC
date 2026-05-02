"""Stage 6-followup-3 task 4 — real-registry end-to-end smoke test.

Drives `replay_validator` + `candidate_generator` + recommendation
CLI against the **real** shadow-artefact registry under
``/Users/christopher/HedgeRock/policy_registry``. Skips when the
registry is absent (CI) so this remains portable.

Invariants asserted:

  * Real registry's shadow_artefacts directory holds 12 JSON files
    (3 per candidate × 4 menu candidates).
  * Replay validator walks all 4 candidate dirs, reading artefacts
    without mutating them; per-candidate
    ``ReplayValidationReport.n_artefacts_read >= 1``.
  * Real audit log records the 4 lost-SHA violation; G8 ABSTAINs
    every candidate when the report CLI is pointed at the live
    audit log via a tmp registry root.
  * `safety_bounds_template.yaml` parses cleanly; G6 PASSes every
    candidate when fed the template.
  * Recommendation report carries the NOT-LIVE banners and lists
    every menu candidate.
  * Mtime sweep: every JSON under
    ``policy_registry/shadow_artefacts`` is byte-identical
    pre/post (catches stray writes).
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
from smc.hedgerock.evolution.replay_validator import summarise_replay


_REPO = Path(__file__).resolve().parents[3]
_REAL_REGISTRY = Path("/Users/christopher/HedgeRock/policy_registry")
_REAL_SHADOW = _REAL_REGISTRY / "shadow_artefacts"
_REAL_AUDIT_LOG = _REAL_SHADOW / "_audit.md"
_TEMPLATE = _REPO / "config" / "safety_bounds_template.yaml"


def _import_recommend_cli():
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_recommend as cli  # type: ignore
    finally:
        sys.path.pop(0)
    return cli


def _import_report_cli():
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_report as cli  # type: ignore
    finally:
        sys.path.pop(0)
    return cli


def _shadow_artefact_state() -> dict[Path, tuple[int, int, int]]:
    """Return a snapshot of (mtime_ns, size, sha-like-hash via mtime)
    for every JSON under the real shadow_artefacts tree."""
    state: dict[Path, tuple[int, int, int]] = {}
    if not _REAL_SHADOW.exists():
        return state
    for p in _REAL_SHADOW.rglob("*.json"):
        st = p.stat()
        # Tuple is (mtime_ns, size, ctime_ns) — three independent
        # signals that any rewrite would disturb.
        state[p] = (st.st_mtime_ns, st.st_size, st.st_ctime_ns)
    return state


@pytest.fixture
def _real_registry_present() -> Path:
    if not _REAL_SHADOW.exists() or not _REAL_AUDIT_LOG.exists():
        pytest.skip("real registry / audit log missing; smoke test "
                    "requires the production fixtures")
    return _REAL_SHADOW


# ---------------------------------------------------------------------------
# 1. Inventory: 12 JSONs + 4 candidate subdirs.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_real_shadow_root_has_12_jsons_in_4_candidate_dirs(
    _real_registry_present: Path,
) -> None:
    json_count = sum(1 for _ in _REAL_SHADOW.rglob("*.json"))
    assert json_count == 12
    cand_dirs = sorted(
        d.name for d in _REAL_SHADOW.iterdir() if d.is_dir()
    )
    assert len(cand_dirs) == 4
    menu_ids = sorted(c.candidate_id for c in CANDIDATE_MENU_V0)
    assert cand_dirs == menu_ids


# ---------------------------------------------------------------------------
# 2. Replay validator runs across every candidate without mutation.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_replay_validator_runs_for_every_candidate_without_mutation(
    _real_registry_present: Path,
) -> None:
    pre = _shadow_artefact_state()

    reports = []
    for c in CANDIDATE_MENU_V0:
        r = summarise_replay(
            candidate_id=c.candidate_id,
            shadow_artefacts_root=_REAL_SHADOW,
        )
        reports.append(r)
        assert r.n_artefacts_read >= 1, (
            f"{c.candidate_id} replay read 0 artefacts; expected >= 1"
        )

    # Mtime/size/ctime sweep — no JSON disturbed.
    post = _shadow_artefact_state()
    assert pre == post, "replay validator mutated a shadow artefact"


# ---------------------------------------------------------------------------
# 3. Report CLI with REAL audit log → every G8 ABSTAINs with violation.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_report_cli_with_real_audit_log_blocks_every_candidate(
    tmp_path: Path, _real_registry_present: Path,
) -> None:
    report_cli = _import_report_cli()
    pre = _shadow_artefact_state()

    atlas = tmp_path / "atlas.md"
    atlas.write_text("# atlas\n", encoding="utf-8")
    avail = tmp_path / "availability.md"
    avail.write_text("""
# availability
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
""", encoding="utf-8")
    wf = tmp_path / "wf.md"
    wf.write_text("# wf\n", encoding="utf-8")
    bounds = tmp_path / "safety_bounds.yaml"  # absent on purpose
    registry_root = tmp_path / "registry"
    report_path = tmp_path / "report.md"

    rc = report_cli.main([
        "--atlas-report", str(atlas),
        "--data-availability-report", str(avail),
        "--walk-forward-report", str(wf),
        "--safety-bounds", str(bounds),
        "--registry-root", str(registry_root),
        "--report-path", str(report_path),
        "--registry-audit-log", str(_REAL_AUDIT_LOG),
    ])
    assert rc == 0
    body = report_path.read_text(encoding="utf-8")
    assert "registry_append_only_violation" in body
    assert "lost_sha_count" in body
    # 4 G8 ABSTAIN lines (one per candidate).
    g8_lines = [ln for ln in body.splitlines()
                if ln.strip().startswith("- G8:")]
    assert len(g8_lines) == 4
    for ln in g8_lines:
        assert "**ABSTAIN**" in ln
        assert "registry_append_only_violation" in ln
        assert str(_REAL_AUDIT_LOG) in ln

    # Real shadow artefacts unchanged.
    assert _shadow_artefact_state() == pre


# ---------------------------------------------------------------------------
# 4. End-to-end: recommendation CLI + REAL audit log → all
#    NO_RECOMMENDATION/evidence_chain_invalid; visualisations rendered.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recommend_cli_against_real_audit_log(
    tmp_path: Path, _real_registry_present: Path,
) -> None:
    rec_cli = _import_recommend_cli()
    pre = _shadow_artefact_state()

    atlas = tmp_path / "atlas.md"
    atlas.write_text("# atlas\n", encoding="utf-8")
    avail = tmp_path / "availability.md"
    avail.write_text("""
# availability
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
""", encoding="utf-8")
    wf = tmp_path / "wf.md"
    wf.write_text("# wf\n", encoding="utf-8")
    bounds = tmp_path / "safety_bounds.yaml"
    registry_root = tmp_path / "registry"
    report_path = tmp_path / "phase-d-evolution-report.md"
    rec_path = tmp_path / "rec.md"

    with redirect_stdout(io.StringIO()):
        rc = rec_cli.main([
            "--atlas-report", str(atlas),
            "--data-availability-report", str(avail),
            "--walk-forward-report", str(wf),
            "--safety-bounds", str(bounds),
            "--registry-root", str(registry_root),
            "--report-path", str(report_path),
            "--recommendation-path", str(rec_path),
            "--registry-audit-log", str(_REAL_AUDIT_LOG),
        ])
    assert rc == 0
    body = rec_path.read_text(encoding="utf-8")

    assert "**NOT LIVE**" in body
    assert "**NOT APPROVED**" in body
    assert "**NOT DEPLOYED**" in body
    # Every menu candidate appears in the report.
    for c in CANDIDATE_MENU_V0:
        assert c.candidate_id in body
    # Visualisations (round-2 task 3) present in real-registry output too.
    assert "Parameter comparison" in body
    assert "Gate matrix" in body
    assert "Heat ranking" in body
    # Violation cascade: every candidate is NO_RECOMMENDATION /
    # evidence_chain_invalid because the real audit log records a
    # violation.
    assert body.count("decision: `NO_RECOMMENDATION`") == 4
    assert "evidence_chain_invalid" in body

    # Real artefacts unchanged.
    assert _shadow_artefact_state() == pre


# ---------------------------------------------------------------------------
# 5. Safety-bounds template makes G6 PASS for every candidate.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_template_lifts_g6_for_every_candidate(
    _real_registry_present: Path,
) -> None:
    """Operator copies the template to live on Day 0; this test
    confirms G6 stops returning safety_bound_undefined for every
    real menu candidate when fed the template."""
    report_cli = _import_report_cli()
    cfg = report_cli.load_safety_bounds(_TEMPLATE)
    from smc.hedgerock.evolution.promotion_gates import g6_safety_bounds
    for c in CANDIDATE_MENU_V0:
        result = g6_safety_bounds(candidate=c, bounds=cfg)
        assert result.status.value == "PASS", (
            f"{c.candidate_id}: G6 returned {result.status.value} "
            f"({result.reason})"
        )


# ---------------------------------------------------------------------------
# 6. Approved/, pointer.json absence sentinel (round 3 must not have
#    accidentally created either).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_approved_and_pointer_still_absent_in_real_registry(
    _real_registry_present: Path,
) -> None:
    assert not (_REAL_REGISTRY / "approved").exists()
    assert not (_REAL_REGISTRY / "pointer.json").exists()

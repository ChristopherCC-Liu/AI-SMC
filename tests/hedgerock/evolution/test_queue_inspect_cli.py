"""Stage 6-followup task 2 — queue-inspect CLI tests (read-only).

Pinned guarantees:

  * The CLI reads the JSONL queue produced by Stage 5 and renders a
    markdown summary. It NEVER mutates the queue file (no
    ``a``/``w`` open mode against the queue path inside the script).
  * Empty / missing queue file → exit 0 with a clear "queue empty"
    summary, not a traceback.
  * Each rendered entry surfaces: candidate_id, parameter target,
    parameter class, baseline → proposed, status, queued_at,
    required_windows, required_tests, blocking_conditions.
  * The summary carries explicit "report-only" / "no promotion"
    banners.
  * The CLI rejects any output path that lands under
    ``policy_registry/approved/`` or ``policy_registry/pointer.json``.
  * Source-level isolation: no live runtime imports.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from smc.hedgerock.evolution.candidate_generator import (
    CandidateProposal, DECISION_RECOMMEND,
)
from smc.hedgerock.evolution.shadow_test_queue import ShadowTestQueue


_REPO = Path(__file__).resolve().parents[3]


def _import_cli():
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_queue_inspect as cli  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return cli


def _recommend(cid: str) -> CandidateProposal:
    return CandidateProposal(
        candidate_id=cid,
        parameter_target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        parameter_class="confidence_threshold_observe",
        baseline_value=0.55,
        proposed_value=0.50,
        triggered_by=("G6_safety_bound_undefined",),
        expected_improvement="micro-relax observe floor",
        risks=("possible false-positive uptick",),
        next_validation=("XAUUSD shadow run",),
        decision=DECISION_RECOMMEND,
        decision_reason="",
    )


# ---------------------------------------------------------------------------
# 1. Populated queue → report mentions every entry's id + status.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_queue_inspect_renders_every_entry(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue" / "shadow_test_queue.jsonl"
    audit_log = tmp_path / "_audit.md"
    audit_log.write_text("# audit\n", encoding="utf-8")

    queue = ShadowTestQueue(path=queue_path, audit_log_path=audit_log)
    queue.enqueue_proposals([
        _recommend("c1-lower-observe-floor-0.50"),
        _recommend("c4-range2-conf-0.70"),
    ])

    cli = _import_cli()
    report_path = tmp_path / "queue_report.md"
    rc = cli.main([
        "--queue-path", str(queue_path),
        "--report-path", str(report_path),
    ])
    assert rc == 0
    body = report_path.read_text(encoding="utf-8")

    assert "c1-lower-observe-floor-0.50" in body
    assert "c4-range2-conf-0.70" in body
    assert body.count("status: `QUEUED`") == 2
    assert "required_windows" in body
    assert "blocking_conditions" in body or "blocking conditions" in body.lower()
    assert "**NOT LIVE**" in body
    assert "**NOT APPROVED**" in body


# ---------------------------------------------------------------------------
# 2. Empty queue → graceful "queue empty" summary, exit 0.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_queue_inspect_handles_empty_queue(tmp_path: Path) -> None:
    queue_path = tmp_path / "shadow_test_queue.jsonl"
    queue_path.touch()
    cli = _import_cli()
    report_path = tmp_path / "report.md"
    rc = cli.main([
        "--queue-path", str(queue_path),
        "--report-path", str(report_path),
    ])
    assert rc == 0
    body = report_path.read_text(encoding="utf-8")
    assert "queue empty" in body.lower() or "no entries" in body.lower()
    assert "**NOT LIVE**" in body


# ---------------------------------------------------------------------------
# 3. Missing queue file → graceful exit 0 with "queue not found".
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_queue_inspect_handles_missing_queue(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.jsonl"
    cli = _import_cli()
    report_path = tmp_path / "report.md"
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        rc = cli.main([
            "--queue-path", str(missing),
            "--report-path", str(report_path),
        ])
    assert rc == 0
    body = report_path.read_text(encoding="utf-8")
    assert "queue not found" in body.lower() or "no queue file" in body.lower()
    out = stdout.getvalue()
    assert str(missing) in out


# ---------------------------------------------------------------------------
# 4. Read-only — queue file unchanged after inspection.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_queue_inspect_does_not_mutate_queue_file(tmp_path: Path) -> None:
    queue_path = tmp_path / "shadow_test_queue.jsonl"
    audit_log = tmp_path / "_audit.md"
    audit_log.write_text("# audit\n", encoding="utf-8")
    queue = ShadowTestQueue(path=queue_path, audit_log_path=audit_log)
    queue.enqueue_proposals([_recommend("c1-lower-observe-floor-0.50")])

    pre_bytes = queue_path.read_bytes()
    pre_mtime = queue_path.stat().st_mtime_ns

    cli = _import_cli()
    report_path = tmp_path / "report.md"
    rc = cli.main([
        "--queue-path", str(queue_path),
        "--report-path", str(report_path),
    ])
    assert rc == 0

    post_bytes = queue_path.read_bytes()
    post_mtime = queue_path.stat().st_mtime_ns
    assert pre_bytes == post_bytes
    assert pre_mtime == post_mtime


# ---------------------------------------------------------------------------
# 5. Report path under approved/ or pointer.json is rejected.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_queue_inspect_rejects_forbidden_report_path(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.jsonl"
    queue_path.touch()
    cli = _import_cli()
    bad = tmp_path / "policy_registry" / "approved" / "report.md"
    rc = cli.main([
        "--queue-path", str(queue_path),
        "--report-path", str(bad),
    ])
    assert rc != 0


# ---------------------------------------------------------------------------
# 6. Source-level isolation — no live runtime imports.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_queue_inspect_script_does_not_import_live_runtime() -> None:
    src = (
        _REPO / "scripts" / "hedgerock_evolution_queue_inspect.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from smc.hedgerock.rule_engine",
        "from smc.hedgerock.decision_server",
        "from smc.hedgerock.phase_d_walk_forward",
    )
    for f in forbidden:
        assert f not in src

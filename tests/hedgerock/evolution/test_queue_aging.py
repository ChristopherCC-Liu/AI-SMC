"""Stage 6-followup-2 task 2 — queue aging tests.

Append-only design: when a queued candidate ages out, the aging pass
appends a NEW JSONL line with status=STALE for that candidate. The
original QUEUED line is never rewritten or removed.

Pinned guarantees:

  * ``mark_stale_entries`` returns a list of newly-appended STALE
    entries; existing QUEUED lines stay byte-identical.
  * Default age threshold is 14 days; configurable via argument.
  * Once an entry has a sibling STALE entry for the same candidate id
    *with the same queued_at*, a second aging pass does NOT re-mark
    it (idempotent).
  * STALE entries carry the original queued_at (so operators can
    correlate) plus a ``stale_at`` timestamp and the reason
    ``aged_out:N_days``.
  * STALE entries leave the JSONL append-only contract intact (no
    writes to approved/, pointer.json, or the queue path's parent
    above the queue file).
  * The CLI ``hedgerock_evolution_queue_age.py`` runs the aging pass
    and prints a summary; no flag exists to delete entries.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from smc.hedgerock.evolution.candidate_generator import (
    CandidateProposal, DECISION_RECOMMEND,
)
from smc.hedgerock.evolution.shadow_test_queue import ShadowTestQueue

from smc.hedgerock.evolution.queue_aging import (
    DEFAULT_STALE_AFTER_DAYS,
    mark_stale_entries,
)


_REPO = Path(__file__).resolve().parents[3]


def _import_age_cli():
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_queue_age as cli  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return cli


def _recommend(cid: str) -> CandidateProposal:
    return CandidateProposal(
        candidate_id=cid,
        parameter_target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        parameter_class="confidence_threshold_observe",
        baseline_value=0.55, proposed_value=0.50,
        triggered_by=("G6_safety_bound_undefined",),
        expected_improvement="micro-relax observe floor",
        risks=("possible false-positive uptick",),
        next_validation=("XAUUSD shadow run",),
        decision=DECISION_RECOMMEND, decision_reason="",
    )


def _seed_queue_with_old_entry(
    *,
    tmp_path: Path,
    queued_days_ago: int,
    candidate_id: str = "c1-lower-observe-floor-0.50",
) -> tuple[Path, str]:
    """Manually emit a queue line with a backdated queued_at so we
    can drive the aging pass deterministically."""
    audit = tmp_path / "_audit.md"
    audit.write_text("# audit\n", encoding="utf-8")
    queue_path = tmp_path / "queue" / "shadow_test_queue.jsonl"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    backdated = (
        datetime.now(timezone.utc) - timedelta(days=queued_days_ago)
    ).isoformat()
    entry = {
        "candidate_id": candidate_id,
        "parameter_target": "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        "parameter_class": "confidence_threshold_observe",
        "baseline_value": 0.55,
        "proposed_value": 0.50,
        "status": "QUEUED",
        "queued_at": backdated,
        "required_windows": 8,
        "required_tests": ["shadow_runner"],
        "blocking_conditions": ["registry_append_only_violation must remain False"],
        "reason": "",
        "audit_log_path": str(audit),
    }
    with queue_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return queue_path, backdated


# ---------------------------------------------------------------------------
# 1. Old entry is marked STALE; original line untouched.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_old_entry_is_marked_stale_and_original_line_is_preserved(
    tmp_path: Path,
) -> None:
    queue_path, queued_at = _seed_queue_with_old_entry(
        tmp_path=tmp_path, queued_days_ago=20,
    )
    pre_lines = queue_path.read_text(encoding="utf-8").splitlines()
    pre_first = pre_lines[0]

    appended = mark_stale_entries(queue_path=queue_path, stale_after_days=14)
    assert len(appended) == 1
    stale = appended[0]
    assert stale["status"] == "STALE"
    assert stale["queued_at"] == queued_at
    assert "aged_out" in stale["reason"]
    assert stale["candidate_id"] == "c1-lower-observe-floor-0.50"

    post_lines = queue_path.read_text(encoding="utf-8").splitlines()
    assert len(post_lines) == 2
    # First line preserved byte-for-byte.
    assert post_lines[0] == pre_first


# ---------------------------------------------------------------------------
# 2. Fresh entry is NOT marked.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fresh_entry_is_not_marked(tmp_path: Path) -> None:
    queue_path, _ = _seed_queue_with_old_entry(
        tmp_path=tmp_path, queued_days_ago=2,
    )
    appended = mark_stale_entries(queue_path=queue_path, stale_after_days=14)
    assert appended == []
    # Queue still has just the one line.
    lines = queue_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


# ---------------------------------------------------------------------------
# 3. Idempotent — running twice doesn't double-mark.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_aging_is_idempotent(tmp_path: Path) -> None:
    queue_path, _ = _seed_queue_with_old_entry(
        tmp_path=tmp_path, queued_days_ago=30,
    )
    first = mark_stale_entries(queue_path=queue_path, stale_after_days=14)
    second = mark_stale_entries(queue_path=queue_path, stale_after_days=14)
    assert len(first) == 1
    assert second == []
    # Queue should have exactly 2 lines (1 QUEUED + 1 STALE).
    assert len(queue_path.read_text(encoding="utf-8").splitlines()) == 2


# ---------------------------------------------------------------------------
# 4. Aging skips entries already in non-QUEUED states (e.g. STALE).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_aging_skips_entries_already_stale(tmp_path: Path) -> None:
    audit = tmp_path / "_audit.md"
    audit.write_text("# audit\n", encoding="utf-8")
    queue_path = tmp_path / "queue.jsonl"
    backdated = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    # Manually write a STALE entry alongside a QUEUED entry.
    queue_path.write_text(
        json.dumps({
            "candidate_id": "c1", "status": "QUEUED",
            "queued_at": backdated, "audit_log_path": str(audit),
        }, sort_keys=True) + "\n"
        + json.dumps({
            "candidate_id": "c1", "status": "STALE",
            "queued_at": backdated,
            "stale_at": (datetime.now(timezone.utc)).isoformat(),
            "reason": "aged_out:30_days",
        }, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    appended = mark_stale_entries(queue_path=queue_path, stale_after_days=14)
    assert appended == []


# ---------------------------------------------------------------------------
# 5. CLI runs and prints summary.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cli_marks_stale_and_prints_summary(tmp_path: Path) -> None:
    queue_path, _ = _seed_queue_with_old_entry(
        tmp_path=tmp_path,
        queued_days_ago=DEFAULT_STALE_AFTER_DAYS + 5,
    )
    cli = _import_age_cli()
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        rc = cli.main(["--queue-path", str(queue_path)])
    assert rc == 0
    out = stdout.getvalue()
    assert "marked 1 stale" in out.lower() or "marked stale: 1" in out.lower()


# ---------------------------------------------------------------------------
# 6. CLI rejects forbidden queue-path edits (no --delete style flag).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cli_has_no_delete_or_remove_flag() -> None:
    cli = _import_age_cli()
    src = (_REPO / "scripts" / "hedgerock_evolution_queue_age.py").read_text(
        encoding="utf-8"
    )
    forbidden_flags = ("--delete", "--remove", "--purge", "--clear", "--unlink")
    for f in forbidden_flags:
        assert f not in src, (
            f"queue-age CLI exposes a removal flag: {f!r}"
        )


# ---------------------------------------------------------------------------
# 7. Source-level isolation.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_queue_aging_module_has_no_live_runtime_imports() -> None:
    src = (
        _REPO / "src" / "smc" / "hedgerock" / "evolution" / "queue_aging.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from smc.hedgerock.rule_engine",
        "from smc.hedgerock.decision_server",
        "from smc.hedgerock.phase_d_walk_forward",
    )
    for f in forbidden:
        assert f not in src


# ---------------------------------------------------------------------------
# 8. Queue path under approved/ or pointer.json is rejected.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_aging_rejects_queue_path_under_approved(tmp_path: Path) -> None:
    bad = tmp_path / "policy_registry" / "approved" / "queue.jsonl"
    bad.parent.mkdir(parents=True)
    bad.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        mark_stale_entries(queue_path=bad, stale_after_days=14)


# ---------------------------------------------------------------------------
# 9. The aging pass refuses to mark when queued_at is unparseable.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unparseable_queued_at_is_left_alone(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.jsonl"
    queue_path.write_text(
        json.dumps({
            "candidate_id": "c1", "status": "QUEUED",
            "queued_at": "not-a-timestamp",
        }) + "\n",
        encoding="utf-8",
    )
    appended = mark_stale_entries(queue_path=queue_path, stale_after_days=14)
    assert appended == []


# ---------------------------------------------------------------------------
# 10. Two distinct old candidates each get one STALE marker.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_multiple_old_candidates_each_get_one_stale_marker(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "_audit.md"
    audit.write_text("# audit\n", encoding="utf-8")
    queue_path = tmp_path / "queue.jsonl"
    backdated = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    queue_path.write_text(
        json.dumps({"candidate_id": "c1", "status": "QUEUED",
                    "queued_at": backdated,
                    "audit_log_path": str(audit)}, sort_keys=True) + "\n"
        + json.dumps({"candidate_id": "c4", "status": "QUEUED",
                      "queued_at": backdated,
                      "audit_log_path": str(audit)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    appended = mark_stale_entries(queue_path=queue_path, stale_after_days=14)
    cids = {e["candidate_id"] for e in appended}
    assert cids == {"c1", "c4"}
    # Original two lines + two STALE = 4 lines total.
    assert len(queue_path.read_text(encoding="utf-8").splitlines()) == 4

"""Stage 5 — shadow-test queue tests (report-only sidecar).

Pinned guarantees:

  * Queue is a sidecar JSON-lines file. Each line carries a single
    queue entry (one per candidate proposal that flipped to RECOMMEND).
  * Entries are append-only. The queue API exposes ``enqueue`` but
    NEVER ``dequeue``, ``unlink``, or any in-place rewrite.
  * ``RECOMMEND`` proposals get queued; ``NO_RECOMMENDATION`` proposals
    do NOT enter the queue.
  * Queue path is rejected when it points under
    ``policy_registry/approved/`` or ``policy_registry/pointer.json``.
  * Each queue entry carries: candidate_id, parameter_target,
    parameter_class, baseline_value, proposed_value, status='QUEUED',
    queued_at, required_windows, required_tests, blocking_conditions,
    reason, audit_log_path.
  * ``approved/`` and ``pointer.json`` MUST not exist on disk after a
    queue run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smc.hedgerock.evolution.candidate_generator import (
    CandidateProposal,
    DECISION_NO_RECOMMENDATION,
    DECISION_RECOMMEND,
)

from smc.hedgerock.evolution.shadow_test_queue import (
    QueueEntry,
    ShadowTestQueue,
    build_queue_entry,
)


_REPO = Path(__file__).resolve().parents[3]


def _recommend(cid: str = "c1-lower-observe-floor-0.50") -> CandidateProposal:
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


def _no_recommend(cid: str, reason: str) -> CandidateProposal:
    return CandidateProposal(
        candidate_id=cid,
        parameter_target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        parameter_class="confidence_threshold_observe",
        baseline_value=0.55,
        proposed_value=0.55,
        triggered_by=(),
        expected_improvement="",
        risks=(),
        next_validation=(),
        decision=DECISION_NO_RECOMMENDATION,
        decision_reason=reason,
    )


# ---------------------------------------------------------------------------
# 1. RECOMMEND proposal → queue entry written; NO_RECOMMENDATION ignored.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recommend_enters_queue_no_recommend_does_not(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue" / "shadow_test_queue.jsonl"
    queue = ShadowTestQueue(path=queue_path, audit_log_path=tmp_path / "_audit.md")

    p_ok = _recommend("c1-lower-observe-floor-0.50")
    p_no = _no_recommend("c2-halt-expiry-observe-6h", "no_trigger")
    p_ok2 = _recommend("c4-range2-conf-0.70")
    p_violation = _no_recommend("c3-aggressive-floor-0.78", "evidence_chain_invalid")

    queue.enqueue_proposals([p_ok, p_no, p_ok2, p_violation])

    text = queue_path.read_text(encoding="utf-8").splitlines()
    assert len(text) == 2  # only the two RECOMMENDs
    parsed = [json.loads(ln) for ln in text]
    ids = sorted(e["candidate_id"] for e in parsed)
    assert ids == ["c1-lower-observe-floor-0.50", "c4-range2-conf-0.70"]
    for e in parsed:
        assert e["status"] == "QUEUED"
        assert "queued_at" in e
        assert e["required_windows"]
        assert e["required_tests"]
        assert e["blocking_conditions"]


# ---------------------------------------------------------------------------
# 2. Queue is append-only — second enqueue extends, never overwrites.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_queue_is_append_only(tmp_path: Path) -> None:
    queue_path = tmp_path / "shadow_test_queue.jsonl"
    queue = ShadowTestQueue(path=queue_path, audit_log_path=tmp_path / "_audit.md")
    queue.enqueue_proposals([_recommend("c1-lower-observe-floor-0.50")])
    pre_lines = queue_path.read_text(encoding="utf-8").splitlines()
    assert len(pre_lines) == 1

    # Second enqueue with a different candidate APPENDS — does not
    # replace.
    queue.enqueue_proposals([_recommend("c4-range2-conf-0.70")])
    post_lines = queue_path.read_text(encoding="utf-8").splitlines()
    assert len(post_lines) == 2
    # First line is preserved verbatim.
    assert post_lines[0] == pre_lines[0]


# ---------------------------------------------------------------------------
# 3. ShadowTestQueue exposes no removal / mutation API.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_queue_has_no_removal_or_mutation_api() -> None:
    forbidden = {
        "dequeue", "remove", "delete", "unlink", "rewrite", "replace",
        "pop", "clear", "discard",
    }
    public_attrs = {
        a for a in dir(ShadowTestQueue) if not a.startswith("_")
    }
    leaked = forbidden & public_attrs
    assert not leaked, (
        f"ShadowTestQueue leaks removal/mutation API: {leaked}"
    )


# ---------------------------------------------------------------------------
# 4. Queue path under approved/ or pointer.json is rejected.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_queue_path_under_approved_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "policy_registry" / "approved" / "queue.jsonl"
    with pytest.raises(ValueError):
        ShadowTestQueue(path=bad, audit_log_path=tmp_path / "_audit.md")


@pytest.mark.unit
def test_queue_path_at_pointer_json_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "policy_registry" / "pointer.json"
    with pytest.raises(ValueError):
        ShadowTestQueue(path=bad, audit_log_path=tmp_path / "_audit.md")


# ---------------------------------------------------------------------------
# 5. Required-validation envelope on each entry.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_queue_entry_carries_required_validation_envelope(tmp_path: Path) -> None:
    p = _recommend()
    entry = build_queue_entry(
        proposal=p, audit_log_path=tmp_path / "_audit.md",
    )
    assert isinstance(entry, QueueEntry)
    assert entry.candidate_id == p.candidate_id
    assert entry.parameter_class == p.parameter_class
    assert entry.parameter_target == p.parameter_target
    assert entry.baseline_value == p.baseline_value
    assert entry.proposed_value == p.proposed_value
    assert entry.status == "QUEUED"
    assert entry.queued_at  # ISO 8601
    assert entry.required_windows >= 4
    assert "shadow_runner" in " ".join(entry.required_tests).lower()
    assert "registry_append_only_violation" in " ".join(entry.blocking_conditions)
    assert "human_approval" in " ".join(entry.blocking_conditions)
    assert entry.reason == ""  # empty when QUEUED
    assert str(tmp_path / "_audit.md") in entry.audit_log_path


# ---------------------------------------------------------------------------
# 6. ShadowTestQueue does NOT create approved/ or pointer.json.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_enqueue_does_not_create_approved_or_pointer(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue" / "shadow_test_queue.jsonl"
    queue = ShadowTestQueue(path=queue_path, audit_log_path=tmp_path / "_audit.md")
    queue.enqueue_proposals([_recommend()])
    assert not (tmp_path / "approved").exists()
    assert not (tmp_path / "pointer.json").exists()
    assert not (tmp_path / "policy_registry" / "approved").exists()
    assert not (tmp_path / "policy_registry" / "pointer.json").exists()


# ---------------------------------------------------------------------------
# 7. Source-level isolation — module imports no live runtime.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_shadow_test_queue_module_does_not_import_live_runtime() -> None:
    src = (
        _REPO / "src" / "smc" / "hedgerock" / "evolution"
        / "shadow_test_queue.py"
    ).read_text(encoding="utf-8")
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
            f"shadow_test_queue imports a live module: {f!r}"
        )


# ---------------------------------------------------------------------------
# 8. Read-back: reading the queue file produces a parseable list of
#    QueueEntry-shaped dicts. (No QueueEntry objects deserialised; the
#    queue is a read-only ledger from this module's perspective.)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_queue_lines_are_jsonl_parseable(tmp_path: Path) -> None:
    queue_path = tmp_path / "shadow_test_queue.jsonl"
    queue = ShadowTestQueue(path=queue_path, audit_log_path=tmp_path / "_audit.md")
    queue.enqueue_proposals([_recommend("c1-lower-observe-floor-0.50"),
                              _recommend("c4-range2-conf-0.70")])
    for ln in queue_path.read_text(encoding="utf-8").splitlines():
        d = json.loads(ln)
        assert d["status"] == "QUEUED"
        assert isinstance(d["required_windows"], int)
        assert isinstance(d["required_tests"], list)
        assert isinstance(d["blocking_conditions"], list)

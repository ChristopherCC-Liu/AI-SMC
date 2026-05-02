"""Stage 5 — shadow-test queue (report-only sidecar).

A queue entry records a candidate proposal that has been *recommended*
for shadow validation. The queue is an append-only JSON-Lines file
written to a sidecar path of the operator's choosing.

The queue does NOT promote, approve, or deploy anything. It does not
write under ``policy_registry/approved/`` or modify
``policy_registry/pointer.json``. There is no public dequeue / remove
/ rewrite API — entries flow forward through the state machine, never
backward.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from smc.hedgerock.evolution.candidate_generator import (
    CandidateProposal,
    DECISION_RECOMMEND,
)


__all__ = [
    "QueueEntry",
    "ShadowTestQueue",
    "build_queue_entry",
]


_FORBIDDEN_PATH_FRAGMENTS = (
    "policy_registry/approved",
    "policy_registry/pointer.json",
)

_REQUIRED_WINDOWS = 8
_REQUIRED_TESTS = (
    "shadow_runner v0.3.0 multi-window run on XAUUSD",
    "G1-G8 re-evaluation under refreshed bundle",
    "window_coverage gate verdict",
    "registry_audit clean state at queue-pull time",
)
_BLOCKING_CONDITIONS = (
    "registry_append_only_violation must remain False",
    "audit_log_present must be True",
    "human_approval required before promotion",
    "XAUUSD years_passing must remain ≥ 4",
)


@dataclass(frozen=True)
class QueueEntry:
    candidate_id: str
    parameter_target: str
    parameter_class: str
    baseline_value: float
    proposed_value: float
    status: str  # always "QUEUED" — no in-flight states in v0
    queued_at: str
    required_windows: int
    required_tests: tuple[str, ...]
    blocking_conditions: tuple[str, ...]
    reason: str  # empty when QUEUED; populated by future state additions
    audit_log_path: str


def _assert_path_safe(path: Path) -> None:
    text = str(path)
    for fragment in _FORBIDDEN_PATH_FRAGMENTS:
        if fragment in text:
            raise ValueError(
                f"shadow-test queue path would land under a forbidden "
                f"location: {text!r} (matched fragment {fragment!r})"
            )


def build_queue_entry(
    *, proposal: CandidateProposal, audit_log_path: Path,
) -> QueueEntry:
    """Construct a :class:`QueueEntry` for a single RECOMMEND proposal.

    Caller is expected to have filtered out NO_RECOMMENDATION
    proposals; this function does not validate the decision.
    """
    return QueueEntry(
        candidate_id=proposal.candidate_id,
        parameter_target=proposal.parameter_target,
        parameter_class=proposal.parameter_class,
        baseline_value=float(proposal.baseline_value),
        proposed_value=float(proposal.proposed_value),
        status="QUEUED",
        queued_at=datetime.now(timezone.utc).isoformat(),
        required_windows=_REQUIRED_WINDOWS,
        required_tests=_REQUIRED_TESTS,
        blocking_conditions=_BLOCKING_CONDITIONS,
        reason="",
        audit_log_path=str(audit_log_path),
    )


class ShadowTestQueue:
    """Append-only sidecar queue.

    Public surface: ``enqueue_proposals`` and ``path``. There is no
    dequeue / remove / clear / pop method by design — entries flow
    forward through the state machine, never backward.
    """

    __slots__ = ("path", "_audit_log_path")

    def __init__(self, *, path: Path, audit_log_path: Path) -> None:
        _assert_path_safe(Path(path))
        self.path: Path = Path(path)
        self._audit_log_path: Path = Path(audit_log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def enqueue_proposals(
        self, proposals: Iterable[CandidateProposal],
    ) -> list[QueueEntry]:
        """Append RECOMMEND proposals to the queue file.

        ``NO_RECOMMENDATION`` proposals are silently skipped — the
        recommendation report carries the full audit of why they were
        not queued.
        """
        added: list[QueueEntry] = []
        with self.path.open("a", encoding="utf-8") as fh:
            for p in proposals:
                if p.decision != DECISION_RECOMMEND:
                    continue
                entry = build_queue_entry(
                    proposal=p, audit_log_path=self._audit_log_path,
                )
                fh.write(
                    json.dumps(
                        asdict(entry), ensure_ascii=False, sort_keys=True,
                    ) + "\n"
                )
                added.append(entry)
        return added

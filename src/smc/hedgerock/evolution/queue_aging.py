"""Stage 6-followup-2 task 2 — queue aging (append-only STALE marking).

When a queued candidate has been waiting longer than ``stale_after_days``
(default 14) without a sibling STALE marker, the aging pass appends a
new line with status ``STALE`` to the queue file. The original
``QUEUED`` line is preserved byte-for-byte; nothing is rewritten or
removed.

The aging pass is idempotent: running it twice does not produce two
STALE markers for the same (candidate_id, queued_at) pair.

This module never imports the live trading runtime and refuses to
operate on queue paths that land under
``policy_registry/approved/`` or ``policy_registry/pointer.json``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


__all__ = [
    "DEFAULT_STALE_AFTER_DAYS",
    "mark_stale_entries",
]


DEFAULT_STALE_AFTER_DAYS: int = 14

_FORBIDDEN_PATH_FRAGMENTS = (
    "policy_registry/approved",
    "policy_registry/pointer.json",
)


def _assert_queue_path_safe(path: Path) -> None:
    text = str(path)
    for fragment in _FORBIDDEN_PATH_FRAGMENTS:
        if fragment in text:
            raise ValueError(
                f"queue path lands under a forbidden location: "
                f"{text!r} (matched {fragment!r})"
            )


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def mark_stale_entries(
    *,
    queue_path: Path,
    stale_after_days: int = DEFAULT_STALE_AFTER_DAYS,
    now: datetime | None = None,
) -> list[dict]:
    """Append STALE marker lines for any QUEUED entry older than
    ``stale_after_days`` whose (candidate_id, queued_at) pair is not
    already marked STALE.

    Returns the list of newly-appended entries (as dicts). Never
    rewrites or removes existing lines.
    """
    qp = Path(queue_path)
    _assert_queue_path_safe(qp)
    if not qp.exists():
        return []

    now_dt = now or datetime.now(timezone.utc)

    # Read current state. We need:
    #   - Set of (candidate_id, queued_at) pairs that are already STALE
    #   - List of QUEUED entries with parseable queued_at older than threshold
    already_stale: set[tuple[str, str]] = set()
    candidates_to_age: list[dict] = []

    for ln in qp.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        cid = d.get("candidate_id")
        qa = d.get("queued_at")
        status = d.get("status")
        if not cid or not qa:
            continue
        key = (cid, qa)
        if status == "STALE":
            already_stale.add(key)
        elif status == "QUEUED":
            qa_dt = _parse_iso(qa)
            if qa_dt is None:
                continue
            age_days = (now_dt - qa_dt).total_seconds() / 86400.0
            if age_days >= stale_after_days:
                d["__age_days"] = age_days  # transient; not written back
                candidates_to_age.append(d)

    appended: list[dict] = []
    if not candidates_to_age:
        return appended

    with qp.open("a", encoding="utf-8") as fh:
        for d in candidates_to_age:
            cid = d["candidate_id"]
            qa = d["queued_at"]
            if (cid, qa) in already_stale:
                continue
            age_days = d.get("__age_days", 0.0)
            stale_entry = {
                "candidate_id": cid,
                "parameter_target": d.get("parameter_target", ""),
                "parameter_class": d.get("parameter_class", ""),
                "baseline_value": d.get("baseline_value"),
                "proposed_value": d.get("proposed_value"),
                "status": "STALE",
                "queued_at": qa,
                "stale_at": now_dt.isoformat(),
                "reason": f"aged_out:{int(age_days)}_days",
                "required_windows": d.get("required_windows"),
                "required_tests": d.get("required_tests", []),
                "blocking_conditions": d.get("blocking_conditions", []),
                "audit_log_path": d.get("audit_log_path", ""),
            }
            fh.write(
                json.dumps(stale_entry, ensure_ascii=False, sort_keys=True)
                + "\n"
            )
            appended.append(stale_entry)
            already_stale.add((cid, qa))
    return appended

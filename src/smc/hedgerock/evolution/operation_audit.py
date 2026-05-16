"""Stage 6-followup-2 task 4 — operation audit trail.

Append-only JSONL ledger that records what the report-only
self-evolution loop did, when, and on whose behalf. Distinct from
the registry append-only audit log (`_audit.md`) — this trail tracks
*operator actions*, not *artefact incidents*.

Every entry: ``{timestamp, operation, result, operator, details}``.
The trail file MUST NOT live under ``policy_registry/approved/``,
``policy_registry/pointer.json``, or anywhere inside
``policy_registry/shadow_artefacts/`` (mixing trail and registry
audit log would muddle two different operator contracts).

This module imports no live trading runtime.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


__all__ = [
    "OperationAuditEntry",
    "append_operation",
    "read_trail",
]


_FORBIDDEN_PATH_FRAGMENTS = (
    "policy_registry/approved",
    "policy_registry/pointer.json",
    "policy_registry/shadow_artefacts",
)


def _assert_trail_path_safe(path: Path) -> None:
    text = str(path)
    for fragment in _FORBIDDEN_PATH_FRAGMENTS:
        if fragment in text:
            raise ValueError(
                f"operation-audit trail lands under a forbidden "
                f"location: {text!r} (matched {fragment!r}). "
                "The operator action trail must stay separate from "
                "the shadow-artefact registry audit log."
            )


def _resolve_operator(operator: str | None) -> str:
    if operator:
        return operator
    env = os.environ.get("USER")
    if env:
        return env
    return "anonymous"


@dataclass(frozen=True)
class OperationAuditEntry:
    timestamp: str  # ISO 8601 UTC
    operation: str
    result: str  # "ok" | "fail" | other short token
    operator: str
    details: dict[str, Any] = field(default_factory=dict)


def append_operation(
    *,
    trail_path: Path,
    operation: str,
    result: str,
    operator: str | None,
    details: dict[str, Any] | None = None,
) -> OperationAuditEntry:
    """Append one entry to the trail; return the structured entry.

    ``details`` is JSON-serialised verbatim; callers must keep it
    JSON-safe.
    """
    tp = Path(trail_path)
    _assert_trail_path_safe(tp)
    tp.parent.mkdir(parents=True, exist_ok=True)
    entry = OperationAuditEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        operation=operation,
        result=result,
        operator=_resolve_operator(operator),
        details=dict(details or {}),
    )
    with tp.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True)
            + "\n"
        )
    return entry


def read_trail(trail_path: Path) -> list[dict[str, Any]]:
    tp = Path(trail_path)
    if not tp.exists():
        return []
    out: list[dict[str, Any]] = []
    for ln in tp.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out

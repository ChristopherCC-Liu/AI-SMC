"""Ticket 4 v2 follow-on T4-F1 — registry append-only audit state.

The shadow-artefact registry under
``policy_registry/shadow_artefacts/`` is append-only by contract.
When this contract is violated mid-session — for example, when an
artefact is deleted because of a gate-correctness fix — the
incident is recorded in
``policy_registry/shadow_artefacts/_audit.md``. The loader below
parses that markdown into a frozen :class:`RegistryAuditState`
which the evidence bundle, G8, the active multi-window PASS
evaluator and the report renderer all consume.

The point of T4-F1 is to make sure G8 sees the same red-line
status as the report's hard-boundary footer. Operators must not
be able to read "registry_append_only_violation: YES" in the
report and then approve a candidate because G8 silently let it
through.

Pure: this module never mutates the audit log and never writes
anywhere on disk. ``load_registry_audit_state`` is a forgiving
parser — it returns a clean state with ``violation=False`` when
the log is missing or malformed; callers must verify
``audit_log_path.exists()`` if they care about the difference
between "no log" and "log says no violations."
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


__all__ = [
    "DEFAULT_REGISTRY_AUDIT_LOG",
    "REGISTRY_VIOLATION_GATE_REASON_PREFIX",
    "RegistryAuditState",
    "load_registry_audit_state",
]


def _hedgerock_home() -> Path:
    raw = os.environ.get("HEDGEROCK_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "HedgeRock"


#: Default location of the operator-team registry audit log.
#: Override with ``$HEDGEROCK_HOME``; defaults to
#: ``$HOME/HedgeRock/policy_registry/shadow_artefacts/_audit.md``.
DEFAULT_REGISTRY_AUDIT_LOG: Path = (
    _hedgerock_home() / "policy_registry" / "shadow_artefacts" / "_audit.md"
)


#: Reason-string prefix emitted by G8 / the active PASS evaluator
#: when blocking a candidate because of a registry-append-only
#: violation. Used by tests and the report renderer for matching.
REGISTRY_VIOLATION_GATE_REASON_PREFIX: str = "registry_append_only_violation"


# A line that announces the deletion-section header in the audit
# log. Detection of this exact substring (case-insensitive) flips
# the audit state to violation=True. The 2026-05-02 incident wrote
# this header verbatim; future incidents are expected to follow
# the same pattern.
_DELETION_HEADER_PATTERN = re.compile(
    r"^##\s+\d{4}-\d{2}-\d{2}\s+—\s+Stale\s+v0\.3\.0\s+artefacts\s+deleted",
    flags=re.IGNORECASE | re.MULTILINE,
)


# A 64-hex-digit run anywhere in the log is treated as a lost SHA.
# The audit log records each lost SHA verbatim in a markdown table
# so the regex-only approach is good enough — false positives
# (e.g. a SHA quoted in a different context) are flagged as part
# of the lost-set, which is fail-closed by design.
_SHA256_PATTERN = re.compile(r"\b[0-9a-f]{64}\b")


@dataclass(frozen=True)
class RegistryAuditState:
    """Frozen registry-audit state for one evaluation cycle.

    Fields:

    * ``audit_log_path`` — absolute filesystem path of the audit log
      that produced this state.
    * ``audit_log_present`` — whether the audit log file exists on
      disk at evaluation time.
    * ``stale_v030_deleted_during_this_session`` — True when the
      log contains a "Stale v0.3.0 artefacts deleted" section.
    * ``lost_sha_count`` — number of distinct SHA-256 strings
      recorded in the log (one per lost artefact).
    * ``lost_sha256`` — tuple of those SHAs in stable iteration
      order. Defensive copy; never mutated.
    * ``registry_append_only_violation`` — True when ANY of the
      above flags indicate a violation. The G8 gate / active PASS
      evaluator MUST refuse PASS when this is True regardless of
      the candidate's own metric verdict.
    """

    audit_log_path: str
    audit_log_present: bool
    stale_v030_deleted_during_this_session: bool
    lost_sha_count: int
    lost_sha256: tuple[str, ...]
    registry_append_only_violation: bool


def load_registry_audit_state(
    audit_log_path: Path | str = DEFAULT_REGISTRY_AUDIT_LOG,
) -> RegistryAuditState:
    """Parse the audit log into a frozen :class:`RegistryAuditState`.

    Forgiving on missing / malformed logs: returns a clean state
    with ``violation=False`` when the file is absent or unreadable
    rather than raising. Callers that need to distinguish
    "no log" from "log says clean" inspect ``audit_log_present``.
    """
    p = Path(audit_log_path)
    path_str = str(p)
    if not p.exists():
        return RegistryAuditState(
            audit_log_path=path_str,
            audit_log_present=False,
            stale_v030_deleted_during_this_session=False,
            lost_sha_count=0,
            lost_sha256=(),
            registry_append_only_violation=False,
        )
    try:
        body = p.read_text(encoding="utf-8")
    except OSError:
        return RegistryAuditState(
            audit_log_path=path_str,
            audit_log_present=True,
            stale_v030_deleted_during_this_session=False,
            lost_sha_count=0,
            lost_sha256=(),
            registry_append_only_violation=False,
        )

    deletion_section_present = bool(_DELETION_HEADER_PATTERN.search(body))
    sha_matches = _SHA256_PATTERN.findall(body)
    # De-duplicate while preserving order — the audit log is
    # canonical and SHAs already appear unique, but this guards
    # against a future log that re-quotes a SHA in the corrective-
    # action narrative.
    seen: set[str] = set()
    distinct_shas: list[str] = []
    for s in sha_matches:
        if s not in seen:
            seen.add(s)
            distinct_shas.append(s)

    violation = deletion_section_present or bool(distinct_shas)
    return RegistryAuditState(
        audit_log_path=path_str,
        audit_log_present=True,
        stale_v030_deleted_during_this_session=deletion_section_present,
        lost_sha_count=len(distinct_shas),
        lost_sha256=tuple(distinct_shas),
        registry_append_only_violation=violation,
    )

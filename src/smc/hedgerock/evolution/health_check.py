"""P2 — Self-healing health-check tree for the evolution sidecar.

Subsystem checks live as ``check_*`` functions; each returns a
:class:`SubsystemReport`. :func:`diagnose` aggregates everything
into a :class:`HealthReport`. :func:`auto_recover` applies the
SAFE recovery actions (no destructive operations — only "create
missing dir", "seed an empty append-only file", etc.).

Pure data-side: this module never imports ``rule_engine`` or the
Tier-1 unsealed prod modules. All paths come from arguments — no
environment lookups inside the helpers themselves so the operator
can audit which paths were checked.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


__all__ = [
    "HealthReport",
    "HealthStatus",
    "RecoveryAction",
    "RecoveryResult",
    "SubsystemReport",
    "auto_recover",
    "check_audit_log",
    "check_calibrator_state",
    "check_config",
    "check_ledger",
    "check_queue",
    "check_registry",
    "diagnose",
]


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    DOWN = "DOWN"


_RANK = {
    HealthStatus.HEALTHY: 0,
    HealthStatus.DEGRADED: 1,
    HealthStatus.CRITICAL: 2,
    HealthStatus.DOWN: 3,
}


def _rank(s: HealthStatus) -> int:
    return _RANK[s]


@dataclass(frozen=True)
class SubsystemReport:
    name: str
    status: HealthStatus
    details: str
    suggested_recovery: str = ""
    auto_recoverable: bool = False


@dataclass(frozen=True)
class HealthReport:
    overall: HealthStatus
    subsystems: tuple[SubsystemReport, ...]
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def from_subsystems(
        cls, subsystems: Iterable[SubsystemReport],
    ) -> "HealthReport":
        items = tuple(subsystems)
        if not items:
            return cls(overall=HealthStatus.HEALTHY, subsystems=())
        worst = max(items, key=lambda r: _rank(r.status)).status
        return cls(overall=worst, subsystems=items)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecoveryAction:
    subsystem: str
    description: str
    succeeded: bool
    error: str = ""


@dataclass(frozen=True)
class RecoveryResult:
    actions: tuple[RecoveryAction, ...]
    post_status: HealthStatus

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-subsystem checks
# ---------------------------------------------------------------------------


def check_registry(*, registry_root: Path) -> SubsystemReport:
    """Operator-team registry root. Missing → DEGRADED on a fresh
    machine, NOT critical (the sidecar can run without it; the
    real-registry-smoke tests skip)."""
    p = Path(registry_root)
    if not p.exists():
        return SubsystemReport(
            name="registry",
            status=HealthStatus.DEGRADED,
            details=f"registry root absent at {p}",
            suggested_recovery=(
                "Set $HEDGEROCK_HOME or run with the operator-team "
                "registry checked out."
            ),
            auto_recoverable=False,
        )
    if not p.is_dir():
        return SubsystemReport(
            name="registry", status=HealthStatus.CRITICAL,
            details=f"registry path is not a directory: {p}",
            suggested_recovery="Remove the file and recreate as a directory.",
        )
    shadow = p / "shadow_artefacts"
    if not shadow.exists():
        return SubsystemReport(
            name="registry", status=HealthStatus.DEGRADED,
            details=(
                f"registry root present but shadow_artefacts/ missing "
                f"at {shadow}"
            ),
            suggested_recovery=(
                "Create shadow_artefacts/ — the sidecar's append-only "
                "data dir."
            ),
            auto_recoverable=True,
        )
    return SubsystemReport(
        name="registry", status=HealthStatus.HEALTHY,
        details=f"registry root + shadow_artefacts present at {p}",
    )


def check_audit_log(*, audit_log_path: Path) -> SubsystemReport:
    p = Path(audit_log_path)
    if not p.exists():
        return SubsystemReport(
            name="audit", status=HealthStatus.DEGRADED,
            details=f"audit log absent at {p}",
            suggested_recovery=(
                "Either point --registry-audit-log at the operator's "
                "real audit log, or accept that the registry-audit "
                "guard runs in 'absent ≠ clean' mode."
            ),
            auto_recoverable=False,
        )
    try:
        body = p.read_text(encoding="utf-8")
    except OSError as e:
        return SubsystemReport(
            name="audit", status=HealthStatus.CRITICAL,
            details=f"audit log unreadable: {e}",
            suggested_recovery="Check filesystem permissions.",
        )
    if not body.strip():
        return SubsystemReport(
            name="audit", status=HealthStatus.DEGRADED,
            details=f"audit log present but empty at {p}",
            suggested_recovery=(
                "Treat as 'no recorded violations'. Operators MUST "
                "still verify the registry hasn't been deleted."
            ),
        )
    return SubsystemReport(
        name="audit", status=HealthStatus.HEALTHY,
        details=f"audit log present + non-empty ({len(body)} bytes)",
    )


def check_queue(*, queue_path: Path) -> SubsystemReport:
    p = Path(queue_path)
    if not p.exists():
        return SubsystemReport(
            name="queue", status=HealthStatus.DEGRADED,
            details=f"queue file absent at {p}",
            suggested_recovery=(
                "Run the recommend CLI to seed a queue, or create an "
                "empty file."
            ),
            auto_recoverable=True,
        )
    bad = 0
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        n += 1
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            bad += 1
    if bad:
        return SubsystemReport(
            name="queue", status=HealthStatus.CRITICAL,
            details=f"{bad}/{n} queue lines fail JSON parsing",
            suggested_recovery=(
                "Inspect manually. Append-only invariant means we "
                "never auto-fix — broken lines record an incident."
            ),
        )
    return SubsystemReport(
        name="queue", status=HealthStatus.HEALTHY,
        details=f"queue parses cleanly ({n} entries)",
    )


def check_ledger(*, ledger_path: Path) -> SubsystemReport:
    p = Path(ledger_path)
    if not p.exists():
        return SubsystemReport(
            name="ledger", status=HealthStatus.DEGRADED,
            details=f"paper-test ledger absent at {p}",
            suggested_recovery=(
                "Empty ledger is fine for a fresh checkout — the "
                "demo seeds one when --seed-paper-trades is set."
            ),
            auto_recoverable=True,
        )
    bad = 0
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        n += 1
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            bad += 1
    if bad:
        return SubsystemReport(
            name="ledger", status=HealthStatus.CRITICAL,
            details=f"{bad}/{n} ledger lines fail JSON parsing",
            suggested_recovery="Fix manually — append-only contract.",
        )
    return SubsystemReport(
        name="ledger", status=HealthStatus.HEALTHY,
        details=f"ledger parses cleanly ({n} entries)",
    )


def check_config(*, config_path: Path | None) -> SubsystemReport:
    if config_path is None:
        return SubsystemReport(
            name="config", status=HealthStatus.HEALTHY,
            details="no config path supplied; relying on argparse defaults",
        )
    p = Path(config_path)
    if not p.exists():
        return SubsystemReport(
            name="config", status=HealthStatus.DEGRADED,
            details=f"config file missing at {p}",
            suggested_recovery=(
                "Drop in a safety_bounds template or use the bundled "
                "config/safety_bounds_template.yaml."
            ),
        )
    try:
        body = p.read_text(encoding="utf-8")
    except OSError as e:
        return SubsystemReport(
            name="config", status=HealthStatus.CRITICAL,
            details=f"config unreadable: {e}",
        )
    if not body.strip():
        return SubsystemReport(
            name="config", status=HealthStatus.DEGRADED,
            details=f"config present but empty: {p}",
        )
    return SubsystemReport(
        name="config", status=HealthStatus.HEALTHY,
        details=f"config present + non-empty ({len(body)} bytes)",
    )


def check_calibrator_state(*, state_path: Path | None) -> SubsystemReport:
    if state_path is None:
        return SubsystemReport(
            name="calibrator", status=HealthStatus.HEALTHY,
            details="no calibrator state requested; uninformative prior in use",
        )
    p = Path(state_path)
    if not p.exists():
        return SubsystemReport(
            name="calibrator", status=HealthStatus.DEGRADED,
            details=f"calibrator state file absent at {p}",
            suggested_recovery=(
                "Will fall back to uninformative Beta(1, 1) priors. "
                "Run a paper-test cycle to seed real posteriors."
            ),
        )
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return SubsystemReport(
            name="calibrator", status=HealthStatus.CRITICAL,
            details=f"calibrator state unparseable: {e}",
            suggested_recovery="Move it aside; uninformative prior will reseed.",
        )
    schema = obj.get("schema") if isinstance(obj, dict) else None
    if schema != "bayesian_calibrator/v1":
        return SubsystemReport(
            name="calibrator", status=HealthStatus.DEGRADED,
            details=f"unknown calibrator schema: {schema}",
        )
    return SubsystemReport(
        name="calibrator", status=HealthStatus.HEALTHY,
        details=f"calibrator state parses ({len(obj.get('priors', []))} priors)",
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def diagnose(
    *,
    registry_root: Path | None = None,
    audit_log_path: Path | None = None,
    queue_path: Path | None = None,
    ledger_path: Path | None = None,
    config_path: Path | None = None,
    calibrator_state_path: Path | None = None,
) -> HealthReport:
    """Run every supplied check. Subsystems with ``None`` paths skip
    their check entirely (so an operator can scope diagnose() to the
    surfaces they actually use)."""
    subs: list[SubsystemReport] = []
    if registry_root is not None:
        subs.append(check_registry(registry_root=registry_root))
    if audit_log_path is not None:
        subs.append(check_audit_log(audit_log_path=audit_log_path))
    if queue_path is not None:
        subs.append(check_queue(queue_path=queue_path))
    if ledger_path is not None:
        subs.append(check_ledger(ledger_path=ledger_path))
    subs.append(check_config(config_path=config_path))
    subs.append(check_calibrator_state(state_path=calibrator_state_path))
    return HealthReport.from_subsystems(subs)


# ---------------------------------------------------------------------------
# Auto-recovery — only the safe paths.
# ---------------------------------------------------------------------------


def auto_recover(
    report: HealthReport,
    *,
    registry_root: Path | None = None,
    queue_path: Path | None = None,
    ledger_path: Path | None = None,
) -> RecoveryResult:
    """Best-effort safe recovery. Touch missing files/dirs that the
    rest of the sidecar would otherwise fail to open. NEVER
    overwrites existing content."""
    actions: list[RecoveryAction] = []
    for sub in report.subsystems:
        if not sub.auto_recoverable:
            continue
        if sub.name == "registry" and registry_root is not None:
            shadow = Path(registry_root) / "shadow_artefacts"
            try:
                shadow.mkdir(parents=True, exist_ok=True)
                actions.append(RecoveryAction(
                    subsystem="registry", succeeded=True,
                    description=f"created {shadow}",
                ))
            except OSError as e:
                actions.append(RecoveryAction(
                    subsystem="registry", succeeded=False,
                    description="create shadow_artefacts dir",
                    error=str(e),
                ))
        elif sub.name == "queue" and queue_path is not None:
            qp = Path(queue_path)
            try:
                qp.parent.mkdir(parents=True, exist_ok=True)
                if not qp.exists():
                    qp.touch()
                actions.append(RecoveryAction(
                    subsystem="queue", succeeded=True,
                    description=f"seeded empty queue at {qp}",
                ))
            except OSError as e:
                actions.append(RecoveryAction(
                    subsystem="queue", succeeded=False,
                    description="seed empty queue",
                    error=str(e),
                ))
        elif sub.name == "ledger" and ledger_path is not None:
            lp = Path(ledger_path)
            try:
                lp.parent.mkdir(parents=True, exist_ok=True)
                if not lp.exists():
                    lp.touch()
                actions.append(RecoveryAction(
                    subsystem="ledger", succeeded=True,
                    description=f"seeded empty ledger at {lp}",
                ))
            except OSError as e:
                actions.append(RecoveryAction(
                    subsystem="ledger", succeeded=False,
                    description="seed empty ledger",
                    error=str(e),
                ))
    # Re-diagnose to get post status.
    post = diagnose(
        registry_root=registry_root,
        queue_path=queue_path,
        ledger_path=ledger_path,
    )
    return RecoveryResult(actions=tuple(actions), post_status=post.overall)

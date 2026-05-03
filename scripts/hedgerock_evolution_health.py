"""HedgeRock evolution sidecar — health-check CLI.

Prints a markdown table of subsystem statuses. Exit codes:
  0 — overall HEALTHY
  1 — overall DEGRADED
  2 — overall CRITICAL
  3 — overall DOWN

Optional ``--auto-recover`` runs safe recovery actions (create
missing dirs, touch empty append-only files) and re-emits the
report.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from smc.hedgerock.evolution.health_check import (
    HealthStatus,
    auto_recover,
    diagnose,
)


_EXIT_BY_STATUS = {
    HealthStatus.HEALTHY: 0,
    HealthStatus.DEGRADED: 1,
    HealthStatus.CRITICAL: 2,
    HealthStatus.DOWN: 3,
}


def _hedgerock_home() -> Path:
    raw = os.environ.get("HEDGEROCK_HOME")
    return Path(raw).expanduser() if raw else Path.home() / "HedgeRock"


def _print_report(report) -> None:
    print(f"## HedgeRock evolution health — overall = {report.overall.value}")
    print()
    print("| Subsystem | Status | Details |")
    print("|---|---|---|")
    for sub in report.subsystems:
        print(f"| {sub.name} | {sub.status.value} | {sub.details} |")
    print()
    suggestions = [s for s in report.subsystems if s.suggested_recovery]
    if suggestions:
        print("### Suggested recovery")
        print()
        for s in suggestions:
            print(f"- **{s.name}**: {s.suggested_recovery}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_registry = _hedgerock_home() / "policy_registry"
    default_audit = default_registry / "shadow_artefacts" / "_audit.md"
    parser.add_argument("--registry-root", type=Path, default=default_registry)
    parser.add_argument("--audit-log", type=Path, default=default_audit)
    parser.add_argument("--queue-path", type=Path, default=None)
    parser.add_argument("--ledger-path", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--calibrator-state", type=Path, default=None)
    parser.add_argument("--auto-recover", action="store_true")
    args = parser.parse_args(argv)

    report = diagnose(
        registry_root=args.registry_root,
        audit_log_path=args.audit_log,
        queue_path=args.queue_path,
        ledger_path=args.ledger_path,
        config_path=args.config,
        calibrator_state_path=args.calibrator_state,
    )
    _print_report(report)

    if args.auto_recover:
        result = auto_recover(
            report,
            registry_root=args.registry_root,
            queue_path=args.queue_path,
            ledger_path=args.ledger_path,
        )
        print("### Auto-recovery actions")
        print()
        if not result.actions:
            print("- (no auto-recoverable subsystems)")
        else:
            for a in result.actions:
                tag = "OK" if a.succeeded else f"FAIL ({a.error})"
                print(f"- [{a.subsystem}] {a.description} — {tag}")
        print()
        print(f"### Post-recovery overall = {result.post_status.value}")
        return _EXIT_BY_STATUS.get(result.post_status, 3)

    return _EXIT_BY_STATUS.get(report.overall, 3)


if __name__ == "__main__":
    sys.exit(main())

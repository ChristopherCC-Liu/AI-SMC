"""HedgeRock SASL daily-cycle CLI (REPORT-ONLY by default).

Wraps :class:`smc.hedgerock.evolution.sasl_orchestrator.SASLOrchestrator`
with a thin operator-facing CLI. The default invocation runs in
**propose-only** mode — drift is detected, purification archives stale
artefacts, but parameter adjustments are merely listed in the report.
``--apply-adjustments`` arms the auto-adjuster (still gated by the
circuit breaker).

Usage::

    python scripts/hedgerock_evolution_sasl.py \\
        --workspace /tmp/sasl \\
        [--apply-adjustments] \\
        [--breaker-state /tmp/sasl/cb_state.jsonl]

Exit codes::

    0 — cycle ran (regardless of drift severity)
    2 — workspace under a forbidden parent
    3 — fatal stage exception (auto-adjust raised on FORBIDDEN_PARAMS)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _ai_smc_home() -> Path:
    raw = os.environ.get("AI_SMC_HOME")
    return Path(raw).expanduser() if raw else Path(__file__).resolve().parents[1]


def _default_workspace() -> Path:
    return _ai_smc_home() / "tmp" / "sasl" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workspace", type=Path, default=_default_workspace())
    parser.add_argument(
        "--apply-adjustments", action="store_true",
        help="Arm AutoAdjuster (default: propose only).",
    )
    parser.add_argument(
        "--breaker-state", type=Path, default=None,
        help="JSONL file persisting the circuit-breaker history.",
    )
    parser.add_argument(
        "--breaker-max-adjustments", type=int, default=4,
        help="Max adjustments inside the breaker window (default 4).",
    )
    parser.add_argument(
        "--breaker-window-days", type=int, default=7,
        help="Rolling window length in days (default 7).",
    )
    parser.add_argument(
        "--event", default=None,
        help="When set, runs run_event_triggered(<event>, ...) instead.",
    )
    args = parser.parse_args(argv)

    try:
        from smc.hedgerock.evolution.sasl_circuit_breaker import (
            SASLCircuitBreaker,
        )
        from smc.hedgerock.evolution.sasl_orchestrator import (
            SASLOrchestrator,
        )
    except Exception as e:
        print(f"FAILED: SASL imports broken: {e!r}", file=sys.stderr)
        return 3

    breaker = None
    if args.breaker_state is not None:
        breaker = SASLCircuitBreaker(
            max_adjustments=args.breaker_max_adjustments,
            window_days=args.breaker_window_days,
            state_path=args.breaker_state,
        )

    try:
        orch = SASLOrchestrator(
            workspace=args.workspace,
            circuit_breaker=breaker,
            apply_adjustments=args.apply_adjustments,
        )
    except ValueError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 2

    if args.event:
        report = orch.run_event_triggered(event_type=args.event)
    else:
        report = orch.run_daily_cycle()

    print("== SASL cycle complete ==")
    print(f"  workspace: {orch.workspace}")
    print(f"  trigger:   {report.trigger}")
    print(f"  health:    {report.health_overall}")
    print(f"  drift:     {report.drift_overall_severity}")
    print(f"  archived:  {report.purified_archived_total}")
    print(
        f"  proposals: applied={report.proposals_applied} "
        f"rejected={report.proposals_rejected}"
    )
    if report.circuit_breaker_frozen:
        print(f"  breaker:   FROZEN — {report.circuit_breaker_reason}")
    print("  status: NOT LIVE / NOT APPROVED / NOT DEPLOYED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

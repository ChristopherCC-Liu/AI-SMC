"""Stage 6-followup-4 task 3 — metrics dashboard JSON export.

Read-only snapshot of the report-only self-evolution loop in JSON
(``metrics/v0`` schema). Designed for ingestion by Grafana / any
generic visualisation tool — the layout is stable and small.

Reads:
  - shadow-test queue (`--queue-path`)
  - paper-test ledger (`--paper-test-ledger`)
  - registry-audit log (`--registry-audit-log`)
  - shadow_artefacts root (`--shadow-artefacts-root`)

Writes:
  - one JSON snapshot at `--out`. Refuses to write under
    ``policy_registry/approved/`` or ``policy_registry/pointer.json``.

Never mutates the input files. Never imports live runtime modules.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
from smc.hedgerock.evolution.paper_test_ledger import PaperTestLedger
from smc.hedgerock.evolution.registry_audit import load_registry_audit_state
from smc.hedgerock.evolution.replay_validator import summarise_replay


_FORBIDDEN_PATH_FRAGMENTS = (
    "policy_registry/approved",
    "policy_registry/pointer.json",
)


def _assert_out_path_safe(path: Path) -> None:
    text = str(path)
    for fragment in _FORBIDDEN_PATH_FRAGMENTS:
        if fragment in text:
            raise ValueError(
                f"--out lands under a forbidden location: {text!r} "
                f"(matched {fragment!r})"
            )


def _tally_queue(queue_path: Path) -> dict[str, int]:
    if not queue_path.exists():
        return {"entries_total": 0, "queued": 0, "stale": 0, "other": 0}
    queued = stale = other = 0
    total = 0
    for ln in queue_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        total += 1
        s = d.get("status", "")
        if s == "QUEUED":
            queued += 1
        elif s == "STALE":
            stale += 1
        else:
            other += 1
    return {
        "entries_total": total, "queued": queued,
        "stale": stale, "other": other,
    }


def _shadow_artefacts_summary(
    *, shadow_root: Path,
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for c in CANDIDATE_MENU_V0:
        if not shadow_root.exists():
            out[c.candidate_id] = {"n_artefacts": 0, "n_windows": 0}
            continue
        report = summarise_replay(
            candidate_id=c.candidate_id,
            shadow_artefacts_root=shadow_root,
        )
        out[c.candidate_id] = {
            "n_artefacts": int(report.n_artefacts_read),
            "n_windows": int(report.n_windows_replayed),
        }
    return out


def _paper_test_summary(
    *, ledger_path: Path, audit_log_path: Path,
) -> dict[str, Any]:
    if not ledger_path.exists():
        return {"candidates": {}}
    led = PaperTestLedger(path=ledger_path, audit_log_path=audit_log_path)
    summary = led.summarise()
    # Cast to JSON-safe primitives.
    cleaned: dict[str, dict[str, float]] = {}
    for cid, body in summary.items():
        cleaned[cid] = {
            "trades": int(body.get("trades", 0)),
            "pnl_sum": float(body.get("pnl_sum", 0.0)),
            "max_drawdown": float(body.get("max_drawdown", 0.0)),
        }
    return {"candidates": cleaned}


def build_snapshot(
    *,
    queue_path: Path,
    ledger_path: Path,
    audit_log_path: Path,
    shadow_root: Path,
) -> dict[str, Any]:
    audit_state = load_registry_audit_state(audit_log_path)
    snap: dict[str, Any] = {
        "schema": "metrics/v0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry_audit": {
            "audit_log_path": str(getattr(audit_state, "audit_log_path", audit_log_path)),
            "audit_log_present": bool(audit_state.audit_log_present),
            "registry_append_only_violation": bool(
                audit_state.registry_append_only_violation
            ),
            "lost_sha_count": int(audit_state.lost_sha_count or 0),
        },
        "queue": _tally_queue(Path(queue_path)),
        "paper_test": _paper_test_summary(
            ledger_path=Path(ledger_path),
            audit_log_path=Path(audit_log_path),
        ),
        "shadow_artefacts": _shadow_artefacts_summary(
            shadow_root=Path(shadow_root),
        ),
        "candidates": [c.candidate_id for c in CANDIDATE_MENU_V0],
    }
    return snap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-path", type=Path, required=True)
    parser.add_argument("--paper-test-ledger", type=Path, required=True)
    parser.add_argument("--registry-audit-log", type=Path, required=True)
    parser.add_argument("--shadow-artefacts-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        _assert_out_path_safe(Path(args.out))
    except ValueError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    snap = build_snapshot(
        queue_path=args.queue_path,
        ledger_path=args.paper_test_ledger,
        audit_log_path=args.registry_audit_log,
        shadow_root=args.shadow_artefacts_root,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(snap, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote metrics snapshot → {args.out}")
    print(f"  schema: {snap['schema']}")
    print(f"  queue: {snap['queue']}")
    print(f"  registry_append_only_violation: "
          f"{snap['registry_audit']['registry_append_only_violation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

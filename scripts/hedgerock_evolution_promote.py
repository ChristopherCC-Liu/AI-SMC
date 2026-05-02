"""Stage 6-followup task 4 — human promotion helper (DRY-RUN ONLY).

This CLI never promotes anything. It produces a "promotion packet"
markdown document that bundles all of the evidence a human operator
needs to manually copy a candidate into
``policy_registry/approved/`` and update
``policy_registry/pointer.json`` themselves.

There is no ``--apply`` mode — passing ``--apply`` aborts the run
with a message naming the manual escalation path.

Pre-requisites checked before producing a packet:
  1. queue entry exists for the named candidate
  2. paper-test ledger contains ≥ ``--min-paper-trades`` (default 20)
     trades for the candidate, with aggregate ``pnl_sum > 0`` and
     ``max_drawdown >= --paper-drawdown-floor`` (default -50.0)
  3. registry-audit log records no append-only violation
  4. operator passed the literal confirmation sentinel via
     ``--operator-confirmation``

Any prerequisite failure → exit code 1, no packet written.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from smc.hedgerock.evolution.paper_test_ledger import PaperTestLedger
from smc.hedgerock.evolution.registry_audit import load_registry_audit_state


CONFIRMATION_SENTINEL = (
    "I-have-reviewed-the-shadow-evidence-and-paper-test-ledger"
)

_FORBIDDEN_PATH_FRAGMENTS = (
    "policy_registry/approved",
    "policy_registry/pointer.json",
    "config/safety_bounds.yaml",
)

_DEFAULT_MIN_PAPER_TRADES = 20
_DEFAULT_DRAWDOWN_FLOOR = -50.0


def _assert_packet_path_safe(path: Path) -> None:
    text = str(path)
    for fragment in _FORBIDDEN_PATH_FRAGMENTS:
        if fragment in text:
            raise ValueError(
                f"packet path lands under a forbidden location: "
                f"{text!r} (matched {fragment!r})"
            )


def _read_queue_entry(*, queue_path: Path, candidate_id: str) -> dict | None:
    if not queue_path.exists():
        return None
    for ln in queue_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if d.get("candidate_id") == candidate_id:
            return d
    return None


def _render_packet(
    *,
    candidate_id: str,
    queue_entry: dict,
    paper_summary: dict,
    audit_log_path: Path,
    audit_present: bool,
    audit_violation: bool,
    min_paper_trades: int,
    drawdown_floor: float,
) -> str:
    out: list[str] = []
    out.append("# HedgeRock Promotion Packet (DRY-RUN — manual approval required)")
    out.append("")
    out.append("> This packet is **NOT LIVE**, **NOT APPROVED**, "
               "**NOT DEPLOYED**. The CLI that produced it does not "
               "write to `policy_registry/approved/` or "
               "`policy_registry/pointer.json`. Promotion is a "
               "**manual** step the human operator carries out after "
               "reading this packet end-to-end.")
    out.append("")
    out.append("## Candidate")
    out.append("")
    out.append(f"- candidate_id: `{candidate_id}`")
    out.append(f"- parameter target: `{queue_entry.get('parameter_target', '?')}`")
    out.append(f"- parameter class: `{queue_entry.get('parameter_class', '?')}`")
    out.append(f"- baseline → proposed: "
               f"`{queue_entry.get('baseline_value', '?')}` → "
               f"`{queue_entry.get('proposed_value', '?')}`")
    out.append(f"- queue status: `{queue_entry.get('status', '?')}`")
    out.append(f"- queued_at: `{queue_entry.get('queued_at', '?')}`")
    out.append("")
    out.append("## Paper-test summary")
    out.append("")
    out.append(f"- trades: **{paper_summary.get('trades', 0)}** "
               f"(threshold: ≥ {min_paper_trades})")
    out.append(f"- pnl_sum: **{paper_summary.get('pnl_sum', 0.0)}** "
               f"(must be > 0)")
    out.append(f"- max_drawdown: **{paper_summary.get('max_drawdown', 0.0)}** "
               f"(floor: ≥ {drawdown_floor})")
    out.append("")
    out.append("## Registry audit")
    out.append("")
    out.append(f"- audit_log_path: `{audit_log_path}`")
    out.append(f"- audit_log_present: **{audit_present}**")
    out.append(f"- registry_append_only_violation: **{audit_violation}**")
    out.append("")
    out.append("## Manual escalation steps (human only)")
    out.append("")
    out.append("1. Re-read the recommendation report and the "
               "diagnostic evolution report linked from the queue "
               "entry.")
    out.append("2. Spot-check at least 3 paper-test trades against "
               "the live order book records.")
    out.append("3. Diff the proposed value against the current live "
               "constant in the rule_engine module (read-only).")
    out.append("4. If approving: manually copy the candidate manifest "
               "into `policy_registry/approved/<candidate_id>.json` "
               "(no automation handles this step) and update "
               "`policy_registry/pointer.json` with the new approved "
               "candidate id.")
    out.append("5. Append a line to the registry audit log naming the "
               "approving operator and the timestamp.")
    out.append("")
    out.append("## Boundary boilerplate")
    out.append("")
    out.append("- this CLI never wrote `policy_registry/approved/`")
    out.append("- this CLI never wrote `policy_registry/pointer.json`")
    out.append("- this CLI never wrote any module under "
               "`src/smc/hedgerock/`")
    out.append("- this CLI never modified `config/safety_bounds.yaml`")
    out.append("")
    out.append(f"Generated at: {datetime.now(timezone.utc).isoformat()}")
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--queue-path", type=Path, required=True)
    parser.add_argument("--paper-test-ledger", type=Path, required=True)
    parser.add_argument("--registry-audit-log", type=Path, required=True)
    parser.add_argument("--packet-path", type=Path, required=True)
    parser.add_argument("--operator-confirmation", default="")
    parser.add_argument("--min-paper-trades", type=int,
                        default=_DEFAULT_MIN_PAPER_TRADES)
    parser.add_argument("--paper-drawdown-floor", type=float,
                        default=_DEFAULT_DRAWDOWN_FLOOR)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    if args.apply:
        print(
            "FAILED: --apply is unavailable. Promotion is a manual "
            "human step. Re-run without --apply to produce a "
            "dry-run packet, then perform the manual escalation "
            "steps printed inside it.",
            file=sys.stderr,
        )
        return 1

    if args.operator_confirmation != CONFIRMATION_SENTINEL:
        print(
            "FAILED: --operator-confirmation must equal the sentinel "
            f"phrase. Set it to the literal value documented in the "
            f"helper script (CONFIRMATION_SENTINEL).",
            file=sys.stderr,
        )
        return 1

    try:
        _assert_packet_path_safe(Path(args.packet_path))
    except ValueError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    queue_entry = _read_queue_entry(
        queue_path=Path(args.queue_path), candidate_id=args.candidate_id,
    )
    if queue_entry is None:
        print(
            f"FAILED: candidate {args.candidate_id!r} not found in "
            f"queue {args.queue_path!s}",
            file=sys.stderr,
        )
        return 1

    # Paper-test summary.
    audit_log_path = Path(args.registry_audit_log)
    ledger = PaperTestLedger(
        path=Path(args.paper_test_ledger), audit_log_path=audit_log_path,
    )
    summary = ledger.summarise()
    cand_summary = summary.get(args.candidate_id, {
        "trades": 0, "pnl_sum": 0.0, "max_drawdown": 0.0,
    })
    if cand_summary["trades"] < args.min_paper_trades:
        print(
            f"FAILED: paper-test trades {cand_summary['trades']} < "
            f"min {args.min_paper_trades}",
            file=sys.stderr,
        )
        return 1
    if cand_summary["pnl_sum"] <= 0:
        print(
            f"FAILED: paper-test pnl_sum {cand_summary['pnl_sum']} "
            f"must be > 0",
            file=sys.stderr,
        )
        return 1
    if cand_summary["max_drawdown"] < args.paper_drawdown_floor:
        print(
            f"FAILED: paper-test max_drawdown {cand_summary['max_drawdown']} "
            f"breached floor {args.paper_drawdown_floor}",
            file=sys.stderr,
        )
        return 1

    # Registry-audit state.
    audit_state = load_registry_audit_state(audit_log_path)
    if audit_state.registry_append_only_violation:
        print(
            "FAILED: registry append-only violation detected in "
            f"{audit_log_path}; cannot produce a promotion packet "
            "until a clean session is restarted",
            file=sys.stderr,
        )
        return 1

    body = _render_packet(
        candidate_id=args.candidate_id,
        queue_entry=queue_entry,
        paper_summary=cand_summary,
        audit_log_path=audit_log_path,
        audit_present=audit_state.audit_log_present,
        audit_violation=audit_state.registry_append_only_violation,
        min_paper_trades=args.min_paper_trades,
        drawdown_floor=args.paper_drawdown_floor,
    )
    Path(args.packet_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.packet_path).write_text(body, encoding="utf-8")
    print(f"wrote promotion packet → {args.packet_path}")
    print("Remember: this packet is DRY-RUN. The CLI did not write to "
          "policy_registry/approved/ or pointer.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Stage 6-followup task 2 — shadow-test queue inspector (read-only).

Operator-facing CLI that reads the JSONL queue produced by Stage 5
(:class:`ShadowTestQueue`) and renders a markdown summary.

Read-only by design:

  * Never opens the queue file in append/write mode.
  * Never writes under ``policy_registry/approved/`` or
    ``policy_registry/pointer.json``.
  * Never imports the live trading runtime.

Exit codes:
  * 0 — report rendered (queue may be empty or missing; that is
        surfaced explicitly in the report body).
  * 2 — bad argv.
  * 3 — report path lands under a forbidden registry location.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


_FORBIDDEN_PATH_FRAGMENTS = (
    "policy_registry/approved",
    "policy_registry/pointer.json",
)


def _assert_report_path_safe(path: Path) -> None:
    text = str(path)
    for fragment in _FORBIDDEN_PATH_FRAGMENTS:
        if fragment in text:
            raise ValueError(
                f"report path would write under a forbidden location: "
                f"{text!r} (matched {fragment!r})"
            )


def _read_queue(queue_path: Path) -> list[dict]:
    if not queue_path.exists():
        return []
    out: list[dict] = []
    for ln in queue_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def _render_entry(entry: dict) -> list[str]:
    out: list[str] = []
    out.append(f"### {entry.get('candidate_id', '?')}")
    out.append("")
    out.append(f"- parameter target: `{entry.get('parameter_target', '?')}`")
    out.append(f"- parameter class: `{entry.get('parameter_class', '?')}`")
    out.append(
        f"- baseline → proposed: `{entry.get('baseline_value', '?')}` → "
        f"`{entry.get('proposed_value', '?')}`"
    )
    out.append(f"- status: `{entry.get('status', '?')}`")
    out.append(f"- queued_at: `{entry.get('queued_at', '?')}`")
    out.append(f"- required_windows: **{entry.get('required_windows', '?')}**")
    out.append("- required_tests:")
    for t in entry.get("required_tests", ()):
        out.append(f"    - {t}")
    out.append("- blocking_conditions:")
    for b in entry.get("blocking_conditions", ()):
        out.append(f"    - {b}")
    if entry.get("reason"):
        out.append(f"- reason: `{entry['reason']}`")
    out.append(f"- audit_log_path: `{entry.get('audit_log_path', '?')}`")
    out.append("")
    return out


def _render_report(*, queue_path: Path, entries: list[dict]) -> str:
    out: list[str] = []
    out.append("# HedgeRock Shadow-Test Queue — Inspection Report (READ-ONLY)")
    out.append("")
    out.append("> Read-only snapshot of the append-only shadow-test "
               "queue. Promotion remains a manual human step downstream "
               "of paper-test validation.")
    out.append("")
    out.append("## Banner block (machine-greppable)")
    out.append("")
    out.append("- status: **NOT LIVE**")
    out.append("- approval: **NOT APPROVED**")
    out.append("- deployment: **NOT DEPLOYED**")
    out.append("")
    out.append("## Source")
    out.append("")
    out.append(f"- queue_path: `{queue_path}`")
    if not queue_path.exists():
        out.append("- status: **queue not found** (no queue file at the path)")
    elif not entries:
        out.append("- status: **queue empty** (file present, no entries)")
    else:
        out.append(f"- status: **{len(entries)} entr"
                   f"{'y' if len(entries) == 1 else 'ies'}**")
    out.append("")
    out.append("## Entries")
    out.append("")
    if not entries:
        out.append("(none)")
        out.append("")
    else:
        for e in entries:
            out.extend(_render_entry(e))
    out.append("## Boundary boilerplate")
    out.append("")
    out.append("- this CLI never modifies the queue file")
    out.append("- this CLI never writes under `policy_registry/approved/`, "
               "`policy_registry/pointer.json`, `config/safety_bounds.yaml`, "
               "or live source modules")
    out.append("- queued candidates remain QUEUED until a paper-test pass "
               "+ manual human approval moves them forward")
    out.append("")
    out.append(f"Generated at: {datetime.now(timezone.utc).isoformat()}")
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--queue-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        _assert_report_path_safe(Path(args.report_path))
    except ValueError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 3

    queue_path = Path(args.queue_path)
    if not queue_path.exists():
        print(f"queue not found: {queue_path}")
    entries = _read_queue(queue_path)

    body = _render_report(queue_path=queue_path, entries=entries)
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).write_text(body, encoding="utf-8")

    print(f"queue entries: {len(entries)}")
    for e in entries:
        print(f"  {e.get('candidate_id', '?')}: {e.get('status', '?')}")
    print(f"wrote inspection report → {args.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

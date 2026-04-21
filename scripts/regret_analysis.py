"""R5 M3 CLI: per-trade regret analysis → data/reports/regret_YYYY-MM-DD.jsonl.

Usage
-----
::

    python scripts/regret_analysis.py
    python scripts/regret_analysis.py --date 2026-04-20
    python scripts/regret_analysis.py --symbols XAUUSD
    python scripts/regret_analysis.py --reports-root /tmp/regret

Writes one JSONL line per trade and a trailing summary line with deltas and
verdicts (guardrail_helped / guardrail_cost / neutral).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from smc.monitor.digest_report import (  # noqa: E402
    collect_journal_paths,
    scan_structured_events,
)
from smc.monitor.regret_analysis import (  # noqa: E402
    build_regret_records,
    count_anti_stack_blocks,
    extract_closures_by_ticket,
    load_day_trades,
    synthesize_unassigned_closures,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Round 5 M3 regret analysis.")
    p.add_argument("--date", default=None, help="UTC date YYYY-MM-DD (default: today UTC).")
    p.add_argument("--symbols", default="XAUUSD,BTCUSD")
    p.add_argument("--data-root", default="data")
    p.add_argument("--log-root", default="logs")
    p.add_argument("--reports-root", default="data/reports")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        target = (
            date.fromisoformat(args.date)
            if args.date else datetime.now(timezone.utc).date()
        )
    except ValueError:
        print(f"[regret_analysis] invalid --date: {args.date!r}", file=sys.stderr)
        return 2

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    data_root = Path(args.data_root)
    log_root = Path(args.log_root)
    reports_root = Path(args.reports_root)
    reports_root.mkdir(parents=True, exist_ok=True)

    journal_paths = list(collect_journal_paths(data_root, symbols=symbols).values())
    trades = load_day_trades(journal_paths, target)
    events = scan_structured_events(log_root, target)
    anti_stack_blocks = count_anti_stack_blocks(events)
    closures = extract_closures_by_ticket(events)

    # DELEGATED_TO_EA bridge: orphan closures (ticket not in any journal row
    # due to pre-execute write ordering) are synthesised as unassigned rows
    # so day-total PnL stays honest.
    trades = trades + synthesize_unassigned_closures(
        trades, closures, target_date=target,
    )

    records = build_regret_records(
        trades,
        anti_stack_blocks=anti_stack_blocks,
        closures_by_ticket=closures,
    )

    out_path = reports_root / f"regret_{target.isoformat()}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"[regret_analysis] wrote {out_path} ({len(records)} records)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""R5 M4 CLI: send the daily summary to Telegram via alert_critical.

Designed to be invoked by the Windows Scheduled Task
``AI-SMC-Daily-Summary`` at 00:05 UTC.  The 5-minute delay after midnight
gives rotating log files (structured.jsonl -> structured.jsonl.YYYY-MM-DD)
time to settle and ensures all end-of-day trade closes have landed.

Usage::

    python scripts/send_daily_summary.py                    # today UTC
    python scripts/send_daily_summary.py --date 2026-04-20
    python scripts/send_daily_summary.py --dry-run          # print only

Exit codes:
  0  message composed + dispatched (or logged with no Telegram creds)
  2  invalid arguments
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from smc.monitor.critical_alerter import alert_critical  # noqa: E402
from smc.monitor.daily_summary import format_daily_summary  # noqa: E402
from smc.monitor.digest_report import build_multi_symbol_digest  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R5 M4 daily Telegram summary.")
    p.add_argument(
        "--date", default=None,
        help="UTC date YYYY-MM-DD (default: yesterday UTC — intended for 00:05 UTC cron).",
    )
    p.add_argument("--symbols", default="XAUUSD,BTCUSD")
    p.add_argument("--data-root", default="data")
    p.add_argument("--log-root", default="logs")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the composed message to stdout and exit (no Telegram).",
    )
    return p.parse_args(argv)


def _default_target_date(now: datetime | None = None) -> date:
    """Default to *yesterday* UTC, since this runs at 00:05 UTC for "yesterday"."""
    now = now or datetime.now(timezone.utc)
    return (now - timedelta(hours=1)).date()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        target = (
            date.fromisoformat(args.date) if args.date else _default_target_date()
        )
    except ValueError:
        print(f"[send_daily_summary] invalid --date: {args.date!r}", file=sys.stderr)
        return 2

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    digest = build_multi_symbol_digest(
        target,
        data_root=Path(args.data_root),
        log_root=Path(args.log_root),
        symbols=symbols,
    )
    msg = format_daily_summary(digest)

    if args.dry_run:
        print(msg)
        return 0

    # alert_critical emits [CRIT] log + best-effort Telegram.  Pass the full
    # message via the dedicated "message" context key so the alerter doesn't
    # flatten it into a 1-line "k=v" dump.
    alert_critical(
        "daily_summary",
        send_telegram=True,
        date=target.isoformat(),
        body=msg,
    )
    print(f"[send_daily_summary] dispatched for {target.isoformat()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

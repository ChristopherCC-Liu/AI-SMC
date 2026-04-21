"""Round 5 M1 CLI: generate today's daily digest CSV + Markdown.

Usage
-----
::

    python scripts/build_daily_digest.py
    python scripts/build_daily_digest.py --date 2026-04-20
    python scripts/build_daily_digest.py --symbols XAUUSD,BTCUSD
    python scripts/build_daily_digest.py --data-root data --reports-root data/reports

Designed to be called by Windows Scheduled Task (or cron) at the end of the
trading day.  See also ``scripts/send_daily_summary.py`` (R5 M4) for the
Telegram blast that consumes the same digest.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

# Ensure src/ is importable when run from repo root.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from smc.monitor.digest_report import (  # noqa: E402
    build_multi_symbol_digest,
    write_digest_report,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the R5 daily digest.")
    parser.add_argument(
        "--date", default=None,
        help="UTC date YYYY-MM-DD (default: today UTC).",
    )
    parser.add_argument(
        "--symbols", default="XAUUSD,BTCUSD",
        help="Comma-separated symbol list (default: XAUUSD,BTCUSD).",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--reports-root", default="data/reports")
    parser.add_argument(
        "--print-json", action="store_true",
        help="Also print the digest JSON to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        target = (
            date.fromisoformat(args.date)
            if args.date
            else datetime.now(timezone.utc).date()
        )
    except ValueError:
        print(f"[build_daily_digest] Invalid --date: {args.date!r}", file=sys.stderr)
        return 2

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    data_root = Path(args.data_root)
    log_root = Path(args.log_root)
    reports_root = Path(args.reports_root)

    digest = build_multi_symbol_digest(
        target,
        data_root=data_root,
        log_root=log_root,
        symbols=symbols,
    )
    csv_path, md_path = write_digest_report(digest, reports_root=reports_root)

    print(f"[build_daily_digest] wrote {csv_path}", file=sys.stderr)
    print(f"[build_daily_digest] wrote {md_path}", file=sys.stderr)
    if args.print_json:
        print(json.dumps(digest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

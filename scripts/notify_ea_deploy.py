"""Send a Telegram notification after ``scripts/deploy_ea.ps1`` compiles the EA.

Thin wrapper around :func:`smc.monitor.critical_alerter.alert_critical` so the
PowerShell deploy script can fire one Telegram line without reimplementing bot
I/O in .ps1. Called by deploy_ea.ps1; also safe to run manually for testing.

Plain text only (parse_mode fixed 2026-04-20 — no HTML/Markdown).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the repo's src/ is importable when the script is invoked directly
# on the VPS (where the project is checked out to C:\AI-SMC but not
# installed as a package via pip).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _build_message(
    version: str,
    ex5_size_kb: float,
    compile_sec: float,
    include_count: int = 0,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    include_line = (
        f"Includes: {include_count} .mqh synced"
        if include_count > 0
        else "Includes: none"
    )
    lines = [
        f"[EA-DEPLOY] AISMCReceiver v{version} compiled",
        f"Size: {ex5_size_kb:.1f} KB  |  Compile: {compile_sec:.1f}s",
        include_line,
        f"When: {now}",
        "",
        "Action: detach + re-attach the EA on XAUUSD and BTCUSD charts.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post-deploy Telegram notifier for AISMCReceiver.")
    parser.add_argument("--version", required=True, help='EA #property version string, e.g. "2.01"')
    parser.add_argument("--ex5-size-kb", type=float, required=True, help="Compiled .ex5 size in KB")
    parser.add_argument("--compile-sec", type=float, required=True, help="Compile duration in seconds")
    parser.add_argument(
        "--include-count",
        type=int,
        default=0,
        help="Number of .mqh files synced to MQL5/Include/ (R7+; default 0 for backwards compat)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print message + skip Telegram")
    args = parser.parse_args(argv)

    message = _build_message(
        version=args.version,
        ex5_size_kb=args.ex5_size_kb,
        compile_sec=args.compile_sec,
        include_count=args.include_count,
    )

    if args.dry_run:
        print("[dry-run] would send Telegram message:")
        print(message)
        return 0

    # We use the structured logger + Telegram sender directly instead of
    # ``alert_critical`` because the latter formats Telegram output as
    # ``"[CRIT] event | k=v | ..."`` which would flatten our multi-line
    # body. The structured JSON line still lands in logs/structured.jsonl
    # so the audit trail is preserved.
    try:
        from smc.monitor.structured_log import log_event
    except ImportError as exc:
        print(f"[notify_ea_deploy] cannot import smc.monitor.structured_log: {exc}", file=sys.stderr)
        print(message)
        return 2

    log_event(
        "CRIT",
        "ea_deployed",
        version=args.version,
        ex5_size_kb=round(args.ex5_size_kb, 1),
        compile_sec=round(args.compile_sec, 1),
        include_count=args.include_count,
    )

    import os
    token = os.getenv("SMC_TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("SMC_TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[notify_ea_deploy] Telegram env vars missing — printed only")
        print(message)
        return 0

    try:
        import asyncio
        from smc.monitor.alerter import TelegramAlerter

        alerter = TelegramAlerter(bot_token=token, chat_id=chat_id)
        ok = asyncio.run(alerter.send_text(message))
        if ok:
            print(f"[notify_ea_deploy] Telegram OK (version={args.version})")
            return 0
        print("[notify_ea_deploy] Telegram send returned False (rate-limit or API error)", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"[notify_ea_deploy] Telegram send exception: {exc}", file=sys.stderr)
        print(message)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())

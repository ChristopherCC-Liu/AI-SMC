"""Stage 6-followup-2 task 2 — shadow-test queue aging CLI.

Runs an append-only aging pass over the queue: any QUEUED entry
older than ``--stale-after-days`` (default 14) gets a sibling STALE
marker line. Original entries are preserved byte-for-byte. The CLI
exposes NO removal flag.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from smc.hedgerock.evolution.queue_aging import (
    DEFAULT_STALE_AFTER_DAYS,
    mark_stale_entries,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-path", type=Path, required=True)
    parser.add_argument(
        "--stale-after-days", type=int, default=DEFAULT_STALE_AFTER_DAYS,
    )
    args = parser.parse_args(argv)

    try:
        appended = mark_stale_entries(
            queue_path=args.queue_path,
            stale_after_days=args.stale_after_days,
        )
    except ValueError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 3
    print(f"marked {len(appended)} stale entr"
          f"{'y' if len(appended) == 1 else 'ies'} "
          f"(threshold: {args.stale_after_days} days)")
    for e in appended:
        print(f"  {e['candidate_id']}: queued_at={e['queued_at']} "
              f"reason={e['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

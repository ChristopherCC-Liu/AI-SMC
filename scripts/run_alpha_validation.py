"""Run Phase 5.1 Alpha Validation over a XAUUSD time slice.

Usage:
    python scripts/run_alpha_validation.py --start 2024-01-01 --end 2024-12-31

Default behaviour uses synthetic positive-edge trades to verify the
6 acceptance criteria architecturally — production wire-up to a real
trade stream lands in Stage C / Stage E.

Exit code:
- 0 — every AC passed
- 1 — at least one AC failed (sentinel triggered or reverse_pf >= 1.0)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from smc.hedgerock.alpha_validation import (
    AlphaValidationConfig,
    format_validation_report,
    run_alpha_validation,
)


def _parse_iso_date(value: str) -> datetime:
    if "T" in value or " " in value:
        dt = datetime.fromisoformat(value)
    else:
        dt = datetime.fromisoformat(f"{value}T00:00:00+00:00")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", default="XAUUSD")
    parser.add_argument(
        "--start",
        type=_parse_iso_date,
        default=_parse_iso_date("2024-01-01"),
    )
    parser.add_argument(
        "--end",
        type=_parse_iso_date,
        default=_parse_iso_date("2024-12-31"),
    )
    parser.add_argument(
        "--lot-factor-mismatch",
        type=float,
        default=0.3,
        help="Lot factor applied when regime mismatch detected (default 0.3 = 30%)",
    )
    parser.add_argument(
        "--p99-budget-ms",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--min-cache-hit",
        type=float,
        default=0.90,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = AlphaValidationConfig(
        instrument=args.instrument,
        start=args.start,
        end=args.end,
        lot_factor_when_mismatch=args.lot_factor_mismatch,
        p99_latency_budget_ms=args.p99_budget_ms,
        min_cache_hit_rate=args.min_cache_hit,
    )
    result = run_alpha_validation(config)
    print(format_validation_report(result))
    return 0 if result.pass_all_criteria else 1


if __name__ == "__main__":
    raise SystemExit(main())

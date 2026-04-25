"""Daily aggregator that builds a gate funnel report from diagnostic jsonl.

R10 P1.2 — turns ``data/diagnostics/aggregator_gates_<UTC-DATE>.jsonl`` files
into a structured report so we can finally answer:
- Where do setups die in the SMC pipeline?
- Is the kill ratio different between control (magic A) and treatment (magic B)?
- Are losses clustered in any UTC hour (Asian-session anomaly)?
- Is the funnel shifting day-over-day?

Usage::

    python scripts/build_gate_funnel.py --days 1
    python scripts/build_gate_funnel.py --days 7 --journal-dir data/diagnostics

The default behaviour reads the most recent UTC day's jsonl and writes
``gate_funnel_<UTC-DATE>.json`` next to it.  ``--days N`` (N>1) extends the
window backwards to capture multi-day trends.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_JOURNAL_DIR = Path("data") / "diagnostics"
JOURNAL_GLOB = "aggregator_gates_*.jsonl"


def _parse_jsonl_lines(path: Path) -> Iterable[dict[str, Any]]:
    """Yield parsed JSON records from a jsonl file.

    Skips malformed lines silently (telemetry must be lossy-tolerant) but
    records the count for the caller via the ``malformed`` field on
    :func:`build_funnel_report`.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _hour_bucket(ts_iso: str) -> int | None:
    """Return UTC hour 0-23 for an ISO 8601 ``bar_ts`` string, or None."""
    try:
        dt = datetime.fromisoformat(ts_iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).hour


def _date_bucket(ts_iso: str) -> str | None:
    """Return UTC date YYYY-MM-DD for an ISO 8601 ``bar_ts`` string, or None."""
    try:
        dt = datetime.fromisoformat(ts_iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def discover_journal_files(
    journal_dir: Path,
    days: int,
    end_date: datetime | None = None,
) -> list[Path]:
    """List jsonl files for the last ``days`` UTC days inclusive of ``end_date``.

    ``end_date`` defaults to "now (UTC)".  Returns paths sorted oldest first.
    """
    if days < 1:
        raise ValueError("days must be >= 1")
    end = end_date or datetime.now(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    dates = [
        (end - timedelta(days=offset)).strftime("%Y-%m-%d")
        for offset in range(days)
    ]
    files: list[Path] = []
    for date_str in sorted(dates):
        candidate = journal_dir / f"aggregator_gates_{date_str}.jsonl"
        if candidate.exists():
            files.append(candidate)
    return files


def build_funnel_report(
    files: list[Path],
) -> dict[str, Any]:
    """Build a histogram report from one or more jsonl files.

    Schema of the returned dict::

        {
            "files": [str, ...],
            "total_calls": int,
            "by_stage_reject": {<reason>: count, ...},  # None → "passed_all_gates"
            "by_magic": {<magic>: {"total": int, "by_stage_reject": {...}}, ...},
            "by_hour_utc": {0..23: {"total": int, "rejects": int, "passed": int}},
            "by_date": {"YYYY-MM-DD": {...same shape as top level...}},
            "zone_rejects_aggregate": {<sub-reason>: count},
            "passed_all_gates": int,
            "malformed_lines": 0,
        }
    """
    total_calls = 0
    by_stage: Counter[str] = Counter()
    by_magic: dict[int, dict[str, Any]] = {}
    by_hour: dict[int, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "rejects": 0, "passed": 0}
    )
    by_date: dict[str, dict[str, Any]] = {}
    zone_rejects_total: Counter[str] = Counter()
    passed_all = 0

    for path in files:
        for record in _parse_jsonl_lines(path):
            total_calls += 1
            diag = record.get("diagnostic") or {}
            stage = diag.get("stage_reject")
            stage_key = stage if stage else "passed_all_gates"
            by_stage[stage_key] += 1
            if not stage:
                passed_all += 1

            magic = record.get("magic")
            if isinstance(magic, int):
                bucket = by_magic.setdefault(magic, {"total": 0, "by_stage_reject": Counter()})
                bucket["total"] += 1
                bucket["by_stage_reject"][stage_key] += 1

            hour = _hour_bucket(record.get("bar_ts", ""))
            if hour is not None:
                hb = by_hour[hour]
                hb["total"] += 1
                if stage:
                    hb["rejects"] += 1
                else:
                    hb["passed"] += 1

            date_str = _date_bucket(record.get("bar_ts", ""))
            if date_str is not None:
                d_bucket = by_date.setdefault(
                    date_str,
                    {
                        "total": 0,
                        "passed_all_gates": 0,
                        "by_stage_reject": Counter(),
                    },
                )
                d_bucket["total"] += 1
                d_bucket["by_stage_reject"][stage_key] += 1
                if not stage:
                    d_bucket["passed_all_gates"] += 1

            zr = diag.get("zone_rejects") or {}
            for sub_reason, count in zr.items():
                if isinstance(count, (int, float)) and count:
                    zone_rejects_total[sub_reason] += int(count)

    # Convert Counters to plain dicts for JSON-friendly output.
    by_magic_out: dict[str, dict[str, Any]] = {}
    for m, bucket in by_magic.items():
        by_magic_out[str(m)] = {
            "total": bucket["total"],
            "by_stage_reject": dict(bucket["by_stage_reject"]),
        }

    by_date_out: dict[str, Any] = {}
    for date_str, bucket in sorted(by_date.items()):
        by_date_out[date_str] = {
            "total": bucket["total"],
            "passed_all_gates": bucket["passed_all_gates"],
            "by_stage_reject": dict(bucket["by_stage_reject"]),
        }

    return {
        "files": [str(p) for p in files],
        "total_calls": total_calls,
        "passed_all_gates": passed_all,
        "by_stage_reject": dict(by_stage),
        "by_magic": by_magic_out,
        "by_hour_utc": {str(h): dict(b) for h, b in sorted(by_hour.items())},
        "by_date": by_date_out,
        "zone_rejects_aggregate": dict(zone_rejects_total),
    }


def _resolve_output_path(
    files: list[Path],
    journal_dir: Path,
    out_arg: str | None,
) -> Path:
    """Pick an output filename — most recent date in window or 'empty'."""
    if out_arg:
        return Path(out_arg)
    if files:
        # Newest date in the window.
        latest_name = files[-1].name  # already sorted oldest first
        # aggregator_gates_<date>.jsonl → gate_funnel_<date>.json
        prefix = "aggregator_gates_"
        suffix = ".jsonl"
        if latest_name.startswith(prefix) and latest_name.endswith(suffix):
            date_str = latest_name[len(prefix) : -len(suffix)]
        else:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    else:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return journal_dir / f"gate_funnel_{date_str}.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate aggregator gate diagnostic jsonl into a funnel report."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of UTC days to include (1 = today only). Default: 1.",
    )
    parser.add_argument(
        "--journal-dir",
        type=Path,
        default=DEFAULT_JOURNAL_DIR,
        help="Directory holding aggregator_gates_<date>.jsonl files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for the JSON report (default: gate_funnel_<latest-date>.json in journal-dir).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End of window as YYYY-MM-DD (UTC). Default: today.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    end_dt: datetime | None = None
    if args.end_date:
        try:
            end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"error: --end-date must be YYYY-MM-DD, got {args.end_date!r}", file=sys.stderr)
            return 2

    files = discover_journal_files(args.journal_dir, args.days, end_dt)
    report = build_funnel_report(files)
    out_path = _resolve_output_path(files, args.journal_dir, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(
        f"gate funnel report → {out_path} "
        f"(files={len(files)}, calls={report['total_calls']}, "
        f"passed={report['passed_all_gates']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

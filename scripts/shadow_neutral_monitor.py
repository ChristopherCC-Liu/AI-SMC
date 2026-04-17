"""Shadow neutral monitor — Sonnet F equivalent (written by skeptic-lead).

Monitors DirectionEngine output across N cycles to decide Point 8 cache TTL.
Reads the live AIDirection output file from ``live_demo.py`` and aggregates:

  - neutral hit rate (count of direction=='neutral' / total cycles)
  - reasoning_tag distribution (macro_free_capped / analyst_disagreement / ...)
  - confidence histogram (validates post-filter cap 0.5 when macro-free)
  - TTL recommendation based on neutral hit rate

REJECT conditions per skeptic-lead Round 4 checklist:
  - MUST include every cycle (no filter-out of neutral samples)
  - MUST report raw counts + percentages (no opaque smoothing)

Usage::

    python -m scripts.shadow_neutral_monitor \\
        --source data/ai_analysis.json \\
        --cycles 24 \\
        --out data/shadow_report.json

After live_demo.py writes N cycle's AI assessments to a JSONL file, this
script reads the tail N lines, computes stats, and writes a report.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


NEUTRAL_HIGH_THRESHOLD = 0.50   # >50% neutral → recommend TTL 1h
NEUTRAL_LOW_THRESHOLD = 0.20    # <20% neutral → keep TTL 4h
MACRO_FREE_CAP = 0.5            # post-filter cap (must not exceed when tag set)


def read_cycles(source: Path, n_cycles: int) -> list[dict[str, Any]]:
    """Read the last n_cycles records from a JSONL file.

    Raises FileNotFoundError if source missing (caller decides what to do).
    """
    if not source.exists():
        raise FileNotFoundError(f"Shadow source not found: {source}")

    lines = source.read_text().strip().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    if len(records) < n_cycles:
        print(
            f"[WARN] Only {len(records)} cycles available, requested {n_cycles}",
            file=sys.stderr,
        )
    return records[-n_cycles:]


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute neutral rate + reasoning_tag + confidence histogram.

    NO filter-out of neutral samples (REJECT 4 compliance). Every cycle counts.
    """
    total = len(records)
    if total == 0:
        return {"total": 0, "error": "no cycles available"}

    directions = Counter(r.get("direction", "unknown") for r in records)
    tags = Counter(r.get("reasoning_tag") or "none" for r in records)
    confs = [float(r.get("confidence", 0.0)) for r in records]

    # Validate post-filter cap: any cycle tagged macro_free_capped must have
    # confidence <= 0.5 (post-filter enforcement check).
    post_filter_violations = [
        r for r in records
        if r.get("reasoning_tag") == "macro_free_capped"
        and float(r.get("confidence", 0.0)) > MACRO_FREE_CAP + 1e-9
    ]

    neutral_rate = directions.get("neutral", 0) / total

    # TTL recommendation
    if neutral_rate > NEUTRAL_HIGH_THRESHOLD:
        ttl_recommendation = "1h — neutral hit rate >50%, Point 7 fix insufficient"
    elif neutral_rate < NEUTRAL_LOW_THRESHOLD:
        ttl_recommendation = "4h (keep current) — neutral hit rate <20%, Point 7 fix works"
    else:
        ttl_recommendation = "4h (keep current, re-review next sprint) — neutral rate in 20-50% band"

    # Confidence histogram (5 buckets)
    buckets = {"0.0-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0}
    for c in confs:
        if c < 0.3:
            buckets["0.0-0.3"] += 1
        elif c < 0.5:
            buckets["0.3-0.5"] += 1
        elif c < 0.7:
            buckets["0.5-0.7"] += 1
        elif c < 0.9:
            buckets["0.7-0.9"] += 1
        else:
            buckets["0.9-1.0"] += 1

    return {
        "total_cycles": total,
        "direction_counts": dict(directions),
        "neutral_rate": round(neutral_rate, 4),
        "reasoning_tag_counts": dict(tags),
        "confidence_stats": {
            "mean": round(mean(confs), 4),
            "median": round(median(confs), 4),
            "min": round(min(confs), 4),
            "max": round(max(confs), 4),
        },
        "confidence_histogram": buckets,
        "post_filter_violations": [
            {"cycle_idx": i, "confidence": r["confidence"], "direction": r["direction"]}
            for i, r in enumerate(records)
            if r in post_filter_violations
        ],
        "ttl_recommendation": ttl_recommendation,
        "honest_accounting": {
            "filter_applied": "none — every cycle counted",
            "neutral_samples_excluded": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/ai_analysis.json"))
    parser.add_argument("--cycles", type=int, default=24)
    parser.add_argument("--out", type=Path, default=Path("data/shadow_report.json"))
    args = parser.parse_args()

    try:
        records = read_cycles(args.source, args.cycles)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    report = aggregate(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))

    print(f"Shadow report written to {args.out}")
    print(f"  total_cycles: {report['total_cycles']}")
    print(f"  neutral_rate: {report.get('neutral_rate', 'n/a')}")
    print(f"  ttl_recommendation: {report.get('ttl_recommendation', 'n/a')}")
    if report.get("post_filter_violations"):
        print(
            f"  [WARN] {len(report['post_filter_violations'])} post-filter violations — "
            "macro_free_capped tag exists but confidence > 0.5",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Ticket 2 Step 7 — shadow_run CLI.

DIAGNOSTIC ONLY. Read-only with respect to live runtime: no
broker calls, no live HTTP, no decision_server import, no writes
under approved/ / pointer.json / src/ / config/ / .mq5.

Pipeline:
    1. Resolve the data lake (default: AI-SMC parquet root).
    2. For every candidate in CANDIDATE_MENU_V0, invoke the shadow
       runner once with the (symbol, start, end) window.
    3. Each call writes one ShadowArtefact under
       <out_dir>/<candidate_id>/<run_id>.json (immutable, mode 0444,
       wrapped envelope).

Outputs only under ``--out-dir``. The report CLI later joins these
artefacts via ``--shadow-artefacts <out_dir>`` (R5 double-key join).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
from smc.hedgerock.evolution.shadow_runner import (
    SHADOW_RUNNER_VERSION,
    run_shadow_for_candidate,
)


def _hedgerock_home() -> Path:
    raw = os.environ.get("HEDGEROCK_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "HedgeRock"


def _ai_smc_home() -> Path:
    raw = os.environ.get("AI_SMC_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parents[1]


_DEFAULT_LAKE = _ai_smc_home() / "data" / "parquet"
_DEFAULT_OUT = (
    _hedgerock_home() / "policy_registry" / "shadow_artefacts"
)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def run(
    *,
    lake: Any,
    symbol: str,
    start: datetime,
    end: datetime,
    out_dir: Path,
) -> list[Path]:
    """Library entry point. Returns the list of artefact paths
    produced (one per menu candidate)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for cand in CANDIDATE_MENU_V0:
        p = run_shadow_for_candidate(
            candidate=cand, lake=lake, symbol=symbol,
            start=start, end=end, out_dir=out_dir,
        )
        paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--start", type=_parse_date,
                        default=datetime(2021, 1, 1, tzinfo=timezone.utc))
    parser.add_argument("--end", type=_parse_date,
                        default=datetime(2025, 1, 1, tzinfo=timezone.utc))
    parser.add_argument("--data-lake-root", type=Path, default=_DEFAULT_LAKE)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.data_lake_root.exists():
        print(f"data lake root not found: {args.data_lake_root}",
              file=sys.stderr)
        return 2

    from smc.data.lake import ForexDataLake
    lake = ForexDataLake(args.data_lake_root)

    print(
        f"shadow_run: lake={args.data_lake_root}, symbol={args.symbol}, "
        f"window={args.start.date()}..{args.end.date()}, "
        f"out_dir={args.out_dir}, runner_version={SHADOW_RUNNER_VERSION}"
    )
    paths = run(
        lake=lake, symbol=args.symbol, start=args.start, end=args.end,
        out_dir=args.out_dir,
    )
    print(f"  produced {len(paths)} artefact(s):")
    for p in paths:
        print(f"    {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

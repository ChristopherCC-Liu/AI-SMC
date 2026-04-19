"""A/B Comparison Script — Round 4 Alt-B W3.

Reads two paper trading journals (control vs macro treatment) and produces
a side-by-side performance summary.

Usage::

    # From the AI-SMC root directory:
    python scripts/ab_compare.py
    python scripts/ab_compare.py --symbol XAUUSD --days 30
    python scripts/ab_compare.py --journal-a data/XAUUSD/journal \
                                  --journal-b data/XAUUSD/journal_macro \
                                  --output .scratch/audit-r4/ab_report_$(date +%Y%m%d).md

Journals are JSONL files (``live_trades.jsonl``) where each line is a JSON
object with at minimum::

    {"time": "<ISO>", "action": "RANGE BUY|RANGE SELL|BUY|SELL|HOLD",
     "mode": "PAPER|...", "rr_ratio": <float>, ...}

PnL is approximated as:
    WIN  when ``result`` field is present and > 0, OR when ``rr_ratio`` > 0
         and ``result`` field is absent (assume 1R win for paper trades).
    LOSS when ``result`` field is present and <= 0, OR rr_ratio <= 0.
    HOLD entries are excluded from trade counting.

Profit Factor = gross_wins / abs(gross_losses) using rr_ratio as proxy for
PnL units when the ``result`` field is absent.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class TradeRow(NamedTuple):
    """Parsed trade row from a journal JSONL line."""

    time: datetime
    action: str
    direction: str   # long / short / unknown
    pnl_units: float  # approximated from result or rr_ratio
    is_win: bool
    # audit-r4 v5 Option B: balance snapshot for dual-magic A/B analysis.
    # Populated from journal fields when present; None when journal predates
    # Option B (backward-compat).
    account_balance_usd: float | None = None
    virtual_balance_usd: float | None = None
    magic: int | None = None


class LegStats(NamedTuple):
    """Performance statistics for one journal leg."""

    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float       # 0.0 – 1.0
    profit_factor: float  # gross wins / abs(gross losses)
    avg_pnl: float        # average pnl_units per trade
    gross_wins: float
    gross_losses: float   # positive number (absolute value)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_time(raw: str) -> datetime:
    """Parse ISO 8601 timestamp to UTC-aware datetime."""
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_optional_float(raw: object) -> float | None:
    """Parse to float or None — tolerant of string / None / missing fields."""
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _parse_optional_int(raw: object) -> int | None:
    """Parse to int or None — tolerant of string / None / missing fields."""
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _is_trade_action(action: str) -> bool:
    """Return True if this journal row is an actual trade (not HOLD)."""
    if not action:
        return False
    upper = action.upper()
    return not upper.startswith("HOLD")


def _parse_pnl(row: dict) -> tuple[float, bool]:
    """Extract (pnl_units, is_win) from a journal row.

    Priority:
    1. ``result`` field (actual closed PnL in USD or points, broker-reported).
    2. ``rr_ratio`` field — proxy: positive rr_ratio → 1R win; 0/missing → 1R loss.
    """
    result = row.get("result")
    if result is not None:
        pnl = float(result)
        return pnl, pnl > 0

    rr = row.get("rr_ratio", 0.0)
    if rr is None:
        rr = 0.0
    rr = float(rr)
    if rr > 0:
        return rr, True
    return -1.0, False


def load_journal(path: Path, cutoff_days: int | None = None) -> list[TradeRow]:
    """Load trades from a JSONL journal file.

    Parameters
    ----------
    path:
        Path to the ``live_trades.jsonl`` file.
    cutoff_days:
        If given, only include trades within the last N calendar days (UTC).
        None means include all trades.

    Returns
    -------
    List of TradeRow sorted by time ascending.
    """
    if not path.exists():
        return []

    cutoff_dt: datetime | None = None
    if cutoff_days is not None:
        cutoff_dt = datetime.now(tz=timezone.utc) - timedelta(days=cutoff_days)

    rows: list[TradeRow] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"  WARN: skipping malformed line {line_no} in {path}: {exc}",
                    file=sys.stderr,
                )
                continue

            action = str(obj.get("action", ""))
            if not _is_trade_action(action):
                continue

            time_raw = obj.get("time", "")
            if not time_raw:
                continue
            try:
                ts = _parse_time(time_raw)
            except (ValueError, TypeError):
                continue

            if cutoff_dt is not None and ts < cutoff_dt:
                continue

            direction = str(obj.get("direction", "unknown"))
            pnl_units, is_win = _parse_pnl(obj)
            rows.append(TradeRow(
                time=ts,
                action=action,
                direction=direction,
                pnl_units=pnl_units,
                is_win=is_win,
                account_balance_usd=_parse_optional_float(obj.get("account_balance_usd")),
                virtual_balance_usd=_parse_optional_float(obj.get("virtual_balance_usd")),
                magic=_parse_optional_int(obj.get("magic")),
            ))

    rows.sort(key=lambda r: r.time)
    return rows


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(rows: list[TradeRow]) -> LegStats:
    """Compute performance statistics for a list of trade rows."""
    if not rows:
        return LegStats(
            trade_count=0, win_count=0, loss_count=0,
            win_rate=0.0, profit_factor=0.0, avg_pnl=0.0,
            gross_wins=0.0, gross_losses=0.0,
        )

    win_count = sum(1 for r in rows if r.is_win)
    loss_count = len(rows) - win_count
    gross_wins = sum(r.pnl_units for r in rows if r.is_win)
    gross_losses = abs(sum(r.pnl_units for r in rows if not r.is_win))
    total_pnl = sum(r.pnl_units for r in rows)

    win_rate = win_count / len(rows)
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    avg_pnl = total_pnl / len(rows)

    return LegStats(
        trade_count=len(rows),
        win_count=win_count,
        loss_count=loss_count,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_pnl=avg_pnl,
        gross_wins=gross_wins,
        gross_losses=gross_losses,
    )


def _weekly_buckets(rows: list[TradeRow]) -> dict[str, list[TradeRow]]:
    """Group trade rows by ISO week string (e.g. '2026-W17')."""
    buckets: dict[str, list[TradeRow]] = defaultdict(list)
    for row in rows:
        week_key = f"{row.time.isocalendar()[0]}-W{row.time.isocalendar()[1]:02d}"
        buckets[week_key].append(row)
    return dict(buckets)


def _daily_buckets(rows: list[TradeRow]) -> dict[str, list[TradeRow]]:
    """Group trade rows by date string (e.g. '2026-04-19')."""
    buckets: dict[str, list[TradeRow]] = defaultdict(list)
    for row in rows:
        day_key = row.time.date().isoformat()
        buckets[day_key].append(row)
    return dict(buckets)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt_pf(pf: float) -> str:
    if pf == float("inf"):
        return "∞"
    return f"{pf:.2f}"


def _fmt_pct(v: float) -> str:
    return f"{v:.1%}"


def _latest_balance_snapshot(rows: list[TradeRow]) -> tuple[str, str]:
    """Return (account_balance_str, virtual_balance_str) from latest trade row.

    Empty journals or legacy rows (pre-Option-B, no balance fields) yield
    ``("n/a", "n/a")`` so the markdown renders cleanly.
    """
    if not rows:
        return ("n/a", "n/a")
    # Most recent first
    sorted_rows = sorted(rows, key=lambda r: r.time, reverse=True)
    for r in sorted_rows:
        if r.account_balance_usd is not None or r.virtual_balance_usd is not None:
            acc = f"${r.account_balance_usd:.2f}" if r.account_balance_usd is not None else "n/a"
            vir = f"${r.virtual_balance_usd:.2f}" if r.virtual_balance_usd is not None else "n/a"
            return (acc, vir)
    return ("n/a", "n/a")


def _summary_table(
    label_a: str, stats_a: LegStats, label_b: str, stats_b: LegStats,
    rows_a: list[TradeRow] | None = None, rows_b: list[TradeRow] | None = None,
) -> str:
    """Return a Markdown table comparing two legs.

    audit-r4 v5 Option B: includes account-balance and virtual-balance
    snapshots from the latest trade in each leg so dual-magic A/B runs
    can quickly confirm the virtual-balance split was honoured.
    """
    acc_a, vir_a = _latest_balance_snapshot(rows_a or [])
    acc_b, vir_b = _latest_balance_snapshot(rows_b or [])

    rows = [
        ("Trade count",   str(stats_a.trade_count),               str(stats_b.trade_count)),
        ("Win count",     str(stats_a.win_count),                  str(stats_b.win_count)),
        ("Loss count",    str(stats_a.loss_count),                 str(stats_b.loss_count)),
        ("Win rate",      _fmt_pct(stats_a.win_rate),              _fmt_pct(stats_b.win_rate)),
        ("Profit factor", _fmt_pf(stats_a.profit_factor),          _fmt_pf(stats_b.profit_factor)),
        ("Avg PnL (R)",   f"{stats_a.avg_pnl:+.3f}",              f"{stats_b.avg_pnl:+.3f}"),
        ("Gross wins",    f"{stats_a.gross_wins:.3f}",             f"{stats_b.gross_wins:.3f}"),
        ("Gross losses",  f"{stats_a.gross_losses:.3f}",           f"{stats_b.gross_losses:.3f}"),
        ("Account balance (latest)", acc_a, acc_b),
        ("Virtual balance (latest)", vir_a, vir_b),
    ]

    col_w = max(len(label_a), len(label_b), 12)
    header = f"| {'Metric':<22} | {label_a:<{col_w}} | {label_b:<{col_w}} |"
    sep    = f"| {'-'*22} | {'-'*col_w} | {'-'*col_w} |"
    lines  = [header, sep]
    for metric, val_a, val_b in rows:
        lines.append(f"| {metric:<22} | {val_a:<{col_w}} | {val_b:<{col_w}} |")
    return "\n".join(lines)


def _rolling_table(
    label_a: str, rows_a: list[TradeRow],
    label_b: str, rows_b: list[TradeRow],
    bucket_fn,
    heading: str,
) -> str:
    """Return a Markdown table of per-bucket stats for two legs."""
    buckets_a = bucket_fn(rows_a)
    buckets_b = bucket_fn(rows_b)
    all_keys = sorted(set(buckets_a) | set(buckets_b))
    if not all_keys:
        return f"*No {heading.lower()} data.*"

    col_w = max(len(label_a), len(label_b), 8)
    header = (
        f"| {'Period':<12} | "
        f"{label_a+' PF':<{col_w+3}} | {label_a+' WR':<{col_w+3}} | "
        f"{label_b+' PF':<{col_w+3}} | {label_b+' WR':<{col_w+3}} |"
    )
    sep = (
        f"| {'-'*12} | "
        f"{'-'*(col_w+3)} | {'-'*(col_w+3)} | "
        f"{'-'*(col_w+3)} | {'-'*(col_w+3)} |"
    )
    lines = [header, sep]
    for key in all_keys:
        sa = compute_stats(buckets_a.get(key, []))
        sb = compute_stats(buckets_b.get(key, []))
        lines.append(
            f"| {key:<12} | "
            f"{_fmt_pf(sa.profit_factor):<{col_w+3}} | {_fmt_pct(sa.win_rate):<{col_w+3}} | "
            f"{_fmt_pf(sb.profit_factor):<{col_w+3}} | {_fmt_pct(sb.win_rate):<{col_w+3}} |"
        )
    return "\n".join(lines)


def build_report(
    label_a: str, rows_a: list[TradeRow],
    label_b: str, rows_b: list[TradeRow],
    cutoff_days: int | None,
    generated_at: datetime,
) -> str:
    """Build the full Markdown comparison report."""
    stats_a = compute_stats(rows_a)
    stats_b = compute_stats(rows_b)

    period_note = f"last {cutoff_days} days" if cutoff_days else "all time"
    lines = [
        f"# AI-SMC A/B Comparison Report",
        f"",
        f"**Generated:** {generated_at.strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Period:** {period_note}  ",
        f"**Leg A (control):** `{label_a}`  ",
        f"**Leg B (treatment, macro ON):** `{label_b}`  ",
        f"",
        f"## Overall Summary",
        f"",
        _summary_table(label_a, stats_a, label_b, stats_b, rows_a=rows_a, rows_b=rows_b),
        f"",
        f"## Weekly Rolling Comparison",
        f"",
        _rolling_table(label_a, rows_a, label_b, rows_b, _weekly_buckets, "Weekly"),
        f"",
        f"## Daily Rolling Comparison",
        f"",
        _rolling_table(label_a, rows_a, label_b, rows_b, _daily_buckets, "Daily"),
        f"",
        f"---",
        f"*PnL unit = R (risk-reward ratio proxy). Result field used when available.*",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _resolve_journal(symbol: str, suffix: str, data_root: Path) -> Path:
    """Resolve the journal JSONL path from symbol + suffix."""
    return data_root / symbol / f"journal{suffix}" / "live_trades.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare A/B paper trading journals (control vs macro treatment)."
    )
    parser.add_argument(
        "--symbol", default="XAUUSD",
        help="Trading symbol (default: XAUUSD)",
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Only include trades from the last N days. Default: all time.",
    )
    parser.add_argument(
        "--data-root", default="data",
        help="Root data directory (default: data)",
    )
    parser.add_argument(
        "--journal-a", default=None,
        help="Explicit path to control journal JSONL (overrides --symbol + --data-root).",
    )
    parser.add_argument(
        "--journal-b", default=None,
        help="Explicit path to treatment journal JSONL.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Write report to this file. Default: print to stdout.",
    )
    parser.add_argument(
        "--label-a", default="control",
        help="Label for leg A (default: control).",
    )
    parser.add_argument(
        "--label-b", default="macro",
        help="Label for leg B (default: macro).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)

    path_a = Path(args.journal_a) if args.journal_a else _resolve_journal(
        args.symbol, "", data_root
    )
    path_b = Path(args.journal_b) if args.journal_b else _resolve_journal(
        args.symbol, "_macro", data_root
    )

    print(f"[ab_compare] Loading journal A: {path_a}", file=sys.stderr)
    print(f"[ab_compare] Loading journal B: {path_b}", file=sys.stderr)

    rows_a = load_journal(path_a, cutoff_days=args.days)
    rows_b = load_journal(path_b, cutoff_days=args.days)

    print(f"[ab_compare] Trades — A: {len(rows_a)}, B: {len(rows_b)}", file=sys.stderr)

    now = datetime.now(tz=timezone.utc)
    report = build_report(
        label_a=args.label_a,
        rows_a=rows_a,
        label_b=args.label_b,
        rows_b=rows_b,
        cutoff_days=args.days,
        generated_at=now,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"[ab_compare] Report written to {out_path}", file=sys.stderr)
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())

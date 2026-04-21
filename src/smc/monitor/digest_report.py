"""R5 M1: multi-symbol daily digest aggregator + CSV/Markdown report writer.

Layered on top of ``daily_digest.build_daily_digest`` (backward compatible)
and ``digest_enrich.*`` (R5 additions).  Produces:

  1. ``build_multi_symbol_digest`` — one dict that contains per-symbol
     legacy digest blobs **plus** cross-leg fields (per_leg, regime_distribution,
     ai_debate, handle_resets).  Safe to serve from the dashboard API.
  2. ``write_digest_report`` — writes CSV + Markdown at
     ``data/reports/digest_YYYY-MM-DD.{csv,md}``.  Used by scheduled tasks.

Why separate from daily_digest.py:
  - daily_digest.py is per-symbol and consumed by a live FastAPI endpoint.
    The endpoint can't afford to scan BTC + XAUUSD + all journals on every
    hit.  This module is scheduled / batch only.
  - CSV/MD generation is pure filesystem I/O; keeping it outside the
    importable core lets tests inject a tmp_path trivially.
"""
from __future__ import annotations

import csv
import json
from datetime import date, datetime, time, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

from smc.monitor.daily_digest import build_daily_digest
from smc.monitor.digest_enrich import (
    build_ai_debate_stats,
    build_leg_breakdown,
    build_regime_distribution,
    count_handle_resets,
)


__all__ = [
    "build_multi_symbol_digest",
    "collect_journal_paths",
    "render_digest_markdown",
    "render_digest_csv",
    "write_digest_report",
    "scan_structured_events",
]


# ---------------------------------------------------------------------------
# Journal path resolution
# ---------------------------------------------------------------------------


def collect_journal_paths(
    data_root: Path,
    symbols: Iterable[str] = ("XAUUSD", "BTCUSD"),
) -> dict[str, Path]:
    """Discover per-leg journal files under ``data_root/{symbol}/journal{,_macro}/``.

    Only includes paths whose directories exist on disk — missing legs are
    silently skipped (BTCUSD has no macro treatment today; that's fine).
    """
    out: dict[str, Path] = {}
    for sym in symbols:
        sym_root = data_root / sym
        control = sym_root / "journal" / "live_trades.jsonl"
        if control.parent.exists():
            out[f"{sym}:control"] = control
        treatment = sym_root / "journal_macro" / "live_trades.jsonl"
        if treatment.parent.exists():
            out[f"{sym}:treatment"] = treatment
    return out


# ---------------------------------------------------------------------------
# Structured-log scanning for R5 enrichments
# ---------------------------------------------------------------------------


def scan_structured_events(
    log_root: Path, target_date: date,
) -> list[dict[str, Any]]:
    """Yield all structured.jsonl events on ``target_date`` as dicts.

    Mirrors daily_digest's rotation handling (includes ``structured.jsonl``
    + ``structured.jsonl.YYYY-MM-DD`` for today & tomorrow).
    Returns a *list* (not generator) so callers can iterate multiple times.
    """
    events: list[dict[str, Any]] = []
    files = _structured_log_candidates(log_root, target_date)
    if not files:
        return events

    day_start = datetime.combine(target_date, time(0, 0), tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    for path in files:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            payload = _parse_structured_line(line)
            if payload is None:
                continue
            ts = _parse_ts(payload.get("ts"))
            if ts is None or ts < day_start or ts >= day_end:
                continue
            events.append(payload)
    return events


def _structured_log_candidates(log_root: Path, target_date: date) -> list[Path]:
    if not log_root.exists():
        return []
    out: list[Path] = []
    current = log_root / "structured.jsonl"
    if current.exists():
        out.append(current)
    for suffix in (target_date.isoformat(), (target_date + timedelta(days=1)).isoformat()):
        rotated = log_root / f"structured.jsonl.{suffix}"
        if rotated.exists() and rotated not in out:
            out.append(rotated)
    return out


def _parse_structured_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line or not line.startswith("["):
        return None
    rbracket = line.find("] ")
    if rbracket < 0:
        return None
    try:
        payload = json.loads(line[rbracket + 2:])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Multi-symbol digest aggregator
# ---------------------------------------------------------------------------


def build_multi_symbol_digest(
    target_date: date,
    *,
    data_root: Path,
    log_root: Path,
    symbols: Iterable[str] = ("XAUUSD", "BTCUSD"),
    now: datetime | None = None,
) -> dict[str, Any]:
    """Assemble the full R5 daily digest (all symbols + enrichments).

    Returns::

        {
            "date": "2026-04-20",
            "generated_at": "2026-04-21T00:05:00+00:00",
            "per_symbol": {
                "XAUUSD": { ... legacy build_daily_digest dict ... },
                "BTCUSD": { ... },
            },
            "per_leg": [ { leg, trades, WR, PF, ... }, ... ],
            "regime_distribution": { TRANSITION: N, TREND_UP: N, ... },
            "ai_debate": { cycles_ran, p50_elapsed_ms, ... },
            "handle_resets": int,
        }
    """
    current = now if now is not None else datetime.now(timezone.utc)
    per_symbol: dict[str, dict[str, Any]] = {}
    for sym in symbols:
        sym_root = data_root / sym
        if not sym_root.exists():
            continue
        per_symbol[sym] = build_daily_digest(
            sym, target_date,
            data_root=sym_root, log_root=log_root, now=current,
        )

    events = scan_structured_events(log_root, target_date)
    journal_paths = collect_journal_paths(data_root, symbols=symbols)

    return {
        "date": target_date.isoformat(),
        "generated_at": current.isoformat(),
        "per_symbol": per_symbol,
        "per_leg": build_leg_breakdown(journal_paths, target_date),
        "regime_distribution": build_regime_distribution(events),
        "ai_debate": build_ai_debate_stats(events),
        "handle_resets": count_handle_resets(events),
    }


# ---------------------------------------------------------------------------
# CSV + Markdown rendering
# ---------------------------------------------------------------------------


_CSV_LEG_FIELDS = [
    "leg",
    "trades",
    "wins",
    "losses",
    "win_rate_pct",
    "total_pnl_usd",
    "max_drawdown_usd",
    "avg_win_usd",
    "avg_loss_usd",
    "payoff_ratio",
    "profit_factor",
]


def render_digest_csv(digest: dict[str, Any]) -> str:
    """Render the per-leg breakdown as CSV.

    One row per leg; includes trailing blank line + regime/AI summary as
    comment rows prefixed with ``#`` so CSV parsers can still consume it
    (pandas ``comment="#"``).
    """
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_LEG_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in digest.get("per_leg", []):
        writer.writerow(row)

    buf.write("\n# Regime distribution (ai_regime_classified counts):\n")
    for name, count in digest.get("regime_distribution", {}).items():
        buf.write(f"# {name},{count}\n")
    ai = digest.get("ai_debate", {}) or {}
    buf.write(
        f"# AI debate: cycles={ai.get('cycles_ran', 0)} "
        f"p50_ms={ai.get('p50_elapsed_ms')} "
        f"p90_ms={ai.get('p90_elapsed_ms')} "
        f"cost_usd={ai.get('total_cost_usd', 0)}\n"
    )
    buf.write(f"# Handle resets: {digest.get('handle_resets', 0)}\n")
    return buf.getvalue()


def render_digest_markdown(digest: dict[str, Any]) -> str:
    """Render the digest as a one-page operator-friendly Markdown report."""
    lines: list[str] = []
    lines.append(f"# AI-SMC Daily Digest — {digest.get('date')}")
    lines.append("")
    lines.append(f"**Generated:** {digest.get('generated_at')}")
    lines.append("")

    lines.append("## Per-leg performance")
    lines.append("")
    per_leg = digest.get("per_leg") or []
    if not per_leg:
        lines.append("*No closed trades today across any leg.*")
    else:
        lines.append(
            "| Leg | Trades | W | L | WR% | Total PnL | Max DD | Avg Win | Avg Loss | Payoff | PF |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for row in per_leg:
            lines.append(
                f"| {row['leg']} | {row['trades']} | {row['wins']} | {row['losses']} | "
                f"{_fmt_num(row.get('win_rate_pct'))} | "
                f"{_fmt_usd(row.get('total_pnl_usd'))} | "
                f"{_fmt_usd_unsigned(row.get('max_drawdown_usd'))} | "
                f"{_fmt_usd(row.get('avg_win_usd'))} | "
                f"{_fmt_usd(row.get('avg_loss_usd'))} | "
                f"{_fmt_num(row.get('payoff_ratio'))} | "
                f"{_fmt_num(row.get('profit_factor'))} |"
            )
    lines.append("")

    lines.append("## Regime distribution")
    lines.append("")
    dist = digest.get("regime_distribution") or {}
    if not dist:
        lines.append("*No ai_regime_classified events today.*")
    else:
        lines.append("| Regime | Count |")
        lines.append("|---|---:|")
        for name, count in dist.items():
            lines.append(f"| {name} | {count} |")
    lines.append("")

    lines.append("## AI debate")
    lines.append("")
    ai = digest.get("ai_debate") or {}
    lines.append(f"- **Cycles ran:** {ai.get('cycles_ran', 0)}")
    lines.append(f"- **p50 elapsed ms:** {ai.get('p50_elapsed_ms')}")
    lines.append(f"- **p90 elapsed ms:** {ai.get('p90_elapsed_ms')}")
    lines.append(f"- **Total cost (USD):** {ai.get('total_cost_usd', 0)}")
    lines.append("")

    lines.append("## Stability")
    lines.append("")
    lines.append(f"- **MT5 handle resets today:** {digest.get('handle_resets', 0)}")
    lines.append("")

    warnings: list[str] = []
    for sym, blob in (digest.get("per_symbol") or {}).items():
        for w in blob.get("warnings", []) or []:
            warnings.append(f"{sym}:{w}")
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


def _fmt_num(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and value != value:  # NaN
        return "—"
    return str(value)


def _fmt_usd(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"${float(value):+,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_usd_unsigned(value: Any) -> str:
    """Render a non-negative magnitude (e.g. drawdown) without sign prefix."""
    if value is None:
        return "—"
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Filesystem writer
# ---------------------------------------------------------------------------


def write_digest_report(
    digest: dict[str, Any],
    *,
    reports_root: Path,
) -> tuple[Path, Path]:
    """Write CSV + Markdown report files.  Returns ``(csv_path, md_path)``.

    File names follow ``digest_YYYY-MM-DD.{csv,md}`` using ``digest['date']``.
    Creates ``reports_root`` if missing.  Overwrites existing files.
    """
    date_str = digest.get("date") or datetime.now(timezone.utc).date().isoformat()
    reports_root.mkdir(parents=True, exist_ok=True)
    csv_path = reports_root / f"digest_{date_str}.csv"
    md_path = reports_root / f"digest_{date_str}.md"
    csv_path.write_text(render_digest_csv(digest), encoding="utf-8")
    md_path.write_text(render_digest_markdown(digest), encoding="utf-8")
    return csv_path, md_path

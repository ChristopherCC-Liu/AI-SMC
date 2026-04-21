"""R5 M1: per-leg / regime / AI-debate / handle-reset enrichments for daily_digest.

Pure functions that scan the same data sources as ``daily_digest.py`` and
return structured extensions for the top-level digest dict.  Kept in a
separate module so ``build_daily_digest`` stays small and the existing
backward-compatible contract (fields for the dashboard/JSON API) is not
disturbed — callers who don't need the new fields can simply ignore them.

Inputs mirror ``daily_digest``:
  - journal(s) JSONL  — one per leg (magic suffix => ``journal_macro``)
  - structured.jsonl  — for ai_regime_classified + trade_reconciled +
                        mt5_handle_reset + ai_debate_completed events

Design notes:
  - Per-leg partitioning is driven by *journal paths*, not by the ``magic``
    field on closure events.  This matches the physical file layout
    (``journal/`` vs ``journal_macro/``) and lets us work with BTC where no
    treatment leg exists.
  - ``ai_debate_completed`` event schema is flexible: we coerce any of
    ``elapsed_ms`` / ``latency_ms`` / ``duration_ms`` and ``total_cost_usd`` /
    ``cost_usd``.  stability-lead emits its own schema in parallel and we
    don't want a tight coupling.
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


__all__ = [
    "build_leg_breakdown",
    "build_regime_distribution",
    "build_ai_debate_stats",
    "count_handle_resets",
    "percentile",
]


# ---------------------------------------------------------------------------
# Per-leg trade breakdown
# ---------------------------------------------------------------------------


def build_leg_breakdown(
    journal_paths: dict[str, Path],
    target_date: date,
    *,
    closures_by_ticket: dict[int, float] | None = None,
    closure_events: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return a list of per-leg stat rows.

    Parameters
    ----------
    journal_paths:
        Mapping of leg label -> journal JSONL file.  Example::

            {
                "XAUUSD:control":   data_root/"XAUUSD"/"journal"/"live_trades.jsonl",
                "XAUUSD:treatment": data_root/"XAUUSD"/"journal_macro"/"live_trades.jsonl",
                "BTCUSD:control":   data_root/"BTCUSD"/"journal"/"live_trades.jsonl",
            }
    target_date:
        UTC calendar date to summarise.
    closures_by_ticket:
        Optional: map ticket -> closed_pnl_usd from structured.jsonl.  When
        supplied, we prefer this over the journal ``result`` field (the
        structured log is authoritative for closed PnL).  Empty / None means
        fall back to journal ``result``.
    closure_events:
        Optional: raw trade_reconciled / trade_closed events from
        structured.jsonl on ``target_date``.  In the Round 4 v5
        DELEGATED_TO_EA architecture, journal rows capture entry *signals*
        and never carry closed PnL (EA executes and reconciler emits a
        separate structured event).  When tickets can't be matched back
        to a journal row (VPS journal's ``mt5_ticket`` is populated only
        post-execute and may be null at write time), these closures would
        be lost.  We aggregate them into an ``"unassigned:closures"`` leg
        so daily totals don't silently drop 7 trades on the floor.

    Returns
    -------
    A list of dicts, one per non-empty leg, with fields::

        {
            "leg": "XAUUSD:control",
            "trades": int,
            "wins": int,
            "losses": int,
            "win_rate_pct": float | None,
            "total_pnl_usd": float,
            "max_drawdown_usd": float,
            "avg_win_usd": float | None,
            "avg_loss_usd": float | None,
            "payoff_ratio": float | None,
            "profit_factor": float | None,
        }
    """
    day_start = datetime.combine(target_date, time(0, 0), tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)
    closures_by_ticket = closures_by_ticket or {}

    # Pass 1: legs sourced from journal (strict leg attribution).
    out: list[dict[str, Any]] = []
    assigned_tickets: set[int] = set()
    for label, path in journal_paths.items():
        rows = _scan_leg_journal(
            path, day_start, day_end, closures_by_ticket, assigned_tickets,
        )
        if not rows:
            continue
        out.append(_leg_stats(label, rows))

    # Pass 2: unattributed closures that didn't match any journal ticket.
    # Critical for DELEGATED_TO_EA: journal row often has mt5_ticket=null
    # when it was written (EA assigns ticket only after OrderSend succeeds).
    if closure_events:
        unassigned_pnls = _collect_unassigned_closure_pnls(
            closure_events, day_start, day_end, assigned_tickets,
        )
        if unassigned_pnls:
            out.append(_leg_stats("unassigned:closures", unassigned_pnls))

    return out


def _collect_unassigned_closure_pnls(
    events: Iterable[dict[str, Any]],
    day_start: datetime,
    day_end: datetime,
    assigned_tickets: set[int],
) -> list[float]:
    """Return pnl_usd for trade_reconciled events whose ticket wasn't
    attributed to a leg in pass 1.  ``trade_closed`` is skipped because it
    duplicates ``trade_reconciled`` (both fire on every close)."""
    seen: set[int] = set()
    pnls: list[float] = []
    for ev in events:
        if ev.get("event") != "trade_reconciled":
            continue
        ts = _parse_ts(ev.get("ts"))
        if ts is None or ts < day_start or ts >= day_end:
            continue
        ticket = ev.get("ticket")
        if ticket is None:
            continue
        try:
            t = int(ticket)
        except (TypeError, ValueError):
            continue
        if t in assigned_tickets or t in seen:
            continue
        seen.add(t)
        pnl = _coerce_float(ev.get("pnl_usd"))
        if pnl is None:
            continue
        pnls.append(pnl)
    return pnls


def _scan_leg_journal(
    path: Path,
    day_start: datetime,
    day_end: datetime,
    closures_by_ticket: dict[int, float],
    assigned_tickets: set[int],
) -> list[float]:
    """Return list of per-trade PnL (USD) for closed trades on the day.

    Records any successfully-attributed ticket into ``assigned_tickets`` so
    the second pass (unattributed closures) can skip them.
    """
    if not path.exists():
        return []
    pnls: list[float] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = _parse_ts(entry.get("time") or entry.get("ts"))
        if ts is None or ts < day_start or ts >= day_end:
            continue
        mode = str(entry.get("mode", "")).upper()
        action = str(entry.get("action", "")).upper()
        # Only count opened-and-closed trades; skip HOLD rows, pre-write gate
        # trips (mode=MARGIN_GATED / PAPER) unless they have a result field.
        if action.startswith("HOLD"):
            continue
        if mode not in {"PAPER", "LIVE_EXEC"}:
            continue
        pnl = _extract_pnl(entry, closures_by_ticket)
        if pnl is None:
            continue
        pnls.append(pnl)
        ticket = entry.get("ticket") or entry.get("mt5_ticket")
        if ticket is not None:
            try:
                assigned_tickets.add(int(ticket))
            except (TypeError, ValueError):
                pass
    return pnls


def _extract_pnl(
    entry: dict[str, Any], closures_by_ticket: dict[int, float],
) -> float | None:
    """Prefer closures-by-ticket (from structured.jsonl), else journal result."""
    ticket = entry.get("ticket")
    if ticket is not None and ticket in closures_by_ticket:
        return closures_by_ticket[ticket]
    result = entry.get("result")
    if result is None:
        return None
    try:
        return float(result)
    except (TypeError, ValueError):
        return None


def _leg_stats(label: str, pnls: list[float]) -> dict[str, Any]:
    """Compute per-leg summary stats from a PnL series (USD)."""
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    n = len(pnls)
    w_count = len(wins)
    l_count = len(losses)
    win_rate = (w_count / n * 100.0) if n > 0 else None
    avg_win = (sum(wins) / w_count) if wins else None
    avg_loss = (sum(losses) / l_count) if losses else None
    gross_wins = sum(wins)
    gross_losses = abs(sum(losses))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else None
    payoff_ratio = (
        (avg_win / abs(avg_loss)) if (avg_win is not None and avg_loss is not None and avg_loss != 0)
        else None
    )
    return {
        "leg": label,
        "trades": n,
        "wins": w_count,
        "losses": l_count,
        "win_rate_pct": round(win_rate, 1) if win_rate is not None else None,
        "total_pnl_usd": round(sum(pnls), 2),
        "max_drawdown_usd": round(_max_drawdown(pnls), 2),
        "avg_win_usd": round(avg_win, 2) if avg_win is not None else None,
        "avg_loss_usd": round(avg_loss, 2) if avg_loss is not None else None,
        "payoff_ratio": round(payoff_ratio, 2) if payoff_ratio is not None else None,
        "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
    }


def _max_drawdown(pnls: list[float]) -> float:
    """Running-sum max drawdown over the day (USD).

    DD := max over t of (peak[0..t] - equity[t]) where equity[t] = sum(pnls[0..t]).
    Returned as a non-negative number (0 if no drawdown).
    """
    peak = 0.0
    equity = 0.0
    dd = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        drop = peak - equity
        if drop > dd:
            dd = drop
    return dd


# ---------------------------------------------------------------------------
# Regime distribution from ai_regime_classified events
# ---------------------------------------------------------------------------


def build_regime_distribution(
    structured_events: Iterable[dict[str, Any]],
) -> dict[str, int]:
    """Count ai_regime_classified events by regime label.

    Always returns all 5 canonical regime keys (zero when absent) plus a
    synthetic ``UNKNOWN`` bucket for unexpected values, so downstream
    consumers can render a stable row set.
    """
    buckets = {
        "TRANSITION": 0,
        "TREND_UP": 0,
        "TREND_DOWN": 0,
        "CONSOLIDATION": 0,
        "ATH_BREAKOUT": 0,
        "UNKNOWN": 0,
    }
    for ev in structured_events:
        if ev.get("event") != "ai_regime_classified":
            continue
        regime = str(ev.get("regime", "")).upper() or "UNKNOWN"
        if regime in buckets:
            buckets[regime] += 1
        else:
            buckets["UNKNOWN"] += 1
    return buckets


# ---------------------------------------------------------------------------
# AI debate stats (cycles, latency percentiles, cost)
# ---------------------------------------------------------------------------


_AI_DEBATE_EVENT_NAMES = frozenset({
    # Prod schema observed in VPS structured.jsonl (Apr 2026).
    "ai_regime_debate_result",
    # Plan name + alternatives tolerated for forward compat.
    "ai_debate_completed",
    "ai_debate",
})


def build_ai_debate_stats(
    structured_events: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate AI debate events.

    Tolerates schema drift across event names and latency unit (ms vs s):
      - event in {ai_regime_debate_result, ai_debate_completed, ai_debate}
      - latency via elapsed_ms / latency_ms / duration_ms / elapsed_s
        (``elapsed_s`` is converted to ms)
      - cost via total_cost_usd / cost_usd
    """
    latencies_ms: list[float] = []
    total_cost = 0.0
    count = 0
    for ev in structured_events:
        if ev.get("event") not in _AI_DEBATE_EVENT_NAMES:
            continue
        count += 1
        latency_ms = _coerce_float(
            ev.get("elapsed_ms")
            or ev.get("latency_ms")
            or ev.get("duration_ms")
        )
        if latency_ms is None:
            # Fall back to seconds if that's what the producer emits.
            latency_s = _coerce_float(ev.get("elapsed_s") or ev.get("duration_s"))
            if latency_s is not None:
                latency_ms = latency_s * 1000.0
        if latency_ms is not None:
            latencies_ms.append(latency_ms)
        cost = _coerce_float(
            ev.get("total_cost_usd") if ev.get("total_cost_usd") is not None
            else ev.get("cost_usd")
        )
        if cost is not None:
            total_cost += cost

    return {
        "cycles_ran": count,
        "p50_elapsed_ms": round(percentile(latencies_ms, 50), 1) if latencies_ms else None,
        "p90_elapsed_ms": round(percentile(latencies_ms, 90), 1) if latencies_ms else None,
        "total_cost_usd": round(total_cost, 4),
    }


# ---------------------------------------------------------------------------
# Handle-reset count (from stability-lead R1 watchdog)
# ---------------------------------------------------------------------------


def count_handle_resets(
    structured_events: Iterable[dict[str, Any]],
) -> int:
    """Count mt5_handle_reset events.  Safe when stability-lead's R1 watchdog
    is not yet deployed — returns 0 if no such events exist.
    """
    return sum(1 for ev in structured_events if ev.get("event") == "mt5_handle_reset")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (0..100).  Empty list -> 0.0."""
    if not values:
        return 0.0
    srt = sorted(values)
    if len(srt) == 1:
        return float(srt[0])
    k = (pct / 100.0) * (len(srt) - 1)
    lo = int(k)
    hi = min(lo + 1, len(srt) - 1)
    frac = k - lo
    return float(srt[lo] + (srt[hi] - srt[lo]) * frac)


def _parse_ts(value: Any) -> datetime | None:
    """Parse ISO-8601 string (tolerant of trailing Z) -> UTC datetime."""
    if not isinstance(value, str) or not value:
        return None
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

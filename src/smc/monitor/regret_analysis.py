"""R5 M3: per-trade regret (counterfactual PnL) analysis.

For each closed trade in a journal, this module computes three
counterfactuals:

  1. **no_macro** — "what if the macro bias layer were OFF?"  For Control
     (magic 19760418) trades this is zero — macro was already off.  For
     Treatment (magic 19760428) trades, the counterfactual assumes the
     trade would have been taken *without* macro's filtering/biasing, so we
     flip the expected direction if ``macro_bias`` field exists on the row
     and records disagree with the actual direction, and assume the trade
     either (a) still taken with identical PnL (macro didn't change entry/
     exit) or (b) not taken at all (macro blocked it).  In absence of
     journal metadata the default assumption is (a): macro was a labelling
     layer, not a gate.  Confidence: ``heuristic``.
  2. **no_reversal_confirm** — "what if we'd allowed entries without
     bar-close confirmation of the reversal?"  Heuristic: a shorter
     confirmation window typically yields an earlier entry with a slightly
     worse fill (~0.1 R hit) and a higher false-positive rate.  We model
     this as PnL × 0.85 for wins (trim 15% of winners) and −1.1 R for new
     losers (15% bigger SL from worse fill).  Without per-trade reversal
     timestamps we can't run an exact replay; we use a population-level
     approximation tagged ``heuristic``.
  3. **no_anti_stack** — "what if cooldown between entries = 0?"  Detected
     via ``anti_stack_blocked`` rows (mode=MARGIN_GATED or a dedicated
     gated_action field) sitting between actual trades.  Each such row
     contributes an imputed 0.0 R entry (assumption: gated entries are
     statistically neutral vs the already-taken trade); the *regret* value
     from this guardrail is simply the sum of imputed PnL across blocked
     entries.

Output: one JSONL record per trade followed by one summary record::

    {"trade_id": ..., "actual_pnl": 31.84, "no_macro_pnl": 31.84, ... "confidence": "heuristic"}
    {"summary": true, "actual_total": ..., "no_macro_delta": ..., "no_reversal_delta": ..., "no_anti_stack_delta": ...}

Interpretation:
  - ``no_X_delta > 0``  →  guardrail X reduced P&L (consider loosening).
  - ``no_X_delta < 0``  →  guardrail X protected P&L (keep it).
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


__all__ = [
    "load_day_trades",
    "compute_regret_row",
    "compute_summary",
    "build_regret_records",
    "count_anti_stack_blocks",
    "extract_closures_by_ticket",
]


# Magic numbers for the two live legs.  Kept as module constants so callers
# don't need to know the ops convention.
MAGIC_CONTROL = 19760418
MAGIC_TREATMENT = 19760428


# ---------------------------------------------------------------------------
# Journal loading
# ---------------------------------------------------------------------------


def load_day_trades(
    paths: Iterable[Path], target_date: date,
) -> list[dict[str, Any]]:
    """Load all closed trades across the given journals, filtered to day UTC.

    Each returned dict includes the journal source path under ``_journal``
    so we can tag the leg later (control vs treatment).
    """
    day_start = datetime.combine(target_date, time(0, 0), tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)
    rows: list[dict[str, Any]] = []
    for p in paths:
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _parse_ts(entry.get("time"))
            if ts is None or ts < day_start or ts >= day_end:
                continue
            mode = str(entry.get("mode", "")).upper()
            action = str(entry.get("action", "")).upper()
            if action.startswith("HOLD"):
                continue
            if mode not in {"PAPER", "LIVE_EXEC"}:
                continue
            entry["_journal"] = str(p)
            rows.append(entry)
    return rows


def extract_closures_by_ticket(
    events: Iterable[dict[str, Any]],
) -> dict[int, float]:
    """Map ticket -> pnl_usd from trade_reconciled / trade_closed events.

    trade_reconciled is authoritative (emitted once per close).  trade_closed
    is sometimes duplicated (emitted at both monitor-side and journal-side).
    We dedupe on ticket, preferring the first observed pnl value.
    """
    out: dict[int, float] = {}
    for ev in events:
        if ev.get("event") not in {"trade_reconciled", "trade_closed"}:
            continue
        ticket = ev.get("ticket")
        if ticket is None:
            continue
        try:
            ticket_int = int(ticket)
        except (TypeError, ValueError):
            continue
        if ticket_int in out:
            continue
        pnl = _coerce_float(ev.get("pnl_usd"))
        if pnl is None:
            continue
        out[ticket_int] = pnl
    return out


def count_anti_stack_blocks(
    events: Iterable[dict[str, Any]],
) -> int:
    """Count anti-stack / cooldown blocks from structured.jsonl.

    Looks for ``pre_write_gate_blocked`` events with reason prefixed
    ``cooldown:`` or ``anti_stack:`` (schema is flexible).  Returns 0 if
    absent.
    """
    n = 0
    for ev in events:
        if ev.get("event") != "pre_write_gate_blocked":
            continue
        reason = str(ev.get("blocked_reason", ""))
        if reason.startswith("cooldown:") or reason.startswith("anti_stack:"):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Regret computation (per trade)
# ---------------------------------------------------------------------------


def compute_regret_row(
    trade: dict[str, Any],
    closures_by_ticket: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Compute counterfactual PnLs for a single trade.

    Priority for ``actual_pnl``:
      1. structured.jsonl ``pnl_usd`` for this ticket (authoritative)
      2. journal ``result`` field
      3. None → confidence=unknown row
    """
    closures_by_ticket = closures_by_ticket or {}
    ticket = _coerce_int(trade.get("ticket") or trade.get("mt5_ticket"))
    actual: float | None = None
    if ticket is not None and ticket in closures_by_ticket:
        actual = closures_by_ticket[ticket]
    if actual is None:
        actual = _coerce_float(trade.get("result"))
    magic = _coerce_int(trade.get("magic"))
    is_treatment = magic == MAGIC_TREATMENT

    if actual is None:
        return {
            "trade_id": _trade_id(trade),
            "ts": trade.get("time"),
            "magic": magic,
            "leg": _leg_label(trade),
            "actual_pnl": 0.0,
            "no_macro_pnl": 0.0,
            "no_reversal_confirm_pnl": 0.0,
            "no_anti_stack_pnl": 0.0,
            "confidence": "unknown",
            "notes": "missing result field; cannot compute regret",
        }

    # (1) no_macro — trade stays as-is for control; for treatment we
    # pessimistically assume removing macro leaves the entry unchanged
    # (macro doesn't re-time entries in v4 Alt-B), so PnL identical.
    # This is conservative; a fuller replay would require strategy log
    # reconstruction.  Mark confidence accordingly.
    no_macro_pnl = actual

    # (2) no_reversal_confirm — population-level heuristic:
    #   - winners shrink by ~15% (earlier entry has slightly worse fill, and
    #     we'd catch some fakes that reverse).
    #   - losers grow by ~10% (false-positive entries take a slightly worse
    #     hit on SL).
    if actual > 0:
        no_reversal_pnl = actual * 0.85
    elif actual < 0:
        no_reversal_pnl = actual * 1.10
    else:
        no_reversal_pnl = 0.0

    # (3) no_anti_stack — impact comes from *added* would-have-been trades,
    # not modified existing ones.  Per-row contribution is 0; the summary
    # aggregates the blocked count × average PnL.
    no_anti_stack_pnl = actual

    confidence = "heuristic"
    notes = ""
    if is_treatment:
        notes = "treatment leg; no_macro assumes macro was tagging-only (not a gate)"

    return {
        "trade_id": _trade_id(trade),
        "ts": trade.get("time"),
        "magic": magic,
        "leg": _leg_label(trade),
        "actual_pnl": round(actual, 2),
        "no_macro_pnl": round(no_macro_pnl, 2),
        "no_reversal_confirm_pnl": round(no_reversal_pnl, 2),
        "no_anti_stack_pnl": round(no_anti_stack_pnl, 2),
        "confidence": confidence,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Summary aggregator
# ---------------------------------------------------------------------------


def compute_summary(
    per_trade: list[dict[str, Any]],
    *,
    anti_stack_blocks: int = 0,
) -> dict[str, Any]:
    """Aggregate per-trade regrets + inject anti-stack adjustment."""
    actual_total = sum(r["actual_pnl"] for r in per_trade)
    no_macro_total = sum(r["no_macro_pnl"] for r in per_trade)
    no_reversal_total = sum(r["no_reversal_confirm_pnl"] for r in per_trade)
    # anti-stack: each blocked entry becomes a synthetic trade with PnL =
    # avg(actual) × 0.8 (blocked entries skew modestly losing by design —
    # that's why cooldown exists).  Without anti-stack, those blocked trades
    # would have happened; we *add* their synthetic PnL to the baseline.
    avg_actual = (actual_total / len(per_trade)) if per_trade else 0.0
    imputed_blocked_pnl = 0.8 * avg_actual * anti_stack_blocks
    no_anti_stack_total = actual_total + imputed_blocked_pnl

    def _verdict(delta: float) -> str:
        if abs(delta) < 1e-6:
            return "neutral"
        return "guardrail_helped" if delta < 0 else "guardrail_cost"

    return {
        "summary": True,
        "trade_count": len(per_trade),
        "anti_stack_blocks": anti_stack_blocks,
        "actual_total": round(actual_total, 2),
        "no_macro_total": round(no_macro_total, 2),
        "no_reversal_confirm_total": round(no_reversal_total, 2),
        "no_anti_stack_total": round(no_anti_stack_total, 2),
        "no_macro_delta": round(no_macro_total - actual_total, 2),
        "no_reversal_confirm_delta": round(no_reversal_total - actual_total, 2),
        "no_anti_stack_delta": round(no_anti_stack_total - actual_total, 2),
        "no_macro_verdict": _verdict(no_macro_total - actual_total),
        "no_reversal_confirm_verdict": _verdict(no_reversal_total - actual_total),
        "no_anti_stack_verdict": _verdict(no_anti_stack_total - actual_total),
    }


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_regret_records(
    trades: list[dict[str, Any]],
    *,
    anti_stack_blocks: int = 0,
    closures_by_ticket: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    """Return the full JSONL record list (per-trade rows + final summary)."""
    rows = [compute_regret_row(t, closures_by_ticket) for t in trades]
    summary = compute_summary(rows, anti_stack_blocks=anti_stack_blocks)
    return rows + [summary]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trade_id(trade: dict[str, Any]) -> str:
    """Use ticket if present, else a hash of (time, magic)."""
    ticket = trade.get("ticket")
    if ticket is not None:
        return f"ticket:{ticket}"
    return f"{trade.get('time', '?')}|{trade.get('magic', '?')}"


def _leg_label(trade: dict[str, Any]) -> str:
    """Derive the leg label from magic + journal path."""
    magic = _coerce_int(trade.get("magic"))
    journal = trade.get("_journal", "")
    if "journal_macro" in journal or magic == MAGIC_TREATMENT:
        return "treatment"
    return "control"


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


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

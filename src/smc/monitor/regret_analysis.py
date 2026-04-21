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
    "synthesize_unassigned_closures",
    "compute_regret_row",
    "compute_regret_for_ticket",
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


def synthesize_unassigned_closures(
    trades: list[dict[str, Any]],
    closures_by_ticket: dict[int, float],
    *,
    target_date: date,
) -> list[dict[str, Any]]:
    """Return synthetic trade rows for closures whose ticket didn't match
    any journal row in ``trades``.

    In the Round 4 v5 DELEGATED_TO_EA architecture, the journal captures
    entry *signals* and never a closed PnL.  When the journal row's
    ``mt5_ticket`` is still null at write time (EA assigns it later),
    matching by ticket fails and we'd lose the closure entirely.  This
    function emits a placeholder trade per unmatched closure so regret can
    still be computed over the day's realised P&L — even if we can't
    attribute each closure to control vs treatment (leg="unassigned").
    """
    journal_tickets: set[int] = set()
    for t in trades:
        tk = t.get("ticket") or t.get("mt5_ticket")
        if tk is None:
            continue
        try:
            journal_tickets.add(int(tk))
        except (TypeError, ValueError):
            pass
    ts_default = datetime.combine(target_date, time(12, 0), tzinfo=timezone.utc).isoformat()
    out: list[dict[str, Any]] = []
    for ticket, pnl in closures_by_ticket.items():
        if ticket in journal_tickets:
            continue
        out.append({
            "time": ts_default,
            "action": "CLOSED",
            "mode": "LIVE_EXEC",
            "direction": "unknown",
            "magic": None,
            "ticket": ticket,
            "result": pnl,
            "_journal": "<unassigned>",
        })
    return out


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
# Single-ticket regret (O3 Telegram feed — R6 B2)
# ---------------------------------------------------------------------------


def compute_regret_for_ticket(
    ticket: int,
    closure: dict[str, Any],
    closures_by_ticket: dict[int, float] | None = None,
) -> float | None:
    """Scalar ``regret_delta`` in USD for a single reconciled closure.

    Definition: ``regret_delta = no_macro_pnl - actual_pnl``.

    Sign convention:
      - ``> 0``  →  macro filtering *hurt* this trade (counterfactual without
        macro would have earned more). Treatment leg only.
      - ``< 0``  →  macro *helped* (counterfactual would have been worse).
      - ``== 0.0`` →  Control leg by construction (macro was already off) or a
        Treatment trade where the heuristic concluded macro was neutral.
      - ``None`` →  cannot compute: ticket absent from closure, magic missing /
        unresolvable, no pnl in either ``closure`` or ``closures_by_ticket``.

    The caller (``trade_close_enrichment.build_trade_close_context``) passes
    the raw ``trade_reconciled`` / ``trade_closed`` event dict as ``closure``;
    ``closures_by_ticket`` is an optional pre-built map used when the full
    day's events have already been indexed.

    Kept deliberately thin: real regret math lives in ``compute_regret_row``;
    this wrapper just shapes a per-closure event into the journal-row shape
    that function expects, then extracts the ``no_macro - actual`` scalar.
    """
    try:
        ticket_int = int(ticket)
    except (TypeError, ValueError):
        return None

    closures_by_ticket = dict(closures_by_ticket) if closures_by_ticket else {}
    pnl = _coerce_float(closure.get("pnl_usd"))
    if pnl is not None and ticket_int not in closures_by_ticket:
        closures_by_ticket[ticket_int] = pnl

    if ticket_int not in closures_by_ticket:
        return None

    magic = _coerce_int(closure.get("magic"))
    if magic is None:
        return None
    if magic not in {MAGIC_CONTROL, MAGIC_TREATMENT}:
        return None

    synthetic_trade: dict[str, Any] = {
        "time": closure.get("ts") or closure.get("time"),
        "action": "CLOSED",
        "mode": "LIVE_EXEC",
        "direction": closure.get("direction") or "unknown",
        "magic": magic,
        "ticket": ticket_int,
    }

    row = compute_regret_row(synthetic_trade, closures_by_ticket=closures_by_ticket)
    if row.get("confidence") == "unknown":
        return None

    actual = row.get("actual_pnl")
    no_macro = row.get("no_macro_pnl")
    if actual is None or no_macro is None:
        return None
    return round(float(no_macro) - float(actual), 2)


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
    if journal == "<unassigned>":
        return "unassigned"
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

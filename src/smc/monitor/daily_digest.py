"""Round 3 Sprint 1 ops-usability: daily digest builder.

Assembles a single-page "today" summary for the dashboard by scanning:
  - journal JSONL (data/{SYMBOL}/journal/live_trades.jsonl)       trades + gate trips
  - structured.jsonl (logs/structured.jsonl)                        closed deals + warn/crit events
  - state JSON files (data/{SYMBOL}/*.json)                         halt snapshots
  - live_state.json + PID file                                      freshness + uptime

Pure functions with no FastAPI coupling so dashboard_server.py stays thin.
Every data source is tolerated missing — missing inputs yield zeros/nulls
and append a string to the ``warnings`` list; never raises.

See `.scratch/audit-r2/ops-daily-digest-spec.md` for the authoritative spec.
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def build_daily_digest(
    symbol: str,
    target_date: date,
    *,
    data_root: Path,
    log_root: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a DailyDigest dict for *symbol* on *target_date* (UTC).

    Parameters
    ----------
    symbol:
        "XAUUSD" / "BTCUSD" (validated by caller against SYMBOL_REGISTRY).
    target_date:
        UTC calendar date to summarise.
    data_root:
        ``data/{SYMBOL}/`` directory.
    log_root:
        ``logs/`` directory (structured.jsonl + rotations live here).
    now:
        Override current UTC timestamp — for deterministic tests. Defaults
        to ``datetime.now(timezone.utc)``.
    """
    current = now if now is not None else datetime.now(timezone.utc)
    warnings: list[str] = []

    journal = _scan_journal(
        data_root / "journal" / "live_trades.jsonl", target_date, warnings,
    )
    closures = _scan_structured_log(log_root, target_date, warnings)
    guards = _build_guards_snapshot(data_root, target_date, warnings)
    freshness = _read_freshness(data_root / "live_state.json", current, warnings)
    uptime_hours = _read_pid_uptime(data_root / "live_demo.pid", current)

    wins = closures["wins"]
    losses = closures["losses"]
    closed = wins + losses + closures["breakeven"]
    win_rate = (wins / closed * 100.0) if closed > 0 else None

    return {
        "symbol": symbol,
        "date": target_date.isoformat(),
        "trades_opened": journal["trades_opened"],
        "trades_closed": closed,
        "wins": wins,
        "losses": losses,
        "breakeven": closures["breakeven"],
        "gross_pnl_usd": round(closures["pnl_sum"], 2),
        "win_rate_pct": round(win_rate, 1) if win_rate is not None else None,
        "avg_win_usd": _round_or_none(closures["avg_win"]),
        "avg_loss_usd": _round_or_none(closures["avg_loss"]),
        "pre_write_gate_blocks": closures["gate_blocks"],
        "margin_blocks_count": closures["margin_blocks"],
        "asian_quota_blocks_count": closures["asian_quota_blocks"],
        "margin_cap_gate_trips": journal["margin_cap_trips"],
        "mt5_order_fails": closures["order_fails"],
        "consec_loss_halt_tripped": guards["consec_tripped_today"],
        "consec_loss_halt_tripped_at": guards["consec_tripped_at"],
        "phase1a_breaker_tripped": guards["phase1a_tripped_today"],
        "phase1a_breaker_tripped_at": guards["phase1a_tripped_at"],
        # ``drawdown_halt_active`` intentionally mirrors the ``_tripped`` suffix
        # of the consec/phase1a fields — all three are "did X fire today?".
        # Kept distinct from ``guards_current.drawdown`` which is a live
        # snapshot (currently null; populated when live_state.json grows a
        # ``drawdown_snapshot`` field in a later sprint).
        "drawdown_halt_active": guards["drawdown_active"],
        "drawdown_halt_reason": guards["drawdown_reason"],
        "guards_current": guards["snapshot"],
        "last_state_ts": freshness["ts"],
        "last_state_age_sec": freshness["age_sec"],
        "uptime_hours": uptime_hours,
        "cycles_today": freshness["cycle"],
        # Explicit note for dashboard tooltips — live_state.cycle is
        # process-start relative, not strict today-only. Dashboard can
        # surface this so the operator knows not to interpret it as
        # "bars processed today".
        "cycles_today_note": "process-start relative; not strict today-only",
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Journal scanner
# ---------------------------------------------------------------------------


def _scan_journal(
    path: Path, target_date: date, warnings: list[str],
) -> dict[str, int]:
    """Count trades opened and margin_cap gate trips on *target_date*."""
    result = {"trades_opened": 0, "margin_cap_trips": 0}
    if not path.exists():
        warnings.append("journal_missing")
        return result
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        warnings.append("journal_unreadable")
        return result

    parse_errors = 0
    day_start = datetime.combine(target_date, time(0, 0), tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        ts = _parse_ts(entry.get("time") or entry.get("ts"))
        if ts is None or ts < day_start or ts >= day_end:
            continue
        mode = str(entry.get("mode", "")).upper()
        if mode in {"PAPER", "LIVE_EXEC"}:
            result["trades_opened"] += 1
        elif mode == "MARGIN_GATED":
            result["margin_cap_trips"] += 1

    if parse_errors > 0:
        warnings.append(f"journal_parse_errors={parse_errors}")
    return result


# ---------------------------------------------------------------------------
# structured.jsonl scanner
# ---------------------------------------------------------------------------


def _scan_structured_log(
    log_root: Path, target_date: date, warnings: list[str],
) -> dict[str, Any]:
    """Aggregate closed-trade P&L + event counts from structured.jsonl.

    Rotated files (``structured.jsonl.YYYY-MM-DD``) are included so
    yesterday's data survives the midnight rotation.
    """
    result: dict[str, Any] = {
        "wins": 0,
        "losses": 0,
        "breakeven": 0,
        "pnl_sum": 0.0,
        "avg_win": None,
        "avg_loss": None,
        "gate_blocks": 0,
        "margin_blocks": 0,
        "asian_quota_blocks": 0,
        "order_fails": 0,
    }

    files = _structured_log_candidates(log_root, target_date)
    if not files:
        warnings.append("structured_log_missing")
        return result

    day_start = datetime.combine(target_date, time(0, 0), tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    win_pnls: list[float] = []
    loss_pnls: list[float] = []
    seen_tickets: set[int] = set()
    parse_errors = 0

    for path in files:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            payload = _parse_structured_line(line)
            if payload is None:
                parse_errors += 1
                continue
            ts = _parse_ts(payload.get("ts"))
            if ts is None or ts < day_start or ts >= day_end:
                continue
            event = payload.get("event", "")
            if event == "trade_reconciled":
                # Dedupe on ticket: trade_reconciled + trade_closed both emit the
                # same ticket.  Prefer the first occurrence.
                ticket = payload.get("ticket")
                if ticket is not None and ticket in seen_tickets:
                    continue
                if ticket is not None:
                    seen_tickets.add(ticket)
                pnl = _coerce_float(payload.get("pnl_usd"))
                if pnl is None:
                    continue
                result["pnl_sum"] += pnl
                if pnl > 0:
                    result["wins"] += 1
                    win_pnls.append(pnl)
                elif pnl < 0:
                    result["losses"] += 1
                    loss_pnls.append(pnl)
                else:
                    result["breakeven"] += 1
            elif event == "pre_write_gate_blocked":
                result["gate_blocks"] += 1
                reason = str(payload.get("blocked_reason", ""))
                if reason.startswith("margin_cap:"):
                    result["margin_blocks"] += 1
                elif reason.startswith("asian_quota:"):
                    result["asian_quota_blocks"] += 1
            elif event == "mt5_order_fail":
                result["order_fails"] += 1

    if win_pnls:
        result["avg_win"] = sum(win_pnls) / len(win_pnls)
    if loss_pnls:
        result["avg_loss"] = sum(loss_pnls) / len(loss_pnls)
    if parse_errors > 0:
        warnings.append(f"structured_log_parse_errors={parse_errors}")
    return result


def _structured_log_candidates(log_root: Path, target_date: date) -> list[Path]:
    """Return list of log files that could contain *target_date* entries.

    Current `structured.jsonl` always considered; rotated files are included
    when the date suffix matches target_date or the day after (rotation at
    midnight leaves the previous day's tail in the rotated file).
    """
    if not log_root.exists():
        return []
    candidates = []
    current = log_root / "structured.jsonl"
    if current.exists():
        candidates.append(current)
    for suffix in (target_date.isoformat(), (target_date + timedelta(days=1)).isoformat()):
        rotated = log_root / f"structured.jsonl.{suffix}"
        if rotated.exists() and rotated not in candidates:
            candidates.append(rotated)
    return candidates


def _parse_structured_line(line: str) -> dict[str, Any] | None:
    """Parse ``[SEVERITY] {json...}`` format."""
    line = line.strip()
    if not line:
        return None
    # Find the end of the severity tag and start of JSON body.
    if not line.startswith("["):
        return None
    rbracket = line.find("] ")
    if rbracket < 0:
        return None
    body = line[rbracket + 2:]
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


# ---------------------------------------------------------------------------
# Guards snapshot
# ---------------------------------------------------------------------------


def _build_guards_snapshot(
    data_root: Path, target_date: date, warnings: list[str],
) -> dict[str, Any]:
    """Read 3 state files; compute "tripped today?" from tripped_at timestamp."""
    consec = _load_json_quiet(data_root / "consec_loss_state.json")
    phase1a = _load_json_quiet(data_root / "phase1a_breaker_state.json")
    quota = _load_json_quiet(data_root / "asian_range_quota_state.json")

    consec_tripped_today = _is_tripped_on(consec, "tripped", "tripped_at", target_date)
    phase1a_tripped_today = _is_tripped_on(phase1a, "tripped", "tripped_at", target_date)

    snapshot = {
        "consec_loss": {
            "tripped": bool(consec.get("tripped", False)),
            "consec_losses": int(consec.get("consec_losses", 0) or 0),
            "tripped_at": consec.get("tripped_at"),
            "last_reset_date": consec.get("last_reset_date"),
        },
        "phase1a_breaker": {
            "tripped": bool(phase1a.get("tripped", False)),
            "losses": int(phase1a.get("losses", 0) or 0),
            "pnl_usd": _coerce_float(phase1a.get("pnl_usd")) or 0.0,
            "tripped_at": phase1a.get("tripped_at"),
            "last_reset_date": phase1a.get("last_reset_date"),
        },
        "asian_quota": {
            "last_open_date": quota.get("last_open_date"),
            "exhausted_today": quota.get("last_open_date") == target_date.isoformat(),
        },
        # DrawdownGuard is not persisted — digest leaves this null and relies
        # on structured.jsonl replay for running daily_pnl instead.
        "drawdown": None,
    }

    return {
        "consec_tripped_today": consec_tripped_today,
        "consec_tripped_at": consec.get("tripped_at") if consec_tripped_today else None,
        "phase1a_tripped_today": phase1a_tripped_today,
        "phase1a_tripped_at": phase1a.get("tripped_at") if phase1a_tripped_today else None,
        # drawdown_halt_active/reason left false/null — not persisted today.
        "drawdown_active": False,
        "drawdown_reason": None,
        "snapshot": snapshot,
    }


def _is_tripped_on(
    state: dict[str, Any], flag_key: str, ts_key: str, target_date: date,
) -> bool:
    """True iff ``state[flag_key]`` is true AND ``state[ts_key]`` falls on target_date."""
    if not state.get(flag_key):
        return False
    tripped_at = _parse_ts(state.get(ts_key))
    if tripped_at is None:
        return False
    return tripped_at.date() == target_date


# ---------------------------------------------------------------------------
# Freshness + uptime
# ---------------------------------------------------------------------------


def _read_freshness(
    path: Path, now: datetime, warnings: list[str],
) -> dict[str, Any]:
    state = _load_json_quiet(path)
    ts_str = state.get("timestamp")
    ts = _parse_ts(ts_str)
    age = int((now - ts).total_seconds()) if ts else None
    if age is not None and age > 5 * 60:
        warnings.append("state_stale_over_5min")
    cycle = state.get("cycle")
    return {
        "ts": ts_str,
        "age_sec": age,
        "cycle": int(cycle) if isinstance(cycle, (int, float)) else None,
    }


def _read_pid_uptime(pid_path: Path, now: datetime) -> float | None:
    """Derive uptime (hours) from PID file mtime. Missing → None."""
    if not pid_path.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(pid_path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None
    delta = now - mtime
    return round(delta.total_seconds() / 3600.0, 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json_quiet(path: Path) -> dict[str, Any]:
    """Read JSON dict from *path*; empty dict on any error."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_ts(value: Any) -> datetime | None:
    """Parse ISO-8601 string (tolerant of trailing Z) → UTC datetime."""
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


def _round_or_none(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None


__all__ = ["build_daily_digest"]

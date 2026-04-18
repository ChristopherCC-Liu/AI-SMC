"""Persist the last reconcile UTC timestamp across restarts.

audit-r2 ops #4 fix:

Before this module, `last_reconcile_ts` was initialised on every startup to
`now - 12h`.  When the process crashed or was restarted:
  1. MT5 reports the same closed deals that were already processed pre-crash.
  2. `fetch_closed_pnl_since(mt5, last_reconcile_ts)` returns them again.
  3. `consec_halt.record(pnl)` and `phase1a_breaker.record_trade_close(pnl)`
     are cumulative-side-effect functions — replaying a deal double-counts
     the loss.  `daily_pnl` also doubles.

Consequences observed by ops-sustain: consec_halt could trip after 1-2
real losses if a restart happened to fall between them, phase1a breaker
could open spuriously.  Silent, hard to diagnose.

Fix: persist the cursor to `data/{SYMBOL}/last_reconcile_ts.json` after
each successful reconcile via atomic rename.  Startup reloads the cursor;
missing / unreadable falls back to the original "now - 12h" default
behaviour (preserves first-boot semantics).

This module is intentionally small and MT5-free so it can be unit-tested
without touching the MetaTrader package.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

# First-boot lookback: how far to scan closed deals when no persisted
# cursor exists.
# ops-sustain Option D (audit-r2 ops #4): 1h chosen over the original
# 12h because:
#   - normal cycle cadence is M15 → any real reconcile gap is << 1h
#   - the cursor is persisted after every successful reconcile, so the
#     fallback only runs on *first ever* boot
#   - shorter window means even if we DO hit the fallback (fresh deploy),
#     the blast radius of a replay is bounded to 1h of history instead
#     of 12h.  With consec_halt / phase1a_breaker being cumulative
#     side-effect functions, shorter-is-safer.
_DEFAULT_LOOKBACK = timedelta(hours=1)


def load_reconcile_cursor(
    path: Path,
    *,
    now: datetime | None = None,
    default_lookback: timedelta = _DEFAULT_LOOKBACK,
) -> datetime:
    """Reload the cursor; fall back to ``now - default_lookback`` on any error.

    Accepts ``path`` not existing, unreadable, containing non-JSON, or
    missing/invalid ``ts`` field.  All failure modes converge to the
    same fallback so a partial write does not make the bug worse.
    """
    current_time = now if now is not None else datetime.now(timezone.utc)
    try:
        raw = Path(path).read_text(encoding="utf-8")
        payload = json.loads(raw)
        ts_str = payload.get("ts") if isinstance(payload, dict) else None
        if not ts_str:
            return current_time - default_lookback
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        # Sanity: never return a future timestamp (would hide newly-closed
        # deals).  Also never return something older than 7 days — if we
        # come back after a week-long outage, a 7-day scan is the safer
        # ceiling than arbitrarily old timestamps.
        if ts > current_time:
            return current_time - default_lookback
        max_lookback = timedelta(days=7)
        if current_time - ts > max_lookback:
            return current_time - max_lookback
        return ts
    except Exception:
        return current_time - default_lookback


def save_reconcile_cursor(path: Path, ts: datetime) -> None:
    """Atomic write of ``{"ts": "..."}`` JSON to *path*.

    Uses the standard atomic_write_json tmp+rename pattern.  On failure
    the destination file is unchanged — caller must handle exceptions if
    they care about persistence.  Typical live_demo caller wraps in try/
    except and log_warn to keep the trading loop alive.
    """
    from smc.monitor.state_io import atomic_write_json

    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    atomic_write_json(Path(path), {"ts": ts.isoformat()})

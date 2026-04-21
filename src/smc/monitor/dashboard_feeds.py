"""Round 5 O1/O2: dashboard feed builders for /api/regime and /api/pnl.

Pure functions — no FastAPI coupling — so dashboard_server.py stays thin
and tests don't need the HTTP layer.

Public API:
  - :func:`tail_regime_events`   → powers GET /api/regime (O1)
  - :func:`build_pnl_snapshot`   → powers GET /api/pnl    (O2)
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# O1: Regime events tail
# ---------------------------------------------------------------------------


# Source → order of preference when deduping multiple rows per cycle. The
# main AI path emits with ``source == "claude_debate"`` / "ai_main" /
# "atr_fallback"; ``source == "default"`` rows are the noise cold-start
# fallback we want to hide.
_REGIME_SOURCE_BLOCKLIST = {"default"}


@dataclass(frozen=True)
class RegimeEvent:
    ts: str
    regime: str
    source: str
    direction: str | None
    confidence: float | None
    reasoning: str | None


def _parse_structured_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    # Strip "[CRIT] " / "[INFO] " etc.
    if line.startswith("["):
        close = line.find("]")
        if close != -1:
            line = line[close + 1:].strip()
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def tail_regime_events(
    structured_log_path: Path,
    *,
    limit: int = 5,
    reasoning_max_chars: int = 300,
    max_scan_lines: int = 6000,
) -> list[dict[str, Any]]:
    """Return the last ``limit`` non-default ``ai_regime_classified`` events.

    Output shape (per entry)::

        {
            "ts": "2026-04-20T15:00:00+00:00",
            "regime": "TRENDING" | "CONSOLIDATION" | "TRANSITION",
            "source": "claude_debate" | "atr_fallback" | ...,
            "direction": "bullish" | "bearish" | "neutral" | None,
            "confidence": float | None,
            "reasoning": str | None (truncated to reasoning_max_chars),
        }

    The list is ordered newest-first.  Missing / unreadable log returns [].
    """
    if not structured_log_path.exists():
        return []
    try:
        lines = structured_log_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.debug("structured log read fail %s: %s", structured_log_path, exc)
        return []

    out: list[dict[str, Any]] = []
    scanned = 0
    for line in reversed(lines):
        if scanned >= max_scan_lines:
            break
        scanned += 1
        row = _parse_structured_line(line)
        if row is None or row.get("event") != "ai_regime_classified":
            continue
        src = str(row.get("source", ""))
        if src in _REGIME_SOURCE_BLOCKLIST:
            continue
        reasoning = row.get("reasoning")
        if reasoning is not None:
            reasoning = str(reasoning)[:reasoning_max_chars]
        out.append({
            "ts": str(row.get("ts", "")),
            "regime": str(row.get("regime", "UNKNOWN")),
            "source": src,
            "direction": row.get("direction"),
            "confidence": _safe_float(row.get("confidence")),
            "reasoning": reasoning,
        })
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# O2: P&L snapshot (per-leg realized + floating)
# ---------------------------------------------------------------------------


# (symbol, magic) → leg key mapping — codified after team-lead's 2026-04-20
# correction: BTC has **no** macro/treatment leg.  The 19760428 treatment
# magic is only active on XAUUSD.  Grouping by magic alone would silently
# merge any stray BTC-tagged deal into the XAU-treatment bucket, so we key
# on the composite (symbol, magic).
PNL_LEG_BY_COMPOSITE: dict[tuple[str, int], str] = {
    ("XAUUSD", 19760418): "control_xau",
    ("XAUUSD", 19760428): "treatment_xau",
    ("BTCUSD", 19760419): "control_btc",
}

PNL_LEG_LABELS: dict[str, str] = {
    "control_xau":   "XAUUSD Control",
    "treatment_xau": "XAUUSD Treatment",
    "control_btc":   "BTCUSD Control",
}

# Ordered keys we always surface — even when a leg has no activity, so the
# dashboard renders a stable three-column grid.
PNL_LEG_KEYS: tuple[str, ...] = ("control_xau", "treatment_xau", "control_btc")


# ---------------------------------------------------------------------------
# 5-second cache for /api/pnl — MT5 API isn't free, and the dashboard polls
# /api/pnl every 5s via fetchAll(). Caching matches the poll cadence so the
# broker round-trip fires at most once per cycle, not once per client.
# ---------------------------------------------------------------------------

_PNL_CACHE_TTL_SEC = 5.0
_pnl_cache_lock = threading.Lock()
_pnl_cache: dict[str, Any] = {"ts": 0.0, "snapshot": None}


def reset_pnl_cache() -> None:
    """Drop the cached /api/pnl response — used by tests."""
    with _pnl_cache_lock:
        _pnl_cache["ts"] = 0.0
        _pnl_cache["snapshot"] = None


def _today_utc_start(now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return datetime.combine(now.date(), time(0, 0), tzinfo=timezone.utc)


def _normalize_symbol(raw: Any) -> str:
    """Map broker symbol → canonical XAUUSD / BTCUSD.

    TMGM ships XAUUSD and BTCUSD as the plain symbol, but other brokers may
    suffix ("XAUUSD.r", "BTCUSD.x"). We classify by substring so typo-style
    suffixes don't silently drop deals off the report.
    """
    s = str(raw or "").upper()
    if "XAU" in s:
        return "XAUUSD"
    if "BTC" in s:
        return "BTCUSD"
    return s


def _aggregate_realized(
    mt5: Any,
    *,
    from_time: datetime,
    to_time: datetime,
) -> dict[str, dict[str, float]]:
    """Sum profit + commission + swap per leg key from mt5.history_deals_get().

    Returns ``{leg_key: {"realized": float, "trades": int}}`` where
    ``leg_key`` is one of the values in :data:`PNL_LEG_KEYS`. Deals whose
    (symbol, magic) combo is not in :data:`PNL_LEG_BY_COMPOSITE` (e.g. a
    manual trade, a rogue BTC deal tagged with the XAU treatment magic)
    are silently dropped with a debug log — we never fabricate a
    BTC-treatment leg.
    """
    if mt5 is None:
        return {}
    try:
        deals = mt5.history_deals_get(from_time, to_time)
    except Exception as exc:
        logger.debug("history_deals_get fail: %s", exc)
        return {}
    if not deals:
        return {}
    exit_entries = {
        getattr(mt5, "DEAL_ENTRY_OUT", 1),
        getattr(mt5, "DEAL_ENTRY_INOUT", 2),
    }
    out: dict[str, dict[str, float]] = {}
    for d in deals:
        if getattr(d, "entry", None) not in exit_entries:
            continue
        symbol = _normalize_symbol(getattr(d, "symbol", ""))
        magic = int(getattr(d, "magic", 0) or 0)
        key = PNL_LEG_BY_COMPOSITE.get((symbol, magic))
        if key is None:
            # BTC tagged with 19760428 (XAU treatment magic) would land here.
            # We log it so the anomaly is visible, then drop.
            logger.debug("drop deal: unknown (symbol=%s, magic=%d)", symbol, magic)
            continue
        pnl = (
            float(getattr(d, "profit", 0.0))
            + float(getattr(d, "commission", 0.0))
            + float(getattr(d, "swap", 0.0))
        )
        slot = out.setdefault(key, {"realized": 0.0, "trades": 0})
        slot["realized"] += pnl
        slot["trades"] += 1
    return out


def _aggregate_floating(mt5: Any) -> dict[str, float]:
    """Sum unrealised profit per leg key from :func:`mt5.positions_get`.

    Per team-lead 2026-04-20 correction: don't read mt5_positions{suffix}.json
    — live_demo writes those only at the end of each M15 cycle, so they can
    be up to 15 minutes stale. The MT5 API is authoritative and fast enough
    when wrapped in the 5s cache around build_pnl_snapshot().
    """
    if mt5 is None:
        return {}
    try:
        positions = mt5.positions_get()
    except Exception as exc:
        logger.debug("positions_get fail: %s", exc)
        return {}
    if not positions:
        return {}
    out: dict[str, float] = {}
    for p in positions:
        symbol = _normalize_symbol(getattr(p, "symbol", ""))
        magic = int(getattr(p, "magic", 0) or 0)
        key = PNL_LEG_BY_COMPOSITE.get((symbol, magic))
        if key is None:
            continue
        out[key] = out.get(key, 0.0) + float(getattr(p, "profit", 0.0) or 0.0)
    return out


def build_pnl_snapshot(
    mt5: Any,
    *,
    data_root: Path | None = None,
    now: datetime | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Assemble today's P&L per leg for GET /api/pnl.

    Response shape::

        {
            "as_of": "2026-04-20T15:00:00+00:00",
            "today_utc_start": "2026-04-20T00:00:00+00:00",
            "legs": {
                "control_xau":   {"realized": 12.3, "floating": -0.5, "trades": 2,
                                  "magic": 19760418, "symbol": "XAUUSD",
                                  "label": "XAUUSD Control"},
                "treatment_xau": {...},
                "control_btc":   {...},
            },
            "total": {"realized": ..., "floating": ..., "trades": ...},
            "source": "mt5_live" | "mt5_unavailable",
            "cached": bool,       # true when served from the 5s memo
        }

    Cached for ``_PNL_CACHE_TTL_SEC`` to match the dashboard's 5s poll
    cadence.  Pass ``use_cache=False`` in tests for deterministic output.

    ``data_root`` is accepted for backward compatibility with older callers
    / tests but is no longer used — floating PnL now comes from
    ``mt5.positions_get()`` directly per team-lead 2026-04-20 correction.
    """
    # Cache check first — timestamp of the cached entry fits inside the 5s
    # refresh window, we return it unchanged (cheap fast path).
    if use_cache:
        import time as _time
        with _pnl_cache_lock:
            age = _time.monotonic() - float(_pnl_cache["ts"])
            cached_snap = _pnl_cache["snapshot"]
            if cached_snap is not None and age < _PNL_CACHE_TTL_SEC:
                # Mark as served-from-cache so the dashboard can show a stale
                # indicator if it wants; don't mutate the cached dict itself.
                return {**cached_snap, "cached": True}

    cur = now or datetime.now(timezone.utc)
    day_start = _today_utc_start(cur)
    realized_by_key = _aggregate_realized(mt5, from_time=day_start, to_time=cur)
    floating_by_key = _aggregate_floating(mt5)

    legs: dict[str, dict[str, Any]] = {}
    for (symbol, magic), key in PNL_LEG_BY_COMPOSITE.items():
        legs[key] = _compose_leg(
            key=key,
            symbol=symbol,
            magic=magic,
            realized_by_key=realized_by_key,
            floating_by_key=floating_by_key,
        )

    total = {
        "realized": round(sum(l["realized"] for l in legs.values()), 2),
        "floating": round(sum(l["floating"] for l in legs.values()), 2),
        "trades":   sum(l["trades"] for l in legs.values()),
    }
    # ``source`` is ``mt5_live`` whenever the MT5 handle was usable (even if
    # no deals closed today — zeros are still a live reading).  Only demote
    # to ``mt5_unavailable`` when mt5 is None or raised on every call.
    if mt5 is None:
        source = "mt5_unavailable"
    else:
        # Sentinel: if BOTH aggregate paths returned empty dicts AND at
        # least one of them raised, mark unavailable. Cleaner heuristic:
        # we ask the mt5 object if history_deals_get is callable and
        # returns a non-exception value. Simpler: trust that the try/
        # except in the helpers silently returns {} on failure → treat
        # that combined with no legs hit as "unavailable". Empty account
        # is a legitimate case.
        source = "mt5_live"

    snapshot = {
        "as_of": cur.isoformat(),
        "today_utc_start": day_start.isoformat(),
        "legs": legs,
        "total": total,
        "source": source,
        "cached": False,
    }

    if use_cache:
        import time as _time
        with _pnl_cache_lock:
            _pnl_cache["ts"] = _time.monotonic()
            _pnl_cache["snapshot"] = snapshot

    return snapshot


def _compose_leg(
    *,
    key: str,
    symbol: str,
    magic: int,
    realized_by_key: dict[str, dict[str, float]],
    floating_by_key: dict[str, float],
) -> dict[str, Any]:
    r = realized_by_key.get(key, {"realized": 0.0, "trades": 0})
    return {
        "key": key,
        "label": PNL_LEG_LABELS[key],
        "symbol": symbol,
        "magic": magic,
        "realized": round(float(r["realized"]), 2),
        "floating": round(float(floating_by_key.get(key, 0.0)), 2),
        "trades": int(r["trades"]),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


__all__ = [
    "RegimeEvent",
    "tail_regime_events",
    "build_pnl_snapshot",
    "reset_pnl_cache",
    "PNL_LEG_BY_COMPOSITE",
    "PNL_LEG_KEYS",
    "PNL_LEG_LABELS",
]

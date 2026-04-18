"""Round 3 Sprint 2: live guards snapshot for dashboard traffic-light card.

Assembles the 4 guard states operators need to see at-a-glance:
  - consec_loss_halt     — "亏 3 单停" streak halter
  - phase1a_breaker      — Asian session circuit breaker
  - asian_range_quota    — 1/UTC-day Asian range open cap
  - drawdown_guard       — %-based daily loss + total drawdown backstop

Reads state JSON files and live_state.json; no MT5 coupling so pytest can
exercise the builder without the MetaTrader package.  Every source is
tolerated missing — absent inputs yield sensible defaults plus a warnings
list so the dashboard can surface "why is this zero?".

Traffic-light status (per-guard):
  - ``green``   — can_trade, well under limits
  - ``amber``   — can_trade but within early-warning band (approaching limit)
  - ``red``     — tripped / cannot trade
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Status = Literal["green", "amber", "red"]

# Amber thresholds — at what fraction of a limit do we start warning the
# operator?  Chosen conservatively so dashboard turns yellow before anything
# actually stops working.
_CONSEC_AMBER_FRACTION = 0.66  # e.g. 2/3 losses out of 3 → amber
_DAILY_LOSS_AMBER_FRACTION = 0.66
_DRAWDOWN_AMBER_FRACTION = 0.66
# Phase1a uses both count + PnL tripwires — amber if either is > this fraction
_PHASE1A_LOSS_AMBER_FRACTION = 0.66
_PHASE1A_PNL_AMBER_FRACTION = 0.66

# Defaults matching the live DrawdownGuard instantiation in live_demo.py:
#   DrawdownGuard(max_daily_loss_pct=3.0, max_drawdown_pct=10.0)
# Exposed as kwargs so callers can override per deploy (e.g. BTC strict mode).
_DEFAULT_MAX_DAILY_LOSS_PCT = 3.0
_DEFAULT_MAX_DRAWDOWN_PCT = 10.0

# Phase1a limits (mirror Phase1aCircuitBreaker _LOSS_LIMIT / _PNL_LIMIT_USD):
_PHASE1A_LOSS_LIMIT = 3
_PHASE1A_PNL_LIMIT_USD = -20.0


def build_guards_snapshot(
    symbol: str,
    *,
    data_root: Path,
    consec_loss_limit: int = 3,
    max_daily_loss_pct: float = _DEFAULT_MAX_DAILY_LOSS_PCT,
    max_drawdown_pct: float = _DEFAULT_MAX_DRAWDOWN_PCT,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a live guards snapshot dict for *symbol*.

    Parameters
    ----------
    symbol:
        Trading symbol name (XAUUSD / BTCUSD); used only for context.
    data_root:
        Path to ``data/{SYMBOL}/`` directory.
    consec_loss_limit:
        Symbol-specific consec-loss streak limit (audit-r3 R4).
    max_daily_loss_pct / max_drawdown_pct:
        Drawdown guard thresholds (live_demo ships defaults 3 / 10).
    now:
        Override current UTC datetime — for deterministic tests.
    """
    current = now if now is not None else datetime.now(timezone.utc)
    today = current.date()
    warnings: list[str] = []

    consec = _load_json_quiet(data_root / "consec_loss_state.json", warnings, "consec_loss_state")
    phase1a = _load_json_quiet(data_root / "phase1a_breaker_state.json", warnings, "phase1a_breaker_state")
    quota = _load_json_quiet(data_root / "asian_range_quota_state.json", warnings, "asian_range_quota_state")
    live = _load_json_quiet(data_root / "live_state.json", warnings, "live_state")

    consec_state = _build_consec_state(consec, consec_loss_limit)
    phase1a_state = _build_phase1a_state(phase1a)
    quota_state = _build_quota_state(quota, today)
    drawdown_state = _build_drawdown_state(
        live,
        max_daily_loss_pct=max_daily_loss_pct,
        max_drawdown_pct=max_drawdown_pct,
    )

    # Any red guard blocks trading; amber is informational.
    any_tripped = any(s["status"] == "red" for s in (consec_state, phase1a_state, quota_state, drawdown_state))

    return {
        "symbol": symbol,
        "ts": current.isoformat(),
        "can_trade": not any_tripped,
        "consec_halt": consec_state,
        "phase1a_breaker": phase1a_state,
        "asian_range_quota": quota_state,
        "drawdown_guard": drawdown_state,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Per-guard builders
# ---------------------------------------------------------------------------


def _build_consec_state(state: dict[str, Any], limit: int) -> dict[str, Any]:
    tripped = bool(state.get("tripped", False))
    losses = int(state.get("consec_losses", 0) or 0)
    status: Status
    if tripped:
        status = "red"
    elif limit > 0 and losses / limit >= _CONSEC_AMBER_FRACTION:
        status = "amber"
    else:
        status = "green"
    return {
        "status": status,
        "tripped": tripped,
        "losses": losses,
        "max_losses": limit,
        "tripped_at": state.get("tripped_at"),
        "last_reset_date": state.get("last_reset_date"),
    }


def _build_phase1a_state(state: dict[str, Any]) -> dict[str, Any]:
    tripped = bool(state.get("tripped", False))
    losses = int(state.get("losses", 0) or 0)
    pnl = _coerce_float(state.get("pnl_usd")) or 0.0
    status: Status
    if tripped:
        status = "red"
    else:
        loss_frac = losses / _PHASE1A_LOSS_LIMIT if _PHASE1A_LOSS_LIMIT > 0 else 0.0
        pnl_frac = (pnl / _PHASE1A_PNL_LIMIT_USD) if _PHASE1A_PNL_LIMIT_USD != 0 else 0.0
        if loss_frac >= _PHASE1A_LOSS_AMBER_FRACTION or pnl_frac >= _PHASE1A_PNL_AMBER_FRACTION:
            status = "amber"
        else:
            status = "green"
    return {
        "status": status,
        "tripped": tripped,
        "losses": losses,
        "pnl_usd": round(pnl, 2),
        "max_losses": _PHASE1A_LOSS_LIMIT,
        "pnl_limit_usd": _PHASE1A_PNL_LIMIT_USD,
        "tripped_at": state.get("tripped_at"),
        "last_reset_date": state.get("last_reset_date"),
    }


def _build_quota_state(state: dict[str, Any], today) -> dict[str, Any]:
    last_open_iso = state.get("last_open_date")
    exhausted = last_open_iso == today.isoformat()
    status: Status = "red" if exhausted else "green"
    return {
        "status": status,
        "exhausted": exhausted,
        "last_open_date": last_open_iso,
    }


def _build_drawdown_state(
    live: dict[str, Any],
    *,
    max_daily_loss_pct: float,
    max_drawdown_pct: float,
) -> dict[str, Any]:
    """Derive drawdown state from live_state.drawdown_snapshot (soft-dep).

    Currently live_demo does NOT write drawdown_snapshot (Round 3 Sprint 2
    soft dep).  We fall back to status=green with null numbers, plus a
    ``source=unavailable`` flag so the dashboard can render 灰 not 绿.
    """
    snap = live.get("drawdown_snapshot") if isinstance(live, dict) else None
    if not isinstance(snap, dict):
        return {
            "status": "green",
            "source": "unavailable",
            "can_trade": True,
            "daily_pnl_usd": None,
            "daily_loss_pct": None,
            "peak_balance": None,
            "balance": None,
            "total_drawdown_pct": None,
            "max_daily_loss_pct": max_daily_loss_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "rejection_reason": None,
        }

    daily_pnl = _coerce_float(snap.get("daily_pnl_usd"))
    daily_loss_pct = _coerce_float(snap.get("daily_loss_pct")) or 0.0
    total_dd_pct = _coerce_float(snap.get("total_drawdown_pct")) or 0.0
    can_trade = bool(snap.get("can_trade", True))
    rejection_reason = snap.get("rejection_reason")

    status: Status
    if not can_trade:
        status = "red"
    else:
        # amber if either daily loss or total dd has eaten >66% of the limit
        amber = False
        if max_daily_loss_pct > 0 and (daily_loss_pct / max_daily_loss_pct) >= _DAILY_LOSS_AMBER_FRACTION:
            amber = True
        if max_drawdown_pct > 0 and (total_dd_pct / max_drawdown_pct) >= _DRAWDOWN_AMBER_FRACTION:
            amber = True
        status = "amber" if amber else "green"

    return {
        "status": status,
        "source": "live_state",
        "can_trade": can_trade,
        "daily_pnl_usd": round(daily_pnl, 2) if daily_pnl is not None else None,
        "daily_loss_pct": round(daily_loss_pct, 2),
        "peak_balance": _coerce_float(snap.get("peak_balance")),
        "balance": _coerce_float(snap.get("balance")),
        "total_drawdown_pct": round(total_dd_pct, 2),
        "max_daily_loss_pct": max_daily_loss_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "rejection_reason": rejection_reason,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json_quiet(
    path: Path, warnings: list[str], key: str,
) -> dict[str, Any]:
    if not path.exists():
        warnings.append(f"{key}_missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        warnings.append(f"{key}_unreadable")
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["build_guards_snapshot"]

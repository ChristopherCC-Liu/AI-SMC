"""AI-SMC strategy signal server — serves cached live_state.json to MQL5 EA.

Decouples Python strategy layer from MT5 IPC. live_demo.py computes signals
and writes data/{symbol}/live_state.json; this FastAPI server exposes them
over HTTP so the MQL5 EA (AISMCReceiver.mq5) running inside the MT5
terminal can poll + execute orders.

Bind: 127.0.0.1:8080 (localhost only; MT5 WebRequest URL whitelist must
include http://127.0.0.1:8080).

Endpoints:
  GET  /healthz              — liveness probe
  GET  /signal?symbol=...    — unified signals array (control + treatment)
  POST /test/signal          — inject a test signal without polluting live_state.json
  GET  /test/signal?symbol=  — retrieve last injected test signal
  GET  /status               — per-leg health (age, fresh, cycle) for dashboards
  GET  /                     — service info

/signal response shape (audit-r4 v5 Option B unified signals array):

  {
    "symbol": "XAUUSD",
    "ts": "<most-recent-leg-ts>",
    "fresh": true,
    "signals": [
      {
        "leg": "",                 // journal_suffix ("" = control, "_macro" = treatment)
        "magic": 19760418,
        "cycle": 42,
        "ts": "2026-04-18T10:45:00Z",
        "age_sec": 12.3,
        "fresh": true,
        "action": "RANGE SELL",
        "trading_mode": "ranging",
        "direction": "short",
        "entry": 4850.0,
        "sl": 4900.0,
        "tp": 4800.0,
        "confidence": 0.65,
        "lot": 0.02,
        "signal_id": "XAUUSD__42_2026-04-18T10:45:00Z"
      },
      {
        "leg": "_macro",
        "magic": 19760428,
        ...
      }
    ],
    // Backward-compat flat fields (mirror of signals[0] — control leg).
    // Preserved so legacy EAs that only read top-level keys still function
    // during staged rollout.
    "action": "RANGE SELL",
    "trading_mode": "ranging",
    ...
  }
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn


app = FastAPI(title="AI-SMC Signal Server", docs_url=None, redoc_url=None)
DATA_ROOT = Path("data")

SYMBOL_WHITELIST = {"XAUUSD", "BTCUSD"}
SIGNAL_FRESH_MAX_AGE_SEC = 900  # one M15 cycle
MIN_LOT = 0.01  # MT5 minimum lot fallback

# Per-symbol default magic — must match instruments/{symbol}.py.  Kept local
# so the server stays decoupled from the smc package import chain (avoids
# MetaTrader5 import on import of strategy_server).
_DEFAULT_MAGIC: dict[str, int] = {
    "XAUUSD": 19760418,
    "BTCUSD": 19760419,
}

# audit-r4 v5 Option B: leg suffix → macro magic override (treatment leg).
# The control leg uses the per-symbol default from _DEFAULT_MAGIC; the
# treatment leg uses SMC_MACRO_MAGIC so broker reconcile can split deals
# per-leg on the same TMGM Demo account.
_MACRO_MAGIC_DEFAULT = 19760428


def _macro_magic() -> int:
    """Resolve the treatment-leg magic at request time (env-overridable)."""
    try:
        return int(os.environ.get("SMC_MACRO_MAGIC", _MACRO_MAGIC_DEFAULT))
    except (TypeError, ValueError):
        return _MACRO_MAGIC_DEFAULT


# Legs to expose on /signal.  Order is deterministic — control first so
# backward-compat flat fields mirror the control signal.
_LEGS: tuple[str, ...] = ("", "_macro")

# In-memory test signal store: symbol → payload.  Not persisted; never
# touches live_state.json or the live EA polling path (/signal).
_TEST_SIGNALS: dict[str, dict[str, Any]] = {}


def _iso_age_seconds(iso_ts: str | None) -> float:
    if not iso_ts:
        return 1e9
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(iso_ts)).total_seconds()
    except Exception:
        return 1e9


def _load_state(symbol: str, suffix: str = "") -> dict[str, Any] | None:
    """Read the live_state{suffix}.json for the given symbol.

    Returns None when the file is missing or unparseable — callers treat
    this as "no signal yet" rather than a hard error.
    """
    path = DATA_ROOT / symbol / f"live_state{suffix}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _resolve_magic(symbol: str, suffix: str) -> int:
    """Pick the magic number for (symbol, suffix) leg."""
    if suffix:
        return _macro_magic()
    return _DEFAULT_MAGIC.get(symbol, _MACRO_MAGIC_DEFAULT)


def _build_leg_signal(symbol: str, suffix: str) -> dict[str, Any]:
    """Project a live_state{suffix}.json into the wire-format signal dict.

    Always returns a complete dict (never raises): missing files or
    legacy states produce an explicit HOLD leg with ``fresh=false`` so the
    EA can distinguish "leg unconfigured" from "leg saying HOLD today".
    """
    magic = _resolve_magic(symbol, suffix)
    state = _load_state(symbol, suffix)
    if state is None:
        return {
            "leg": suffix,
            "magic": magic,
            "cycle": None,
            "ts": None,
            "age_sec": 1e9,
            "fresh": False,
            "action": "HOLD",
            "trading_mode": None,
            "direction": None,
            "entry": None,
            "sl": None,
            "tp": None,
            "confidence": 0.0,
            "lot": None,
            "signal_id": f"{symbol}_{suffix}_none",
            "reason": "no_state_yet",
        }

    ts_iso = state.get("timestamp")
    age_sec = _iso_age_seconds(ts_iso)
    best = state.get("best_setup") or {}
    action = state.get("action", "HOLD")

    # Resolve lot: prefer best_setup.position_size_lots; fall back to MIN_LOT.
    # On HOLD signals, expose null so the EA can distinguish "no trade" explicitly.
    if action == "HOLD":
        lot: float | None = None
    else:
        raw_lot = best.get("position_size_lots")
        lot = float(raw_lot) if raw_lot is not None and float(raw_lot) > 0 else MIN_LOT

    # Round 5 A-track Task #8: regime-dynamic trailing SL.  Forward the
    # per-regime activate_r + distance_r that live_demo computed from the
    # AI regime (TREND → 0.3/0.5, CONSOLIDATION → 0.5/0.3, TRANSITION →
    # 0.5/0.5, ATH_BREAKOUT → 0.8/0.7).  Backward-compat: when absent the
    # EA falls back to its own TrailActivateR/TrailDistanceR inputs.
    trail_activate_r = best.get("trail_activate_r")
    trail_distance_r = best.get("trail_distance_r")
    regime_label = best.get("regime_label")

    return {
        "leg": suffix,
        "magic": magic,
        "cycle": state.get("cycle"),
        "ts": ts_iso,
        "age_sec": round(age_sec, 1),
        "fresh": age_sec < SIGNAL_FRESH_MAX_AGE_SEC,
        "action": action,
        "trading_mode": state.get("trading_mode"),
        "direction": best.get("direction"),
        "entry": best.get("entry"),
        "sl": best.get("sl"),
        "tp": best.get("tp1") or best.get("tp"),
        "confidence": best.get("confluence") or best.get("confidence") or 0.0,
        "lot": lot,
        # signal_id must be per-leg unique so the EA dedupes independently
        # per magic (control cooldown must not interfere with treatment).
        "signal_id": f"{symbol}_{suffix}_{state.get('cycle', 0)}_{ts_iso or ''}",
        # Round 5 A-track Task #8: regime-aware trailing SL per-ticket.
        "trail_activate_r": trail_activate_r,
        "trail_distance_r": trail_distance_r,
        "regime_label": regime_label,
    }


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}


@app.get("/signal")
def get_signal(symbol: str = Query(..., min_length=3, max_length=12)) -> JSONResponse:
    """Unified signals array — returns one entry per leg (control + treatment).

    Backward-compat: flat top-level fields mirror the control-leg (suffix="")
    signal so legacy EAs that read `action`, `entry`, etc. directly still
    function during staged rollout.  New EAs iterate the ``signals`` array
    and send orders with signal["magic"].
    """
    sym = symbol.upper()
    if sym not in SYMBOL_WHITELIST:
        raise HTTPException(400, f"symbol must be one of {sorted(SYMBOL_WHITELIST)}")

    signals = [_build_leg_signal(sym, leg) for leg in _LEGS]
    control = signals[0]  # suffix="" is always first by _LEGS ordering

    # Aggregate freshness — envelope is fresh if ANY leg is fresh so the EA
    # keeps polling the treatment leg even when control is stale (and vice
    # versa).  Most recent timestamp drives the top-level ts.
    any_fresh = any(sig["fresh"] for sig in signals)
    ts_values = [sig["ts"] for sig in signals if sig.get("ts")]
    envelope_ts = max(ts_values) if ts_values else None

    return JSONResponse({
        "symbol": sym,
        "ts": envelope_ts,
        "fresh": any_fresh,
        "signals": signals,
        # --- backward-compat flat fields (mirror of control leg) ---
        "cycle": control.get("cycle"),
        "age_sec": control.get("age_sec"),
        "action": control.get("action"),
        "trading_mode": control.get("trading_mode"),
        "direction": control.get("direction"),
        "entry": control.get("entry"),
        "sl": control.get("sl"),
        "tp": control.get("tp"),
        "confidence": control.get("confidence"),
        "lot": control.get("lot"),
        "signal_id": control.get("signal_id"),
    })


@app.get("/status")
def get_status() -> JSONResponse:
    """Per-leg health for dashboards / ops.

    Returns cycle + age + fresh flag for each (symbol, leg) so operators can
    spot a legconfiguration drift at a glance (e.g. treatment frozen while
    control keeps ticking).
    """
    body: dict[str, Any] = {"ts": datetime.now(timezone.utc).isoformat(), "legs": []}
    for sym in sorted(SYMBOL_WHITELIST):
        for leg in _LEGS:
            state = _load_state(sym, leg)
            ts_iso = state.get("timestamp") if state else None
            age_sec = _iso_age_seconds(ts_iso)
            body["legs"].append({
                "symbol": sym,
                "leg": leg,
                "magic": _resolve_magic(sym, leg),
                "cycle": state.get("cycle") if state else None,
                "ts": ts_iso,
                "age_sec": round(age_sec, 1),
                "fresh": age_sec < SIGNAL_FRESH_MAX_AGE_SEC,
                "action": state.get("action") if state else None,
            })
    return JSONResponse(body)


@app.post("/test/signal")
def post_test_signal(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Lead/QA use: inject a test signal WITHOUT polluting live_state.json.

    Body: {"symbol": "BTCUSD", "action": "BUY", "entry": ..., "sl": ...,
           "tp": ..., "confidence": ..., "lot": ...}

    The EA's production path (/signal) is unaffected; test signals are only
    retrievable via GET /test/signal?symbol=...
    """
    sym = str(payload.get("symbol", "")).upper()
    if sym not in SYMBOL_WHITELIST:
        raise HTTPException(400, f"symbol must be one of {sorted(SYMBOL_WHITELIST)}")
    now_iso = datetime.now(timezone.utc).isoformat()
    _TEST_SIGNALS[sym] = {
        **payload,
        "symbol": sym,
        "ts": now_iso,
        "fresh": True,
        "test_mode": True,
        "signal_id": f"TEST_{sym}_{int(datetime.now(timezone.utc).timestamp())}",
    }
    return {"ok": True, "symbol": sym}


@app.get("/test/signal")
def get_test_signal(symbol: str = Query(..., min_length=3, max_length=12)) -> JSONResponse:
    """Retrieve last injected test signal for the given symbol (in-memory only)."""
    sym = symbol.upper()
    if sym not in _TEST_SIGNALS:
        return JSONResponse({
            "symbol": sym,
            "action": "HOLD",
            "fresh": False,
            "reason": "no_test_signal",
        })
    return JSONResponse(_TEST_SIGNALS[sym])


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "AI-SMC Signal Server",
        "endpoints": [
            "/signal?symbol=XAUUSD",
            "/signal?symbol=BTCUSD",
            "/status",
            "/healthz",
            "POST /test/signal",
            "GET  /test/signal?symbol=XAUUSD",
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")

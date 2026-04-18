"""AI-SMC strategy signal server — serves cached live_state.json to MQL5 EA.

Decouples Python strategy layer from MT5 IPC. live_demo.py computes signals
and writes data/{symbol}/live_state.json; this FastAPI server exposes them
over HTTP so the MQL5 EA (AISMCReceiver.mq5) running inside the MT5
terminal can poll + execute orders.

Bind: 127.0.0.1:8080 (localhost only; MT5 WebRequest URL whitelist must
include http://127.0.0.1:8080).

Endpoints:
  GET  /healthz              — liveness probe
  GET  /signal?symbol=...    — latest signal for given symbol (reads live_state.json)
  POST /test/signal          — inject a test signal without polluting live_state.json
  GET  /test/signal?symbol=  — retrieve last injected test signal
  GET  /                     — service info

Signal format (JSON):
  {
    "symbol": "XAUUSD",
    "cycle": 42,
    "ts": "2026-04-18T10:45:00Z",
    "age_sec": 12.3,
    "fresh": true,             // age < 900s (1 M15 cycle)
    "action": "RANGE SELL",    // HOLD / BUY / SELL / RANGE BUY / RANGE SELL
    "trading_mode": "ranging",
    "direction": "short",
    "entry": 4850.0,
    "sl": 4900.0,
    "tp": 4800.0,
    "confidence": 0.65,
    "lot": 0.02,               // position size from live_state best_setup; null on HOLD
    "signal_id": "XAUUSD_42_2026-04-18T10:45:00Z"  // EA dedupes
  }
"""
from __future__ import annotations
import json
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


def _load_state(symbol: str) -> dict[str, Any] | None:
    path = DATA_ROOT / symbol / "live_state.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}


@app.get("/signal")
def get_signal(symbol: str = Query(..., min_length=3, max_length=12)) -> JSONResponse:
    sym = symbol.upper()
    if sym not in SYMBOL_WHITELIST:
        raise HTTPException(400, f"symbol must be one of {sorted(SYMBOL_WHITELIST)}")
    state = _load_state(sym)
    if state is None:
        return JSONResponse({
            "symbol": sym,
            "action": "HOLD",
            "fresh": False,
            "reason": "no_state_yet",
        })

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

    return JSONResponse({
        "symbol": sym,
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
        "signal_id": f"{sym}_{state.get('cycle', 0)}_{ts_iso or ''}",
    })


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
            "/healthz",
            "POST /test/signal",
            "GET  /test/signal?symbol=XAUUSD",
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")

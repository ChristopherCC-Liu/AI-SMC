"""
AI-SMC Dashboard Web Server (MVP).

轻量 FastAPI server, 读 data/{SYMBOL}/*.json 暴露 REST API + serve dashboard/index.html.
端口 8765 (PRD § 7 契约)。

Stage 4: 支持 `?symbol=XAUUSD|BTCUSD` 切换，单 port 8765 服务双 symbol。
默认 symbol: XAUUSD (兼容旧行为)。

启动:
    python scripts/dashboard_server.py
然后浏览器打开 http://localhost:8765 或 http://localhost:8765?symbol=BTCUSD
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DASHBOARD_HTML = ROOT / "dashboard" / "index.html"

STALE_THRESHOLD_SEC = 5 * 60  # 5 分钟无更新视为 stale

app = FastAPI(title="AI-SMC Dashboard", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _symbol_data_root(symbol: str) -> Path:
    """Validate symbol against SYMBOL_REGISTRY; return data/{SYMBOL}/ path.

    Falls back to XAUUSD with a warning if the symbol is unknown so that
    stale/invalid query params never crash the server.
    """
    from smc.instruments import get_instrument_config  # lazy import — avoids circular at module load
    try:
        get_instrument_config(symbol)
    except KeyError:
        logger.warning("Unknown symbol %r — falling back to XAUUSD", symbol)
        symbol = "XAUUSD"
    return DATA / symbol


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _freshness(state: dict | None) -> dict:
    if not state or not state.get("timestamp"):
        return {"fresh": False, "age_sec": None, "stale": True}
    try:
        ts = datetime.fromisoformat(state["timestamp"].replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return {"fresh": False, "age_sec": None, "stale": True}
    age = (datetime.now(timezone.utc) - ts).total_seconds()
    return {"fresh": age <= STALE_THRESHOLD_SEC, "age_sec": int(age), "stale": age > STALE_THRESHOLD_SEC}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/symbols")
def list_symbols() -> JSONResponse:
    """Return all registered trading symbols."""
    from smc.instruments import SYMBOL_REGISTRY  # lazy import
    return JSONResponse({"symbols": sorted(SYMBOL_REGISTRY.keys())})


@app.get("/api/state")
def get_state(symbol: str = Query(default="XAUUSD")) -> JSONResponse:
    root = _symbol_data_root(symbol)
    state = _read_json(root / "live_state.json")
    ai = _read_json(root / "ai_analysis.json")
    pause_flag = root / "trading_paused.flag"
    return JSONResponse({
        "state": state,
        "ai": ai,
        "freshness": _freshness(state),
        "trading_paused": pause_flag.exists(),
        "server_time": datetime.now(timezone.utc).isoformat(),
    })


@app.get("/api/positions")
def get_positions(symbol: str = Query(default="XAUUSD")) -> JSONResponse:
    """Real-time broker positions + halt state for the Hero card.

    ``live_demo`` writes ``data/{SYMBOL}/mt5_positions.json`` at the end of
    each M15 cycle via the MT5 positions adapter. Empty list (not 404) on a
    missing file keeps the dashboard usable during cold starts / market-closed.

    Round 6 B4: each position row is enriched with ``trail_activate_r`` +
    ``trail_distance_r`` + ``regime_at_open`` — derived from
    :func:`smc.ai.param_router.get_trail_params` by tail-scanning
    ``logs/structured.jsonl`` for the last ``ai_regime_classified`` event
    whose ``ts <= position.open_time``. Unknown / missing regime → fields
    are ``None`` and the UI renders an empty pill. Never raises: attachment
    failure falls back to the raw broker rows.
    """
    from smc.monitor.dashboard_feeds import attach_trail_params_to_positions

    root = _symbol_data_root(symbol)
    data = _read_json(root / "mt5_positions.json") or {}
    halt = _read_json(root / "consec_loss_state.json") or {}
    raw_positions = data.get("positions", []) or []
    try:
        enriched_positions = attach_trail_params_to_positions(
            raw_positions,
            structured_log_path=ROOT / "logs" / "structured.jsonl",
        )
    except Exception as exc:
        logger.warning("trail attach fail, returning raw positions: %s", exc)
        enriched_positions = raw_positions
    return JSONResponse({
        "ts": data.get("ts"),
        "positions": enriched_positions,
        "halt": {
            "tripped": bool(halt.get("tripped", False)),
            "consec_losses": int(halt.get("consec_losses", 0)),
            "tripped_at": halt.get("tripped_at"),
        },
    })


@app.get("/api/journal")
def get_journal(
    symbol: str = Query(default="XAUUSD"),
    limit: int = Query(default=50, ge=1, le=500),
) -> JSONResponse:
    root = _symbol_data_root(symbol)
    journal_path = root / "journal" / "live_trades.jsonl"
    if not journal_path.exists():
        return JSONResponse({"trades": []})
    try:
        lines = journal_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return JSONResponse({"trades": []})
    trades = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            trades.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return JSONResponse({"trades": trades})


@app.get("/healthz")
def healthz() -> JSONResponse:
    """Round 3 Sprint 2: liveness probe for watchdog_smart DashboardWeb branch.

    Returns 200 when:
      - FastAPI process is responding
      - JSON parser is working (round-trip check)
      - Data root is accessible

    Never calls MT5 or heavy aggregations so it stays cheap (<5ms).  Parallel
    to strategy_server.py:/healthz (audit-r1 P0-2).
    """
    ok = True
    probes: dict[str, object] = {}
    try:
        json.loads("{}")
        probes["json_parser"] = "ok"
    except Exception as exc:  # pragma: no cover — json.loads("{}") never fails
        ok = False
        probes["json_parser"] = f"fail: {exc}"
    try:
        probes["data_root_exists"] = DATA.exists()
        if not DATA.exists():
            ok = False
    except Exception as exc:
        ok = False
        probes["data_root_exists"] = f"fail: {exc}"
    payload = {
        "ok": ok,
        "service": "AI-SMC Dashboard",
        "ts": datetime.now(timezone.utc).isoformat(),
        "probes": probes,
    }
    return JSONResponse(payload, status_code=200 if ok else 503)


@app.get("/api/guards")
def get_guards(symbol: str = Query(default="XAUUSD")) -> JSONResponse:
    """Round 3 Sprint 2: live guards traffic-light snapshot.

    Returns 4 guard states (consec / phase1a / asian_quota / drawdown) with
    per-guard status ``green`` / ``amber`` / ``red`` so the dashboard can
    render "why can't I trade right now" at a glance.
    """
    from smc.instruments import get_instrument_config
    from smc.monitor.guards_snapshot import build_guards_snapshot

    root = _symbol_data_root(symbol)
    # Per-symbol consec_loss_limit (R4) — fall back to 3 if cfg is missing
    # (_symbol_data_root already normalises unknown symbols to XAUUSD).
    try:
        cfg = get_instrument_config(symbol)
        consec_limit = getattr(cfg, "consec_loss_limit", 3)
    except KeyError:
        consec_limit = 3
    snap = build_guards_snapshot(symbol, data_root=root, consec_loss_limit=consec_limit)
    return JSONResponse(snap)


@app.get("/api/daily_digest")
def get_daily_digest(
    symbol: str = Query(default="XAUUSD"),
    date_str: str | None = Query(default=None, alias="date"),
) -> JSONResponse:
    """Round 3 Sprint 1: one-page "today" summary for老板 5min scan.

    Response schema per `.scratch/audit-r2/ops-daily-digest-spec.md` §2.
    Builder tolerates missing data sources; returns zeros + warnings list.
    """
    from datetime import date as date_cls
    from smc.monitor.daily_digest import build_daily_digest

    try:
        target = date_cls.fromisoformat(date_str) if date_str else datetime.now(timezone.utc).date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format, expected YYYY-MM-DD, got: {date_str!r}",
        )
    root = _symbol_data_root(symbol)
    digest = build_daily_digest(symbol, target, data_root=root, log_root=ROOT / "logs")
    return JSONResponse(digest)


@app.get("/api/regime")
def get_regime(limit: int = Query(default=5, ge=1, le=20)) -> JSONResponse:
    """Round 5 O1: tail last N ``ai_regime_classified`` events.

    Cross-symbol (regime events are written by whichever leg runs the AI
    classifier — not partitioned per symbol). Filter out ``source=="default"``
    cold-start noise so the dashboard only surfaces real regime calls.
    """
    from smc.monitor.dashboard_feeds import tail_regime_events

    events = tail_regime_events(ROOT / "logs" / "structured.jsonl", limit=limit)
    return JSONResponse({"events": events, "server_time": datetime.now(timezone.utc).isoformat()})


@app.get("/api/pnl")
def get_pnl() -> JSONResponse:
    """Round 5 O2: today's P&L per leg (control_xau / treatment_xau / control_btc).

    Aggregates realized P&L from ``mt5.history_deals_get(today_00Z, now)``
    grouped by composite (symbol, magic) key — BTC has no treatment leg,
    so ``treatment_btc`` never appears.  Floating P&L comes from
    ``mt5.positions_get()`` directly.  Not symbol-scoped — response
    carries all three legs side-by-side.  5s cache inside
    ``build_pnl_snapshot()`` matches the dashboard poll cadence so MT5
    is hit at most once per cycle.
    """
    from smc.monitor.dashboard_feeds import build_pnl_snapshot

    try:
        import MetaTrader5 as mt5_mod  # type: ignore
    except ImportError:
        mt5_mod = None
    snapshot = build_pnl_snapshot(mt5_mod, data_root=DATA)
    return JSONResponse(snapshot)


@app.get("/api/config")
def get_config(symbol: str = Query(default="XAUUSD")) -> JSONResponse:
    root = _symbol_data_root(symbol)
    cfg = _read_json(root / "user_config.json") or {}
    return JSONResponse(cfg)


@app.post("/api/config")
async def set_config(
    request: Request,
    symbol: str = Query(default="XAUUSD"),
) -> JSONResponse:
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    root = _symbol_data_root(symbol)
    config_path = root / "user_config.json"
    body["updated_at"] = datetime.now(timezone.utc).isoformat()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    return JSONResponse({"ok": True, "path": str(config_path)})


@app.post("/api/toggle_trading")
async def toggle_trading(
    request: Request,
    symbol: str = Query(default="XAUUSD"),
) -> JSONResponse:
    body = await request.json()
    paused = bool(body.get("paused", False))
    root = _symbol_data_root(symbol)
    flag = root / "trading_paused.flag"
    if paused:
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    else:
        flag.unlink(missing_ok=True)
    return JSONResponse({"paused": paused})


@app.get("/")
def index() -> FileResponse:
    if not DASHBOARD_HTML.exists():
        raise HTTPException(status_code=404, detail=f"dashboard/index.html not found at {DASHBOARD_HTML}")
    return FileResponse(DASHBOARD_HTML, media_type="text/html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")

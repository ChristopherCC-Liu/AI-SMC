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
    """
    root = _symbol_data_root(symbol)
    data = _read_json(root / "mt5_positions.json") or {}
    halt = _read_json(root / "consec_loss_state.json") or {}
    return JSONResponse({
        "ts": data.get("ts"),
        "positions": data.get("positions", []),
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
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date: {exc}")
    root = _symbol_data_root(symbol)
    digest = build_daily_digest(symbol, target, data_root=root, log_root=ROOT / "logs")
    return JSONResponse(digest)


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

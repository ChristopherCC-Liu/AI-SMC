"""
AI-SMC Dashboard Web Server (MVP).

轻量 FastAPI server, 读 data/*.json 暴露 REST API + serve dashboard/index.html.
端口 8765 (PRD § 7 契约)。

启动:
    python scripts/dashboard_server.py
然后浏览器打开 http://localhost:8765
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DASHBOARD_HTML = ROOT / "dashboard" / "index.html"

STATE_PATH = DATA / "live_state.json"
AI_PATH = DATA / "ai_analysis.json"
JOURNAL_PATH = DATA / "journal" / "live_trades.jsonl"
USER_CONFIG_PATH = DATA / "user_config.json"
PAUSE_FLAG_PATH = DATA / "trading_paused.flag"
# Round 5 T1 F2: real-time broker positions written by live_demo each cycle.
MT5_POSITIONS_PATH = DATA / "mt5_positions.json"
# Round 5 T1 F3: daily consecutive-loss halt state.
CONSEC_LOSS_PATH = DATA / "consec_loss_state.json"

STALE_THRESHOLD_SEC = 5 * 60  # 5 分钟无更新视为 stale

app = FastAPI(title="AI-SMC Dashboard", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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


@app.get("/api/state")
def get_state() -> JSONResponse:
    state = _read_json(STATE_PATH)
    ai = _read_json(AI_PATH)
    return JSONResponse({
        "state": state,
        "ai": ai,
        "freshness": _freshness(state),
        "trading_paused": PAUSE_FLAG_PATH.exists(),
        "server_time": datetime.now(timezone.utc).isoformat(),
    })


@app.get("/api/positions")
def get_positions() -> JSONResponse:
    """Round 5 T1 F2: real broker positions + halt state for the Hero card.

    ``live_demo`` writes ``data/mt5_positions.json`` at the end of each M15
    cycle via the MT5 positions adapter. Empty list (not 404) on a missing
    file keeps the dashboard usable during cold starts / market-closed.
    """
    data = _read_json(MT5_POSITIONS_PATH) or {}
    halt = _read_json(CONSEC_LOSS_PATH) or {}
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
def get_journal(limit: int = 50) -> JSONResponse:
    if not JOURNAL_PATH.exists():
        return JSONResponse({"trades": []})
    try:
        lines = JOURNAL_PATH.read_text(encoding="utf-8").splitlines()
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


@app.get("/api/config")
def get_config() -> JSONResponse:
    cfg = _read_json(USER_CONFIG_PATH) or {}
    return JSONResponse(cfg)


@app.post("/api/config")
async def set_config(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    body["updated_at"] = datetime.now(timezone.utc).isoformat()
    USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_CONFIG_PATH.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    return JSONResponse({"ok": True, "path": str(USER_CONFIG_PATH)})


@app.post("/api/toggle_trading")
async def toggle_trading(request: Request) -> JSONResponse:
    body = await request.json()
    paused = bool(body.get("paused", False))
    if paused:
        PAUSE_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        PAUSE_FLAG_PATH.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    else:
        PAUSE_FLAG_PATH.unlink(missing_ok=True)
    return JSONResponse({"paused": paused})


@app.get("/")
def index() -> FileResponse:
    if not DASHBOARD_HTML.exists():
        raise HTTPException(status_code=404, detail=f"dashboard/index.html not found at {DASHBOARD_HTML}")
    return FileResponse(DASHBOARD_HTML, media_type="text/html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")

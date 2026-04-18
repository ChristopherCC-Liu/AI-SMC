@echo off
REM AI-SMC strategy signal server (FastAPI on 127.0.0.1:8080).
REM Polled by the MQL5 EA (AISMCReceiver.mq5) running in the MT5 terminal.
REM
REM Reads data/{SYMBOL}/live_state.json written by live_demo.py and exposes
REM the latest signal over HTTP. No MT5 IPC dependency.
REM
REM Uses uvicorn CLI with workers=2 for resilience.
REM NOTE: --loop uvloop is POSIX-only; Windows uses asyncio (default).

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d C:\AI-SMC
.venv\Scripts\uvicorn.exe scripts.strategy_server:app --host 127.0.0.1 --port 8080 --workers 2 --loop asyncio --log-level warning >> logs\strategy_server_stdout.log 2>> logs\strategy_server_stderr.log

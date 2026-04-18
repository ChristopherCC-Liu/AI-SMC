@echo off
REM AI-SMC XAUUSD live_demo in SIGNAL-ONLY mode.
REM Computes strategy + writes data/XAUUSD/live_state.json for the MQL5 EA.
REM Does NOT call mt5.order_send from Python (--no-execute). EA handles
REM execution, avoiding Python ↔ MT5 multi-client IPC instability.

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
REM --no-execute short-circuits order_send even if SMC_MT5_EXECUTE=1
cd /d C:\AI-SMC
.venv\Scripts\python.exe scripts\live_demo.py --symbol XAUUSD --no-execute >> logs\live_stdout.log 2>> logs\live_stderr.log

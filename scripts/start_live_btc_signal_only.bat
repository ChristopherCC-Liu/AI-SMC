@echo off
REM AI-SMC BTCUSD live_demo in SIGNAL-ONLY mode.
REM Mirror of start_live_signal_only.bat but for BTCUSD.

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d C:\AI-SMC
.venv\Scripts\python.exe scripts\live_demo.py --symbol BTCUSD --no-execute >> logs\live_btc_stdout.log 2>> logs\live_btc_stderr.log
